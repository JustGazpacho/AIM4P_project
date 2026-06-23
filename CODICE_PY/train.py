"""
train.py

    Train a DQN or PPO agent and save artefacts under output_dir.

    Args:
        algo:                "dqn" or "ppo".
        total_timesteps:     total environment steps to train for.
        eval_freq:           evaluation interval in steps.
        seed:                global random seed.
        output_dir:          root folder for all outputs.
        turbulence_severity: turbulence level for training envs.
        n_path_waypoints:    number of waypoints per episode path.
        lr_schedule:         "linear" (decay to zero) or "constant".
        initial_ep_steps:    recorded but not yet wired; see set_max_steps().
        curriculum:          if True, attaches CurriculumCallback to shift
                             severity weights from easy to hard during training.

    Returns:
        Path to the final saved model (without .zip suffix).
    ...
"""

import os
os.environ["MPLBACKEND"] = "Agg"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import argparse
import heapq
import json
import time
import numpy as np

from pathlib import Path
from datetime import datetime
from typing import List, Optional

from stable_baselines3 import DQN, PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, EvalCallback
from stable_baselines3.common.env_util   import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor    import Monitor
from stable_baselines3.common.vec_env    import DummyVecEnv, VecEnv

from aircraft_pitch_env import AircraftPitchEnv
from config_loader import load_best_params, PPO_N_ENVS
from logger import JSONLLogger

_N_WAYPOINTS   = 6
_EPISODE_STEPS = AircraftPitchEnv.MAX_STEPS


def linear_schedule(initial_value: float):
    """Return a learning-rate schedule that decays linearly to zero."""
    def schedule(progress_remaining: float) -> float:
        return max(progress_remaining * initial_value, 1e-8)
    return schedule


class TopNCheckpoint:
    """
    Keeps only the N best model checkpoints on disk, discarding the worst
    whenever the heap exceeds capacity.
    """

    def __init__(self, save_dir: Path, algo: str, n: int = 3):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.algo  = algo
        self.n     = n
        self._heap = []

    def update(self, model, mean_reward: float, step: int) -> None:
        """Save model and evict the worst checkpoint if the heap is full."""
        if not np.isfinite(mean_reward):
            return
        path = self.save_dir / f"{self.algo}_ckpt_{step}_{mean_reward:.1f}.zip"
        model.save(str(path.with_suffix("")))
        heapq.heappush(self._heap, (mean_reward, path))
        if len(self._heap) > self.n:
            _, worst = heapq.heappop(self._heap)
            try:
                if worst.exists():
                    worst.unlink()
            except Exception:
                pass


class CurriculumCallback(BaseCallback):
    """
    Gradually shifts the turbulence-severity sampling distribution from
    mostly-easy to mostly-hard over the course of training.

    Weights are interpolated linearly between WEIGHTS_START (at step 0) and
    WEIGHTS_END (at the final step).  Light and moderate severities never
    drop to zero, avoiding catastrophic forgetting of easier conditions.
    Requires each environment to expose set_severity_weights().
    """

    WEIGHTS_START = np.array([0.35, 0.50, 0.15])   # [light, moderate, severe]
    WEIGHTS_END   = np.array([0.01, 0.09, 0.90])
    UPDATE_EVERY_STEPS = 2_000

    def __init__(self, verbose: int = 0):
        """
        Args:
            verbose: if 1, prints weight updates to stdout at each update step.
        """
        super().__init__(verbose)
        self._warned_unsupported = False
        self._last_update_step   = -1

    def _weights_for_progress(self, frac_done: float) -> np.ndarray:
        """Linearly interpolate between start and end weight vectors."""
        frac_done = np.clip(frac_done, 0.0, 1.0)
        w = self.WEIGHTS_START + frac_done * (self.WEIGHTS_END - self.WEIGHTS_START)
        return w / w.sum()

    def _on_step(self) -> bool:
        """
        Called at every training step. Updates severity weights every
        UPDATE_EVERY_STEPS steps based on training progress fraction.
        Always returns True (training continues).
        """
        if self.num_timesteps - self._last_update_step < self.UPDATE_EVERY_STEPS:
            return True
        self._last_update_step = self.num_timesteps
        total_timesteps = self.locals.get("total_timesteps", self.model._total_timesteps)
        frac_done = self.num_timesteps / max(total_timesteps, 1)
        weights   = self._weights_for_progress(frac_done)
        env = self.training_env
        try:
            if isinstance(env, VecEnv):
                env.env_method("set_severity_weights", weights)
            else:
                env.set_severity_weights(weights)
            if self.verbose:
                print(f"[CurriculumCallback] step={self.num_timesteps:,} "
                      f"frac={frac_done:.2f} -> weights(L/M/S)="
                      f"{weights[0]:.2f}/{weights[1]:.2f}/{weights[2]:.2f}")
        except AttributeError:
            if not self._warned_unsupported:
                print("[CurriculumCallback] WARNING: env does not expose "
                      "set_severity_weights(); curriculum is a no-op.")
                self._warned_unsupported = True
        return True


class RewardLogger(BaseCallback):
    """Collects per-episode rewards from SB3 Monitor info dicts for plotting."""

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []

    def _on_step(self) -> bool:
        """Collect episode reward from Monitor info dict if an episode just ended."""
        for info in self.locals.get("infos", []):
            if isinstance(info, dict) and "episode" in info:
                self.episode_rewards.append(info["episode"]["r"])
        return True


class RichEvalCallback(EvalCallback):
    """
    EvalCallback extended with crash-rate / stability logging and an optional
    joint stop criterion (reward threshold AND maximum crash rate).

    On each evaluation the callback computes crash_rate and pct_stable over
    n_eval_episodes deterministic rollouts and logs them to JSONLLogger.
    The best model is persisted across runs via a best_reward.json sidecar
    file so a second training run only overwrites the checkpoint if it truly
    improves on the previous best.

    Args:
        reward_threshold: stop when mean_reward >= this value (None = never stop early).
        max_crash_rate:   stop only if crash_rate <= this fraction (None = ignore).
    """

    def __init__(self, *args,
                 jsonl_logger:     Optional[JSONLLogger] = None,
                 top3:             Optional[TopNCheckpoint] = None,
                 reward_threshold: Optional[float] = None,
                 max_crash_rate:   Optional[float] = None,
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._jlogger          = jsonl_logger
        self._top3             = top3
        self._last_logged      = float("nan")
        self._reward_threshold = reward_threshold
        self._max_crash_rate   = max_crash_rate

        # Initialise best_mean_reward from a previous run if available.
        if self.best_model_save_path is not None:
            meta_path = Path(self.best_model_save_path) / "best_reward.json"
            if meta_path.exists():
                try:
                    with open(meta_path, encoding="utf-8") as mf:
                        saved = json.load(mf)
                    prev_best = float(saved.get("best_mean_reward", float("-inf")))
                    if np.isfinite(prev_best):
                        self.best_mean_reward = prev_best
                        print(f"[RichEvalCallback] Previous best: {prev_best:.2f} — "
                              "best model updated only if this run exceeds it.")
                except Exception as exc:
                    print(f"[RichEvalCallback] Could not read best_reward.json: {exc}")

    def _rollout_stats(self, n_episodes: int) -> tuple[float, float]:
        """
        Run n_episodes deterministic rollouts on eval_env and return
        (crash_rate, pct_stable).  Returns (nan, nan) on any error so the
        training run is never interrupted by a stats failure.
        """
        try:
            n_crashed    = 0
            stable_fracs = []
            obs = self.eval_env.reset()
            if isinstance(obs, tuple):
                obs = obs[0]
            for _ in range(n_episodes):
                done = False; total = 0; stable_ct = 0; crashed = False
                while not done:
                    action, _ = self.model.predict(obs, deterministic=True)
                    step_result = self.eval_env.step([int(np.asarray(action).flat[0])])
                    if len(step_result) == 5:
                        obs, _, term_v, trunc_v, info_v = step_result
                        done = bool(term_v[0]) or bool(trunc_v[0])
                    else:
                        obs, _, term_v, info_v = step_result
                        done = bool(term_v[0])
                    inf = info_v[0]; total += 1
                    if inf.get("stable", False):
                        stable_ct += 1
                    if bool(term_v[0]) and not inf.get("TimeLimit.truncated", False):
                        crashed = True
                    if done:
                        break
                n_crashed    += int(crashed)
                stable_fracs.append(stable_ct / max(total, 1))
            return n_crashed / max(n_episodes, 1), 100.0 * float(np.mean(stable_fracs))
        except Exception as exc:
            print(f"[RichEvalCallback] stat rollout failed: {exc}")
            return float("nan"), float("nan")

    def _on_step(self) -> bool:
        """
        Called after each EvalCallback evaluation trigger.
        Logs crash_rate and pct_stable, updates top-3 checkpoints,
        persists best_reward.json, and checks the joint stopping criterion.
        """
        result = super()._on_step()
        mean_r = self.last_mean_reward
        if np.isnan(mean_r) or mean_r == self._last_logged:
            return result
        self._last_logged = mean_r

        crash_rate, pct_stable = self._rollout_stats(self.n_eval_episodes)

        if self._jlogger:
            self._jlogger.log({
                "step": self.num_timesteps, "mean_reward": float(mean_r),
                "crash_rate": crash_rate, "pct_stable": pct_stable,
            })
        if self._top3:
            self._top3.update(self.model, mean_r, self.num_timesteps)

        # Persist the best reward to disk so future runs can compare against it.
        if self.best_model_save_path is not None and np.isfinite(self.best_mean_reward):
            meta_path = Path(self.best_model_save_path) / "best_reward.json"
            try:
                Path(self.best_model_save_path).mkdir(parents=True, exist_ok=True)
                with open(meta_path, "w", encoding="utf-8") as mf:
                    json.dump({"best_mean_reward": float(self.best_mean_reward)}, mf)
            except Exception as exc:
                print(f"[RichEvalCallback] Could not write best_reward.json: {exc}")

        if self._reward_threshold is not None and mean_r >= self._reward_threshold:
            crash_ok = (self._max_crash_rate is None
                        or np.isnan(crash_rate)
                        or crash_rate <= self._max_crash_rate)
            if crash_ok:
                if self.verbose >= 1:
                    print(f"[RichEvalCallback] Stopping: mean_reward={mean_r:.2f} "
                          f">= {self._reward_threshold:.2f} and "
                          f"crash_rate={crash_rate:.2%} <= {self._max_crash_rate}")
                return False
            elif self.verbose >= 1:
                print(f"[RichEvalCallback] Threshold met ({mean_r:.2f}) but "
                      f"crash_rate={crash_rate:.2%} > {self._max_crash_rate:.2%} — continuing.")
        return result


def _plot_training(rewards: List[float], algo: str, save_path: Path) -> None:
    """Save a two-panel reward-curve plot (raw + smoothed, rolling mean +/- std)."""
    rewards  = np.array(rewards)
    window   = max(1, len(rewards) // 50)
    smoothed = np.convolve(rewards, np.ones(window) / window, mode="valid")
    eps      = np.arange(len(rewards))
    sm_x     = np.arange(len(smoothed))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(f"{algo.upper()} Training", fontsize=13, fontweight="bold")

    axes[0].plot(eps, rewards,   alpha=0.25, color="#4a90d9", label="Episode reward")
    axes[0].plot(sm_x, smoothed, color="#e74c3c", lw=2, label=f"Smoothed w={window}")
    axes[0].set_xlabel("Episode"); axes[0].set_ylabel("Total reward")
    axes[0].set_title("Reward curve"); axes[0].legend(); axes[0].grid(alpha=0.3)

    if len(rewards) >= window:
        cs  = np.cumsum(np.insert(rewards, 0, 0))
        rm  = (cs[window:] - cs[:-window]) / window
        pad = np.array([rewards[:i+1].mean() for i in range(min(window-1, len(rewards)))])
        rm  = np.concatenate([pad, rm])
        rs  = np.array([rewards[max(0, i-window):i+1].std() for i in range(len(rewards))])
        axes[1].fill_between(eps, rm - rs, rm + rs, alpha=0.2, color="#27ae60")
        axes[1].plot(eps, rm, color="#27ae60", lw=2)
    axes[1].set_xlabel("Episode"); axes[1].set_ylabel("Mean reward rolling")
    axes[1].set_title("Rolling mean +/- std"); axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Reward plot saved -> {save_path}")


def train(
    algo:                str,
    total_timesteps:     int,
    eval_freq:           int,
    seed:                int,
    output_dir:          Path,
    turbulence_severity: str  = "moderate",
    n_path_waypoints:    int  = _N_WAYPOINTS,
    lr_schedule:         str  = "linear",
    initial_ep_steps:    int  = 600,
    curriculum:          bool = False,
) -> Path:
    """
    Train a DQN or PPO agent and save artefacts under output_dir.

    Returns the Path to the final saved model (without .zip suffix).
    The best model (by eval reward) is saved to models/<algo>_best/best_model.
    The top-3 checkpoints by reward are kept in models/<algo>_top3/.
    A JSONL log of evaluation metrics is written to logs/<algo>_runs.jsonl.
    A training-curve plot is saved to plots/.
    A JSON config summary is saved to reports/.
    """
    run_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    OUTPUT  = Path(output_dir).resolve()
    MODELS  = OUTPUT / "models"
    LOGS    = OUTPUT / "logs"
    PLOTS   = OUTPUT / "plots"
    REPORTS = OUTPUT / "reports"
    for folder in [OUTPUT, MODELS, LOGS, PLOTS, REPORTS]:
        folder.mkdir(parents=True, exist_ok=True)

    n_envs        = PPO_N_ENVS if algo == "ppo" else 1
    dqn_init_steps = initial_ep_steps if algo == "dqn" else _EPISODE_STEPS

    if curriculum and turbulence_severity != "random":
        print(f"[train] --curriculum forces turbulence_severity='random' "
              f"(was '{turbulence_severity}').")
        turbulence_severity = "random"

    def make_env_fn():
        """Factory returning a monitored AircraftPitchEnv for vectorised training."""
        return Monitor(AircraftPitchEnv(
            turbulence_severity=turbulence_severity,
            n_path_waypoints=n_path_waypoints,
            initial_max_steps=_EPISODE_STEPS,
        ))

    train_env = make_vec_env(make_env_fn, n_envs=n_envs, seed=seed)

    eval_severity = "random" # instead of evaluating at a single fixed turbulence level
    eval_env = DummyVecEnv([lambda: Monitor(AircraftPitchEnv(
        turbulence_severity=eval_severity,
        n_path_waypoints=n_path_waypoints,
        initial_max_steps=_EPISODE_STEPS,
    ))])
    eval_env.env_method("set_severity_weights", [0.10, 0.30, 0.60])

    params, params_source, params_path = load_best_params(
        algo, OUTPUT, return_source=True
    )
    if params_source == "tuned":
        print(f"[train] Config: iperparametri da Optuna tuning -> {params_path}")
    elif params_source == "default_missing":
        print(f"[train] Config: iperparametri DEFAULT (file non trovato: {params_path})")
    elif params_source == "default_error":
        print(f"[train] Config: iperparametri DEFAULT (file presente ma illeggibile/corrotto: {params_path})")
    params.pop("policy", None)
    if algo == "dqn":
        params["learning_starts"] = AircraftPitchEnv.DQN_LEARNING_STARTS

    base_lr = params.get("learning_rate", 3e-4 if algo == "ppo" else 1e-3)
    if lr_schedule == "linear":
        params["learning_rate"] = linear_schedule(base_lr)

    common = dict(
        policy="MlpPolicy", env=train_env,
        tensorboard_log=str(LOGS / algo), seed=seed, **params,
    )

    if algo == "dqn":
        model = DQN(**common)
    elif algo == "ppo":
        common.setdefault("max_grad_norm", 0.5)
        model = PPO(**common)
    else:
        raise ValueError(f"Unknown algorithm: {algo}")

    reward_logger = RewardLogger()
    jsonl_logger  = JSONLLogger(OUTPUT, algo)
    top3          = TopNCheckpoint(MODELS / f"{algo}_top3", algo, n=3)

    rich_eval = RichEvalCallback(
        eval_env,
        best_model_save_path = str(MODELS / f"{algo}_best"),
        log_path             = str(LOGS   / f"{algo}_eval"),
        eval_freq            = max(eval_freq // n_envs, 1),
        n_eval_episodes      = 20,
        deterministic        = True,
        verbose              = 1,
        jsonl_logger         = jsonl_logger,
        top3                 = top3,
        reward_threshold     = 7000,   
        max_crash_rate       = 0.05    
    )

    callbacks: list = [rich_eval, reward_logger]
    if curriculum:
        callbacks.append(CurriculumCallback(verbose=1))

    ep_s_init = dqn_init_steps * AircraftPitchEnv.dt
    ep_s_full = _EPISODE_STEPS * AircraftPitchEnv.dt
    dqn_ws    = params.get("learning_starts", AircraftPitchEnv.DQN_LEARNING_STARTS)

    print(f"\n{'='*60}")
    print(f"  Training {algo.upper()} — {total_timesteps:,} timesteps")
    if algo == "dqn":
        print(f"  Episode : {_EPISODE_STEPS} steps = {ep_s_full:.0f} s  "
              f"(initial_ep_steps={dqn_init_steps} not yet wired)")
        print(f"  Warmup  : {dqn_ws:,} steps")
    else:
        print(f"  Episode : {_EPISODE_STEPS} steps = {ep_s_full:.0f} s")
    print(f"  Turb    : {turbulence_severity}" +
          (f"  [curriculum: {CurriculumCallback.WEIGHTS_START.tolist()} "
           f"-> {CurriculumCallback.WEIGHTS_END.tolist()} (L/M/S)]" if curriculum else ""))
    print(f"  Eval    : every {eval_freq:,} steps  [fixed: {eval_severity}]")
    print(f"  n_envs  : {n_envs}  |  LR: {lr_schedule}  base={base_lr:.2e}")
    print(f"  Actions : {AircraftPitchEnv.N_ACTIONS}")
    print(f"  Params  : {params_source}")
    print(f"{'='*60}\n")

    t0 = time.time()
    model.learn(total_timesteps=total_timesteps,
                callback=CallbackList(callbacks), progress_bar=True)
    elapsed = time.time() - t0
    print(f"\nTraining complete in {elapsed:.1f}s  ({elapsed/60:.1f} min)")

    model_path = MODELS / f"{run_id}_{algo}_{total_timesteps}"
    model.save(model_path)
    print(f"Model saved -> {model_path}.zip")

    final_eval_env = Monitor(AircraftPitchEnv(
        turbulence_severity=eval_severity,
        n_path_waypoints=n_path_waypoints,
    ))
    final_eval_env.env.set_severity_weights([0.20, 0.40, 0.40])
    mean_r, std_r = evaluate_policy(model, final_eval_env,
                                    n_eval_episodes=20, deterministic=True)
    final_eval_env.close()

    jsonl_logger.log({
        "algo": algo, "timesteps": total_timesteps,
        "mean_reward": float(mean_r), "std_reward": float(std_r),
        "n_envs": n_envs, "eval_freq": eval_freq,
        "turbulence": turbulence_severity,
        "waypoints": n_path_waypoints,
        "lr_schedule": lr_schedule,
        "initial_ep_steps": dqn_init_steps,
        "eval_severity": eval_severity,
    })
    print(f"\nFinal eval (20 ep): {mean_r:.2f} +/- {std_r:.2f}")

    with open(REPORTS / f"{run_id}_{algo}_config.json", "w", encoding="utf-8") as f:
        json.dump({
            "algorithm": algo, "timesteps": total_timesteps, "seed": seed,
            "n_envs": n_envs, "eval_freq": eval_freq, "lr_schedule": lr_schedule,
            "base_lr": base_lr, "training_duration_s": round(elapsed, 1),
            "turbulence_severity": turbulence_severity,
            "eval_severity": eval_severity,
            "n_path_waypoints": n_path_waypoints,
            "episode_steps_initial": dqn_init_steps,
            "episode_steps_full": _EPISODE_STEPS,
            "episode_seconds_initial": ep_s_init,
            "episode_seconds_full": ep_s_full,
            "n_actions": AircraftPitchEnv.N_ACTIONS,
            "hdg_roll_coupling": AircraftPitchEnv.HDG_ROLL_COUPLING,
            "dqn_learning_starts": dqn_ws,
            "gamma": params.get("gamma", "from_config"),
            "params_source": params_source,
        }, f, indent=4)

    if reward_logger.episode_rewards:
        _plot_training(reward_logger.episode_rewards, algo,
                       PLOTS / f"{run_id}_{algo}_training.png")

    train_env.close(); eval_env.close()
    return model_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo",             default="dqn",     choices=["dqn", "ppo"])
    parser.add_argument("--timesteps",        default=1_000_000, type=int)
    parser.add_argument("--eval-freq",        default=20_000,    type=int)
    parser.add_argument("--seed",             default=42,        type=int)
    parser.add_argument("--turbulence",       default="moderate",
                        choices=["none", "light", "moderate", "severe", "random"])
    parser.add_argument("--waypoints",        default=_N_WAYPOINTS, type=int)
    parser.add_argument("--lr-schedule",      default="linear", choices=["linear", "constant"])
    parser.add_argument("--output-dir",       default="OUTPUT")
    parser.add_argument("--initial-ep-steps", default=600, type=int,
                        help="DQN: starting episode length in steps. Ignored for PPO.")
    parser.add_argument("--curriculum",       action="store_true",
                        help="Gradually increase turbulence severity during training.")
    args = parser.parse_args()

    train(
        algo                = args.algo,
        total_timesteps     = args.timesteps,
        eval_freq           = args.eval_freq,
        seed                = args.seed,
        output_dir          = Path(args.output_dir).resolve(),
        turbulence_severity = args.turbulence,
        n_path_waypoints    = args.waypoints,
        lr_schedule         = args.lr_schedule,
        initial_ep_steps    = args.initial_ep_steps,
        curriculum          = args.curriculum,
    )