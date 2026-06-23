"""
tune.py
=======
Hyperparameter search with Optuna (Bayesian TPE sampler + MedianPruner).
Finds optimal DQN or PPO hyperparameters and saves results under --output-dir.

Usage:
    python tune.py --algo dqn --trials 25 --timesteps 400000
    python tune.py --algo ppo --trials 25 --timesteps 400000

Outputs:
    OUTPUT/reports/best_params_<algo>.json
    OUTPUT/reports/optuna_<algo>.db
    OUTPUT/plots/optuna_study_<algo>.png
"""

import argparse
import json
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="stable_baselines3")

import numpy as np
import optuna
from optuna.samplers import TPESampler
from optuna.pruners  import MedianPruner
from optuna.trial    import TrialState
from pathlib import Path

from stable_baselines3 import DQN, PPO
from stable_baselines3.common.env_util   import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor    import Monitor

from aircraft_pitch_env import AircraftPitchEnv
from config_loader import PPO_N_ENVS

_EPISODE_STEPS    = AircraftPitchEnv.MAX_STEPS
_N_WAYPOINTS      = 6

# PPO tuning uses 8 envs: a compromise between trial speed (fewer envs = shorter
# wall time) and consistency with training at PPO_N_ENVS=16.
_TUNE_PPO_N_ENVS = 8


def make_env():
    """Single monitored environment for tuning (shared by DQN and PPO)."""
    return Monitor(AircraftPitchEnv(
        n_path_waypoints=_N_WAYPOINTS,
        initial_max_steps=_EPISODE_STEPS,
    ))


def dqn_objective(trial: optuna.Trial, timesteps: int) -> float:
    """
    Optuna objective for DQN.
    Trains a model for `timesteps` steps, reports intermediate mean reward
    every episode for pruning, and returns the final 10-episode mean reward.
    """
    lr          = trial.suggest_float("lr",          5e-5,  5e-3, log=True)
    buffer_size = trial.suggest_categorical("buffer", [100_000, 200_000, 300_000])
    batch_size  = trial.suggest_categorical("batch",  [32, 64, 128])
    gamma       = trial.suggest_categorical("gamma",  [0.95, 0.97, 0.99, 0.995])
    exp_frac    = trial.suggest_float("exp_frac",     0.10, 0.40)
    net_depth   = trial.suggest_int("net_depth",      1, 3)
    net_width   = trial.suggest_categorical("net_width", [128, 256, 512])
    net_arch    = [net_width] * net_depth
    trial.set_user_attr("algo", "dqn")

    env      = make_vec_env(make_env, n_envs=1, seed=trial.number)
    eval_env = Monitor(make_env())

    tune_learning_starts = 4_000 # lower than production (24k) to speed up tuning trials

    model = DQN(
        "MlpPolicy", env,
        learning_rate         = lr,
        buffer_size           = buffer_size,
        batch_size            = batch_size,
        gamma                 = gamma,
        exploration_fraction  = exp_frac,
        exploration_final_eps = 0.02,
        learning_starts       = tune_learning_starts,
        policy_kwargs         = dict(net_arch=net_arch),
        verbose               = 0,
        seed                  = trial.number,
    )

    report_interval = _EPISODE_STEPS
    first_chunk     = max(report_interval, tune_learning_starts + report_interval)
    steps_done      = 0
    is_first_chunk  = True
    while steps_done < timesteps:
        chunk = first_chunk if is_first_chunk else report_interval
        chunk = min(chunk, timesteps - steps_done)
        is_first_chunk = False
        model.learn(total_timesteps=chunk, reset_num_timesteps=(steps_done == 0),
                    progress_bar=False)
        steps_done += chunk
        mean_r, _ = evaluate_policy(model, eval_env, n_eval_episodes=3,
                                    deterministic=True, warn=False)
        trial.report(mean_r, step=steps_done)
        if trial.should_prune():
            env.close(); eval_env.close()
            raise optuna.exceptions.TrialPruned()

    mean_reward, _ = evaluate_policy(model, eval_env, n_eval_episodes=10,
                                     deterministic=True, warn=False)
    env.close(); eval_env.close()
    trial.set_user_attr("timesteps", timesteps)
    trial.set_user_attr("reward",    float(mean_reward))
    return mean_reward


def ppo_objective(trial: optuna.Trial, timesteps: int) -> float:
    """
    Optuna objective for PPO.
    Trains with _TUNE_PPO_N_ENVS parallel environments, reports every
    2 full policy updates for pruning, and returns the 10-episode mean reward.
    Trials with batch_size > n_steps * n_envs are immediately pruned.
    """
    lr         = trial.suggest_float("lr",            5e-5, 1e-3, log=True)
    n_steps    = trial.suggest_categorical("n_steps",  [512, 1024, 2048])
    batch_size = trial.suggest_categorical("batch",    [64, 128, 256])
    gamma      = trial.suggest_categorical("gamma",    [0.95, 0.97, 0.99, 0.995])
    gae_lambda = trial.suggest_float("gae_lambda",     0.85, 0.99)
    clip_range = trial.suggest_float("clip_range",     0.1,  0.4)
    ent_coef   = trial.suggest_float("ent_coef",       1e-5, 0.02, log=True)
    net_depth  = trial.suggest_int("net_depth",        1, 3)
    net_width  = trial.suggest_categorical("net_width", [128, 256, 512])
    net_arch   = [net_width] * net_depth
    trial.set_user_attr("algo", "ppo")

    n_envs_tune = _TUNE_PPO_N_ENVS
    if batch_size > n_steps * n_envs_tune:
        raise optuna.exceptions.TrialPruned()

    env      = make_vec_env(make_env, n_envs=n_envs_tune, seed=trial.number)
    eval_env = Monitor(make_env())

    model = PPO(
        "MlpPolicy", env,
        learning_rate  = lr,
        n_steps        = n_steps,
        batch_size     = batch_size,
        gamma          = gamma,
        gae_lambda     = gae_lambda,
        clip_range     = clip_range,
        ent_coef       = ent_coef,
        n_epochs       = 10,
        max_grad_norm  = 0.5,
        policy_kwargs  = dict(net_arch=dict(pi=net_arch, vf=net_arch)),
        verbose        = 0,
        seed           = trial.number,
    )

    # Report every 2 policy updates to balance pruner granularity against
    # the high noise of individual PPO updates.
    report_interval = n_steps * n_envs_tune * 2
    steps_done      = 0
    while steps_done < timesteps:
        chunk = min(report_interval, timesteps - steps_done)
        model.learn(total_timesteps=chunk, reset_num_timesteps=(steps_done == 0),
                    progress_bar=False)
        steps_done += chunk
        mean_r, _ = evaluate_policy(model, eval_env, n_eval_episodes=3,
                                    deterministic=True, warn=False)
        trial.report(mean_r, step=steps_done)
        if trial.should_prune():
            env.close(); eval_env.close()
            raise optuna.exceptions.TrialPruned()

    mean_reward, _ = evaluate_policy(model, eval_env, n_eval_episodes=10,
                                     deterministic=True, warn=False)
    env.close(); eval_env.close()
    trial.set_user_attr("timesteps", timesteps)
    trial.set_user_attr("reward",    float(mean_reward))
    return mean_reward


def main():
    """
    CLI entry point for tune.py.
    Creates an Optuna study, runs the requested number of trials for the
    chosen algorithm, saves the best hyperparameters to JSON, and plots
    trial rewards and best-so-far curves.
    """
    parser = argparse.ArgumentParser(description="Hyperparameter tuning with Optuna")
    parser.add_argument("--algo",       default="dqn",  choices=["dqn", "ppo"])
    parser.add_argument("--trials",     default=25,     type=int)
    parser.add_argument("--timesteps",  default=None,   type=int)
    parser.add_argument("--jobs",       default=1,      type=int)
    parser.add_argument("--output-dir", default="OUTPUT")
    args = parser.parse_args()

    if args.timesteps is None:
        args.timesteps = 400_000

    OUTPUT  = Path(args.output_dir).resolve()
    PLOTS   = OUTPUT / "plots"
    REPORTS = OUTPUT / "reports"
    for folder in [PLOTS, REPORTS]:
        folder.mkdir(parents=True, exist_ok=True)

    objective_fn = dqn_objective if args.algo == "dqn" else ppo_objective

    db_path = REPORTS / f"optuna_{args.algo}.db"
    storage = f"sqlite:///{db_path}"
    if db_path.exists():
        print(f"[tune] Removing stale study DB: {db_path}")
        db_path.unlink()

    sampler_startup = max(5, args.trials // 3)

    # Pruner warm-up: DQN reports every ~2 400 steps so warmup=20 activates
    # the pruner after ~48 000 steps (after learning has started).  PPO reports
    # less frequently so warmup=8 is sufficient.
    pruner_warmup = 20 if args.algo == "dqn" else 8
    study = optuna.create_study(
        direction  = "maximize",
        sampler    = TPESampler(seed=42, n_startup_trials=sampler_startup),
        pruner     = MedianPruner(n_startup_trials=sampler_startup,
                                   n_warmup_steps=pruner_warmup),
        study_name = f"{args.algo}_aircraft_pitch",
        storage    = storage,
    )

    study.optimize(
        lambda trial: objective_fn(trial, args.timesteps),
        n_trials          = args.trials,
        n_jobs            = args.jobs,
        show_progress_bar = True,
    )

    best = study.best_params
    print(f"\n{'='*50}")
    print(f"Best trial:  value = {study.best_value:.2f}")
    print(f"Best params: {json.dumps(best, indent=2)}")

    out = REPORTS / f"best_params_{args.algo}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"best_value": study.best_value, "params": best}, f, indent=2)
    print(f"Saved -> {out}")

    try:
        import matplotlib.pyplot as plt
        complete = [t for t in study.trials
                    if t.state == TrialState.COMPLETE and t.value is not None]
        pruned   = [t for t in study.trials if t.state == TrialState.PRUNED]
        values   = [t.value for t in complete]
        if not values:
            print("No complete trials to plot.")
            return
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].plot(values, marker="o", ms=4, color="#2980b9")
        axes[0].set_title("Trial rewards"); axes[0].set_xlabel("Trial"); axes[0].grid(alpha=0.3)
        best_so_far = np.maximum.accumulate(values)
        axes[1].plot(best_so_far, color="#e74c3c", lw=2)
        axes[1].set_title("Best reward so far"); axes[1].set_xlabel("Trial"); axes[1].grid(alpha=0.3)
        fig.suptitle(
            f"{args.algo.upper()} search ({len(complete)} complete, {len(pruned)} pruned)",
            fontweight="bold",
        )
        plt.tight_layout()
        plot_path = PLOTS / f"optuna_study_{args.algo}.png"
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"Plot saved -> {plot_path}")
    except Exception as e:
        print(f"Could not save plot: {e}")


if __name__ == "__main__":
    main()