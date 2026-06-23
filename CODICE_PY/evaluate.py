"""
evaluate.py
===========
Evaluate and visualise a trained RL agent on AircraftPitchEnv.

Usage:
    python evaluate.py --model OUTPUT/models/dqn_best/best_model --algo dqn
    python evaluate.py --model OUTPUT/models/ppo_best/best_model --algo ppo --episodes 20
    python evaluate.py --model OUTPUT/models/dqn_best/best_model --algo dqn --turbulence severe
    python evaluate.py --model OUTPUT/models/dqn_best/best_model --algo dqn --curriculum

Outputs (under --output-dir):
    plots/evaluation_episode_<i>.png        - per-episode time-series
    plots/evaluation_summary.png            - aggregate statistics
    plots/curriculum_summary.png            - cross-level comparison (--curriculum only)
"""

import os
os.environ["MPLBACKEND"] = "Agg"

import warnings
import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D

import argparse
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional

from stable_baselines3 import DQN, PPO
from aircraft_pitch_env import AircraftPitchEnv


TURBULENCE_LEVELS = ["light", "moderate", "severe", "random"]


def load_model(model_path: str, algo: str = None):
    """
    Load a DQN or PPO model from disk.

    Args:
        model_path: path to the saved model (without .zip).
        algo:       "dqn" or "ppo". Inferred from the path if omitted.

    Returns:
        (model, algo_str) tuple.
    """
    if algo is None:
        lower = model_path.lower()
        algo  = "ppo" if "ppo" in lower else "dqn"
        print(f"[WARNING] --algo not supplied; inferred '{algo}' from path. "
              f"Pass --algo explicitly to avoid mistakes.")
    algo = algo.lower()
    if algo == "ppo":
        print(f"Loading PPO model from {model_path}")
        return PPO.load(model_path), "ppo"
    print(f"Loading DQN model from {model_path}")
    return DQN.load(model_path), "dqn"


def run_episode(
    model,
    env: AircraftPitchEnv,
    deterministic: bool = True,
    seed: Optional[int] = None,
) -> dict:
    """
    Run one complete episode and collect per-step telemetry.

    Args:
        model:         trained SB3 model.
        env:           an AircraftPitchEnv instance (not wrapped).
        deterministic: if True, use the greedy policy (recommended for evaluation).
        seed:          optional seed passed to env.reset().

    Returns:
        dict with arrays for each telemetry key plus scalar summaries:
        total_reward, length, pct_stable, crashed, stable_thresh_deg.
    """
    obs, _ = env.reset(seed=seed)
    dt            = AircraftPitchEnv.dt
    stable_thresh = np.degrees(AircraftPitchEnv.STABLE_THRESH)

    data: dict = {k: [] for k in [
        "theta_deg", "theta_dot", "torque", "reward", "time", "stable",
        "altitude", "target_alt", "heading", "target_hdg", "required_pitch",
        "alt_error", "hdg_error", "gust_torque", "shear_torque", "thermal_torque",
    ]}
    t = 0.0
    total_reward = 0.0

    while True:
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, terminated, truncated, info = env.step(int(action))

        for key in data:
            if key == "time":
                data["time"].append(t)
            elif key == "theta_dot":
                data["theta_dot"].append(np.degrees(info["theta_dot"]))
            else:
                data[key].append(info.get(key, 0.0) if key not in ("reward", "stable")
                                 else (reward if key == "reward" else info.get("stable", False)))

        total_reward += reward
        t += dt
        if terminated or truncated:
            break

    for k in data:
        data[k] = np.array(data[k])

    data["total_reward"]      = total_reward
    data["length"]            = len(data["time"])
    data["pct_stable"]        = 100.0 * data["stable"].mean() if len(data["stable"]) > 0 else 0.0
    data["crashed"]           = bool(terminated)
    data["stable_thresh_deg"] = stable_thresh
    return data


def evaluate_curriculum(
    model,
    algo:        str,
    episodes:    int = 10,
    output_dir:  Path = None,
    render_mode: str  = None,
    no_save:     bool = False,
    seed:        Optional[int] = None,
) -> Dict[str, List[dict]]:
    """
    Evaluate the model across all turbulence levels (light, moderate, severe, random).

    For each level runs `episodes` episodes, prints a summary, optionally saves
    per-episode plots and a summary plot, and returns all episode data keyed by level.
    """
    results: Dict[str, List[dict]] = {}

    for level in TURBULENCE_LEVELS:
        print(f"\n{'─'*50}")
        print(f"  Turbulence: {level.upper()}  ({episodes} episodes)")
        print(f"{'─'*50}")

        level_data: List[dict] = []

        for i in range(episodes):
            ep_seed = None if seed is None else seed + i

            # Fresh env every episode — prevents native context corruption on Windows
            env  = AircraftPitchEnv(turbulence_severity=level, render_mode=render_mode)
            data = run_episode(model, env, seed=ep_seed)
            env.close()

            level_data.append(data)
            status = "CRASH" if data["crashed"] else "OK"
            print(f"  Ep {i + 1:3d}  reward={data['total_reward']:7.1f}  "
                  f"stable={data['pct_stable']:5.1f}%  {status}")
            if not no_save and output_dir is not None:
                plots_dir = output_dir / "plots" / f"curriculum_{level}"
                plots_dir.mkdir(parents=True, exist_ok=True)
                plot_episode(data, i, turbulence=level,
                             save_path=str(plots_dir / f"episode_{i + 1:02d}.png"))

        rewards = [d["total_reward"] for d in level_data]
        crashes = sum(d["crashed"]   for d in level_data)
        stab    = np.mean([d["pct_stable"] for d in level_data])
        print(f"  -> mean reward {np.mean(rewards):.1f} +/- {np.std(rewards):.1f}  "
              f"crashes {crashes}/{episodes}  stability {stab:.1f}%")

        if not no_save and output_dir is not None:
            plot_summary(level_data, algo, turbulence=level,
                         save_path=str(output_dir / "plots" / f"curriculum_{level}_summary.png"))

        results[level] = level_data

    if not no_save and output_dir is not None:
        plot_curriculum_comparison(results, algo,
                                   save_path=str(output_dir / "plots" / "curriculum_summary.png"))

    return results


# ------------------------------------------------------------------ #
#  Colour palettes                                                      #
# ------------------------------------------------------------------ #

COLORS = {
    "theta":    "#2980b9",
    "required": "#27ae60",
    "rate":     "#c0392b",
    "torque":   "#8e44ad",
    "reward":   "#27ae60",
    "stable":   "#f39c12",
    "altitude": "#16a085",
    "gust":     "#e74c3c",
    "shear":    "#f39c12",
    "thermal":  "#2ecc71",
}

LEVEL_COLORS = {
    "light":    "#3498db",
    "moderate": "#f39c12",
    "severe":   "#e74c3c",
    "random":   "#9b59b6",
}


def plot_episode(
    data:        dict,
    episode_idx: int,
    turbulence:  str  = "moderate",
    save_path:   Optional[str] = None,
) -> None:
    """Six-panel time-series plot for a single episode."""
    status = "CRASHED" if data["crashed"] else "Completed"
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(
        f"Episode {episode_idx + 1}  |  Turbulence: {turbulence}  |  "
        f"Total reward: {data['total_reward']:.1f}  |  "
        f"Stable: {data['pct_stable']:.1f}%  |  {status}",
        fontsize=12, fontweight="bold",
        color="#c0392b" if data["crashed"] else "#2c3e50",
    )
    gs   = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)
    t    = data["time"]
    band = data["stable_thresh_deg"]

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(t, data["theta_deg"],      color=COLORS["theta"],    lw=1.8, label="actual")
    ax1.plot(t, data["required_pitch"], color=COLORS["required"], lw=1.5, ls="--", label="required")
    ax1.fill_between(t, data["required_pitch"] - band, data["required_pitch"] + band,
                     alpha=0.12, color="#2ecc71", label=f"+/-{band:.0f} [deg] stable band")
    ax1.set_title("Pitch Angle"); ax1.set_xlabel("Time [s]"); ax1.set_ylabel("[deg]")
    ax1.legend(fontsize=8); ax1.grid(alpha=0.3)

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(t, data["theta_dot"], color=COLORS["rate"], lw=1.8)
    ax2.axhline(0, color="k", lw=0.5, ls="--")
    ax2.set_title("Pitch Rate"); ax2.set_xlabel("Time [s]"); ax2.set_ylabel("[deg/s]")
    ax2.grid(alpha=0.3)

    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(t, data["altitude"],   color=COLORS["altitude"], lw=1.8, label="Actual")
    ax3.plot(t, data["target_alt"], color="k",                lw=1.0, ls="--", label="Target")
    ax3.set_title("Altitude"); ax3.set_xlabel("Time [s]"); ax3.set_ylabel("[m]")
    ax3.legend(fontsize=8); ax3.grid(alpha=0.3)

    ax4 = fig.add_subplot(gs[1, 0])
    ax4.step(t, data["torque"], color=COLORS["torque"], lw=1.5, where="post")
    ax4.axhline(0, color="k", lw=0.5, ls="--")
    ax4.set_title("Control Torque"); ax4.set_xlabel("Time [s]"); ax4.set_ylabel("[N*m]")
    ax4.grid(alpha=0.3)

    ax5 = fig.add_subplot(gs[1, 1])
    ax5.plot(t, data["alt_error"], label="Altitude Error")
    ax5.plot(t, data["hdg_error"], label="Heading Error")
    ax5.axhline(0, color="k", lw=0.5, ls="--")
    ax5.set_title("Tracking Errors"); ax5.set_xlabel("Time [s]"); ax5.set_ylabel("Error")
    ax5.legend(fontsize=8); ax5.grid(alpha=0.3)

    ax6 = fig.add_subplot(gs[1, 2])
    ax6.plot(t, data["heading"],    color="#8e44ad", lw=1.8, label="Actual")
    ax6.plot(t, data["target_hdg"], color="k",       lw=1.0, ls="--", label="Target")
    ax6.set_title("Heading"); ax6.set_xlabel("Time [s]"); ax6.set_ylabel("[deg]")
    ax6.legend(fontsize=8); ax6.grid(alpha=0.3)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved -> {save_path}")
    else:
        plt.show()
    plt.close(fig)


def plot_summary(
    all_data:   List[dict],
    algo:       str,
    turbulence: str  = "moderate",
    save_path:  Optional[str] = None,
) -> None:
    """Three-panel aggregate summary across all evaluation episodes."""
    rewards    = [d["total_reward"] for d in all_data]
    lengths    = [d["length"]       for d in all_data]
    pct_stable = [d["pct_stable"]   for d in all_data]
    crashes    = sum(d["crashed"]   for d in all_data)
    n          = len(all_data)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(
        f"{algo.upper()} Evaluation Summary - {n} episodes  [turbulence: {turbulence}]\n"
        f"Crashes: {crashes}/{n}  |  Mean reward: {np.mean(rewards):.1f} +/- {np.std(rewards):.1f}  |  "
        f"Mean stability: {np.mean(pct_stable):.1f}%",
        fontsize=11, fontweight="bold",
    )

    axes[0].hist(rewards, bins=15, color=COLORS["reward"], edgecolor="white", alpha=0.85)
    axes[0].axvline(np.mean(rewards), color="k", ls="--", lw=1.5,
                    label=f"Mean = {np.mean(rewards):.1f}")
    axes[0].set_title("Episode Total Reward"); axes[0].set_xlabel("Reward"); axes[0].legend()

    stab_colors = ["#e74c3c" if d["crashed"] else "#27ae60" for d in all_data]
    axes[1].scatter(range(n), pct_stable, c=stab_colors, s=30, alpha=0.7)
    axes[1].axhline(np.mean(pct_stable), color="k", ls="--", lw=1.5,
                    label=f"Mean = {np.mean(pct_stable):.1f}%")
    axes[1].legend(handles=[
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#27ae60", markersize=7, label="OK"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#e74c3c", markersize=7, label="CRASH"),
        Line2D([0], [0], color="k", ls="--", lw=1.5, label=f"Mean = {np.mean(pct_stable):.1f}%"),
    ])
    axes[1].set_title("Stability per Episode"); axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("% Steps in Stable Band")
    _s_lo = max(0,   min(pct_stable) - 10)
    _s_hi = min(100, max(pct_stable) + 10)
    axes[1].set_ylim(_s_lo, _s_hi)

    axes[2].plot(range(n), lengths, color=COLORS["theta"], lw=1.5,
                 marker="o", markersize=4, alpha=0.8)
    axes[2].axhline(np.mean(lengths), color="k", ls="--", lw=1.5,
                    label=f"Mean = {np.mean(lengths):.0f}")
    axes[2].set_title("Episode Length"); axes[2].set_xlabel("Episode")
    axes[2].set_ylabel("Steps"); axes[2].legend()

    for ax in axes:
        ax.grid(alpha=0.3)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Summary saved -> {save_path}")
    else:
        plt.show()
    plt.close(fig)


def plot_curriculum_comparison(
    results:   Dict[str, List[dict]],
    algo:      str,
    save_path: Optional[str] = None,
) -> None:
    """
    Four-panel cross-level comparison for curriculum evaluation.
    Panels: mean reward +/- std, crash rate, mean stability, reward distributions.
    """
    levels   = [lv for lv in TURBULENCE_LEVELS if lv in results]
    n_levels = len(levels)

    mean_rewards = [np.mean([d["total_reward"] for d in results[lv]]) for lv in levels]
    std_rewards  = [np.std( [d["total_reward"] for d in results[lv]]) for lv in levels]
    crash_rates  = [100.0 * sum(d["crashed"] for d in results[lv]) / max(len(results[lv]), 1)
                    for lv in levels]
    mean_stab    = [np.mean([d["pct_stable"] for d in results[lv]]) for lv in levels]
    colors       = [LEVEL_COLORS[lv] for lv in levels]
    x            = np.arange(n_levels)

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    n_eps = min(len(results[lv]) for lv in levels)
    fig.suptitle(f"{algo.upper()} Curriculum Evaluation — {n_eps} episodes per level",
                 fontsize=13, fontweight="bold")

    for ax, vals, errs, title, ylabel in [
        (axes[0], mean_rewards, std_rewards, "Mean Reward +/- Std", "Total Reward"),
        (axes[1], crash_rates,  None,        "Crash Rate",         "%"),
        (axes[2], mean_stab,    None,        "Mean Stability",     "% Steps in Stable Band"),
    ]:
        bars = ax.bar(x, vals, color=colors, edgecolor="white", alpha=0.85)
        if errs:
            ax.errorbar(x, vals, yerr=errs, fmt="none", color="k", capsize=5, lw=1.5)
        ax.set_xticks(x); ax.set_xticklabels(levels)
        ax.set_title(title); ax.set_ylabel(ylabel); ax.grid(axis="y", alpha=0.3)
        if title != "Mean Reward +/- Std":
            ax.set_ylim(0, 105)
        for bar, val in zip(bars, vals):
            suffix = "%" if title != "Mean Reward +/- Std" else ""
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                    f"{val:.0f}{suffix}", ha="center", va="bottom", fontsize=9)

    all_rewards = [[d["total_reward"] for d in results[lv]] for lv in levels]
    for lv, rew, col in zip(levels, all_rewards, colors):
        axes[3].hist(rew, bins=12, alpha=0.55, color=col, label=lv, edgecolor="white")
    axes[3].set_title("Reward Distributions"); axes[3].set_xlabel("Total Reward")
    axes[3].set_ylabel("Count"); axes[3].legend(fontsize=9); axes[3].grid(alpha=0.3)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Curriculum comparison saved -> {save_path}")
    else:
        plt.show()
    plt.close(fig)


def main() -> None:
    """
    CLI entry point.
    Parses arguments, loads the model, runs evaluation (single level or full
    curriculum), saves plots and appends a JSONL record to the output directory.
    """
    parser = argparse.ArgumentParser(description="Evaluate a trained aircraft-pitch RL agent")
    parser.add_argument("--model",      required=True,
                        help="Path to saved model (without .zip).")
    parser.add_argument("--algo",       default=None, choices=["dqn", "ppo"],
                        help="Algorithm ('dqn' or 'ppo'). Inferred from path if omitted.")
    parser.add_argument("--episodes",   default=10, type=int,
                        help="Episodes per turbulence level.")
    parser.add_argument("--turbulence", default="moderate", choices=TURBULENCE_LEVELS)
    parser.add_argument("--curriculum", action="store_true",
                        help="Evaluate across all turbulence levels.")
    parser.add_argument("--seed",       default=None, type=int)
    parser.add_argument("--render",     action="store_true")
    parser.add_argument("--no-save",    action="store_true")
    parser.add_argument("--output-dir", default="OUTPUT")
    parser.add_argument("--plot-dir",   default=None,
                        help="Override folder where episode plots are saved.")
    args = parser.parse_args()

    OUTPUT = Path(args.output_dir).resolve()
    PLOTS  = OUTPUT / "plots"
    PLOTS.mkdir(parents=True, exist_ok=True)

    model, algo = load_model(args.model, algo=args.algo)

    if args.curriculum:
        print(f"\nCurriculum evaluation: {args.episodes} episodes × {len(TURBULENCE_LEVELS)} levels")
        evaluate_curriculum(
            model, algo=algo, episodes=args.episodes,
            output_dir=None if args.no_save else OUTPUT,
            render_mode="human" if args.render else None,
            no_save=args.no_save, seed=args.seed,
        )
        return

    print(f"\nCollecting {args.episodes} episodes  [turbulence: {args.turbulence}]")
    all_data = []
    plot_folder = None

    for i in range(args.episodes):
        ep_seed = None if args.seed is None else args.seed + i

        # Fresh env every episode — prevents native context corruption on Windows
        env  = AircraftPitchEnv(
            turbulence_severity=args.turbulence,
            render_mode="human" if args.render else None,
        )
        data = run_episode(model, env, seed=ep_seed)
        env.close()

        all_data.append(data)
        status = "CRASH" if data["crashed"] else "OK"
        print(f"  Ep {i + 1:3d}  reward={data['total_reward']:7.1f}  "
              f"stable={data['pct_stable']:5.1f}%  {status}")
        if not args.no_save and not args.render:
            if args.plot_dir:
                plot_folder = Path(args.plot_dir)
            else:
                model_name = Path(args.model).parent.name
                plot_folder = PLOTS / f"{model_name}_{args.turbulence}"
            plot_folder.mkdir(parents=True, exist_ok=True)
            plot_episode(data, i, turbulence=args.turbulence,
                         save_path=str(plot_folder / f"episode_{i + 1:02d}.png"))
        else:
            plot_episode(data, i, turbulence=args.turbulence, save_path=None)

    rewards = [d["total_reward"] for d in all_data]
    print(f"\nEvaluation ({args.episodes} eps): "
          f"{np.mean(rewards):.2f} +/- {np.std(rewards):.2f}  "
          f"crashes={sum(d['crashed'] for d in all_data)}/{args.episodes}")

    summary_path = (None if args.no_save
                    else str(plot_folder / "summary.png") if plot_folder else None)
    plot_summary(all_data, algo, turbulence=args.turbulence, save_path=summary_path)

    if not args.no_save:
        from logger import JSONLLogger
        jlogger = JSONLLogger(OUTPUT, algo)
        jlogger.log({
            "mean_reward": float(np.mean(rewards)),
            "std_reward":  float(np.std(rewards)),
            "turbulence":  args.turbulence,
            "n_episodes":  args.episodes,
        })


if __name__ == "__main__":
    main()