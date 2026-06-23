"""
long_test_episode.py
====================
2-hour test episode with randomly changing turbulence severity.

The 120-minute session is divided into variable-length segments;
each segment uses a different severity than the previous one
(light / moderate / severe), chosen at random. At the end, the
following are saved:
  - a multi-panel plot  ->  OUTPUT/plots/long_test_<timestamp>.png
  - a JSONL record      ->  OUTPUT/logs/long_test_runs.jsonl

Usage (from the command line):
    python long_test_episode.py --model OUTPUT/models/dqn_best/best_model --algo dqn
    python long_test_episode.py --model OUTPUT/models/ppo_best/best_model --algo ppo --seed 7
    python long_test_episode.py --model OUTPUT/models/dqn_best/best_model --algo dqn \
        --seg-min 5 --seg-max 20 --output-dir OUTPUT --no-save

Main parameters:
    --model         path to the saved model (without .zip)
    --algo          dqn | ppo
    --seed          global seed for the severity sequence  (default: 42)
    --seg-min       minimum segment duration in minutes    (default: 8)
    --seg-max       maximum segment duration in minutes    (default: 18)
    --output-dir    root output folder                     (default: OUTPUT)
    --no-save       do not save plot or JSONL (useful for quick tests)
    --deterministic use greedy policy (default: True)
"""

import os
os.environ["MPLBACKEND"] = "Agg"

import argparse
import json
import warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches

from datetime import datetime, timezone
from pathlib import Path
from typing import List, Tuple

from stable_baselines3 import DQN, PPO
from aircraft_pitch_env import AircraftPitchEnv


SEVERITIES    = ["light", "moderate", "severe"]
SEV_COLORS    = {"light": "#639922", "moderate": "#BA7517", "severe": "#E24B4A"}
SEV_ALPHA_BG  = 0.12
TOTAL_MINUTES = 120.0
DT            = AircraftPitchEnv.dt            # 0.05 s
STEPS_PER_MIN = 60.0 / DT                      # 1200 steps / min
TOTAL_STEPS   = int(TOTAL_MINUTES * STEPS_PER_MIN)  # 144 000


# ------------------------------------------------------------------ #
#  Segment generation                                                  #
# ------------------------------------------------------------------ #

def generate_segments(
    seed:    int   = 42,
    seg_min: float = 8.0,
    seg_max: float = 18.0,
) -> List[dict]:
    """
    Divide 120 minutes into segments of random duration in [seg_min, seg_max].
    Each segment's severity is chosen uniformly from the three options, but
    cannot repeat the previous segment's severity.

    Returns:
        List of dicts with keys: start_min, end_min, start_step, end_step, severity.
    """
    rng  = np.random.default_rng(seed)
    segs = []
    t    = 0.0
    prev = None

    while t < TOTAL_MINUTES:
        candidates = [s for s in SEVERITIES if s != prev]
        sev        = rng.choice(candidates)
        dur        = float(rng.uniform(seg_min, seg_max))
        dur        = min(dur, TOTAL_MINUTES - t)
        if dur <= 0:
            break

        start_step = int(round(t          * STEPS_PER_MIN))
        end_step   = int(round((t + dur)   * STEPS_PER_MIN))
        segs.append({
            "start_min":  round(t,      3),
            "end_min":    round(t + dur, 3),
            "start_step": start_step,
            "end_step":   end_step,
            "severity":   sev,
        })
        prev = sev
        t   += dur

    return segs


def severity_at_step(segs: List[dict], step: int) -> str:
    """Returns the active severity at the given step."""
    for s in segs:
        if s["start_step"] <= step < s["end_step"]:
            return s["severity"]
    return segs[-1]["severity"]


# ------------------------------------------------------------------ #
#  Model loading                                                       #
# ------------------------------------------------------------------ #

def load_model(model_path: str, algo: str = None):
    if algo is None:
        algo = "ppo" if "ppo" in model_path.lower() else "dqn"
        print(f"[WARNING] --algo not specified; inferred '{algo}' from path.")
    algo = algo.lower()
    cls  = PPO if algo == "ppo" else DQN
    print(f"Loading {algo.upper()} model from {model_path}")
    return cls.load(model_path), algo


# ------------------------------------------------------------------ #
#  Run                                                                 #
# ------------------------------------------------------------------ #

def run_long_test(
    model,
    segs:         List[dict],
    deterministic: bool = True,
    seed:         int   = 42,
    verbose:      bool  = True,
) -> dict:
    """
    Runs the 2-hour episode, changing the environment severity at each
    segment transition.

    Returns:
        Dict with step-by-step telemetry and aggregate statistics.
    """
    env = AircraftPitchEnv(
        turbulence_severity = segs[0]["severity"],
        initial_max_steps   = TOTAL_STEPS,
    )
    obs, _ = env.reset(seed=seed)
    env.set_turbulence(segs[0]["severity"])

    keys = [
        "theta_deg", "theta_dot", "torque", "reward", "time",
        "altitude",  "target_alt", "heading", "target_hdg",
        "required_pitch", "alt_error", "hdg_error",
        "gust_torque", "shear_torque", "thermal_torque", "severity",
    ]
    data = {k: [] for k in keys}

    current_seg_idx = 0
    total_reward    = 0.0
    step            = 0
    t_sim           = 0.0
    severity_changes: List[dict] = []

    if verbose:
        print(f"\n{'='*60}")
        print(f"  LONG TEST - {TOTAL_MINUTES:.0f} min  |  {len(segs)} segments  |  seed={seed}")
        print(f"{'='*60}")
        for i, s in enumerate(segs):
            print(f"  seg {i+1:2d}: {s['start_min']:6.1f}–{s['end_min']:6.1f} min "
                  f"({s['end_step']-s['start_step']:5d} steps)  {s['severity']}")
        print(f"{'='*60}\n")

    while True:
        # Check for segment transition
        new_seg_idx = 0
        for i, s in enumerate(segs):
            if s["start_step"] <= step < s["end_step"]:
                new_seg_idx = i
                break
        else:
            new_seg_idx = len(segs) - 1

        if new_seg_idx != current_seg_idx:
            new_sev = segs[new_seg_idx]["severity"]
            old_sev = segs[current_seg_idx]["severity"]
            env.set_turbulence(new_sev)
            current_seg_idx = new_seg_idx
            severity_changes.append({
                "step":     step,
                "time_min": round(t_sim / 60.0, 2),
                "from":     old_sev,
                "to":       new_sev,
            })
            if verbose:
                print(f"  t={t_sim/60:.1f} min  severity change: {old_sev} → {new_sev}")

        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, terminated, truncated, info = env.step(int(action))

        sev = segs[current_seg_idx]["severity"]
        data["theta_deg"].append(info["theta_deg"])
        data["theta_dot"].append(np.degrees(info["theta_dot"]))
        data["torque"].append(info["torque"])
        data["reward"].append(reward)
        data["time"].append(t_sim)
        data["altitude"].append(info["altitude"])
        data["target_alt"].append(info["target_alt"])
        data["heading"].append(info["heading"])
        data["target_hdg"].append(info["target_hdg"])
        data["required_pitch"].append(info["required_pitch"])
        data["alt_error"].append(info["alt_error"])
        data["hdg_error"].append(info["hdg_error"])
        data["gust_torque"].append(info["gust_torque"])
        data["shear_torque"].append(info["shear_torque"])
        data["thermal_torque"].append(info["thermal_torque"])
        data["severity"].append(sev)

        total_reward += reward
        step         += 1
        t_sim         = step * DT

        if terminated:
            print("\n[CRASH DEBUG]")
            print(info)
            print("step:", step)
            print("theta:", info["theta_deg"])
            print("altitude:", info["altitude"])
            print("alt_error:", info["alt_error"])
            break

        if terminated or truncated:
            break

    env.close()

    for k in keys:
        data[k] = np.array(data[k])

    stable_thresh = np.degrees(AircraftPitchEnv.STABLE_THRESH)
    stable_mask   = np.abs(data["theta_deg"] - data["required_pitch"]) < stable_thresh

    # per-severity statistics
    per_sev = {}
    for sev in SEVERITIES:
        mask = data["severity"] == sev
        if mask.sum() == 0:
            continue
        per_sev[sev] = {
            "steps":       int(mask.sum()),
            "pct_stable":  float(100.0 * (stable_mask & mask).sum() / mask.sum()),
            "mean_reward": float(data["reward"][mask].mean()),
            "mean_alt_err":float(np.abs(data["alt_error"][mask]).mean()),
        }

    data["total_reward"]      = float(total_reward)
    data["n_steps"]           = int(step)
    data["crashed"]           = bool(terminated)
    data["pct_stable"]        = float(100.0 * stable_mask.mean())
    data["severity_changes"]  = severity_changes
    data["segments"]          = segs
    data["per_severity_stats"]= per_sev
    data["stable_thresh_deg"] = stable_thresh

    if verbose:
        status = "CRASH" if terminated else "completed"
        print(f"\n  Result: {status}")
        print(f"  Total reward    : {total_reward:.1f}")
        print(f"  Steps executed  : {step}")
        print(f"  Global stability: {data['pct_stable']:.1f}%")
        print(f"  Severity changes: {len(severity_changes)}")
        for sev, st in per_sev.items():
            print(f"    {sev:8s}: {st['pct_stable']:.1f}% stable  "
                  f"mean_reward={st['mean_reward']:.2f}  "
                  f"alt_err={st['mean_alt_err']:.1f} m")

    return data


# ------------------------------------------------------------------ #
#  Plot                                                                #
# ------------------------------------------------------------------ #

def plot_long_test(
    data:      dict,
    algo:      str,
    seed:      int,
    save_path: str = None,
) -> None:
    """
    6-panel plot:
      1. Pitch angle vs required  (background shaded by severity)
      2. Severity over time (step chart)
      3. Altitude vs target
      4. Cumulative reward
      5. Total disturbance (gust + shear + thermal)
      6. Control torque
    """
    segs    = data["segments"]
    t_min   = data["time"] / 60.0          # x-axis in minutes
    status  = "CRASH" if data["crashed"] else "completed"
    n_chg   = len(data["severity_changes"])

    fig = plt.figure(figsize=(18, 12))
    fig.suptitle(
        f"{algo.upper()} - Long test 2h  |  seed={seed}  |  "
        f"reward={data['total_reward']:.0f}  |  "
        f"stability={data['pct_stable']:.1f}%  |  "
        f"turb. changes={n_chg}  |  {status}",
        fontsize=11, fontweight="bold",
        color="#c0392b" if data["crashed"] else "#2c3e50",
    )
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.32)

    def shade_background(ax, y_min, y_max):
        for s in segs:
            x0 = s["start_min"]
            x1 = s["end_min"]
            ax.axvspan(x0, x1, alpha=SEV_ALPHA_BG,
                       color=SEV_COLORS[s["severity"]], lw=0)

    # --- 1. Pitch ---
    ax1 = fig.add_subplot(gs[0, 0])
    shade_background(ax1, None, None)
    ax1.plot(t_min, data["theta_deg"],      color="#2980b9", lw=0.8, label="pitch")
    ax1.plot(t_min, data["required_pitch"], color="#27ae60", lw=0.8, ls="--", label="required")
    band = data["stable_thresh_deg"]
    ax1.fill_between(t_min,
                     data["required_pitch"] - band,
                     data["required_pitch"] + band,
                     alpha=0.1, color="#27ae60")
    ax1.set_title("Pitch angle"); ax1.set_xlabel("Time [min]")
    ax1.set_ylabel("[deg]"); ax1.legend(fontsize=8); ax1.grid(alpha=0.25)

    # --- 2. Severity ---
    SEV_NUM = {"light": 1, "moderate": 2, "severe": 3}
    sev_num = np.array([SEV_NUM[s] for s in data["severity"]])
    ax2 = fig.add_subplot(gs[0, 1])
    shade_background(ax2, None, None)
    ax2.step(t_min, sev_num, color="#8e44ad", lw=1.2, where="post")
    ax2.set_yticks([1, 2, 3]); ax2.set_yticklabels(["light", "moderate", "severe"])
    ax2.set_title("Turbulence severity"); ax2.set_xlabel("Time [min]")
    ax2.set_ylim(0.5, 3.5); ax2.grid(alpha=0.25)
    patches = [mpatches.Patch(color=SEV_COLORS[s], alpha=0.5, label=s) for s in SEVERITIES]
    ax2.legend(handles=patches, fontsize=8, loc="upper right")

    # --- 3. Altitude ---
    ax3 = fig.add_subplot(gs[1, 0])
    shade_background(ax3, None, None)
    ax3.plot(t_min, data["altitude"],   color="#16a085", lw=0.8, label="actual")
    ax3.plot(t_min, data["target_alt"], color="k",       lw=0.8, ls="--", label="target", alpha=0.6)
    ax3.set_title("Altitude"); ax3.set_xlabel("Time [min]")
    ax3.set_ylabel("[m]"); ax3.legend(fontsize=8); ax3.grid(alpha=0.25)

    # --- 4. Cumulative reward ---
    ax4 = fig.add_subplot(gs[1, 1])
    shade_background(ax4, None, None)
    ax4.plot(t_min, np.cumsum(data["reward"]), color="#27ae60", lw=1.0)
    ax4.set_title("Cumulative reward"); ax4.set_xlabel("Time [min]")
    ax4.set_ylabel("Σ reward"); ax4.grid(alpha=0.25)

    # --- 5. Total disturbance ---
    total_dist = data["gust_torque"] + data["shear_torque"] + data["thermal_torque"]
    ax5 = fig.add_subplot(gs[2, 0])
    shade_background(ax5, None, None)
    ax5.plot(t_min, data["gust_torque"],    color="#e74c3c", lw=0.6, alpha=0.7, label="gust")
    ax5.plot(t_min, data["shear_torque"],   color="#f39c12", lw=0.6, alpha=0.7, label="shear")
    ax5.plot(t_min, data["thermal_torque"], color="#2ecc71", lw=0.6, alpha=0.7, label="thermal")
    ax5.plot(t_min, total_dist,             color="#2c3e50", lw=0.9, label="total")
    ax5.axhline(0, color="k", lw=0.4, ls="--")
    ax5.set_title("Atmospheric disturbance"); ax5.set_xlabel("Time [min]")
    ax5.set_ylabel("[N·m]"); ax5.legend(fontsize=7); ax5.grid(alpha=0.25)

    # --- 6. Control torque ---
    ax6 = fig.add_subplot(gs[2, 1])
    shade_background(ax6, None, None)
    ax6.step(t_min, data["torque"], color="#8e44ad", lw=0.8, where="post")
    ax6.axhline(0, color="k", lw=0.4, ls="--")
    ax6.set_title("Control torque"); ax6.set_xlabel("Time [min]")
    ax6.set_ylabel("[N·m]"); ax6.grid(alpha=0.25)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved -> {save_path}")
    else:
        plt.show()
    plt.close(fig)


# ------------------------------------------------------------------ #
#  JSONL logging                                                       #
# ------------------------------------------------------------------ #

class _NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):   return int(obj)
        if isinstance(obj, (np.floating,)):  return float(obj)
        if isinstance(obj, np.ndarray):      return obj.tolist()
        return super().default(obj)


def log_jsonl(output_dir: Path, algo: str, record: dict) -> None:
    log_path = output_dir / "logs" / "long_test_runs.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    record["time"] = datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    with log_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, cls=_NumpyEncoder, ensure_ascii=False) + "\n")
    print(f"JSONL saved -> {log_path}")


# ------------------------------------------------------------------ #
#  CLI                                                                 #
# ------------------------------------------------------------------ #

def main() -> None:
    parser = argparse.ArgumentParser(
        description="2-hour test episode with variable turbulence severity"
    )
    parser.add_argument("--model",       required=True,
                        help="Path to the saved model (without .zip).")
    parser.add_argument("--algo",        default=None, choices=["dqn", "ppo"])
    parser.add_argument("--seed",        default=42,   type=int,
                        help="Seed for segment generation.")
    parser.add_argument("--seg-min",     default=8.0,  type=float,
                        help="Minimum segment duration [min] (default 8).")
    parser.add_argument("--seg-max",     default=18.0, type=float,
                        help="Maximum segment duration [min] (default 18).")
    parser.add_argument("--output-dir",  default="OUTPUT")
    parser.add_argument("--no-save",     action="store_true",
                        help="Do not save plot or JSONL.")
    parser.add_argument("--no-det",      action="store_true",
                        help="Use stochastic policy instead of greedy.")
    args = parser.parse_args()

    OUTPUT     = Path(args.output_dir).resolve()
    PLOTS      = OUTPUT / "plots"
    PLOTS.mkdir(parents=True, exist_ok=True)

    model, algo = load_model(args.model, algo=args.algo)

    segs = generate_segments(
        seed    = args.seed,
        seg_min = args.seg_min,
        seg_max = args.seg_max,
    )

    data = run_long_test(
        model,
        segs,
        deterministic = not args.no_det,
        seed          = args.seed,
        verbose       = True,
    )

    if not args.no_save:
        ts        = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        plot_path = str(PLOTS / f"long_test_{algo}_{ts}.png")
        plot_long_test(data, algo, args.seed, save_path=plot_path)

        log_jsonl(OUTPUT, algo, {
            "algo":            algo,
            "seed":            args.seed,
            "seg_min":         args.seg_min,
            "seg_max":         args.seg_max,
            "n_segments":      len(segs),
            "n_steps":         data["n_steps"],
            "total_reward":    data["total_reward"],
            "pct_stable":      data["pct_stable"],
            "crashed":         data["crashed"],
            "n_sev_changes":   len(data["severity_changes"]),
            "per_severity":    data["per_severity_stats"],
            "severity_changes":data["severity_changes"],
            "plot_path":       plot_path,
        })
    else:
        plot_long_test(data, algo, args.seed, save_path=None)


if __name__ == "__main__":
    main()