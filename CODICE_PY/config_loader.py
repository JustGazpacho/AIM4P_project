"""
config_loader.py
================
Loads and builds hyperparameter dicts for DQN and PPO.
Tuned params are read from OUTPUT/reports/best_params_<algo>.json;
if the file is absent or unreadable, built-in defaults are returned.
"""

import json
import logging
from pathlib import Path

log = logging.getLogger(__name__)


# Default hyperparameters used when no tuned file exists.
DQN_DEFAULTS: dict = dict(
    learning_rate         = 1e-3,
    buffer_size           = 200_000,
    batch_size            = 64,
    gamma                 = 0.97,
    exploration_fraction  = 0.20,
    exploration_final_eps = 0.02,
    policy_kwargs         = dict(net_arch=[256, 256, 256]),
)

PPO_DEFAULTS: dict = dict(
    learning_rate  = 3e-4,
    n_steps        = 2048,
    batch_size     = 64,
    gamma          = 0.97,
    gae_lambda     = 0.95,
    clip_range     = 0.2,
    ent_coef       = 0.01,
    n_epochs       = 10,
    max_grad_norm  = 0.5,
    policy_kwargs  = dict(net_arch=dict(pi=[256, 256, 256], vf=[256, 256, 256])),
)

# Number of parallel environments used during PPO training.
PPO_N_ENVS: int = 16


def _clamp(value: float, lo: float, hi: float) -> float:
    """Clamp value to the closed interval [lo, hi]."""
    return max(lo, min(value, hi))


def _parse_arch(params: dict, default_arch: list) -> list:
    """
    Build a network architecture list from net_depth and net_width keys.
    Falls back to default_arch if either key is missing.
    """
    if "net_depth" not in params or "net_width" not in params:
        return default_arch
    depth = _clamp(int(params["net_depth"]), 1, 4)
    width = _clamp(int(params["net_width"]), 32, 512)
    return [width] * depth


def _build_dqn(params: dict) -> dict:
    """Construct a DQN config dict from raw Optuna param keys."""
    cfg = DQN_DEFAULTS.copy()
    cfg["learning_rate"]        = float(params.get("lr",       cfg["learning_rate"]))
    cfg["buffer_size"]          = int(  params.get("buffer",   cfg["buffer_size"]))
    cfg["batch_size"]           = int(  params.get("batch",    cfg["batch_size"]))
    cfg["gamma"]                = float(params.get("gamma",    cfg["gamma"]))
    cfg["exploration_fraction"] = _clamp(
        float(params.get("exp_frac", cfg["exploration_fraction"])), 0.0, 1.0
    )
    if cfg["buffer_size"] < 50_000:
        log.warning("DQN buffer_size %d < 50k; raising to 200k.", cfg["buffer_size"])
        cfg["buffer_size"] = 200_000
    arch = _parse_arch(params, cfg["policy_kwargs"]["net_arch"])
    cfg["policy_kwargs"] = dict(net_arch=arch)
    return cfg


def _build_ppo(params: dict) -> dict:
    """
    Construct a PPO config dict from raw Optuna param keys.
    Ensures batch_size never exceeds n_steps * PPO_N_ENVS.
    """
    cfg = PPO_DEFAULTS.copy()
    n_steps    = _clamp(int(  params.get("n_steps", cfg["n_steps"])),    128,  4096)
    batch_size = _clamp(int(  params.get("batch",   cfg["batch_size"])), 32,   512)
    cfg["learning_rate"] = float(params.get("lr",         cfg["learning_rate"]))
    cfg["n_steps"]       = n_steps
    cfg["gamma"]         = float(params.get("gamma",      cfg["gamma"]))
    cfg["gae_lambda"]    = float(params.get("gae_lambda", cfg["gae_lambda"]))
    cfg["clip_range"]    = float(params.get("clip_range", cfg["clip_range"]))
    cfg["ent_coef"]      = float(params.get("ent_coef",   cfg["ent_coef"]))
    arch = _parse_arch(params, cfg["policy_kwargs"]["net_arch"]["pi"])
    cfg["policy_kwargs"] = dict(net_arch=dict(pi=arch, vf=arch))
    max_batch = n_steps * PPO_N_ENVS
    if batch_size > max_batch:
        log.warning("PPO batch_size %d > n_steps*n_envs=%d; clamping.",
                    batch_size, max_batch)
        batch_size = max_batch
    cfg["batch_size"] = batch_size
    return cfg


def load_best_params(algo: str, output_dir: Path, return_source: bool = False):
    """
    Return the best hyperparameters for algo ("dqn" or "ppo").
    Reads OUTPUT/reports/best_params_<algo>.json if it exists;
    falls back to built-in defaults on any error.

    If return_source=True, returns (cfg, source, params_path) instead of
    just cfg, where source is one of:
        "tuned"            - params loaded from best_params_<algo>.json
        "default_missing"  - file not found, defaults used
        "default_error"    - file found but unreadable/corrupt, defaults used
    """
    _BUILDERS = {
        "dqn": (_build_dqn, DQN_DEFAULTS),
        "ppo": (_build_ppo, PPO_DEFAULTS),
    }
    if algo not in _BUILDERS:
        raise ValueError(f"Unknown algo '{algo}'")

    build_fn, defaults = _BUILDERS[algo]
    params_path = Path(output_dir) / "reports" / f"best_params_{algo}.json"

    def _ret(cfg, source):
        """Return cfg alone or (cfg, source, params_path) depending on return_source."""
        return (cfg, source, params_path) if return_source else cfg

    if not params_path.exists():
        log.info("No tuned params for %s — using defaults.", algo)
        return _ret(defaults.copy(), "default_missing")

    try:
        raw = json.loads(params_path.read_text(encoding="utf-8")).get("params", {})
    except (json.JSONDecodeError, OSError) as exc:
        log.warning("Could not read %s (%s) — using defaults.", params_path, exc)
        return _ret(defaults.copy(), "default_error")

    cfg = build_fn(raw)
    log.info("Loaded tuned params for %s from %s", algo, params_path)
    return _ret(cfg, "tuned")