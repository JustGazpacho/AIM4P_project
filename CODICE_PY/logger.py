"""
logger.py
=========
Append-only JSONL run logger.
Each call to .log() appends one JSON line to OUTPUT/logs/<algo>_runs.jsonl.
The file is human-readable, grep-friendly, and resumable across sessions.
"""

import json
import logging
import numpy as np
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

log = logging.getLogger(__name__)


class _NumpyEncoder(json.JSONEncoder):
    """JSON encoder that silently converts numpy scalars and arrays to Python types."""
    def default(self, obj):
        """Convert numpy scalars/arrays to native Python types for JSON serialisation."""
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


class JSONLLogger:
    """
    Append-only logger that writes one JSON object per line.

    Each record is a flat dict of scalars plus an ISO-8601 UTC timestamp
    under the key "time".  Numpy scalars are silently coerced to Python
    primitives via _NumpyEncoder.

    Args:
        output_dir: root output folder (e.g. Path("OUTPUT"))
        algo:       algorithm name used as part of the filename ("dqn"/"ppo")
    """

    def __init__(self, output_dir: Path, algo: str) -> None:
        """
        Args:
            output_dir: root output folder (e.g. Path("OUTPUT")).
            algo:       algorithm tag used in the filename ("dqn" or "ppo").
                        File will be written to output_dir/logs/<algo>_runs.jsonl.
        """
        self.path = Path(output_dir) / "logs" / f"{algo}_runs.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, data: dict) -> None:
        """Append data as a JSON line with a UTC timestamp. Silently skips on I/O errors."""
        record = {**data, "time": _utc_now()}
        try:
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, cls=_NumpyEncoder, ensure_ascii=False))
                fh.write("\n")
        except OSError as exc:
            log.warning("JSONLLogger: could not write to %s - %s", self.path, exc)

    def read_all(self) -> list[dict]:
        """Return all logged records as a list of dicts."""
        return list(self.iter_records())

    def iter_records(self) -> Iterator[dict]:
        """Yield records one at a time without loading the whole file into memory."""
        if not self.path.exists():
            return iter(())
        with self.path.open("r", encoding="utf-8") as fh:
            for i, raw in enumerate(fh, 1):
                line = raw.strip().replace("\x00", "")
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    log.warning(
                        "Skipping malformed line %d in %s: %s", i, self.path, exc
                    )

    def last(self) -> dict | None:
        """Return the most recent record, or None if the log is empty."""
        record = None
        for record in self.iter_records():
            pass
        return record

    def summary(self) -> dict:
        """
        Return a brief statistical summary across all logged runs.
        Keys: n_runs, mean_reward, best_reward, worst_reward, last_time.
        Returns an empty dict when no runs have been logged.
        """
        rewards   = []
        last_time = None
        for r in self.iter_records():
            if "mean_reward" in r:
                rewards.append(float(r["mean_reward"]))
            last_time = r.get("time")
        if not rewards:
            return {}
        return {
            "n_runs":       len(rewards),
            "mean_reward":  round(float(np.mean(rewards)),  2),
            "best_reward":  round(float(np.max(rewards)),   2),
            "worst_reward": round(float(np.min(rewards)),   2),
            "last_time":    last_time,
        }

    def __len__(self) -> int:
        """Return the total number of records in the log file."""
        return sum(1 for _ in self.iter_records())

    def __repr__(self) -> str:
        """Return a string representation showing the log file path."""
        return f"JSONLLogger(path={self.path!r})"


def _utc_now() -> str:
    """Return the current time as an ISO-8601 UTC string, e.g. 2026-06-07T22:50:00Z."""
    return datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")