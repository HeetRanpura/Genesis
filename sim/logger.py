"""Per-run logging: resolved config, manifest, metrics CSV, event log, summary."""
from __future__ import annotations

import csv
import datetime as dt
import json
import platform
import subprocess
from pathlib import Path

import numpy as np
import yaml

from .config import ROOT
from .constants import DEATH_NAMES, EVENT_NAMES


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "uncommitted"


class RunLogger:
    def __init__(self, run_dir: str | Path, env, meta: dict | None = None):
        self.dir = Path(run_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.env = env
        (self.dir / "config.yaml").write_text(yaml.safe_dump(env.cfg.to_dict(), sort_keys=False))
        import numba
        manifest = dict(seed=env.seed, started=dt.datetime.now().isoformat(timespec="seconds"),
                        git_commit=git_commit(), python=platform.python_version(), numpy=np.__version__,
                        numba=numba.__version__, machine=platform.platform(), world_hash=self._world_hash(),
                        action_names=env.action_names, event_names=EVENT_NAMES, death_names=DEATH_NAMES,
                        **(meta or {}))
        (self.dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
        self._csv = None
        self._writer = None
        self._events: list[np.ndarray] = []

    def _world_hash(self) -> str:
        import hashlib
        return hashlib.sha256(self.env.terrain.tobytes()).hexdigest()[:16]

    def log_metrics(self, extra: dict | None = None):
        row = self.env.metrics()
        if extra:
            row.update(extra)
        if self._writer is None:
            self._csv = open(self.dir / "metrics.csv", "w", newline="")
            self._writer = csv.DictWriter(self._csv, fieldnames=list(row.keys()))
            self._writer.writeheader()
        self._writer.writerow(row)
        return row

    def collect_events(self):
        if self.env.event_log:
            self._events.extend(self.env.event_log)
            self.env.event_log.clear()

    def close(self, summary: dict | None = None):
        self.collect_events()
        if self._csv:
            self._csv.close()
        ev = np.concatenate(self._events) if self._events else np.zeros((0, 7))
        np.save(self.dir / "events.npy", ev)
        s = dict(final=self.env.metrics(), manual_interventions=self.env.manual_log,
                 finished=dt.datetime.now().isoformat(timespec="seconds"), **(summary or {}))
        (self.dir / "summary.json").write_text(json.dumps(s, indent=2, default=float))
        return s


AGENT_LOG = ROOT / "logs" / "AGENT_LOG.md"


def append_agent_log(title: str, lines: list[str]):
    """Append a dated entry to the human-readable log of how agents are doing."""
    AGENT_LOG.parent.mkdir(parents=True, exist_ok=True)
    if not AGENT_LOG.exists():
        AGENT_LOG.write_text("# GENESIS agent log\n\nRunning record of how agents are doing, newest last. "
                             "Entries are appended by experiment scripts; numbers are measured, not estimated.\n")
    stamp = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    with open(AGENT_LOG, "a") as f:
        f.write(f"\n## {stamp} — {title}\n\n" + "\n".join(lines) + "\n")
