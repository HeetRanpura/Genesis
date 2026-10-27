"""Where the pilot's local runs live, with checks that the files there really are the pilot's.

The pilot was trained and evaluated under the run names gru_s0 / ff_s0, and the PPOTrainer replication (and the
partial replication runs gru_s1 / ff_s1) reuse those names under runs/stage2/ and runs/stage2_eval/. Reading those
folders silently mixed replication data into the pilot's figures, so the pilot has its own tree:

  runs/stage2_pilot/<arm>_s0/checkpoints/last.pt   pilot checkpoints (original rppo, pre-format-2)
  runs/stage2_pilot/<name>/train.jsonl, eval.jsonl training logs (gru_s0, ff_s0, gru_s1, ff_s1)
  runs/stage2_pilot/eval/<arm>_s0/s<k>/            held-out recordings of the pilot checkpoints (worlds 0-4 x 2 yr)

On a machine that still holds the pilot under the old names (check before moving: these names are reused):
    mkdir -p runs/stage2_pilot/eval
    mv runs/stage2/gru_s0 runs/stage2/ff_s0 runs/stage2/gru_s1 runs/stage2/ff_s1 runs/stage2_pilot/
    mv runs/stage2_eval/gru_s0 runs/stage2_eval/ff_s0 runs/stage2_pilot/eval/

Checks: a checkpoint must be an rppo checkpoint (state dict under "net", no "format"); a recording's death counts by
cause must equal the pilot's held-out results in reports/stage2/rppo_original/eval_<arm>_s0.json for that world.
"""
from __future__ import annotations

import json
from functools import cache
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
PILOT = ROOT / "runs" / "stage2_pilot"
RESULTS = ROOT / "reports" / "stage2" / "rppo_original"


def _need(p: Path) -> Path:
    if not p.exists():
        raise FileNotFoundError(f"{p} is missing. The pilot's local runs live under {PILOT}; see pilot_runs.py "
                                "for the layout and how to move them from the old run names.")
    return p


def train_log(name: str, kind: str = "train") -> Path:
    """Training log of a pilot-era run (gru_s0, ff_s0, gru_s1, ff_s1); kind is "train" or "eval"."""
    return _need(PILOT / name / f"{kind}.jsonl")


@cache
def checkpoint(arm: str) -> dict:
    """The loaded pilot checkpoint of arm "gru" or "ff"; refuses PPOTrainer (format-2) checkpoints."""
    p = _need(PILOT / f"{arm}_s0" / "checkpoints" / "last.pt")
    ck = torch.load(p, map_location="cpu", weights_only=False)
    if "format" in ck or "net" not in ck:
        raise ValueError(f"{p} is not a pilot checkpoint (format {ck.get('format')}): it was written by PPOTrainer. "
                         "The pilot's rppo checkpoints have no format field.")
    return ck


@cache
def recording(arm: str, seed: int) -> Path:
    """Held-out recording of the pilot's arm on test world `seed`, checked against the pilot's published results."""
    d = _need(PILOT / "eval" / f"{arm}_s0" / f"s{seed}")
    final = json.loads(_need(d / "summary.json").read_text())["final"]
    got = {k[len("deaths_"):]: v for k, v in final.items() if k.startswith("deaths_") and v}
    runs = json.loads((RESULTS / f"eval_{arm}_s0.json").read_text())["per_run"]
    want = next(r["deaths_by_cause"] for r in runs if r["seed"] == seed)
    if got != want:
        raise ValueError(f"{d} is not the pilot's recording: deaths by cause {got} != {want} in "
                         f"{RESULTS / f'eval_{arm}_s0.json'} (seed {seed}).")
    return d
