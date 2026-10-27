"""H2/H3 confirmatory experiment, one command from start to finish (docs/PREREGISTRATION.md, H2/H3 protocol entry).

    .venv/bin/python experiments/run_h2h3.py            # 2 arms x seeds 1-3 x 3,000 iterations, then evaluation
    .venv/bin/python experiments/run_h2h3.py --jobs 1   # one run at a time (fanless laptops)
    make h2h3                                           # same as the first line

1. Trains six runs with experiments/train_ppo.py, --jobs at a time: h2h3_gru_s{1,2,3} (GRU memory) and
   h2h3_ff_s{1,2,3} (memoryless), 3,000 iterations each (98.3M agent-steps). Weights are kept in
   runs/stage2/<run>/checkpoints/iter_XXXXX.pt every 200 iterations and at iteration 900 (29.5M steps, the
   preregistered H2'/H3'/H5 test); last.pt is refreshed every 25 iterations for resuming.
2. Evaluates checkpoints 900 and 3000 of every run on the held-out worlds 0-4 x 2 years (experiments/eval_policy.py;
   the final checkpoints are flight-recorded, the 900 ones are not, to save ~6 GB).
3. Applies the preregistered decision rules: reports/stage2/h2h3_results.json and .md.

Stopping is safe (Ctrl-C, crash, sleep, power loss): run the same command again. Finished runs and evaluations are
skipped, unfinished runs resume from their last.pt. A run that crashes is restarted from its checkpoint up to
--retries times. Progress: one bar per running job plus an overall bar; each job's full output is appended to
runs/h2h3_logs/<job>.log. On macOS the script keeps the machine awake (caffeinate) while it runs; keep it on power.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from analysis.survival import hedges_g  # noqa: E402

PY = sys.executable
ARMS = {"gru": [], "ff": ["--core", "mlp"]}           # ff = memoryless ablation
LABEL = {"gru": "GRU", "ff": "memoryless"}
LOGS = ROOT / "runs" / "h2h3_logs"


def run_name(arm, seed):
    return f"h2h3_{arm}_s{seed}"


def run_dir(name):
    return ROOT / "runs" / "stage2" / name


def eval_name(name, it):
    return f"{name}_it{it:05d}"


def eval_json(ename):
    return ROOT / "reports" / "stage2" / f"eval_{ename}.json"


def jsonl(p: Path):
    if not p.exists():
        return []
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]


@dataclass
class Job:
    name: str
    cmd: list
    total: int
    progress: callable          # () -> units done so far
    done: callable              # () -> finished?
    postfix: callable = lambda: ""
    before: callable = lambda: None
    tries: int = 0
    proc: subprocess.Popen | None = None
    bar: tqdm | None = None
    slot: int = 0
    seen: int = 0
    log: object = field(default=None, repr=False)


def training_jobs(a):
    jobs = []
    for seed in a.seeds:
        for arm in ARMS:
            name = run_name(arm, seed)
            d = run_dir(name)
            cmd = [PY, "experiments/train_ppo.py", "--name", name, "--seed", str(seed), "--iters", str(a.iters),
                   "--agents", str(a.agents), "--resume", "--val-every", str(a.val_every), "--val-years",
                   str(a.val_years), "--record-every", str(a.record_every), "--ckpt-every", "25",
                   "--keep-every", str(a.keep_every), "--keep-at", str(a.confirm_iter), *ARMS[arm]]

            def progress(d=d):
                rows = jsonl(d / "train.jsonl")
                return max((r["iter"] for r in rows), default=-1) + 1

            def postfix(d=d):
                rows, val = jsonl(d / "train.jsonl")[-20:], jsonl(d / "val.jsonl")
                if not rows:
                    return "starting (first run compiles Numba, ~30 s)"
                s = f"train {np.mean([r['deaths_per_agent_year'] for r in rows]):.2f} d/ay"
                s += f", {rows[-1]['agent_steps_per_s']} steps/s"
                return s + (f", val {val[-1]['deaths_per_agent_year']:.2f} @ it {val[-1]['iter']}" if val else "")

            jobs.append(Job(name, cmd, a.iters, progress,
                            lambda d=d: (d / "checkpoints" / f"iter_{a.iters:05d}.pt").exists(), postfix))
    return jobs


def eval_jobs(a):
    jobs = []
    for seed in a.seeds:
        for arm in ARMS:
            name = run_name(arm, seed)
            for it in (a.confirm_iter, a.iters):
                ename = eval_name(name, it)
                ckpt = run_dir(name) / "checkpoints" / f"iter_{it:05d}.pt"
                out = ROOT / "runs" / "stage2_eval" / ename
                cmd = [PY, "experiments/eval_policy.py", str(ckpt.relative_to(ROOT)), "--name", ename,
                       "--seeds", str(a.eval_worlds), "--years", str(a.eval_years), "--agents", str(a.agents)]
                if it != a.iters:
                    cmd.append("--no-record")

                def progress(out=out):
                    return sum("result" in json.loads(p.read_text()) for p in out.glob("s*/summary.json"))

                jobs.append(Job(ename, cmd, a.eval_worlds, progress, lambda e=ename: eval_json(e).exists(),
                                before=lambda out=out: shutil.rmtree(out, ignore_errors=True)))
    return jobs


def run_all(jobs, n_parallel, retries, poll, title, overall_unit):
    """Run jobs, n_parallel at a time, until each is done or has failed retries+1 times. Returns the failures."""
    todo = [j for j in jobs if not j.done()]
    for j in jobs:
        if j.done():
            tqdm.write(f"  {j.name}: already done, skipped")
    if not todo:
        return []
    for j in todo:
        j.before()                                   # e.g. clear a half-written evaluation
    overall = tqdm(total=sum(j.total for j in todo), initial=sum(min(j.progress(), j.total) for j in todo),
                   desc=title, unit=overall_unit, position=0, dynamic_ncols=True, smoothing=0.02)
    free, active, failed = list(range(1, n_parallel + 1)), [], []
    try:
        while todo or active:
            while todo and free:
                j = todo.pop(0)
                j.slot, j.tries = free.pop(0), j.tries + 1
                if j.tries > 1:
                    j.before()
                now = min(j.progress(), j.total)
                if now < j.seen:                         # a restart redoes iterations after its checkpoint
                    overall.update(now - j.seen)
                j.seen = now
                j.bar = tqdm(total=j.total, initial=j.seen, desc=j.name, unit="it" if j.total > 100 else "world",
                             position=j.slot, leave=False, dynamic_ncols=True, smoothing=0.05)
                LOGS.mkdir(parents=True, exist_ok=True)
                j.log = open(LOGS / f"{j.name}.log", "a")
                j.log.write(f"\n=== {time.strftime('%Y-%m-%d %H:%M:%S')} attempt {j.tries}: {' '.join(j.cmd)}\n")
                j.log.flush()
                j.proc = subprocess.Popen(j.cmd, cwd=ROOT, stdout=j.log, stderr=subprocess.STDOUT)
                active.append(j)
            time.sleep(poll)
            for j in list(active):
                now = min(j.progress(), j.total)
                if now != j.seen:
                    j.bar.update(now - j.seen)
                    overall.update(now - j.seen)
                    j.seen = now
                j.bar.set_postfix_str(j.postfix(), refresh=False)
                j.bar.refresh()
                if j.proc.poll() is None:
                    continue
                active.remove(j)
                j.log.close()
                j.bar.close()
                free.append(j.slot)
                free.sort()
                if j.proc.returncode == 0 and j.done():
                    tqdm.write(f"  {j.name}: finished")
                elif j.tries <= retries:
                    tqdm.write(f"  {j.name}: exited with code {j.proc.returncode}; restarting from its checkpoint "
                               f"(attempt {j.tries + 1}/{retries + 1}), see {LOGS / (j.name + '.log')}")
                    todo.insert(0, j)
                else:
                    tqdm.write(f"  {j.name}: FAILED {j.tries} times, giving up; see {LOGS / (j.name + '.log')}")
                    failed.append(j.name)
    except KeyboardInterrupt:
        tqdm.write("\nStopping: terminating running jobs. Run the same command again to resume.")
        for j in active:
            j.proc.terminate()
        for j in active:
            try:
                j.proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                j.proc.kill()
        raise
    finally:
        overall.close()
    return failed


# ------------------------------------------------------------------------------------------- decision rules
def seed_score(ename):
    """One evaluation = one training seed: mean deaths/agent-yr over the worlds (the preregistered unit), plus the
    cause-specific rates and large-prey kills pooled over the worlds."""
    p = eval_json(ename)
    if not p.exists():
        return None
    runs = json.loads(p.read_text())["per_run"]
    ay = sum(r["agent_years"] for r in runs)
    causes = {}
    for r in runs:
        for c, v in r["deaths_by_cause"].items():
            causes[c] = causes.get(c, 0) + v
    kills = 0
    for s in range(len(runs)):
        summ = ROOT / "runs" / "stage2_eval" / ename / f"s{s}" / "summary.json"
        kills += json.loads(summ.read_text())["final"].get("ev_kill_large", 0) if summ.exists() else 0
    return dict(score=float(np.mean([r["deaths_per_agent_year"] for r in runs])), agent_years=ay,
                rates={c: v / ay for c, v in causes.items()}, large_kills_per_1000_ay=1000 * kills / ay)


def separated_below(a, b):
    """Every value of a below every value of b: the only n=3 vs 3 pattern with exact one-sided Mann-Whitney p <= 0.05."""
    return bool(max(a) < min(b))


def summarize(a):
    base = json.loads((ROOT / "reports" / "stage1" / "baselines.json").read_text())["per_run"]
    heur = [r["deaths_per_agent_year"] for r in base if r["policy"] == "heuristic"]
    h_mean = float(np.mean(heur))
    out = dict(protocol="docs/PREREGISTRATION.md (H2/H3 protocol entry)", heuristic=dict(mean=h_mean, per_world=heur),
               checkpoints={})
    md = ["# H2/H3 results (generated by experiments/run_h2h3.py)", "",
          f"Held-out worlds 0-{a.eval_worlds - 1} x {a.eval_years:g} yr, {a.agents} agents. Unit of replication = training "
          f"seed (n = {len(a.seeds)} per arm); a seed's score is its mean deaths/agent-yr over the worlds. "
          f"Heuristic baseline: {h_mean:.2f}.", ""]
    for it in (a.confirm_iter, a.iters):
        tag = "confirmatory (H2', H3', H5)" if it == a.confirm_iter else "extension (H2'', H3'', H5'')"
        arms = {}
        for arm in ARMS:
            res = {s: seed_score(eval_name(run_name(arm, s), it)) for s in a.seeds}
            if any(v is None for v in res.values()):
                arms[arm] = None
                continue
            sc = [res[s]["score"] for s in a.seeds]
            thirst = [res[s]["rates"].get("thirst", 0.0) for s in a.seeds]
            causes = {}
            for s in a.seeds:
                for c, v in res[s]["rates"].items():
                    causes[c] = causes.get(c, 0.0) + v / len(a.seeds)
            arms[arm] = dict(per_seed=res, scores=sc, mean=float(np.mean(sc)), range=[min(sc), max(sc)],
                             hedges_g_vs_heuristic=hedges_g(sc, heur), thirst_rates=thirst,
                             thirst_mean=float(np.mean(thirst)), mean_cause_rates=causes,
                             leading_cause=max(causes, key=causes.get) if causes else "none",
                             large_kills_per_1000_ay=float(np.mean([res[s]["large_kills_per_1000_ay"] for s in a.seeds])))
        steps = it * 128 * a.agents
        md += [f"## Iteration {it} ({steps / 1e6:.1f}M agent-steps): {tag}", ""]
        if any(v is None for v in arms.values()):
            out["checkpoints"][it] = dict(status="pending: evaluations missing")
            md += ["Pending: some evaluations are missing.", ""]
            continue
        g, f = arms["gru"], arms["ff"]
        verdict = dict(
            H2={arm: dict(supported=arms[arm]["mean"] <= h_mean, mean=arms[arm]["mean"], heuristic=h_mean)
                for arm in ARMS},
            H3=dict(thirst_lower_for_gru=g["thirst_mean"] < f["thirst_mean"],
                    thirst_lower_every_seed_pair=separated_below(g["thirst_rates"], f["thirst_rates"]),
                    all_cause_gru_advantage=separated_below(g["scores"], f["scores"]),
                    hedges_g_gru_minus_ff=hedges_g(g["scores"], f["scores"])),
            H5={arm: dict(starvation_leading=arms[arm]["leading_cause"] == "starvation",
                          large_kills_per_1000_ay=arms[arm]["large_kills_per_1000_ay"],
                          supported=arms[arm]["leading_cause"] == "starvation"
                          and arms[arm]["large_kills_per_1000_ay"] <= 1.0) for arm in ARMS})
        out["checkpoints"][it] = dict(agent_steps=steps, arms=arms, verdict=verdict)
        md += ["| arm | seed scores (deaths/agent-yr) | mean | Hedges' g vs heuristic | thirst deaths/agent-yr | "
               "leading cause | large kills / 1,000 agent-yr |", "|---|---|---|---|---|---|---|"]
        for arm in ARMS:
            r = arms[arm]
            md.append(f"| {LABEL[arm]} | {', '.join(f'{x:.2f}' for x in r['scores'])} | **{r['mean']:.2f}** | "
                      f"{r['hedges_g_vs_heuristic']:+.2f} | {r['thirst_mean']:.3f} | {r['leading_cause']} | "
                      f"{r['large_kills_per_1000_ay']:.2f} |")
        h3 = verdict["H3"]
        md += ["", "Decision rules as preregistered:",
               *[f"- H2 ({LABEL[arm]}): mean {r['mean']:.2f} {'<=' if r['supported'] else '>'} heuristic {h_mean:.2f} "
                 f"-> **{'supported' if r['supported'] else 'not supported'}**" for arm, r in verdict["H2"].items()],
               f"- H3 thirst: GRU {g['thirst_mean']:.3f} vs memoryless {f['thirst_mean']:.3f} -> **"
               f"{'GRU lower (direction replicates)' if h3['thirst_lower_for_gru'] else 'GRU not lower'}**"
               f" (complete separation of seeds: {h3['thirst_lower_every_seed_pair']})",
               f"- H3 all-cause: GRU advantage (all GRU seeds below all memoryless seeds): "
               f"**{h3['all_cause_gru_advantage']}**, Hedges' g (GRU - memoryless) {h3['hedges_g_gru_minus_ff']:+.2f}",
               *[f"- H5 ({LABEL[arm]}): starvation leading {r['starvation_leading']}, large kills "
                 f"{r['large_kills_per_1000_ay']:.2f} per 1,000 agent-yr -> **{'supported' if r['supported'] else 'not supported'}**"
                 for arm, r in verdict["H5"].items()], ""]
    (ROOT / "reports" / "stage2" / "h2h3_results.json").write_text(json.dumps(out, indent=1, default=float))
    (ROOT / "reports" / "stage2" / "h2h3_results.md").write_text("\n".join(md) + "\n")
    return "\n".join(md)


# ------------------------------------------------------------------------------------------- main
def preflight(a):
    import torch
    dev = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    free = shutil.disk_usage(ROOT).free / 1e9
    prereg = (ROOT / "docs" / "PREREGISTRATION.md").read_text()
    print(f"GENESIS H2/H3: {len(a.seeds)} seeds x {len(ARMS)} arms x {a.iters} iterations, {a.jobs} at a time, "
          f"device {dev}, torch {torch.__version__}, free disk {free:.0f} GB")
    if dev == "cpu":
        print("  WARNING: no GPU backend found; training on CPU is roughly 10x slower than on Apple-silicon MPS.")
    if free < 15:
        print("  WARNING: under 15 GB free. Training stops flight-recording below 15 GB; the six final evaluations "
              "record ~1.1 GB each. Free some space if you can.")
    if "H2/H3 protocol" not in prereg:
        sys.exit("docs/PREREGISTRATION.md has no H2/H3 protocol entry: pull the latest main before running.")
    if sys.platform == "darwin" and shutil.which("caffeinate"):
        subprocess.Popen(["caffeinate", "-ims", "-w", str(os.getpid())])     # no idle/system sleep while we run
        print("  caffeinate: the Mac will not sleep while this runs (keep it plugged in; closing the lid on battery "
              "still sleeps it).")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jobs", type=int, default=2, help="runs trained (and evaluated) at the same time")
    ap.add_argument("--seeds", default="1,2,3", help="learner seeds (preregistered: 1,2,3)")
    ap.add_argument("--iters", type=int, default=3000, help="iterations per run (preregistered: 3000)")
    ap.add_argument("--confirm-iter", type=int, default=900, help="confirmatory checkpoint (preregistered: 900)")
    ap.add_argument("--keep-every", type=int, default=200, help="keep a .pt checkpoint every k iterations")
    ap.add_argument("--agents", type=int, default=256)
    ap.add_argument("--val-every", type=int, default=50)
    ap.add_argument("--val-years", type=float, default=1.0)
    ap.add_argument("--record-every", type=int, default=20, help="flight-record 1 rollout in k during training")
    ap.add_argument("--eval-worlds", type=int, default=5, help="held-out worlds 0..k-1 (preregistered: 5)")
    ap.add_argument("--eval-years", type=float, default=2.0)
    ap.add_argument("--retries", type=int, default=3, help="restarts from checkpoint after a crash, per job")
    ap.add_argument("--poll", type=float, default=5.0, help="seconds between progress updates")
    ap.add_argument("--skip-eval", action="store_true", help="train only")
    a = ap.parse_args()
    a.seeds = [int(s) for s in a.seeds.split(",")]
    preflight(a)
    t0 = time.time()
    try:
        print("\nPhase 1/3: training")
        failed = run_all(training_jobs(a), a.jobs, a.retries, a.poll, "all training", "it")
        if failed:
            sys.exit(f"Training failed for {failed}. Fix the cause (see the logs), then re-run to resume.")
        if a.skip_eval:
            print("Training complete (--skip-eval).")
            return
        print("\nPhase 2/3: held-out evaluation")
        failed = run_all(eval_jobs(a), a.jobs, a.retries, a.poll, "all evaluations", "world")
        if failed:
            sys.exit(f"Evaluation failed for {failed}. See the logs, then re-run (finished steps are skipped).")
    except KeyboardInterrupt:
        sys.exit(130)
    print("\nPhase 3/3: preregistered decision rules\n")
    print(summarize(a))
    print(f"Done in {(time.time() - t0) / 3600:.1f} h. Results: reports/stage2/h2h3_results.md (and .json), "
          "reports/stage2/eval_h2h3_*.json. Commit those and logs/AGENT_LOG.md (see docs/H2H3_RUNBOOK.md).")


if __name__ == "__main__":
    main()
