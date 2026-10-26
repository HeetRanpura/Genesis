"""Evaluate trained PPO agents on held-out worlds with the baseline protocol, and compare.

    python experiments/eval_agents.py --runs gru mlp --seeds 5 --years 2

Protocol (identical to experiments/run_baselines.py): world seeds 0..S-1 (training used 1000+),
256 agents, fixed population with respawn, stochastic policy. Writes reports/stage2/eval.json,
fig_stage2_survival.png and appends to logs/AGENT_LOG.md.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from agents.policy import ActorCritic, PolicyAgent  # noqa: E402
from analysis.survival import bootstrap_ci, hedges_g, kaplan_meier  # noqa: E402
from run_baselines import run_one  # noqa: E402
from sim.constants import DEATH_NAMES  # noqa: E402
from sim.logger import append_agent_log  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "reports" / "stage2"
COL = {"random": "#9aa0a6", "greedy": "#e6a23c", "heuristic": "#3a7bd5", "ppo_gru": "#2e7d32", "ppo_mlp": "#8e44ad"}


def load_agent(run: str):
    ck = torch.load(ROOT / "runs" / "stage2" / run / "ckpt_latest.pt", weights_only=False, map_location="cpu")
    model = ActorCritic(ck["obs_shapes"], ck["n_actions"], ck["hidden"], ck["core"])
    model.load_state_dict(ck["trainer"]["model"])
    info = dict(iterations=ck["trainer"]["iteration"], agent_steps=ck["trainer"]["agent_steps"], core=ck["core"])
    return model, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", default=["gru", "mlp"])
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--years", type=float, default=2.0)
    ap.add_argument("--agents", type=int, default=256)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--out", default="stage2", help="reports/<out>/ and runs/<out>_eval/")
    args = ap.parse_args()
    global OUT
    OUT = ROOT / "reports" / args.out
    OUT.mkdir(parents=True, exist_ok=True)
    base = json.loads((ROOT / "reports" / "stage1" / "baselines.json").read_text())
    per_run = [r for r in base["per_run"]]
    trained = {}
    for run in args.runs:
        model, info = load_agent(run)
        name = f"ppo_{run}"
        trained[name] = info
        for seed in range(args.seeds):
            r = run_one(name, seed, args.years, args.agents, None, ROOT / "runs" / f"{args.out}_eval",
                        make_policy=lambda env, s: PolicyAgent(model, env.N, seed=s, device=args.device),
                        experiment="stage2_eval")
            per_run.append(r)
            print(f"{name:10s} seed={seed} deaths/agent-yr={r['deaths_per_agent_year']:.2f} "
                  f"S(1yr)={r['km_survival_1yr']:.3f} {r['deaths_by_cause']}", flush=True)
    policies = ["random", "greedy", "heuristic"] + [f"ppo_{r}" for r in args.runs]
    summ = {}
    for p in policies:
        d = [r["deaths_per_agent_year"] for r in per_run if r["policy"] == p]
        s = [r["km_survival_1yr"] for r in per_run if r["policy"] == p]
        summ[p] = dict(deaths_per_agent_year=bootstrap_ci(d), km_survival_1yr=bootstrap_ci(s), n_seeds=len(d))
    for p in policies[3:]:
        mine = [r["deaths_per_agent_year"] for r in per_run if r["policy"] == p]
        for b in ("random", "greedy", "heuristic"):
            other = [r["deaths_per_agent_year"] for r in per_run if r["policy"] == b]
            summ[p][f"hedges_g_vs_{b}"] = hedges_g(mine, other)
    (OUT / "eval.json").write_text(json.dumps(dict(args=vars(args), trained=trained, per_run=per_run,
                                                    summary=summ), indent=2, default=float))
    # figure
    fig, axs = plt.subplots(1, 3, figsize=(17, 4.6))
    for p in policies:
        D, O, C = [], [], []
        src = ROOT / "runs" / ("stage1_baselines" if not p.startswith("ppo") else f"{args.out}_eval")
        for f in sorted(src.glob(f"{p}_s*/lifetimes.npz")):
            z = np.load(f)
            D.append(z["dur"]), O.append(z["observed"]), C.append(z["cause"])
        if not D:
            continue
        D, O, C = np.concatenate(D), np.concatenate(O), np.concatenate(C)
        tt, ss = kaplan_meier(D, O)
        axs[0].step(tt / 120, ss, where="post", c=COL[p], label=p)
        counts = np.array([(C == k).sum() for k in range(len(DEATH_NAMES))], float)
        left = 0.0
        for k, col in enumerate(["#e67e22", "#2980b9", "#7fb3d5", "#e74c3c", "#7b241c", "#7f8c8d", "#000", "#ccc"]):
            frac = counts[k] / max(1, counts.sum())
            axs[2].barh(p, frac, left=left, color=col, label=DEATH_NAMES[k] if p == "random" else None)
            left += frac
    axs[0].set_xlabel("days since spawn"), axs[0].set_ylabel("Kaplan-Meier survival"), axs[0].legend(frameon=False)
    axs[0].set_title(f"Held-out worlds (seeds 0-{args.seeds - 1}), {args.years:g} years, {args.agents} agents")
    m = [summ[p]["deaths_per_agent_year"][0] for p in policies]
    lo = [m[i] - summ[p]["deaths_per_agent_year"][1] for i, p in enumerate(policies)]
    hi = [summ[p]["deaths_per_agent_year"][2] - m[i] for i, p in enumerate(policies)]
    axs[1].bar(policies, m, yerr=[lo, hi], color=[COL[p] for p in policies], capsize=4)
    for i, p in enumerate(policies):
        for r in per_run:
            if r["policy"] == p:
                axs[1].scatter(i + 0.25, r["deaths_per_agent_year"], c="k", s=8)
    axs[1].set_ylabel("deaths / agent-year"), axs[1].set_title("Mortality (mean ± 95% bootstrap CI; dots = seeds)")
    axs[2].set_title("Causes of death (fraction)")
    axs[2].legend(frameon=False, fontsize=7, ncol=4, loc="lower center", bbox_to_anchor=(0.5, -0.32))
    fig.tight_layout()
    fig.savefig(OUT / "fig_stage2_survival.png")
    lines = [f"Held-out evaluation: seeds 0-{args.seeds - 1} (never trained on), {args.years:g} years, {args.agents} "
             "agents, respawn, stochastic policies.", "",
             "| policy | deaths / agent-year (95% CI) | KM survival 1 yr (95% CI) | training |", "|---|---|---|---|"]
    for p in policies:
        d, s = summ[p]["deaths_per_agent_year"], summ[p]["km_survival_1yr"]
        tr = f"{trained[p]['agent_steps'] / 1e6:.1f}M agent-steps" if p in trained else "—"
        lines.append(f"| {p} | {d[0]:.2f} [{d[1]:.2f}, {d[2]:.2f}] | {s[0]:.3f} [{s[1]:.3f}, {s[2]:.3f}] | {tr} |")
    for p in policies[3:]:
        lines.append(f"- {p}: Hedges' g on per-seed mortality vs random {summ[p]['hedges_g_vs_random']:.2f}, "
                     f"vs greedy {summ[p]['hedges_g_vs_greedy']:.2f}, vs heuristic {summ[p]['hedges_g_vs_heuristic']:.2f} "
                     "(negative = fewer deaths)")
    append_agent_log("Stage 2: PPO agents vs baselines on held-out worlds", lines)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
