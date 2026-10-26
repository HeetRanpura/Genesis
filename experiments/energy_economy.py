"""Energy economy: does any foraging strategy beat grazing, i.e. can hunting or fruit pay in this world?

Learners at ~20M agent-steps live on plants, never hunt, and starve slowly (reports/stage2/STAGE2_REPORT.md).
This probe asks whether that is a learning failure or a property of the world: hand-written foragers that differ
ONLY in what a hungry agent seeks (agents/baselines.py heuristic diets) run on validation worlds, and we
compare survival and energy intake by source and season. Trained learners can be added for comparison.

    python experiments/energy_economy.py --seeds 500 501 502 --years 2 [--learners gru_s0 ff_s0]

Worlds 500+ are validation worlds: neither training (1000+) nor test (0-4). Nothing in the world is tuned.
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from agents.baselines import make_baseline  # noqa: E402
from analysis.survival import bootstrap_ci  # noqa: E402
from sim import GenesisEnv, load_config  # noqa: E402
from sim.constants import (DEATH_NAMES, EV_EAT_FRUIT, EV_EAT_MEAT, EV_EAT_PLANT, EV_KILL_LARGE,  # noqa: E402
                           EV_KILL_SMALL)
from sim.logger import append_agent_log  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DIETS = ["heuristic", "heuristic:grazer", "heuristic:fruit", "heuristic:hunter"]
SOURCES = {"plants": EV_EAT_PLANT, "fruit": EV_EAT_FRUIT, "meat / carried food": EV_EAT_MEAT}
SEASONS = ["spring", "summer", "autumn", "winter"]


def run(policy: str, seed: int, years: float, n: int, device: str) -> dict:
    cfg = load_config(overrides={"agents": {"n_initial": n, "max_agents": n}, "logging": {"events": False}})
    env = GenesisEnv(cfg, seed=seed)
    if policy.startswith("ppo:"):
        from agents.policy import load_policy_agent
        pol = load_policy_agent(policy[4:], env.N, seed=seed, device=device)
    else:
        pol = make_baseline(policy, env, seed)
    obs = env.observe()
    intake = {s: np.zeros(4) for s in SOURCES}            # energy by source × season
    exposure = np.zeros(4)                                   # alive agent-ticks by season
    kills = dict(small=0, large=0)
    deaths = 0
    causes0 = env.deaths_by_cause.copy()
    energy_sum = 0.0
    for _ in range(int(years * env.year_len)):
        season = env.season
        n_alive = int(env.alive.sum())
        exposure[season] += n_alive
        energy_sum += float(env.A[env.alive, 0].sum())
        obs, r, died, info = env.step(pol.act(obs, env.alive))
        ev = env.last_events
        if len(ev):
            t = ev[:, 1].astype(int)
            for name, code in SOURCES.items():
                m = t == code
                if m.any():
                    intake[name][season] += float(ev[m, 6].sum())
            kills["small"] += int((t == EV_KILL_SMALL).sum())
            kills["large"] += int((t == EV_KILL_LARGE).sum())
        deaths += int(died.sum())
        if info["born"].any():
            pol.reset(info["born"])
    ay = exposure.sum() / env.year_len                       # agent-years
    ay_season = exposure / env.year_len * 4      # agent-years spent in each season, x4: per agent per season
    return dict(
        policy=policy, seed=seed, agent_years=ay, deaths_per_agent_year=deaths / ay,
        deaths_by_cause={k: int(v) for k, v in zip(DEATH_NAMES, env.deaths_by_cause - causes0) if v},
        intake_per_agent_year={s: float(v.sum() / ay) for s, v in intake.items()},
        intake_per_agent_per_season={s: (v / np.maximum(ay_season, 1e-9)).round(3).tolist()
                                     for s, v in intake.items()},          # the four seasons sum to the yearly rate
        small_kills_per_agent_year=kills["small"] / ay, large_kills_per_agent_year=kills["large"] / ay,
        mean_energy=energy_sum / max(exposure.sum(), 1),
        basal_need_per_agent_year=float(cfg.agents.basal_cost * env.year_len))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[500, 501, 502])
    ap.add_argument("--years", type=float, default=2.0)
    ap.add_argument("--agents", type=int, default=256)
    ap.add_argument("--learners", nargs="*", default=[], help="runs/stage2/<run> checkpoints to include")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--no-diets", action="store_true", help="only the learners (merged into an existing energy_economy.json)")
    a = ap.parse_args()
    policies = ([] if a.no_diets else DIETS) + [f"ppo:{r}" for r in a.learners]
    rows = []
    for p in policies:
        for s in a.seeds:
            r = run(p, s, a.years, a.agents, a.device)
            rows.append(r)
            tot = sum(r["intake_per_agent_year"].values())
            print(f"{p:18s} seed {s}: deaths/agent-yr {r['deaths_per_agent_year']:.2f}  intake {tot:.2f}/agent-yr "
                  f"(basal {r['basal_need_per_agent_year']:.1f})  {', '.join(f'{k} {v:.2f}' for k, v in r['intake_per_agent_year'].items())}"
                  f"  small kills {r['small_kills_per_agent_year']:.2f}/agent-yr  {r['deaths_by_cause']}", flush=True)
    prev = ROOT / "reports" / "stage2" / "energy_economy.json"
    if a.no_diets and prev.exists():          # keep earlier policies' rows, replace re-run ones
        old = [r for r in json.loads(prev.read_text())["per_run"] if r["policy"] not in policies]
        for r in old:                          # older files used a misleading key name for the same quantity
            if "intake_per_agent_year_by_season" in r:
                r["intake_per_agent_per_season"] = r.pop("intake_per_agent_year_by_season")
        rows = old + rows
        policies = list(dict.fromkeys([r["policy"] for r in old] + policies))
    summ = {}
    for p in policies:
        rr = [r for r in rows if r["policy"] == p]
        summ[p] = dict(deaths_per_agent_year=bootstrap_ci([r["deaths_per_agent_year"] for r in rr]),
                       intake_per_agent_year={s: float(np.mean([r["intake_per_agent_year"][s] for r in rr])) for s in SOURCES},
                       intake_by_season={s: np.mean([r["intake_per_agent_per_season"][s] for r in rr], 0).round(3).tolist()
                                         for s in SOURCES},
                       small_kills_per_agent_year=float(np.mean([r["small_kills_per_agent_year"] for r in rr])),
                       starvation_share=float(np.mean([r["deaths_by_cause"].get("starvation", 0) / max(1, sum(r["deaths_by_cause"].values()))
                                                       for r in rr])),
                       mean_energy=float(np.mean([r["mean_energy"] for r in rr])))
    out = ROOT / "reports" / "stage2"
    out.mkdir(parents=True, exist_ok=True)
    (out / "energy_economy.json").write_text(json.dumps(dict(args=vars(a), per_run=rows, summary=summ), indent=2))
    # figure: intake by source, mortality, winter intake
    fig, ax = plt.subplots(1, 3, figsize=(18, 4.6))
    x = np.arange(len(policies))
    bottom = np.zeros(len(policies))
    for s, col in zip(SOURCES, ["#6aa84f", "#e69138", "#a61c00"]):
        v = np.array([summ[p]["intake_per_agent_year"][s] for p in policies])
        ax[0].bar(x, v, bottom=bottom, color=col, label=s)
        bottom += v
    ax[0].axhline(rows[0]["basal_need_per_agent_year"], color="k", ls=":", label="basal metabolism alone")
    ax[0].set(title="Energy eaten per agent-year, by source", ylabel="energy units")
    m = [summ[p]["deaths_per_agent_year"] for p in policies]
    ax[1].bar(x, [v[0] for v in m], yerr=[[v[0] - v[1] for v in m], [v[2] - v[0] for v in m]], capsize=4, color="#5b7db1")
    ax[1].set(title="Mortality on validation worlds (95% CI over seeds)", ylabel="deaths / agent-year")
    for i, p in enumerate(policies):
        tot = np.sum([summ[p]["intake_by_season"][s] for s in SOURCES], 0)
        ax[2].plot(SEASONS, tot, "o-", label=p)
    ax[2].set(title="Energy eaten per agent per season (all sources)", ylabel="energy units per season")
    for a_ in ax[:2]:
        a_.set_xticks(x, [p.replace("heuristic:", "") for p in policies], rotation=20)
    for a_ in ax:
        a_.legend(frameon=False, fontsize=8)
        a_.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "fig_energy_economy.png", dpi=110)
    lines = [f"Validation worlds {a.seeds} (not training, not test), {a.years:g} years, {a.agents} agents. Hand-written "
             "foragers that differ only in what a hungry agent seeks; nothing in the world was changed.", "",
             "| policy | deaths / agent-yr (95% CI) | energy eaten / agent-yr (plants, fruit, meat) | small kills / agent-yr | starvation share |",
             "|---|---|---|---|---|"]
    for p in policies:
        d, it = summ[p]["deaths_per_agent_year"], summ[p]["intake_per_agent_year"]
        lines.append(f"| {p} | {d[0]:.2f} [{d[1]:.2f}, {d[2]:.2f}] | {sum(it.values()):.2f} ({it['plants']:.2f}, {it['fruit']:.2f}, "
                     f"{it['meat / carried food']:.2f}) | {summ[p]['small_kills_per_agent_year']:.2f} | {summ[p]['starvation_share']:.0%} |")
    lines.append(f"Basal metabolism alone costs {rows[0]['basal_need_per_agent_year']:.1f} energy per agent-year (before movement).")
    append_agent_log("Energy economy: can hunting or fruit pay?", lines)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
