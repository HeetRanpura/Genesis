"""Diagrams for the VIP reports and decks (docs/vip/diagrams/), in the style of the preprint's system diagrams.

    .venv/bin/python docs/vip/build/make_vip_diagrams.py        (needs rsvg-convert: brew install librsvg)

Reuses the Diagram helper from reports/stage2/pilot_preprint/diagrams/make_diagrams.py (Advik Divekar), which checks
that every label fits its box. The literature map is a matplotlib chart. Positions on it are a qualitative reading
of each paper's reward or objective design by the authors, and say so on the chart.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "docs" / "vip" / "diagrams"
sys.path.insert(0, str(ROOT / "reports" / "stage2" / "pilot_preprint" / "diagrams"))
import make_diagrams as md  # noqa: E402
from make_diagrams import KIND, Diagram  # noqa: E402

md.OUT = OUT


def problem_context():
    d = Diagram("v01_problem_context", 1060, 430, "The research problem",
                "Most multi-agent RL builds social structure into the objective. GENESIS removes it and measures what is left.")
    d.group(24, 80, 320, 322, "Common MARL designs", "neutral", "social term in the objective")
    d.box(40, 112, 288, 54, "Team reward", ["one shared return for the group"])
    d.box(40, 176, 288, 54, "Prosocial / inequity terms", ["reward depends on others' payoffs"])
    d.box(40, 240, 288, 54, "Influence and comm. losses", ["reward for affecting others"])
    d.box(40, 314, 288, 70, "Question left open", ["did cooperation emerge,", "or was it specified?"], dashed=True)
    d.group(372, 80, 316, 322, "GENESIS design", "sim", "nothing social is rewarded")
    d.box(388, 112, 284, 70, "Own homeostasis only", ["r = D(h_t) - D(h_t+1) - death", "enforced by a unit test"], "sim")
    d.box(388, 192, 284, 70, "Ecological affordances", ["cooperative large prey, carried food,", "scarce water, winters"], "sim")
    d.box(388, 272, 284, 54, "Capabilities off by default", ["memory, communication, reproduction"], "sim")
    d.box(388, 336, 284, 54, "Preregistered measurement", ["held-out worlds, baselines, nulls"], "sim")
    d.group(716, 80, 320, 322, "Research questions", "analysis")
    d.box(732, 112, 288, 70, "RQ1  Survival", ["do survival-only learners beat", "hand-written foragers?"], "analysis")
    d.box(732, 192, 288, 70, "RQ2  Social organisation", ["does grouping, sharing or signalling", "appear with no social reward?"], "analysis")
    d.box(732, 272, 288, 70, "RQ3  Order of emergence", ["compared with human prehistory,", "as a structural analogy only"], "analysis")
    d.box(732, 352, 288, 44, "Evidence so far", ["survival: yes; social behaviour: none yet"], "data")
    d.arrow([(344, 240), (372, 240)])
    d.arrow([(688, 240), (716, 240)])
    d.save()


def subsystems():
    d = Diagram("v02_subsystems", 1060, 470, "System concept: five subsystems",
                "Each subsystem has an owner, an interface and its own tests. The reward rule cuts across all of them.")
    d.box(24, 90, 300, 130, "World engine  (sim/)", ["Numba simulator: terrain, fauna, weather,", "fire, floods, agent physiology",
                                                     "interface: GenesisEnv.step(actions)", "lead: Heet Ranpura"], "sim")
    d.box(380, 90, 300, 130, "Learning  (agents/)", ["shared actor-critic (conv + GRU/MLP)", "PPO trainer, baselines, checkpoints",
                                                     "interface: act(obs) -> actions", "lead: Heet; rppo design: Advik"], "agent")
    d.box(736, 90, 300, 130, "Visualisation  (viewer/)", ["three.js live 3-D viewer, 2-D preview", "dashboard, inspector, disturbances",
                                                          "interface: binary WebSocket frames", "lead: Advik Divekar"], "viewer")
    d.box(202, 290, 300, 130, "Measurement  (analysis/)", ["flight recorder, exact replay", "survival statistics, nulls, behaviour",
                                                           "interface: runs/<run>/ recordings", "leads: Advik (recorder), Heet"], "data")
    d.box(558, 290, 300, 130, "Research operations", ["preregistration, AGENT_LOG, handoffs", "tests, CI, reproducible runs",
                                                      "interface: docs/, reports/, PRs", "shared by both members"], "analysis")
    d.arrow([(324, 140), (380, 140)], "obs, r", at=(352, 133))
    d.arrow([(380, 175), (324, 175)], "actions", at=(352, 193))
    d.arrow([(680, 155), (736, 155)], "state", at=(708, 148))
    d.arrow([(174, 220), (174, 256), (300, 256), (300, 290)], "recordings", at=(237, 250))
    d.arrow([(530, 220), (530, 256), (420, 256), (420, 290)])
    d.note(530, 452, "Research operations governs all five: the hard rules, the logs, the tests and the handoffs.", anchor="middle")
    d.save()


def team():
    d = Diagram("v03_team_structure", 1060, 530, "VIP vertical team structure",
                "Roles follow the VIP model. Dashed boxes are roles not yet filled; names marked 'to fill' are for the team.")
    d.box(380, 80, 300, 62, "Faculty mentor", ["research guidance, milestone reviews"], tag="to fill")
    d.box(380, 178, 300, 62, "Sem VI senior lead", ["system integration, mentoring"], dashed=True, tag="open role")
    d.box(140, 284, 320, 86, "Heet Ranpura  (Sem V)", ["engine, learner, experiments, paper", "integration and PR reviews",
                                                       "machine: MacBook Air M4"], "sim")
    d.box(600, 284, 320, 86, "Advik Divekar  (Sem V)", ["viewer, flight recorder, evaluation", "preregistration, pilot preprint",
                                                        "machine: MacBook Pro M3 Pro"], "agent")
    d.box(380, 420, 300, 62, "Sem IV junior members", ["literature, documentation, tests"], dashed=True, tag="to onboard")
    d.arrow([(530, 142), (530, 178)])
    d.arrow([(530, 240), (530, 262), (300, 262), (300, 284)])
    d.arrow([(530, 262), (760, 262), (760, 284)])
    d.arrow([(460, 322), (600, 322)])
    d.arrow([(600, 340), (460, 340)])
    d.text(530, 314, "PRs + reviews", 10.5, md.INK2, halo=True)
    d.el.append(f'<path d="M 760 370 L 760 398 L 530 398" fill="none" stroke="{md.LINE}" stroke-width="1.6"/>')
    d.arrow([(300, 370), (300, 398), (530, 398), (530, 420)])
    d.text(530, 390, "onboarding guide, TEACHER.md", 10.5, md.INK2, halo=True)
    d.note(530, 512, "Sem VI roles (integration, leadership, innovation output) pass to the current Sem V members next semester.",
           anchor="middle")
    d.save()


def roadmap():
    d = Diagram("v04_roadmap", 1060, 540, "Three-semester roadmap",
                "VIP deliverables per semester. Status as of 27 October 2026.")
    cols = [("Semester IV", "problem, research, team", [
                ("Problem definition report", "done"), ("Literature and technology survey", "done"),
                ("Initial system concept", "done"), ("Team role definition", "done"),
                ("Concept presentation", "done"), ("Design notebook + meeting log", "ongoing")]),
            ("Semester V", "system and prototype", [
                ("World engine (Stages 1, 1.1)", "done"), ("Recurrent PPO learner (Stage 2)", "done"),
                ("Live 3-D viewer (Stage 5)", "done"), ("Flight recorder + exact replay", "done"),
                ("Test plan, CI, coverage", "done"), ("H2/H3 three-seed experiment", "ready"),
                ("Technical presentation", "done")]),
            ("Semester VI", "integration and outputs", [
                ("Stage 3 social metrics", "planned"), ("Capability experiments (Stage 7)", "planned"),
                ("Human-comparison module", "planned"), ("Integrated paper + arXiv", "planned"),
                ("Final demo and report", "planned"), ("Mentor incoming Sem IV members", "planned")])]
    kind = {"done": "data", "ongoing": "analysis", "ready": "analysis", "planned": "neutral"}
    for i, (name, sub, items) in enumerate(cols):
        x = 24 + i * 346
        d.group(x, 80, 320, 436, name, ["sim", "agent", "analysis"][i], sub)
        for j, (t, s) in enumerate(items):
            d.box(x + 16, 118 + j * 54, 288, 40, t, kind=kind[s], dashed=s == "planned", tag=s)
    d.legend(24, 532, [("data", "done"), ("analysis", "ongoing / ready to run"), ("neutral", "planned (dashed)")])
    d.save()


def modules():
    d = Diagram("v05_modules", 1060, 500, "Code structure and dependencies",
                "Arrows point from a package to the packages it imports. The engine imports nothing above it.")
    d.box(24, 80, 320, 96, "experiments/", ["train_ppo, eval_policy, run_h2h3", "run_baselines, energy_economy",
                                            "behaviour_analysis, paper_figures"], "analysis")
    d.box(370, 80, 320, 96, "viewer/", ["server.py: aiohttp + sim thread", "static/: three.js ES modules",
                                        "preview_server.py: stdlib preview"], "viewer")
    d.box(716, 80, 320, 96, "tests/", ["59 tests: rules, determinism,", "learner, recorder, viewer smoke",
                                       "helpers.py: 64x64 test arenas"])
    d.box(150, 236, 340, 96, "agents/", ["policy.py: ActorCritic, PolicyAgent", "ppo.py: PPOTrainer, GAE",
                                         "baselines.py: random, greedy, heuristic"], "agent")
    d.box(570, 236, 340, 96, "analysis/", ["survival.py: lifetimes, KM, bootstrap", "trajectories.py: load, export, replay",
                                           "social.py: planned (Stage 3)"], "data")
    d.group(24, 392, 1012, 92, "sim/", "sim", "world engine: Numba, CPU")
    for k, t in enumerate(["config, layout", "worldgen", "kernels (world_tick)", "env, pz_env", "recorder, logger"]):
        d.box(44 + k * 198, 428, 180, 40, t, kind="sim")
    d.arrow([(240, 176), (240, 236)])
    d.arrow([(450, 176), (450, 236)])
    d.arrow([(640, 176), (640, 206), (530, 206), (530, 392)])
    d.arrow([(800, 176), (800, 236)])
    d.arrow([(1000, 176), (1000, 392)])
    d.arrow([(60, 176), (60, 392)])
    d.arrow([(320, 332), (320, 392)])
    d.arrow([(740, 332), (740, 392)])
    d.save()


def vv():
    d = Diagram("v06_verification_levels", 1060, 480, "Verification and validation levels",
                "The lower levels run on every change (make test, CI). The top level runs once per preregistered experiment.")
    d.box(320, 80, 420, 66, "Scientific validation", ["held-out worlds, baselines, preregistered rules",
                                                       "bootstrap CIs, Hedges' g, null models"], "analysis")
    d.box(250, 160, 560, 66, "System tests  (test_viewer, test_recorder)", [
        "viewer server boots and streams a frame; PettingZoo API conformance",
        "a flight recording replays bit-exactly"], "viewer")
    d.box(180, 240, 700, 66, "Integration tests  (test_engine, test_ppo, test_baselines)", [
        "seeded runs are bit-identical; sequential and parallel kernels agree; obs image matches rebuild",
        "learner: unroll equals stepwise, checkpoint round-trip, CPU determinism, GAE stops at death"], "agent")
    d.box(110, 320, 840, 66, "Unit tests: world rules  (test_world_rules)", [
        "terrain blocks movement, drinking only near fresh water, eating raises energy, occlusion,",
        "large prey need 2 attackers, fire spreads and is put out, reward = the agent's own drive only"], "sim")
    d.box(40, 400, 980, 50, "Process gates", [
        "CI on every push and PR; review checks attribution, append-only logs, data provenance; preregistration before evaluation"])
    d.save()


def ci():
    d = Diagram("v07_ci_pipeline", 1060, 290, "Continuous integration",
                "Every push and pull request to main runs the whole test suite on a clean Linux machine.")
    steps = [("Push / PR", ["to main, or a PR", "from a fork"], "neutral"),
             ("Clean runner", ["ubuntu-latest", "Python 3.11 via uv"], "sim"),
             ("Install", ["-e .[dev,viewer,learn]", "CPU build of torch"], "agent"),
             ("Test", ["pytest with coverage", "Numba compiled fresh"], "analysis"),
             ("Report", ["status on the commit/PR", "coverage report artifact"], "data")]
    for i, (t, ls, k) in enumerate(steps):
        x = 24 + i * 207
        d.box(x, 96, 180, 86, t, ls, k)
        if i:
            d.arrow([(x - 27, 139), (x, 139)])
    d.note(530, 236, "Local gate before every PR: make test runs the same suite. "
                     "Determinism tests fail on any unintended change to the dynamics.", anchor="middle")
    d.save()


def collaboration():
    d = Diagram("v08_collaboration", 1060, 440, "How the sub-teams coordinate",
                "The real workflow behind PRs #1-#4: fork, pull request, review, merge, handoff.")
    d.box(24, 90, 220, 80, "Advik's fork", ["advikdivekar/Genesis", "viewer, recorder, analyses"], "agent")
    d.box(290, 90, 200, 80, "Pull request", ["4 PRs so far", "tests must pass"])
    d.box(536, 90, 220, 80, "Review", ["attribution, append-only logs", "tests, data provenance"], "analysis")
    d.box(802, 90, 234, 80, "Merge to main", ["HeetRanpura/Genesis", "rebase merge, linear history"], "sim")
    for a, b in [(244, 290), (490, 536), (756, 802)]:
        d.arrow([(a, 130), (b, 130)])
    d.box(24, 250, 300, 96, "Heet's machine", ["MacBook Air M4 (fanless)", "engine, analysis, paper, reviews",
                                               "short jobs, one at a time"], "sim")
    d.box(380, 250, 300, 96, "Handoff documents", ["HANDOFF.md, HANDOFF_ADVIK.md:", "state, results, next tasks,",
                                                   "exact commands, rules"], "data")
    d.box(736, 250, 300, 96, "Advik's machine", ["MacBook Pro M3 Pro", "viewer work, long training runs",
                                                 "make h2h3 (about 13-16 h)"], "agent")
    d.arrow([(919, 170), (919, 210), (530, 210), (530, 250)], "after each merge", at=(725, 204))
    d.arrow([(324, 298), (380, 298)])
    d.arrow([(680, 298), (736, 298)])
    d.note(530, 400, "Example: the review of PR #4 found pilot scripts silently reading replication data; "
                     "fixed and verified in commit b248efd.", anchor="middle")
    d.save()


def risks():
    d = Diagram("v09_risk_matrix", 1060, 540, "Risk matrix",
                "Likelihood x impact of the project's main risks. Mitigations are listed in the System Design Document.")
    x0, y0, cw, ch = 210, 90, 270, 124
    fills = {0: "#e5f6ef", 1: "#fdf3dc", 2: "#fdeee7", 3: "#fbdcd0", 4: "#f7c4b2"}
    cells = {(2, 2): ["R1 learners never discover hunting"],
             (1, 2): ["R2 too few seeds to decide H2/H3", "R4 data mix-ups between runs (fixed)"],
             (0, 2): ["R5 over-claiming emergence"],
             (2, 1): ["R3 laptop heat and compute limits"],
             (1, 1): ["R6 disk space for recordings", "R8 handover gaps between members"],
             (2, 0): ["R7 GPU (MPS) not bit-identical"],
             (0, 1): ["R9 unverified citations (fixed)"]}
    for li in range(3):            # likelihood: 2 high (top row) .. 0 low
        for ii in range(3):        # impact: 0 low (left) .. 2 high
            x, y = x0 + ii * cw, y0 + (2 - li) * ch
            d.el.append(f'<rect x="{x}" y="{y}" width="{cw}" height="{ch}" fill="{fills[li + ii]}" stroke="#ffffff" stroke-width="3"/>')
            for k, s in enumerate(cells.get((li, ii), [])):
                assert md.tw(s, 11.5) <= cw - 24, s
                d.text(x + 12, y + 30 + k * 22, s, 11.5, md.INK, anchor="start")
    for i, s in enumerate(["low", "medium", "high"]):
        d.text(x0 + i * cw + cw / 2, y0 + 3 * ch + 22, s, 12, md.INK2)
        d.text(x0 - 14, y0 + (2 - i) * ch + ch / 2 + 4, s, 12, md.INK2, anchor="end")
    d.text(x0 + 1.5 * cw, y0 + 3 * ch + 44, "impact", 12.5, md.INK, bold=True)
    d.text(80, y0 + 1.5 * ch, "likelihood", 12.5, md.INK, bold=True)
    d.save()


def viewer():
    d = Diagram("v10_viewer_architecture", 1060, 300, "Live viewer architecture",
                "The browser only displays. All simulation happens in the server's sim thread, so viewing never changes results.")
    d.box(24, 96, 300, 110, "Sim thread", ["GenesisEnv + policy (baseline or PPO)", "commands applied between ticks",
                                           "speed changes pacing, not results"], "sim")
    d.box(380, 96, 300, 110, "aiohttp server", ["binary WebSocket protocol", "compact per-tick state frames",
                                                "every control -> controls.jsonl"])
    d.box(736, 96, 300, 110, "Browser (three.js)", ["diorama camera, day/night, weather", "dashboard: 7 live charts",
                                                    "inspector, minimap, director"], "viewer")
    d.arrow([(324, 130), (380, 130)], "state", at=(352, 123))
    d.arrow([(380, 172), (324, 172)], "cmds", at=(352, 190))
    d.arrow([(680, 130), (736, 130)], "frames", at=(708, 123))
    d.arrow([(736, 172), (680, 172)], "controls", at=(708, 190))
    d.note(530, 256, "Sessions are logged under runs/viewer/<session>/; manual disturbances are replayable bit-exactly.",
           anchor="middle")
    d.save()


def literature_map():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    pts = [("Sequential social dilemmas\n(Leibo 2017)", 0.30, 0.30), ("Common-pool resources\n(Perolat 2017)", 0.22, 0.42),
           ("Inequity aversion\n(Hughes 2018)", 0.80, 0.30), ("Social influence\n(Jaques 2019)", 0.88, 0.40),
           ("Learning to communicate\n(Foerster 2016)", 0.70, 0.12), ("Referential games\n(Lazaridou 2017)", 0.82, 0.05),
           ("Melting Pot\n(Leibo 2021)", 0.45, 0.52), ("Hide-and-seek\n(Baker 2020)", 0.62, 0.62),
           ("Neural MMO\n(Suarez 2019)", 0.17, 0.66), ("Crafter\n(Hafner 2022)", 0.05, 0.56),
           ("Social learning\n(Ndousse 2021)", 0.55, 0.42), ("Bartering\n(Johanson 2022)", 0.32, 0.56),
           ("Generative agents\n(Park 2023)", 0.90, 0.78), ("PolyWorld / Sugarscape\n(1994 / 1996)", 0.10, 0.77)]
    plt.rcParams["font.family"] = "Arial"
    fig, ax = plt.subplots(figsize=(8.6, 5.6), dpi=200)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for label, x, y in pts:
        ax.scatter(x, y, s=46, color="#8a8984", zorder=2)
        ax.annotate(label, (x, y), xytext=(6, 4), textcoords="offset points", fontsize=7.2, color="#52514e")
    ax.scatter(0.02, 0.95, s=140, color="#2a78d6", zorder=3, edgecolor="white", linewidth=1.2)
    ax.annotate("GENESIS\n(own homeostasis only,\nrich seasonal ecology, RL)", (0.02, 0.95), xytext=(12, -30),
                textcoords="offset points", fontsize=8.5, color="#2a78d6", fontweight="bold")
    ax.set_xlim(-0.03, 1.12)
    ax.set_ylim(-0.03, 1.05)
    ax.set_xlabel("social structure supplied by the designer (reward, objective or prompt)  →", fontsize=9)
    ax.set_ylabel("ecological richness and open-endedness  →", fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Where GENESIS sits in the literature", fontsize=12, loc="left", fontweight="bold")
    fig.text(0.01, 0.01, "Qualitative positions read from each paper's reward/objective design and environment, by the authors. "
             "Not a measured quantity.", fontsize=7, color="#8a8984", style="italic")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(OUT / "v11_literature_map.png")
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for f in (problem_context, subsystems, team, roadmap, modules, vv, ci, collaboration, risks, viewer, literature_map):
        f()
        print("ok", f.__name__)
