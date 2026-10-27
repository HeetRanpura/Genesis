"""SVG architecture/pipeline diagrams for the GENESIS preprint.

    .venv/bin/python reports/stage2/pilot_preprint/diagrams/make_diagrams.py

Writes one standalone .svg per diagram next to this file (and .pdf/.png when rsvg-convert is installed).
Every label is checked to fit its box (rough Helvetica metrics) so edits fail loudly instead of overflowing.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from xml.sax.saxutils import escape

OUT = Path(__file__).resolve().parent

INK, INK2, MUTED, LINE = "#1b1b1a", "#52514e", "#8a8984", "#6f6e69"
KIND = {  # fill, stroke  (hues follow the paper's categorical order)
    "sim": ("#eaf2fc", "#2a78d6"),
    "agent": ("#fdeee7", "#eb6834"),
    "data": ("#e5f6ef", "#1baf7a"),
    "analysis": ("#fdf3dc", "#c98500"),
    "viewer": ("#fcebf2", "#d55181"),
    "neutral": ("#f4f4f2", "#8a8984"),
}
FONT = "Helvetica Neue, Helvetica, Arial, sans-serif"


def tw(s: str, fs: float, bold=False) -> float:
    """Approximate rendered width of s in px."""
    narrow = sum(c in "il.,:;|!'()[]{} ftjrI1" for c in s)
    wide = sum(c in "mwMWG@%" for c in s)
    k = 0.60 if bold else 0.56
    return fs * (k * (len(s) - narrow - wide) + 0.32 * narrow + 0.85 * wide)


class Diagram:
    def __init__(self, name, w, h, title, subtitle=""):
        self.name, self.w, self.h = name, w, h
        self.el: list[str] = []
        self.text(24, 34, title, 18, INK, bold=True, anchor="start")
        if subtitle:
            self.text(24, 56, subtitle, 12.5, INK2, anchor="start")

    # ---- primitives
    def text(self, x, y, s, fs: float = 12, color=INK, bold=False, anchor="middle", italic=False, halo=False):
        st = f'font-size="{fs}" fill="{color}" text-anchor="{anchor}"'
        if bold:
            st += ' font-weight="600"'
        if italic:
            st += ' font-style="italic"'
        if halo:
            st += ' stroke="#ffffff" stroke-width="4" stroke-linejoin="round" paint-order="stroke"'
        self.el.append(f'<text x="{x:.1f}" y="{y:.1f}" {st}>{escape(s)}</text>')

    def box(self, x, y, w, h, title, lines=(), kind="neutral", dashed=False, tag=None, fs: float = 12.5, lfs: float = 11):
        fill, stroke = KIND[kind]
        dash = ' stroke-dasharray="5 4"' if dashed else ""
        self.el.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="9" fill="{fill}" stroke="{stroke}" '
                       f'stroke-width="1.6"{dash}/>')
        need = 22 + 15 * len(lines) + 6 if lines else 26
        assert need <= h + 1, f"{self.name}: box '{title}' too short ({need} > {h})"
        for s, f, b in [(title, fs, True)] + [(l, lfs, False) for l in lines]:
            assert tw(s, f, b) <= w - 14, f"{self.name}: '{s}' too wide ({tw(s, f, b):.0f} > {w - 14})"
        if lines:
            block = 18 + 15 * len(lines)
            y0 = y + (h - block) / 2 + 13
            self.text(x + w / 2, y0, title, fs, INK, bold=True)
            for i, l in enumerate(lines):
                self.text(x + w / 2, y0 + 18 + 15 * i, l, lfs, INK2)
        else:
            self.text(x + w / 2, y + h / 2 + fs * 0.36, title, fs, INK, bold=True)
        if tag:
            tx = x + w - 8
            self.el.append(f'<rect x="{tx - tw(tag, 9.5) - 10:.1f}" y="{y - 9}" width="{tw(tag, 9.5) + 12:.1f}" height="17" '
                           f'rx="8.5" fill="#ffffff" stroke="{stroke}" stroke-width="1.2"/>')
            self.text(tx - tw(tag, 9.5) / 2 - 4, y + 3.5, tag, 9.5, INK2)
        return (x, y, w, h)

    def group(self, x, y, w, h, label, kind="neutral", sub=None):
        fill, stroke = KIND[kind]
        self.el.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="12" fill="{fill}" fill-opacity="0.35" '
                       f'stroke="{stroke}" stroke-width="1.3" stroke-dasharray="7 5"/>')
        self.text(x + 14, y + 22, label, 13, INK, bold=True, anchor="start")
        if sub:
            self.text(x + 14 + tw(label, 13, True) + 10, y + 22, sub, 11, INK2, anchor="start")

    def arrow(self, pts, label=None, at=None, dashed=False, color=LINE, lfs: float = 10.5, anchor="middle"):
        d = "M " + " L ".join(f"{px:.1f} {py:.1f}" for px, py in pts)
        dash = ' stroke-dasharray="5 4"' if dashed else ""
        self.el.append(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="1.6"{dash} marker-end="url(#ah)"/>')
        if label:
            lx, ly = at if at else ((pts[0][0] + pts[-1][0]) / 2, (pts[0][1] + pts[-1][1]) / 2 - 6)
            for i, part in enumerate(label.split("\n")):
                self.text(lx, ly + i * 13, part, lfs, INK2, halo=True, anchor=anchor)

    def legend(self, x, y, items):
        for kind, label in items:
            fill, stroke = KIND[kind]
            self.el.append(f'<rect x="{x}" y="{y - 10}" width="14" height="14" rx="3" fill="{fill}" stroke="{stroke}" stroke-width="1.4"/>')
            self.text(x + 20, y + 1, label, 11, INK2, anchor="start")
            x += 34 + tw(label, 11)

    def note(self, x, y, s, fs: float = 11, anchor="start"):
        self.text(x, y, s, fs, MUTED, italic=True, anchor=anchor)

    def save(self):
        svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.w}" height="{self.h}" viewBox="0 0 {self.w} {self.h}" '
               f'font-family="{FONT}" role="img" aria-label="{escape(self.name)}">\n'
               f'<defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
               f'orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" fill="{LINE}"/></marker></defs>\n'
               f'<rect width="100%" height="100%" fill="#ffffff"/>\n' + "\n".join(self.el) + "\n</svg>\n")
        p = OUT / f"{self.name}.svg"
        p.write_text(svg)
        if shutil.which("rsvg-convert"):
            subprocess.run(["rsvg-convert", "-f", "pdf", "-o", str(p.with_suffix(".pdf")), str(p)], check=True)
            subprocess.run(["rsvg-convert", "-z", "2", "-o", str(p.with_suffix(".png")), str(p)], check=True)
        return p


def c(b, side):
    """Anchor point on a box (x, y, w, h)."""
    x, y, w, h = b
    return {"t": (x + w / 2, y), "b": (x + w / 2, y + h), "l": (x, y + h / 2), "r": (x + w, y + h / 2)}[side]


# =========================================================================================== 1 system
def system():
    d = Diagram("01_system_architecture", 1060, 760, "GENESIS system architecture",
                "One seeded simulator, interchangeable policies, everything recorded; analysis never feeds back into reward.")
    cfg = d.box(380, 78, 300, 54, "Config + seed", ["configs/default.yaml, deep-merged experiment YAML"], "neutral")
    d.group(20, 160, 540, 270, "Simulator", "sim", "sim/ · Numba · CPU")
    wg = d.box(40, 200, 200, 86, "World generation", ["terrain, lakes, rivers, biomes", "fauna seeding, 1-yr spin-up"], "sim")
    d.box(40, 306, 200, 104, "Determinism", ["counter hash RNG", "(seed, tick, index, stream)", "bit-identical runs", "exact replay"], "sim")
    env = d.box(270, 200, 270, 210, "GenesisEnv.step", ["climate tick (Python)", "world_tick: ONE Numba call", "agents → fauna → vegetation", "fire → rewards → births", "masks → events → observations",
                                                       "reward = own drive only", "(enforced by a unit test)"], "sim")
    d.arrow([c(wg, "r"), (env[0], c(wg, "r")[1])])
    d.group(680, 160, 360, 270, "Agents", "agent", "agents/ · PyTorch · MPS")
    d.box(700, 200, 320, 58, "Baselines", ["random · greedy forager · heuristic"], "agent")
    lr = d.box(700, 272, 320, 58, "Learner (shared weights)", ["conv + GRU / MLP actor-critic, per-agent state"], "agent")
    tr = d.box(700, 344, 320, 66, "PPO trainer", ["GAE, clipped PPO, BPTT", "train worlds 1000+  ·  eval worlds 0-4"], "agent")
    d.arrow([(540, 250), (680, 250)], "obs, reward,\ndone, born", at=(610, 230))
    d.arrow([(680, 370), (540, 370)], "actions", at=(610, 362))
    d.arrow([c(cfg, "b"), (530, 146), (290, 146), (290, 160)])
    d.arrow([(530, 146), (860, 146), (860, 160)])
    dat = d.box(20, 466, 1020, 72, "Flight recorder + RunLogger  →  runs/<run>/",
                ["every agent, every tick: state it acted from, action, reward, death and cause, value, action probabilities",
                 "fauna · world totals · events with provider uid · births with genes · manual interventions · start-state hash"], "data")
    d.arrow([(405, 430), (405, 466)])
    d.arrow([(860, 430), (860, 466)])
    xs = [20, 280, 540, 800]
    boxes = [("Evaluation", ["held-out worlds, baselines", "KM, bootstrap CI, Hedges g"], "analysis", False, None),
             ("Emergence analysis", ["social, comms, culture, norms", "against matched null models"], "analysis", True, "planned"),
             ("Human comparison", ["onset order vs a cited", "timeline (analogy only)"], "analysis", True, "planned"),
             ("Live viewer", ["aiohttp WebSocket → three.js", "controls logged, replayable"], "viewer", False, None)]
    bb = []
    for x, (t, l, k, dsh, tag) in zip(xs, boxes):
        b = d.box(x, 574, 240, 74, t, l, k, dashed=dsh, tag=tag)
        bb.append(b)
        d.arrow([(x + 120, 538), (x + 120, 574)])
    rep = d.box(20, 684, 760, 50, "reports/ · PREREGISTRATION.md · AGENT_LOG.md (negative results kept)", (), "neutral", fs=12)
    for b in bb[:3]:
        d.arrow([c(b, "b"), (c(b, "b")[0], 684)])
    d.legend(810, 712, [("sim", "built"), ])
    d.el.append(f'<rect x="870" y="702" width="14" height="14" rx="3" fill="#fdf3dc" stroke="#c98500" stroke-width="1.4" stroke-dasharray="3 2"/>')
    d.text(890, 713, "dashed = planned", 11, INK2, anchor="start")
    return d.save()


# =========================================================================================== 2 worldgen
def worldgen():
    d = Diagram("02_world_generation", 1040, 470, "World generation pipeline",
                "A pure function of the seed (default 256×256). Terrain is static afterwards; everything else evolves.")
    steps = [("1  Seed", ["integer world seed", "train 1000+ · eval 0-4"], "neutral"),
             ("2  Elevation", ["domain-warped fBm", "+ continental falloff"], "sim"),
             ("3  Normalise", ["rank-normalised heights", "sea / rock / mountain levels"], "sim"),
             ("4  Priority-flood", ["fill depressions", "(Barnes et al. 2014) → lakes"], "data"),
             ("5  D8 flow accumulation", ["rivers ≥ 180 upstream cells", "streams ≥ 90 (dry in drought)"], "data"),
             ("6  Moisture", ["noise + distance to", "fresh water"], "data"),
             ("7  Biomes", ["water, sand, grass, meadow,", "forest, barren, rock, mountain"], "sim"),
             ("8  Resources", ["fertility, wood, caves,", "fruit trees (~1.2% density)"], "agent"),
             ("9  Fauna", ["small prey, large prey,", "predators (food-limited)"], "agent"),
             ("10  Spin-up", ["1 year of ecology", "without agents"], "agent"),
             ("11  Line of sight", ["precompute per-cell", "visibility masks"], "neutral"),
             ("12  Spawn agents", ["256 agents on land", "= run tick 0"], "neutral")]
    bw, bh, gx = 226, 76, 26
    x0 = 20
    rows = [90, 210, 330]
    boxes = []
    for i, (t, l, k) in enumerate(steps):
        r, col = divmod(i, 4)
        boxes.append(d.box(x0 + col * (bw + gx), rows[r], bw, bh, t, l, k))
    for i in range(len(boxes) - 1):
        a, b = boxes[i], boxes[i + 1]
        if i % 4 != 3:
            d.arrow([c(a, "r"), c(b, "l")])
        else:
            ya = a[1] + bh + 22
            d.arrow([c(a, "b"), (c(a, "b")[0], ya), (c(b, "t")[0], ya), c(b, "t")])
    d.legend(24, 448, [("sim", "terrain"), ("data", "hydrology (ocean is salt; fresh water ~4-6% of map)"), ("agent", "ecology"), ("neutral", "setup")])
    return d.save()


# =========================================================================================== 3 tick
def tick():
    d = Diagram("03_world_tick", 1080, 520, "One simulation tick",
                "Per-tick work stays inside a single fused Numba call; each extra Python-side dispatch would cost 5-10 µs.")
    clim = d.box(20, 175, 180, 170, "_climate_tick", ["Python", "season, time of day", "temperature", "weather Markov chain", "disturbance timers"], "neutral")
    d.group(230, 76, 640, 384, "kernels.world_tick", "sim", "one Numba call")
    steps = [("1  Agents", ["apply actions; metabolism,", "movement, carrying costs"]),
             ("2  Fauna", ["threat grid → prey proposals", "→ sequential resolve, births"]),
             ("3  Vegetation & trees", ["every 8 ticks, dt-scaled;", "season, rain, drought, famine"]),
             ("4  Fire", ["spread by fuel and dryness,", "rain extinguishes, scars"]),
             ("5  Rewards", ["D(h_t) − D(h_t+1)", "+ death penalty (own body only)"]),
             ("6  Deaths → births / respawn", ["slot refilled; info['born']", "marks new agents"]),
             ("7  Masks + event tally", ["(tick, type, a, b, x, y, value)", "per-cause death counts"]),
             ("8  Observations", ["masked 11×11 crops of the", "incrementally patched image"])]
    bw, bh, gx, gy = 280, 64, 40, 24
    boxes = []
    for i, (t, l) in enumerate(steps):
        r, col = divmod(i, 2)
        col = col if r % 2 == 0 else 1 - col
        boxes.append(d.box(250 + col * (bw + gx), 108 + r * (bh + gy), bw, bh, t, l, "sim"))
    for i in range(7):
        a, b = boxes[i], boxes[i + 1]
        if i % 2 == 0:
            d.arrow([c(a, "r"), c(b, "l")] if a[0] < b[0] else [c(a, "l"), c(b, "r")])
        else:
            d.arrow([c(a, "b"), c(b, "t")])
    out = d.box(900, 175, 160, 170, "Returns", ["obs (grid, self,", "neighbours, signals)", "rewards, done", "info['born']", "events"], "agent")
    d.arrow([c(clim, "r"), (230, 260)])
    d.arrow([(870, 260), c(out, "l")])
    d.note(240, 494, "Determinism: randomness only via hash(seed, tick, index, stream); sequential and parallel kernel twins produce identical outputs (tested).")
    return d.save()

# =========================================================================================== 4 observation
def observation():
    d = Diagram("04_observation_pipeline", 1080, 430, "Observation pipeline (per agent, per tick)",
                "Partial, egocentric, occluded. The same observations feed learners and baselines.")
    img = d.box(20, 92, 220, 92, "World image img[y, x, 16]", ["observer-independent channels,", "patched in place at every", "world mutation"], "sim")
    crop = d.box(280, 92, 200, 92, "Egocentric crop", ["11 × 11 around the agent", "(radius 5; 2 at night,", "3 in fog)"], "sim")
    los = d.box(520, 92, 220, 92, "Line-of-sight mask", ["precomputed per cell;", "blocked by a mountain or", "a second forest cell"], "sim")
    grid = d.box(780, 92, 260, 92, "grid  uint8 [N, 11, 11, 16]", ["visible, blocked, water, forest, rock,", "shelter, plant, fruit, food, prey ×2,", "predator, agent (kin), fire, signal, snow"], "agent", lfs=10.5)
    d.arrow([c(img, "r"), c(crop, "l")])
    d.arrow([c(crop, "r"), c(los, "l")])
    d.arrow([c(los, "r"), c(grid, "l")])
    body = d.box(20, 230, 220, 76, "Own body", ["energy, hydration, health, warmth,", "age, carried items, temperature"], "sim")
    selfv = d.box(280, 230, 200, 76, "self  [13]", ["interoception + light,", "rain, shelter, maturity"], "agent")
    nb = d.box(520, 230, 220, 76, "neighbours  [4 × 9]", ["nearest visible: dx, dy, health,", "food, kin (+ gated features)"], "agent")
    sig = d.box(780, 230, 260, 76, "heard signals  [8]", ["communication switch only;", "radius 9, not occluded"], "agent", dashed=True, tag="off by default")
    d.arrow([c(body, "r"), c(selfv, "l")])
    d.arrow([c(selfv, "r"), c(nb, "l")], "concat", at=(500, 262))
    d.arrow([c(nb, "r"), c(sig, "l")], "concat", at=(760, 262))
    pol = d.box(280, 340, 760, 60, "Policy input", ["grid → float / 255 → conv branch      ·      self ‖ neighbours ‖ signals (57) → FC 64"], "agent")
    d.arrow([(910, 184), (910, 212), (1062, 212), (1062, 370), (1040, 370)])
    d.arrow([(660, 306), (660, 340)])
    d.note(24, 360, "dead slots → zeros")
    d.note(24, 378, "info['born'] → reset")
    d.note(24, 396, "that slot's GRU state")
    return d.save()


# =========================================================================================== 5 network
def network():
    d = Diagram("05_policy_network", 1100, 440, "Actor-critic policy (one set of weights shared by all agents)",
                "≈1.1M parameters. The memory ablation swaps only the core: GRUCell(256) vs a 256-unit feed-forward layer.")
    g = d.box(20, 96, 150, 62, "grid", ["11 × 11 × 16"], "sim")
    c1 = d.box(200, 96, 130, 62, "conv 3×3", ["32, ReLU"], "agent")
    c2 = d.box(360, 96, 140, 62, "conv 3×3", ["64, stride 2, ReLU"], "agent")
    f1 = d.box(530, 96, 120, 62, "flatten", ["→ FC 256"], "agent")
    v = d.box(20, 196, 150, 62, "vector", ["13 + 4×9 + 8 = 57"], "sim")
    f2 = d.box(200, 196, 130, 62, "FC 64", ["ReLU"], "agent")
    cat = d.box(680, 180, 110, 62, "concat", ["256 + 64 = 320"], "neutral", lfs=10)
    t = d.box(680, 290, 110, 62, "FC 256", ["trunk"], "agent")
    core = d.box(830, 280, 220, 82, "core", ["GRUCell(256)   [memory]", "or MLP 256   [memoryless]"], "agent")
    pi = d.box(830, 120, 220, 56, "policy head", ["softmax over 11 actions"], "agent")
    val = d.box(830, 384, 220, 44, "value head  V(o, h)", (), "agent")
    for a, b in [(g, c1), (c1, c2), (c2, f1), (v, f2)]:
        d.arrow([c(a, "r"), c(b, "l")])
    d.arrow([c(f1, "r"), (735, 127), c(cat, "t")])
    d.arrow([c(f2, "r"), (680, 227)])
    d.arrow([c(cat, "b"), c(t, "t")])
    d.arrow([c(t, "r"), (830, 321)])
    d.arrow([(940, 280), c(pi, "b")])
    d.arrow([(940, 362), c(val, "t")])
    d.arrow([(1050, 341), (1066, 341), (1066, 301), (1050, 301)])
    d.text(1072, 325, "h_t", 10.5, INK2, anchor="start")
    d.note(24, 320, "h_t is per agent; zeroed when info['born'] marks a new agent in the slot.")
    d.note(24, 338, "Grid ÷ 255 on the way in. Network on MPS (~1 ms for 256 agents), world on CPU.")
    return d.save()

# =========================================================================================== 6 training
def training():
    d = Diagram("06_training_loop", 1060, 520, "Training loop (one iteration ≈ 32k agent-steps)",
                "Reward is exactly the environment's homeostatic reward: no social term, no shaping, no curiosity bonus.")
    world = d.box(20, 100, 220, 92, "Training world", ["seed 1000 + k", "new world every 40 iterations", "1-year spin-up first"], "sim")
    roll = d.box(290, 100, 230, 92, "Rollout", ["128 ticks × 256 agents", "a ~ π(· | o, h), all in parallel", "reset h where born"], "agent")
    pre = d.box(570, 100, 230, 92, "Advantages", ["GAE  γ 0.995, λ 0.95", "bootstrap cut at death", "(next obs is a newborn)"], "agent")
    upd = d.box(820, 230, 220, 120, "PPO update", ["4 epochs × 4 minibatches", "clip 0.2, value 0.5, ent 0.01", "Adam 3e-4, grad-norm 0.5", "BPTT over the full rollout"], "agent")
    ck = d.box(570, 260, 220, 66, "Checkpoint", ["checkpoints/last.pt", "resumable"], "neutral")
    rec = d.box(290, 260, 230, 66, "Flight recorder", ["obs-free records + value,", "action probs, log-prob"], "data")
    ev = d.box(570, 380, 220, 92, "Held-out mini-evaluation", ["every 100 iterations", "worlds 0-1, 0.5 yr", "progress only, not claims"], "analysis", tag="caveat")
    d.arrow([c(world, "r"), c(roll, "l")])
    d.arrow([c(roll, "r"), c(pre, "l")])
    d.arrow([c(pre, "r"), (930, 146), c(upd, "t")])
    d.arrow([(820, 293), c(ck, "r")])
    d.arrow([c(ck, "t"), (680, 226), (440, 226), (440, 192)], "next iteration", at=(560, 220))
    d.arrow([(340, 192), (340, 260)], "every k-th rollout", at=(346, 230), anchor="start")
    d.arrow([c(ck, "b"), c(ev, "t")])
    d.note(130, 214, "when k mod 40 = 0: next seed", anchor="middle")
    d.note(24, 500, "Pilots: 600 iterations = 19.7M agent-steps, ≈50 min on an M3 Pro (MPS). Replication: 900 iterations × 3 seeds × 2 arms (paused).")
    return d.save()

# =========================================================================================== 7 evaluation
def evaluation():
    d = Diagram("07_evaluation_protocol", 1060, 560, "Evaluation protocol and decision rule",
                "Identical for learners and baselines; hypotheses and the rule are committed before any evaluation.")
    pol = d.box(20, 90, 230, 92, "Policies under test", ["learner checkpoints", "random · greedy · heuristic", "(always all three)"], "agent")
    prot = d.box(290, 90, 300, 92, "Same protocol", ["held-out worlds 0-4, never trained on", "2 simulated years (includes winter)", "256 agents, respawn, sampled actions"], "sim")
    rec = d.box(630, 90, 410, 92, "Recorded runs", ["events.npy, metrics.csv, full trajectories", "lifetimes measured from end of spin-up", "(regression test: the spin-up bug halved mortality)"], "data")
    met = d.box(20, 230, 300, 108, "Metrics", ["deaths per agent-year", "Kaplan-Meier S(1 yr), median lifetime", "cause-specific death rates", "mortality + energy by season"], "analysis")
    st = d.box(360, 230, 280, 108, "Statistics", ["unit: world (pilot) / training seed", "95% bootstrap CI", "Hedges' g vs every baseline", "paired wins per world"], "analysis")
    prereg = d.box(680, 230, 360, 108, "PREREGISTRATION.md", ["timestamped, committed before evaluation", "H1-H4 (pilot), H2', H3', H5-H7 (replication)", "entries never edited; corrections appended"], "neutral")
    dec = d.box(360, 380, 280, 66, "Do the 95% CIs overlap?", (), "neutral")
    yes = d.box(20, 380, 300, 66, "“not distinguishable at n”", ["never reported as a win"], "neutral")
    no = d.box(680, 380, 360, 66, "“A beats B”", ["direction + effect size reported"], "neutral")
    out = d.box(20, 480, 1020, 54, "Report: supported / not supported / mixed  ·  exploratory analyses labelled  ·  negative results logged in AGENT_LOG", (), "neutral", fs=12)
    d.arrow([c(pol, "r"), c(prot, "l")])
    d.arrow([c(prot, "r"), c(rec, "l")])
    d.arrow([c(rec, "b"), (835, 206), (170, 206), c(met, "t")])
    d.arrow([c(met, "r"), c(st, "l")])
    d.arrow([c(st, "b"), c(dec, "t")])
    d.arrow([c(prereg, "b"), (860, 360), (600, 360), (600, 380)])
    d.arrow([c(dec, "l"), c(yes, "r")], "yes", at=(340, 405))
    d.arrow([c(dec, "r"), c(no, "l")], "no", at=(660, 405))
    d.arrow([c(yes, "b"), (170, 480)])
    d.arrow([c(no, "b"), (860, 480)])
    return d.save()


# =========================================================================================== 8 emergence
def emergence():
    d = Diagram("08_emergence_measurement", 1060, 560, "Measuring emergence (Stage 3-4, planned)",
                "A phenomenon counts only if it beats a null that keeps the ecological confounds. Validated on planted ground truth first.")
    src = d.box(20, 92, 230, 92, "Recordings", ["trajectories, events with", "provider uid, births + genes,", "positions, signals"], "data")
    det = d.box(290, 92, 420, 92, "Detectors (operational definitions)", ["aggregation · co-feeding · food transfer · reciprocity", "cooperative hunts · kin bias · signalling · listening", "displacement · traditions · ratchet · norms"], "analysis", dashed=True)
    null = d.box(790, 92, 250, 92, "Null models", ["shuffled timing / identity,", "water-distance relocation,", "baseline agents, rewiring"], "analysis", dashed=True)
    val = d.box(790, 220, 250, 92, "Validation (unit tests)", ["planted clustering, planted", "signal-context MI, sharing;", "baselines must give nulls"], "neutral", dashed=True)
    cmpb = d.box(290, 220, 420, 92, "Detector vs null, over time", ["effect size + 95% band per phenomenon", "onset = first sustained exceedance of the band"], "analysis", dashed=True)
    order = d.box(290, 350, 420, 70, "Emergence order per run", ["ranked onsets, with triggering conditions"], "analysis", dashed=True)
    hum = d.box(790, 350, 250, 70, "Human timeline", ["cited; else UNVERIFIED"], "neutral", dashed=True)
    out = d.box(290, 460, 750, 66, "Alignment: Kendall τ over onset ranks + condition match", ["reported as a structural analogy only; no claims about minds or early humans"], "analysis", dashed=True)
    sw = d.box(20, 220, 230, 200, "Single-variable experiments", ["A memory / world model ×", "   patchy vs uniform", "B social learning on/off", "C communication on/off", "D famine / flood resilience", "E 50-seed rerun-the-tape", "(one switch at a time)"], "agent", dashed=True)
    d.arrow([c(src, "r"), c(det, "l")])
    d.arrow([c(null, "l"), c(det, "r")], "compared with", at=(750, 126), lfs=10)
    d.arrow([c(det, "b"), c(cmpb, "t")])
    d.arrow([c(val, "l"), c(cmpb, "r")], "validates", at=(750, 254), lfs=10)
    d.arrow([c(cmpb, "b"), c(order, "t")])
    d.arrow([c(order, "b"), (500, 460)])
    d.arrow([c(hum, "b"), (915, 460)])
    d.arrow([c(sw, "t"), c(src, "b")])
    return d.save()

# =========================================================================================== 9 reward
def reward():
    d = Diagram("09_homeostatic_reward", 1060, 430, "The only reward: own homeostasis (Keramati & Gutkin 2014)",
                "Nothing about other agents enters the reward. A unit test recomputes it from the body alone every tick.")
    acts = d.box(20, 96, 240, 120, "What the agent does", ["eat, drink, rest, hunt, shelter", "move, carry, (signal, strike,", "build when switched on)"], "agent")
    world = d.box(20, 236, 240, 120, "What the world does", ["metabolism, heat, cold,", "predators, fire, kicks,", "season, weather, drought"], "sim")
    body = d.box(310, 150, 240, 150, "Body state h", ["energy  e", "hydration  w", "health  η", "warmth  θ", "all in [0, 1]"], "sim")
    drive = d.box(600, 96, 440, 92, "Drive", ["D(h) = ‖ max(h* − h, 0) ‖₂", "setpoint h* = (1, 1, 1, 0.8)"], "analysis", fs=13, lfs=12)
    rew = d.box(600, 216, 440, 92, "Reward", ["r_t = D(h_t) − D(h_t+1)  +  c_death · died_t", "c_death = −1   (reproduction bonus 0 by default)"], "analysis", fs=13, lfs=12)
    no = d.box(600, 330, 440, 66, "Never in the reward", ["others' state · sharing · proximity · signals · shaping"], "neutral", dashed=True)
    d.arrow([c(acts, "r"), (310, 190)])
    d.arrow([c(world, "r"), (310, 262)])
    d.arrow([(550, 200), (600, 150)])
    d.arrow([c(drive, "b"), c(rew, "t")])
    d.note(24, 400, "Learners see r_t ÷ running std of discounted returns (scale only; sign and ordering unchanged).")
    return d.save()


if __name__ == "__main__":
    for f in (system, worldgen, tick, observation, network, training, evaluation, emergence, reward):
        print(f().name)
