// The two VIP slide decks: Semester IV concept presentation and Semester V technical presentation.
//   node docs/vip/build/build_decks.js
// Structured decks: one theme, named layouts; result charts are images (make_vip_charts.py) because Keynote does not
// render pptxgenjs charts; numbers come from reports/ (see notes on each slide).
const path = require("path");
const fs = require("fs");
const sharp = require("sharp");
const React = require("react");
const { renderToStaticMarkup } = require("react-dom/server");
const fa = require("react-icons/fa6");
const pptxgen = require("pptxgenjs");
const { applyTheme } = require(process.env.PPTX_SKILL_THEME || "./apply_theme.js");

const ROOT = path.resolve(__dirname, "..", "..", "..");
const OUT = path.resolve(__dirname, "..");
const THEME = {
  name: "GENESIS", headFontFace: "Arial", bodyFontFace: "Calibri",
  colors: { dk1: "1B1B1A", lt1: "FFFFFF", dk2: "0F2A3D", lt2: "EEF3F7", accent1: "2A78D6", accent2: "EB6834",
            accent3: "1BAF7A", accent4: "C98500", accent5: "D55181", accent6: "8A8984", hlink: "2A78D6", folHlink: "6D4C9F" },
};
const HEX = THEME.colors;
const W = 13.333, H = 7.5, M = 0.6;

async function icon(name, color, px = 256) {
  const svg = renderToStaticMarkup(React.createElement(fa[name], { color: "#" + color, size: px }));
  const buf = await sharp(Buffer.from(svg)).resize(px, px).png().toBuffer();
  return "image/png;base64," + buf.toString("base64");
}

async function fitImage(slide, rel, x, y, w, h, opts = {}) {
  const file = path.join(ROOT, rel);
  const m = await sharp(file).metadata();
  const r = Math.min(w / m.width, h / m.height);
  const iw = m.width * r, ih = m.height * r;
  slide.addImage({ path: file, x: x + (w - iw) / 2, y: y + (opts.top ? 0 : (h - ih) / 2), w: iw, h: ih,
                   altText: opts.alt || path.basename(rel), rounding: false });
}

function setup(title) {
  const pres = new pptxgen();
  pres.layout = "LAYOUT_WIDE";
  pres.author = "Heet Ranpura, Advik Divekar";
  pres.company = "MPSTME NMIMS, VIP vertical";
  pres.title = title;
  pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
  const C = pres.SchemeColor;
  pres.defineSlideMaster({
    title: "DARK", background: { color: HEX.dk2 },
    objects: [{ placeholder: { options: { name: "title", type: "title", x: M, y: 2.0, w: 6.6, h: 2.1, fontSize: 38, bold: true, align: "left",
                color: C.background1, valign: "bottom", fontFace: THEME.headFontFace, margin: 0 }, text: "" } },
              { placeholder: { options: { name: "body", type: "body", x: M, y: 4.25, w: 6.6, h: 1.8, fontSize: 18, align: "left",
                color: "C9D6E2", valign: "top", margin: 0 }, text: "" } }],
  });
  pres.defineSlideMaster({
    title: "CONTENT", background: { color: HEX.lt1 },
    objects: [{ placeholder: { options: { name: "title", type: "title", x: M, y: 0.35, w: W - 2 * M, h: 0.85, fontSize: 30, bold: true, align: "left",
                color: C.text2, valign: "middle", fontFace: THEME.headFontFace, margin: 0 }, text: "" } },
              { text: { text: "GENESIS  |  VIP Design Experience", options: { x: M, y: H - 0.42, w: 6, h: 0.3, fontSize: 10, color: HEX.accent6, margin: 0 } } }],
    slideNumber: { x: W - M - 0.6, y: H - 0.42, w: 0.6, h: 0.3, fontSize: 10, color: HEX.accent6, align: "right" },
  });
  return { pres, C };
}

function contentSlide(pres, section, title, notes) {
  const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: section });
  s.addText(title, { placeholder: "title" });
  if (notes) s.addNotes(notes);
  return s;
}

function darkSlide(pres, section, title, body, notes) {
  const s = pres.addSlide({ masterName: "DARK", sectionTitle: section });
  s.addText(title, { placeholder: "title" });
  if (body) s.addText(body, { placeholder: "body" });
  if (notes) s.addNotes(notes);
  return s;
}

// card with an icon in a coloured circle, a bold header and body text
async function iconCard(s, C, x, y, w, h, ic, color, head, body) {
  s.addShape("roundRect", { x, y, w, h, fill: { color: HEX.lt2 }, line: { color: HEX.lt2 }, rectRadius: 0.12, objectName: "card " + head });
  s.addShape("ellipse", { x: x + 0.25, y: y + 0.28, w: 0.62, h: 0.62, fill: { color }, line: { color }, objectName: "icon circle " + head });
  s.addImage({ data: await icon(ic, "FFFFFF"), x: x + 0.38, y: y + 0.41, w: 0.36, h: 0.36, altText: head });
  s.addText(head, { x: x + 1.05, y: y + 0.22, w: w - 1.25, h: 0.45, fontSize: 16, bold: true, color: C.text2, margin: 0, isTextBox: true, valign: "middle" });
  s.addText(body, { x: x + 1.05, y: y + 0.68, w: w - 1.25, h: h - 0.85, fontSize: 14, color: C.text1, margin: 0, isTextBox: true, valign: "top" });
}

function stat(s, C, x, y, w, big, label, color) {
  s.addText(big, { x, y, w, h: 0.95, fontSize: 44, bold: true, color, margin: 0, isTextBox: true, fontFace: THEME.headFontFace });
  s.addText(label, { x, y: y + 0.95, w, h: 0.8, fontSize: 14, color: C.text1, margin: 0, isTextBox: true, valign: "top" });
}

function bulletsBox(s, C, items, x, y, w, h, size = 16) {
  s.addText(items.map((t, i) => ({ text: t, options: { bullet: true, breakLine: i < items.length - 1, paraSpaceAfter: 8 } })),
            { x, y, w, h, fontSize: size, color: C.text1, valign: "top", margin: 0, isTextBox: true });
}

// ------------------------------------------------------------------------------------------- Sem IV deck
async function conceptDeck() {
  const { pres, C } = setup("GENESIS: concept presentation");
  pres.addSection({ title: "Opening" });
  let s = darkSlide(pres, "Opening", "Do learning agents build societies when nothing social is rewarded?",
    "GENESIS  ·  Concept presentation  ·  VIP Design Experience I (Semester IV)\nHeet Ranpura, Advik Divekar  ·  Mentor: [to fill]",
    "Open with the question. GENESIS is a testbed where every agent is paid only for keeping its own body healthy; we measure what social behaviour appears anyway.");
  s.addImage({ path: path.join(ROOT, "reports/viewer/02_overview.jpg"), x: 7.6, y: 0, w: W - 7.6, h: H, sizing: { type: "cover", w: W - 7.6, h: H }, altText: "Live viewer overview of a GENESIS world" });

  pres.addSection({ title: "Problem" });
  s = contentSlide(pres, "Problem", "Most multi-agent AI studies build cooperation into the reward",
    "Team rewards, prosocial bonuses, influence terms and communication losses all specify the social outcome. That makes it impossible to say whether cooperation emerged. GENESIS removes every social term.");
  await fitImage(s, "docs/vip/diagrams/v01_problem_context.png", M, 1.35, W - 2 * M, 5.6);

  s = contentSlide(pres, "Problem", "Why the question matters",
    "Four reasons, one per card. Keep the safety framing concrete: autonomous agents already share markets, roads and the web.");
  const why = [["FaShieldHalved", HEX.accent1, "Multi-agent AI safety", "Agents from different developers already share markets, roads and the web. Which collective behaviours appear without being designed in?"],
               ["FaUsers", HEX.accent2, "Computational social science", "Grow societies bottom-up, like Sugarscape, but with learning instead of hand-written rules."],
               ["FaLandmark", HEX.accent3, "Origins of social organisation", "Compare the order in which structure appears with human prehistory, as a structural analogy only."],
               ["FaScaleBalanced", HEX.accent4, "Trustworthy methods", "Preregistration, held-out worlds and baselines answer known reproducibility problems in deep RL."]];
  for (let i = 0; i < 4; i++) await iconCard(s, C, M + (i % 2) * 6.15, 1.5 + Math.floor(i / 2) * 2.7, 5.95, 2.45, ...why[i]);

  pres.addSection({ title: "Literature" });
  s = contentSlide(pres, "Literature", "Where GENESIS sits among prior work",
    "Positions are our qualitative reading of each paper's reward design and environment, not measurements. All references verified online on 27 Oct 2026.");
  await fitImage(s, "docs/vip/diagrams/v11_literature_map.png", M, 1.3, 7.4, 5.7);
  bulletsBox(s, C, ["Artificial-life worlds (PolyWorld, Sugarscape) are rich but not learned", "Social-dilemma RL uses selfish reward in small, purpose-built games",
    "Cooperation-boosting methods add a social term to the reward", "Generative agents get their social behaviour from language models",
    "**Gap:** survival-only learners in a rich ecology, measured against nulls"].map((t) => t.replace(/\*\*/g, "")), 8.4, 1.6, 4.35, 5.2, 16);

  s = contentSlide(pres, "Literature", "The gap: four ingredients no prior system combines",
    "These four ingredients define the project. Each one maps to a hard rule or a design feature.");
  const gap = [["1", "Own-body reward only", "Homeostatic reward (Keramati & Gutkin 2014); no term depends on others"],
               ["2", "An ecology where cooperation could pay", "Large prey need two hunters; food can be carried; water is scarce; winters bite"],
               ["3", "Capabilities switched on one at a time", "Memory, communication, reproduction, social learning: all off by default"],
               ["4", "Preregistered measurement", "Held-out worlds, three baselines, null models, negative results kept"]];
  gap.forEach(([n, h, b], i) => {
    const y = 1.5 + i * 1.35;
    s.addShape("ellipse", { x: M, y, w: 0.85, h: 0.85, fill: { color: HEX.accent1 }, line: { color: HEX.accent1 }, objectName: "number " + n });
    s.addText(n, { x: M, y, w: 0.85, h: 0.85, fontSize: 24, bold: true, color: C.background1, align: "center", valign: "middle", margin: 0, isTextBox: true });
    s.addText(h, { x: M + 1.15, y: y - 0.02, w: 11, h: 0.45, fontSize: 20, bold: true, color: C.text2, margin: 0, isTextBox: true });
    s.addText(b, { x: M + 1.15, y: y + 0.43, w: 11, h: 0.45, fontSize: 15, color: C.text1, margin: 0, isTextBox: true });
  });

  pres.addSection({ title: "Concept" });
  s = contentSlide(pres, "Concept", "Design principles we will not break",
    "Each rule is enforced in code or tests, not just in documentation.");
  const rules = [["FaLock", HEX.accent2, "Reward = own homeostasis", "Enforced by a unit test"], ["FaToggleOff", HEX.accent1, "Capabilities off by default", "Each one a config switch"],
                 ["FaRotate", HEX.accent3, "Seeded and reproducible", "Bit-identical runs, hash RNG"], ["FaUsers", HEX.accent4, "No scripted agents", "Animals scripted, agents never"],
                 ["FaChartLine", HEX.accent5, "Baselines everywhere", "Random, greedy, heuristic"]];
  for (let i = 0; i < 5; i++) {
    const x = M + i * 2.45;
    s.addShape("roundRect", { x, y: 1.7, w: 2.25, h: 4.6, fill: { color: HEX.lt2 }, line: { color: HEX.lt2 }, rectRadius: 0.12, objectName: "rule card " + i });
    s.addShape("ellipse", { x: x + 0.66, y: 2.05, w: 0.93, h: 0.93, fill: { color: rules[i][1] }, line: { color: rules[i][1] }, objectName: "rule icon " + i });
    s.addImage({ data: await icon(rules[i][0], "FFFFFF"), x: x + 0.89, y: 2.28, w: 0.47, h: 0.47, altText: rules[i][2] });
    s.addText(rules[i][2], { x: x + 0.15, y: 3.3, w: 1.95, h: 1.3, fontSize: 17, bold: true, color: C.text2, align: "center", valign: "top", margin: 0, isTextBox: true });
    s.addText(rules[i][3], { x: x + 0.15, y: 4.75, w: 1.95, h: 1.2, fontSize: 14, color: C.text1, align: "center", valign: "top", margin: 0, isTextBox: true });
  }

  s = contentSlide(pres, "Concept", "The world: a seeded, seasonal island ecology",
    "Every seed gives a different island. Fresh water is only in rivers and lakes, so it is spatially scarce.");
  await fitImage(s, "reports/stage1/fig1_terrain_seeds.png", M, 1.25, W - 2 * M, 3.6);
  bulletsBox(s, C, ["256 x 256 cells, ten terrain types, rivers and lakes", "Plants, fruit trees, small prey, large prey (need 2 hunters), predators",
    "Animals limited by food, not by caps"], M, 5.15, 5.9, 1.9, 16);
  bulletsBox(s, C, ["Day and night, four seasons, rain, fog, droughts, cold snaps", "Wildfires, floods and famines, random or triggered",
    "Ocean is salt: fresh water only in rivers and lakes"], M + 6.2, 5.15, 5.9, 1.9, 16);

  s = contentSlide(pres, "Concept", "The agent is paid only for keeping its own body healthy",
    "Reward is the reduction in the agent's distance below its set-point, minus a death penalty. Nothing in it refers to other agents.");
  await fitImage(s, "reports/stage2/pilot_preprint/diagrams/09_homeostatic_reward.png", M, 1.3, W - 2 * M, 5.7);

  s = contentSlide(pres, "Concept", "System concept: five subsystems with clear interfaces",
    "Owners and interfaces were fixed at concept stage so two people could work in parallel.");
  await fitImage(s, "docs/vip/diagrams/v02_subsystems.png", M, 1.3, W - 2 * M, 5.7);

  pres.addSection({ title: "Feasibility" });
  s = contentSlide(pres, "Feasibility", "Feasible on two laptops, without a GPU cluster",
    "Measured in Stage 1/1.1 and the first learner runs: benchmark on the MacBook Air M4, viewer on the M3 Pro.");
  stat(s, C, M, 1.7, 2.9, "112k", "environment steps per minute with observations, 1,000 agents", HEX.accent1);
  stat(s, C, M + 3.1, 1.7, 2.9, "~1 h", "to train 20M agent-steps of PPO on a laptop", HEX.accent2);
  stat(s, C, M + 6.2, 1.7, 2.9, "63 MB", "to record every agent for one simulated year", HEX.accent3);
  stat(s, C, M + 9.3, 1.7, 2.9, "75-90", "frames per second in the live viewer at 1,000 agents", HEX.accent5);
  s.addShape("roundRect", { x: M, y: 4.4, w: W - 2 * M, h: 1.55, fill: { color: HEX.lt2 }, line: { color: HEX.lt2 }, rectRadius: 0.12, objectName: "risk box" });
  s.addText([{ text: "Main risk at concept stage", options: { bold: true, color: HEX.dk2, breakLine: true, fontSize: 18 } },
             { text: "Learners might never discover cooperation, e.g. group hunting. That is still a result: the testbed measures its absence rigorously, against null models.", options: { fontSize: 16 } }],
            { x: M + 0.3, y: 4.58, w: W - 2 * M - 0.6, h: 1.2, color: C.text1, valign: "top", margin: 0, isTextBox: true });

  s = contentSlide(pres, "Feasibility", "Research questions and how each will be answered",
    "Hypotheses are preregistered in docs/PREREGISTRATION.md before the runs that test them.");
  s.addTable([
    [{ text: "Question", options: { bold: true, fill: { color: "E3EAF4" } } }, { text: "Measured by", options: { bold: true, fill: { color: "E3EAF4" } } }],
    ["RQ1  Do survival-only learners outlive hand-written foragers?", "Deaths per agent-year on held-out worlds vs three baselines"],
    ["RQ2  Does memory matter when resources leave view?", "GRU vs memoryless: thirst and all-cause deaths"],
    ["RQ3  Does social organisation appear without social reward?", "Group hunts, food transfer, aggregation vs null models, blinding ablation"],
    ["RQ4  In what order, under which capabilities?", "Capability switches; cited human timeline as analogy"],
  ], { x: M, y: 1.5, w: W - 2 * M, colW: [6.4, W - 2 * M - 6.4], fontSize: 16, color: HEX.dk1, border: { type: "solid", pt: 0.75, color: "C9C8C3" }, valign: "middle", rowH: 0.85 });

  pres.addSection({ title: "Plan" });
  s = contentSlide(pres, "Plan", "Three-semester roadmap", "Semester IV defines; Semester V builds and validates; Semester VI integrates and publishes.");
  await fitImage(s, "docs/vip/diagrams/v04_roadmap.png", M, 1.3, W - 2 * M, 5.7);
  s = contentSlide(pres, "Plan", "Team and roles in the VIP vertical", "Mentor and open roles are left for the team to fill; nothing here is invented.");
  await fitImage(s, "docs/vip/diagrams/v03_team_structure.png", M, 1.3, W - 2 * M, 5.7);

  darkSlide(pres, "Plan", "Next: build the world engine", "Stage 1: seeded world, ecology, physiology, baselines, tests and benchmarks.\nQuestions?",
    "Close by stating the first build milestone and its acceptance criteria.");
  await pres.writeFile({ fileName: path.join(OUT, "sem4", "06_Concept_Presentation.pptx") });
  await applyTheme(path.join(OUT, "sem4", "06_Concept_Presentation.pptx"), THEME);
  console.log("wrote docs/vip/sem4/06_Concept_Presentation.pptx");
}

// ------------------------------------------------------------------------------------------- Sem V deck
async function technicalDeck() {
  const { pres, C } = setup("GENESIS: technical presentation");
  pres.addSection({ title: "Opening" });
  let s = darkSlide(pres, "Opening", "GENESIS: what we built, and what the agents did",
    "Technical presentation  ·  VIP Design Experience II (Semester V)\nHeet Ranpura, Advik Divekar  ·  Mentor: [to fill]",
    "Recap the question in one sentence, then show the system and the first results.");
  s.addImage({ path: path.join(ROOT, "reports/viewer/03_inspector_follow.jpg"), x: 7.6, y: 0, w: W - 7.6, h: H, sizing: { type: "cover", w: W - 7.6, h: H }, altText: "Viewer following one agent with the inspector open" });

  s = contentSlide(pres, "Opening", "Semester V in numbers", "Counts from the repository on 27 Oct 2026.");
  stat(s, C, M, 1.7, 2.9, "~8,000", "lines of Python, plus 2,600 lines of JavaScript", HEX.accent1);
  stat(s, C, M + 3.1, 1.7, 2.9, "59", "automated tests, run by CI on every change", HEX.accent3);
  stat(s, C, M + 6.2, 1.7, 2.9, "112k", "env steps/min with observations at 1,000 agents", HEX.accent2);
  stat(s, C, M + 9.3, 1.7, 2.9, "4", "pull requests reviewed and merged between sub-teams", HEX.accent5);
  bulletsBox(s, C, ["World engine (Stages 1, 1.1)", "Recurrent PPO learner and memoryless ablation (Stage 2)", "Flight recorder with bit-exact replay",
    "Live 3-D viewer (Stage 5)", "Preregistered evaluation, CI, documentation"], M, 4.3, W - 2 * M, 2.4, 16);

  pres.addSection({ title: "System" });
  s = contentSlide(pres, "System", "Architecture", "One seeded simulator, interchangeable policies, everything recorded; analysis never feeds back into reward. Diagram: Advik Divekar.");
  await fitImage(s, "reports/stage2/pilot_preprint/diagrams/01_system_architecture.png", M, 1.3, W - 2 * M, 5.7);
  s = contentSlide(pres, "System", "One tick is one compiled call", "Fusing all per-tick work into one Numba call removed Python dispatch overhead; observations come from an incrementally patched image.");
  await fitImage(s, "reports/stage2/pilot_preprint/diagrams/03_world_tick.png", M, 1.3, W - 2 * M, 5.7);
  s = contentSlide(pres, "System", "Throughput clears the 100k target", "make bench on the MacBook Air M4, medians of repeated runs. Source: reports/stage1/bench.txt.");
  await fitImage(s, "docs/vip/diagrams/v12_throughput.png", M, 1.3, 7.7, 5.6);
  bulletsBox(s, C, ["Target: 100k steps/min with observations at 1,000 agents", "Measured: 112k (1.87M agent-steps per second)",
    "Incremental observation image and precomputed line of sight were the key changes", "Known regression at 512 x 512 (open issue)"], 8.6, 1.6, 4.15, 5.2, 16);

  s = contentSlide(pres, "System", "Live 3-D viewer", "The browser only renders; simulation stays in the server's sim thread, so watching never changes results.");
  await fitImage(s, "reports/viewer/01_day_closeup.jpg", M, 1.35, 6.0, 4.1);
  await fitImage(s, "reports/viewer/02_overview.jpg", M + 6.15, 1.35, 6.0, 4.1);
  bulletsBox(s, C, ["85-115 fps with 256 agents, 75-90 fps with 1,000 (1600 x 900, M3 Pro)", "Dashboard of 7 live charts, inspector showing each agent's value estimate and action probabilities; every control logged"], M, 5.65, W - 2 * M, 1.3, 15);

  s = contentSlide(pres, "System", "One shared network, per-agent memory", "Conv encoder for the 11 x 11 x 16 view, MLP for body signals, GRU or MLP core; PPO with full backpropagation through 128-tick rollouts.");
  await fitImage(s, "reports/stage2/pilot_preprint/diagrams/05_policy_network.png", M, 1.3, W - 2 * M, 5.7);

  pres.addSection({ title: "Results" });
  s = contentSlide(pres, "Results", "Learners beat random and greedy, but not yet the heuristic",
    "Held-out worlds 0-4 x 2 years x 256 agents; 95% bootstrap CIs in reports/. Replication runs with the unified PPOTrainer, 19.7M agent-steps each.");
  await fitImage(s, "docs/vip/diagrams/v13_mortality.png", M, 1.3, 7.7, 5.6);
  bulletsBox(s, C, ["H1 supported: learners clearly beat random (3.83) and greedy (3.31)", "H2 not supported: memoryless 1.50 [1.30, 1.75] and GRU 1.67 [1.45, 1.92] overlap the heuristic 1.50 [1.04, 2.11]",
    "H3 not supported: memory cut thirst deaths but not overall mortality", "One training seed per arm, so make h2h3 runs three"], 8.6, 1.6, 4.15, 5.2, 15);

  s = contentSlide(pres, "Results", "Why learners plateau: they never hunt, so winter starves them",
    "Energy-economy probe on validation worlds: intake per agent by season. Source: reports/stage2/energy_economy.json.");
  await fitImage(s, "docs/vip/diagrams/v14_energy_by_season.png", M, 1.3, 7.7, 5.6);
  bulletsBox(s, C, ["Plants alone are below basal metabolism: a grazer eats 3.9 per year against 5.0 basal", "Hunting pays: the hunter heuristic eats 5.2 per year and dies far less",
    "Learners eat about twice as much in summer, but under a fifth of the heuristic's intake in winter", "So the plateau is a learning problem, not a limit of the world"], 8.6, 1.6, 4.15, 5.2, 15);

  s = contentSlide(pres, "Results", "What the learners did learn", "Population-level behaviour on test world 0. Source: paper/stats.json.");
  stat(s, C, M, 1.6, 2.9, "0.52", "chance of drinking when parched (0.09 when hydrated)", HEX.accent1);
  stat(s, C, M + 3.1, 1.6, 2.9, "2 cells", "median distance from water (heuristic: 3.6)", HEX.accent3);
  stat(s, C, M + 6.2, 1.6, 2.9, "14-18", "cell home ranges (heuristic: 38): riverside sedentism", HEX.accent2);
  stat(s, C, M + 9.3, 1.6, 2.9, "r = -0.7", "value estimate vs homeostatic drive (-0.65 / -0.75)", HEX.accent5);
  s.addShape("roundRect", { x: M, y: 4.3, w: W - 2 * M, h: 1.55, fill: { color: HEX.lt2 }, line: { color: HEX.lt2 }, rectRadius: 0.12, objectName: "correction box" });
  s.addText([{ text: "A claim we corrected", options: { bold: true, color: HEX.dk2, breakLine: true, fontSize: 18 } },
             { text: "Watching one agent in the viewer suggested learners rest at night. The population data showed no night resting, so the claim was corrected in the paper and the logs.", options: { fontSize: 16 } }],
            { x: M + 0.3, y: 4.48, w: W - 2 * M - 0.6, h: 1.2, color: C.text1, valign: "top", margin: 0, isTextBox: true });

  s = contentSlide(pres, "Results", "No social organisation yet, and we checked carefully",
    "Aggregation is compared with null models that keep each agent's habitat and place preferences; Advik's blinding ablation zeroes the agents' view of each other.");
  await iconCard(s, C, M, 1.5, 3.9, 2.7, "FaUsers", HEX.accent2, "0 group kills", "Large prey need two simultaneous hunters; learners never managed one.");
  await iconCard(s, C, M + 4.1, 1.5, 3.9, 2.7, "FaSeedling", HEX.accent3, "< 0.01 food shared", "Energy received from others per agent-year: essentially none.");
  await iconCard(s, C, M + 8.2, 1.5, 3.9, 2.7, "FaEye", HEX.accent1, "Clustering is habitat", "Nearest-neighbour ratio 0.84 vs habitat null, but 0.98-1.01 vs place preference; blinding leaves it intact.");
  bulletsBox(s, C, ["H4 (no social behaviour at this stage) holds", "The ingredients for cooperation exist in the world; the next semester tests which capabilities unlock them"], M, 4.6, W - 2 * M, 1.8, 17);

  pres.addSection({ title: "Quality" });
  s = contentSlide(pres, "Quality", "Verification: 59 tests, CI on every change", "Coverage: 55% of all lines, 76% of Python-level lines; Numba-compiled kernels run as machine code and cannot be traced.");
  await fitImage(s, "docs/vip/diagrams/v06_verification_levels.png", M, 1.3, 8.0, 5.7);
  bulletsBox(s, C, ["Bit-identical seeded runs catch any unintended change to dynamics", "Replay of any recording is exact", "Defects found and fixed are logged in the Test & Validation Plan",
    "GitHub Actions runs everything on a clean Linux machine"], 9.0, 1.6, 3.75, 5.2, 15);
  s = contentSlide(pres, "Quality", "How the sub-teams work together", "Real workflow: four PRs from Advik's fork, reviewed and merged by Heet, with handoff documents after each merge.");
  await fitImage(s, "docs/vip/diagrams/v08_collaboration.png", M, 1.3, W - 2 * M, 5.7);
  s = contentSlide(pres, "Quality", "Risks we are tracking", "R1 is the main scientific risk; R2 is addressed by the three-seed experiment.");
  await fitImage(s, "docs/vip/diagrams/v09_risk_matrix.png", M, 1.3, W - 2 * M, 5.7);

  pres.addSection({ title: "Next" });
  s = contentSlide(pres, "Next", "Next: settle H2 and H3, then measure social structure", "make h2h3 is preregistered and ready; it runs on the M3 Pro.");
  await iconCard(s, C, M, 1.6, 3.9, 3.2, "FaFlask", HEX.accent1, "Now: make h2h3", "6 runs (GRU and memoryless x seeds 1-3) x 3,000 iterations, about 13-16 h on the M3 Pro; verdicts applied automatically.");
  await iconCard(s, C, M + 4.1, 1.6, 3.9, 3.2, "FaSeedling", HEX.accent3, "Then: hunting probes", "Is hunting an exploration problem? A curriculum and a denser-prey world, each as a separate, labelled configuration.");
  await iconCard(s, C, M + 8.2, 1.6, 3.9, 3.2, "FaUsers", HEX.accent2, "Sem VI: social metrics", "Co-feeding, food transfer, proximity networks, signalling, each tested on synthetic data first; then the capability experiments.");
  darkSlide(pres, "Next", "Thank you", "Repository: github.com/HeetRanpura/Genesis\nLive demo: make demo  ·  Questions?", "Offer the live demo: make demo starts the viewer with the trained GRU learner.");
  const f = path.join(OUT, "sem5", "04_Technical_Presentation.pptx");
  await pres.writeFile({ fileName: f });
  await applyTheme(f, THEME);
  console.log("wrote docs/vip/sem5/04_Technical_Presentation.pptx");
}

(async () => { await conceptDeck(); await technicalDeck(); })().catch((e) => { console.error(e); process.exit(1); });
