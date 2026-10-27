// Semester IV deliverables (VIP Design Experience I): problem definition, literature and technology survey,
// initial system concept, team roles, documentation templates.   node docs/vip/build/build_sem4_reports.js
const { P, H1, H2, H3, bullets, numbered, pageBreak, callout, table, figure, titlePage, save } = require("./lib_docx");
const REFS = require("./refs");

const COURSE4 = "Design Experience I (Semester IV), VIP vertical";
const cite = (...k) => "[" + k.join("; ") + "]";

async function problemDefinition() {
  const c = [
    ...titlePage({ title: "Problem Definition Report", subtitle: "What we study, why it matters, and how success is judged", course: COURSE4 }),
    H1("1. Summary"),
    callout("Problem in one paragraph", [
      "When many AI agents share a world, do they develop social organisation (grouping, sharing, signalling, division of labour) on their own, or only when their designers reward it? Most multi-agent reinforcement-learning studies put a social term into the reward, so the answer is built in. **GENESIS** removes every social term: each agent is rewarded only for keeping its own body near a healthy set-point. We build the world, the learners and the measurement tools needed to see what social structure, if any, appears, and in what order.",
    ]),
    P("This report fixes the problem statement, research questions, scope, constraints and success criteria for the VIP project. The project sits in the VIP domain **Artificial Intelligence & Data Systems**."),
    H1("2. Background and motivation"),
    H2("2.1 Cooperation is usually specified, not discovered"),
    P(`Influential multi-agent RL work studies cooperation by changing the reward: shared team rewards, inequity-averse utilities ${cite("Hughes2018")}, rewards for influencing others ${cite("Jaques2019")}, or communication objectives ${cite("Foerster2016", "Lazaridou2017")}. These designs are useful for engineering cooperative systems, but they cannot tell us whether cooperation would emerge without being asked for. Sequential social dilemmas ${cite("Leibo2017", "Perolat2017")} come closest to selfish rewards, but their environments are built around a specific dilemma.`),
    H2("2.2 Why the question matters beyond the lab"),
    ...bullets([
      "**Multi-agent AI deployments.** Autonomous agents (trading systems, fleets of robots, LLM-based assistants acting on the web) increasingly share environments. Knowing which collective behaviours appear **without** being designed in is a safety question: emergent coordination can be beneficial or harmful.",
      "**Computational social science.** Agent-based models such as Sugarscape " + cite("Epstein1996") + " show that simple rules can produce social patterns. Learning agents let us ask the same question without hand-writing the rules.",
      "**Origins of human social organisation.** Human prehistory shows a broad order of social innovations. A testbed where social structure must emerge from survival pressure allows careful **structural analogies** (never claims of equivalence) with that record.",
      "**Methodology.** Deep RL results are fragile across seeds and reporting choices " + cite("Henderson2018", "Agarwal2021") + ". A testbed with preregistration " + cite("Nosek2018") + ", held-out worlds and baselines in every comparison is itself a contribution.",
    ]),
    ...(await figure("docs/vip/diagrams/v01_problem_context.png", "Figure 1. The research problem: common designs specify social structure in the objective; GENESIS removes it and measures what remains.")),
    H1("3. Problem statement"),
    P("Design, build and validate a reproducible artificial-life testbed in which a population of learning agents, rewarded **only** for their own homeostasis, lives in a seeded, seasonal foraging ecology; and use it to measure, against baselines and null models, whether and how social organisation emerges."),
    H1("4. Research questions and hypotheses"),
    table(["ID", "Question", "How it is answered"], [
      ["RQ1", "Do survival-only learners outlive hand-written foragers?", "Deaths per agent-year on held-out worlds 0-4 vs random, greedy and heuristic baselines (H1, H2)."],
      ["RQ2", "Does memory change behaviour where resources leave view?", "GRU vs memoryless ablation: thirst deaths and all-cause mortality (H3)."],
      ["RQ3", "Does any social organisation appear without social reward?", "Group hunts, food transfer, aggregation against habitat and place-preference nulls, social-blinding ablation (H4, H7)."],
      ["RQ4", "If structure appears, in what order, and under which capabilities?", "Capability switches (communication, reproduction, social learning) and a cited human timeline used as structural analogy (Stages 3-4, Semester VI)."],
    ], [0.08, 0.42, 0.50]),
    P("Hypotheses are preregistered in `docs/PREREGISTRATION.md` with dated entries, written before the runs that test them; corrections are appended, never edited in place."),
    H1("5. Objectives"),
    table(["Semester", "Objective", "Measure of done"], [
      ["IV", "Define the problem, survey literature and tools, fix the concept and team roles", "This report; literature survey; concept report; team charter; concept presentation"],
      ["V", "Build the engine, learners, recorder and viewer; test and validate them", ">= 100k env steps/min with observations at 1,000 agents; bit-identical seeded runs; held-out evaluation; CI"],
      ["V", "Answer RQ1-RQ2 with replication", "Preregistered 3-seed experiment (H2/H3) with decision rules applied"],
      ["VI", "Measure social structure (RQ3-RQ4), integrate, publish", "Stage 3 metrics with synthetic-data tests; arXiv preprint; final demonstration"],
    ], [0.12, 0.48, 0.40]),
    H1("6. Scope and constraints"),
    H2("6.1 In scope"),
    ...bullets(["A procedurally generated 256 x 256 island with terrain, rivers, lakes, vegetation, three animal species, weather, seasons, fire and floods.",
      "Agents with egocentric, occluded vision and a four-variable body (energy, hydration, health, warmth).",
      "Parameter-shared PPO learners with and without memory; three non-learning baselines.",
      "Flight recording of every agent at every tick, exact replay, statistical analysis, a live 3-D viewer."]),
    H2("6.2 Out of scope"),
    ...bullets(["Any reward term that depends on other agents (forbidden by design, enforced by a unit test).",
      "Scripted agent behaviour (animals are scripted because they are the environment; agents never are).",
      "Claims that agents have feelings or human-like minds; human comparisons are structural analogies only.",
      "Large-cluster training: everything must run on student laptops."]),
    H2("6.3 Hard rules (non-negotiable)"),
    ...numbered(["Reward = own homeostasis only, plus a death penalty.",
      "Every capability (memory, communication, reproduction, social learning, ...) is a configuration switch, off by default.",
      "Every run is seeded and reproducible; randomness inside the engine comes only from a counter-based hash RNG.",
      "No scripted agent behaviour.",
      "Baselines appear in every comparison; negative results are logged, never dropped."]),
    H2("6.4 Resource constraints"),
    table(["Resource", "Constraint", "Consequence for the design"], [
      ["Compute", "Two laptops: MacBook Air M4 (fanless, 16 GB) and MacBook Pro M3 Pro (18 GB); no CUDA GPU", "Numba-compiled CPU engine; PyTorch on Apple-silicon GPU (MPS); one shared network for all agents"],
      ["Heat", "The Air throttles 15-30% under sustained load", "Long runs on the M3 Pro; benchmarks compared only as interleaved A/B runs"],
      ["Disk", "~10-15 GB free on the Air", "Recording can be thinned (1 rollout in k) and stops automatically below a free-space threshold"],
      ["Time", "Course semesters; two members", "Staged plan with acceptance criteria per stage; handoff documents between sessions"],
    ], [0.14, 0.43, 0.43]),
    H1("7. Stakeholders"),
    table(["Stakeholder", "Interest"], [
      ["Faculty mentor / VIP evaluators", "Technical depth, rigour, documentation, teamwork"],
      ["MARL and artificial-life researchers", "A testbed where social behaviour is measured, not specified; reusable engine and protocol"],
      ["AI-safety community", "Evidence on which collective behaviours appear without being designed in"],
      ["Future VIP members (Sem IV juniors)", "A documented codebase, study guide (TEACHER.md) and onboarding path"],
    ], [0.38, 0.62]),
    H1("8. Success criteria"),
    ...bullets(["**Engineering:** >= 100k environment steps per minute including observations at 1,000 agents on a laptop; bit-identical results for the same seed; every world rule unit-tested.",
      "**Science:** every claim made on held-out worlds against all three baselines, with bootstrap confidence intervals and effect sizes, under preregistered decision rules.",
      "**Honesty:** emergence is never claimed without measurement against a null model; negative results are reported.",
      "**Team:** documented handoffs, reviewed pull requests, a test suite that runs on every change."]),
    H1("9. Ethical considerations"),
    ...bullets(["No human participants or personal data are involved.",
      "Comparisons with human prehistory are framed as structural analogies and cite their sources.",
      "Language about agents avoids anthropomorphism: value estimates and probe results are functional correlates, never feelings.",
      "Compute is kept small (laptops); experiment scale is stated next to every claim."]),
    H1("References"),
    ...REFS.filter(([k]) => ["Epstein1996", "Foerster2016", "Henderson2018", "Agarwal2021", "Hughes2018", "Jaques2019", "Lazaridou2017", "Leibo2017", "Nosek2018", "Perolat2017"].includes(k))
      .map(([k, t]) => P(`[${k}] ${t}`, { run: { size: 18 } })),
  ];
  await save("sem4/01_Problem_Definition_Report.docx", "Problem Definition Report", c);
}

async function literature() {
  const c = [
    ...titlePage({ title: "Literature Review & Technology Survey", subtitle: "Prior work, the gap GENESIS fills, and the tools chosen", course: COURSE4 }),
    H1("1. Method"),
    P("We surveyed work on (a) artificial life and agent-based societies, (b) social dilemmas and cooperation in multi-agent RL, (c) emergent communication, (d) open-ended multi-agent environments, (e) social and cultural learning, (f) homeostatic reward, and (g) methodology for RL experiments. Sources were found through NeurIPS, ICML, ICLR, AAMAS, eLife and Nature-family journals, arXiv, and the reference lists of key papers."),
    callout("Verification policy", ["Every reference below was checked online on 27 October 2026 against the publisher, proceedings or arXiv record (authors, title, venue, year, and volume/pages where listed). Details that could not be confirmed were removed rather than guessed; the reference table marks each entry's status. An earlier draft of the project paper had 24 references written from memory; this list supersedes it."]),
    H1("2. Themes in the literature"),
    H2("2.1 Artificial life and agent-based societies"),
    P(`PolyWorld ${cite("Yaeger1994")} combined genetics, metabolism, Hebbian neural networks and vision in one artificial ecology and reported emergent "species" with distinct survival strategies, including flocking and foraging. Sugarscape ${cite("Epstein1996")} showed that hand-written agent rules on a resource landscape produce trade, migration and inequality. Both establish the value of growing societies from the bottom up; neither uses modern deep RL, and Sugarscape's behaviour is written by the modeller.`),
    H2("2.2 Social dilemmas in multi-agent RL"),
    P(`Leibo et al. ${cite("Leibo2017")} introduced sequential social dilemmas (Gathering, Wolfpack), in which independent deep Q-learners with selfish rewards show cooperation or conflict depending on resource scarcity. Pérolat et al. ${cite("Perolat2017")} used the same approach to model common-pool resource appropriation, linking exclusion, sustainability and inequality. Later work added social terms to the reward to increase cooperation: inequity aversion ${cite("Hughes2018")} and causal social influence ${cite("Jaques2019")}. Melting Pot ${cite("Leibo2021")} turned many such scenarios into an evaluation suite focused on generalisation to new co-players.`),
    H2("2.3 Emergent communication"),
    P(`Foerster et al. ${cite("Foerster2016")} and Lazaridou et al. ${cite("Lazaridou2017")} showed agents can learn communication protocols when a shared objective rewards successful coordination. Lowe et al. ${cite("Lowe2019")} warned that common metrics can report "communication" where messages have no effect on behaviour, which motivates GENESIS's insistence on causal and null-model checks before any claim.`),
    H2("2.4 Open-ended multi-agent environments"),
    P(`Hide-and-seek ${cite("Baker2020")} produced tool use through an autocurriculum, under team rewards. Neural MMO ${cite("Suarez2019")} provides a persistent world with many agents competing for resources; it found that larger populations push agents to fill different niches. Crafter ${cite("Hafner2022")} benchmarks a spectrum of survival skills for a single agent. Johanson et al. ${cite("Johanson2022")} observed emergent bartering in a market-like environment.`),
    H2("2.5 Social and cultural learning"),
    P(`Ndousse et al. ${cite("Ndousse2021")} found that independent RL agents usually fail to learn from experts unless the environment and an auxiliary loss support it. Bhoopchand et al. ${cite("Bhoopchand2023")} demonstrated few-shot imitation as cultural transmission in 3-D worlds. Duéñez-Guzmán et al. ${cite("Duenez2023")} argue that a social path is needed for human-like AI. These works show what social learning can look like and what support it needs, which GENESIS will test as switchable capabilities.`),
    H2("2.6 Homeostatic reward"),
    P(`Keramati and Gutkin ${cite("Keramati2014")} define reward as the reduction of deviation from a physiological set-point and prove that maximising it is equivalent to maintaining homeostasis. GENESIS adopts this definition directly: r_t = D(h_t) - D(h_{t+1}) minus a death penalty, with D the distance below the set-point. Silver et al. ${cite("Silver2021")} argue that reward maximisation may suffice for intelligence in general; GENESIS tests a narrow, measurable version of that claim for social behaviour.`),
    H2("2.7 LLM-based social simulation"),
    P(`Generative agents ${cite("Park2023")} simulate believable social life with large language models. Their social behaviour comes from language models trained on human text and from prompts, so it is supplied rather than learned from survival pressure; GENESIS is the opposite end of this spectrum.`),
    H2("2.8 Methodology for RL experiments"),
    P(`Henderson et al. ${cite("Henderson2018")} showed that seeds, implementations and reporting choices can change deep RL conclusions. Agarwal et al. ${cite("Agarwal2021")} showed point estimates over few runs mislead, and recommend interval estimates. Nosek et al. ${cite("Nosek2018")} argue for preregistration to separate confirmatory tests from exploration. GENESIS applies all three: multiple training seeds as the unit of replication, bootstrap intervals and effect sizes, and dated preregistration.`),
    H2("2.9 Algorithms used"),
    P(`Learners use proximal policy optimisation ${cite("Schulman2017")} with generalised advantage estimation ${cite("Schulman2016")} and a GRU recurrent core ${cite("Cho2014")}. Survival is analysed with Kaplan-Meier estimates ${cite("Kaplan1958")}, bootstrap intervals ${cite("Efron1993")} and Hedges' g ${cite("Hedges1981")}.`),
    pageBreak(),
    H1("3. Comparison and gap"),
    table(["Work", "Reward / objective", "Environment", "What GENESIS takes / changes"], [
      ["PolyWorld (1994), Sugarscape (1996)", "Evolution / hand-written rules", "Rich ecology", "Takes the ecological richness; replaces scripted rules with RL"],
      ["Sequential social dilemmas (2017)", "Selfish", "Small dilemma games", "Takes selfish reward; removes the built-in dilemma framing"],
      ["Inequity aversion (2018), social influence (2019)", "Social term added", "Dilemma games", "Forbids social terms by rule and test"],
      ["Emergent communication (2016-2017)", "Shared objective", "Referential games", "Communication only as a switch; causal + null checks"],
      ["Hide-and-seek (2020)", "Team reward", "Physics sandbox", "No teams; individual survival only"],
      ["Neural MMO (2019)", "Survival-like", "Persistent MMO world", "Adds seasons, physiology, cooperative prey; preregistered tests"],
      ["Crafter (2022)", "Achievements", "Survival, single agent", "Many agents; social measurement"],
      ["Generative agents (2023)", "LLM prompts", "Sims-like town", "No language prior; behaviour must be learned"],
    ], [0.27, 0.18, 0.2, 0.35], { size: 17 }),
    ...(await figure("docs/vip/diagrams/v11_literature_map.png", "Figure 1. Qualitative positioning of prior work by how much social structure the designer supplies and how rich the ecology is (the authors' reading, not a measurement).", 560)),
    callout("Gap", ["No surveyed system combines (1) learning agents whose reward is strictly their own physiology, (2) an ecology rich enough that cooperation, sharing and grouping could pay (large prey needing two hunters, carried food, scarce water, winters), (3) capabilities that can be switched on one at a time, and (4) a preregistered measurement protocol with baselines and null models. GENESIS is designed to fill that gap."]),
    H1("4. Technology survey"),
    P("Each decision below was scored by the team on the criteria that mattered for this project (scores 1-5, weights in brackets). Scores are team judgements recorded at decision time; the measured outcome is noted where one exists."),
    H2("4.1 Simulation engine"),
    table(["Option", "Laptop speed, no CUDA (0.35)", "Determinism control (0.25)", "Dev speed (0.2)", "Ecosystem (0.2)", "Weighted"], [
      ["**Numba (chosen)**", "5", "5", "4", "4", "**4.6**"],
      ["JAX", "3 (no CUDA here)", "4", "3", "4", "3.5"],
      ["Pure NumPy", "2", "5", "5", "4", "3.8"],
      ["Unity ML-Agents " + cite("Juliani2018"), "3", "2", "3", "5", "3.2"],
      ["C++ engine + bindings", "5", "5", "1", "2", "3.7"],
    ], [0.26, 0.2, 0.17, 0.12, 0.12, 0.13], { size: 17 }),
    P("Outcome: the fused Numba tick reaches about **112,000 environment steps per minute with observations at 1,000 agents** on the MacBook Air M4 (Stage 1.1 benchmark), above the 100k target. Numba " + cite("Lam2015") + " compiles Python functions to machine code and supports multi-core loops."),
    H2("4.2 Learning framework"),
    table(["Option", "Apple GPU (0.35)", "Recurrent PPO support (0.25)", "Team familiarity (0.2)", "Debuggability (0.2)", "Weighted"], [
      ["**PyTorch on MPS (chosen)** " + cite("Paszke2019"), "4", "5", "5", "5", "**4.65**"],
      ["JAX / Flax", "2", "4", "2", "3", "2.75"],
      ["TensorFlow", "3", "3", "2", "3", "2.8"],
    ], [0.28, 0.14, 0.17, 0.15, 0.13, 0.13], { size: 17 }),
    P("Outcome: CPU inference was measured about 13x slower than MPS for the shared network, so training uses MPS while sampling stays on a seeded CPU generator for reproducibility."),
    H2("4.3 Environment API"),
    P(`The batched slot-based API serves the learner; a PettingZoo ParallelEnv wrapper ${cite("Terry2021")} makes the world usable by standard multi-agent libraries and is checked by PettingZoo's API test. Gymnasium ${cite("Towers2024")} conventions are followed for observation and action spaces.`),
    H2("4.4 Live viewer"),
    table(["Option", "3-D diorama look (0.3)", "No build step (0.25)", "Frame rate at 1,000 agents (0.25)", "Effort (0.2)", "Weighted"], [
      ["**three.js (chosen)**", "5", "5", "4", "4", "**4.55**"],
      ["PixiJS (original plan, 2-D)", "2", "4", "5", "4", "3.65"],
      ["Unity WebGL", "5", "1", "4", "2", "3.15"],
    ], [0.24, 0.16, 0.15, 0.2, 0.11, 0.14], { size: 17 }),
    P("Outcome: the three.js viewer was measured at 85-115 fps with 256 agents and 75-90 fps with 1,000 agents at 1600x900 (M3 Pro, headless Chromium). The original plan named PixiJS; the change to three.js was a recorded decision on 26 October 2026."),
    H2("4.5 Supporting tools"),
    table(["Need", "Tool", "Why"], [
      ["Environment management", "uv + Python 3.11 venv", "Fast, reproducible installs (`make setup`)"],
      ["Testing", "pytest, pytest-cov, GitHub Actions", "59 tests on every change; coverage report"],
      ["Statistics", "NumPy/SciPy: bootstrap, Kaplan-Meier, Hedges' g", "Matches the preregistered protocol"],
      ["Paper", "LaTeX via Tectonic", "Single-binary build of the preprint"],
      ["Diagrams", "Hand-built SVG (checked label fit), Graphviz, matplotlib", "Reproducible figures from code"],
    ], [0.25, 0.35, 0.4]),
    pageBreak(),
    H1("5. References"),
    table(["Key", "Reference", "Check"], REFS.map(([k, t, s]) => [k, t, s]), [0.15, 0.68, 0.17], { size: 16 }),
    P("\"verified\": confirmed online on 27 October 2026. \"classic\": widely cited work whose details the team is confident of; re-check page numbers before any submission.", { run: { size: 17, italics: true } }),
  ];
  await save("sem4/02_Literature_Review_and_Technology_Survey.docx", "Literature Review & Technology Survey", c);
}

async function concept() {
  const c = [
    ...titlePage({ title: "Initial System Concept & Feasibility", subtitle: "Subsystems, interfaces, key decisions and whether it fits on two laptops", course: COURSE4 }),
    H1("1. Concept overview"),
    P("GENESIS is one seeded simulator, a set of interchangeable policies (baselines or learners), and a recorder that captures every agent at every tick. Analysis reads the recordings; it never feeds back into reward. A live viewer shows any run in 3-D."),
    ...(await figure("reports/stage2/pilot_preprint/diagrams/01_system_architecture.png", "Figure 1. System architecture (diagram by Advik Divekar, from the project's preprint).")),
    H1("2. Subsystems and interfaces"),
    ...(await figure("docs/vip/diagrams/v02_subsystems.png", "Figure 2. Five subsystems with owners and interfaces.")),
    table(["Subsystem", "Responsibility", "Interface", "Verified by"], [
      ["World engine (`sim/`)", "World generation, ecology, physiology, observations", "`GenesisEnv.step(actions) -> obs, reward, died, info`", "World-rule unit tests, determinism tests, benchmark"],
      ["Learning (`agents/`)", "Shared actor-critic, PPO trainer, baselines", "`PolicyAgent.act(obs, alive)`, checkpoint format 2", "Learner tests: determinism, unroll = stepwise, round-trip"],
      ["Measurement (`analysis/`, `experiments/`)", "Recording, replay, survival statistics, behaviour", "`runs/<run>/` (manifest, metrics, events, trajectory)", "Replay is bit-exact; preregistered protocol"],
      ["Visualisation (`viewer/`)", "Live 3-D view, dashboard, disturbance tools", "Binary WebSocket frames; logged controls", "Server smoke test; measured frame rate"],
      ["Research operations", "Rules, preregistration, logs, handoffs, CI", "`docs/`, `logs/`, `HANDOFF*.md`, pull requests", "Review checklist; CI"],
    ], [0.2, 0.28, 0.3, 0.22], { size: 17 }),
    H1("3. The world and the agent"),
    ...(await figure("reports/stage1/fig1_terrain_seeds.png", "Figure 3. Procedurally generated islands for several seeds (Stage 1 evidence).", 560)),
    ...bullets(["**World:** 256 x 256 cells; elevation from fractal noise; lakes by priority-flood " + cite("Barnes2014") + "; rivers by flow accumulation; ten terrain classes. The ocean is salt water, so fresh water is spatially scarce.",
      "**Ecology:** plants with seasonal regrowth, fruiting trees, small prey, large prey that need two or more simultaneous attackers, and predators. Fauna are limited by food, not by caps.",
      "**Climate:** day/night, four seasons, a weather Markov chain (rain, fog), droughts, cold snaps, famines, wildfires and floods.",
      "**Agent body:** energy, hydration, health and warmth. Eleven actions: wait, move (4), rest, eat, drink, hunt, pick up, drop.",
      "**Perception:** an 11 x 11 egocentric, occluded view with 16 channels, plus 13 body signals and nearby-agent features."]),
    ...(await figure("reports/stage2/pilot_preprint/diagrams/09_homeostatic_reward.png", "Figure 4. Homeostatic reward: the agent is paid for reducing its own distance below the set-point (diagram by Advik Divekar).")),
    H1("4. Key design decisions"),
    table(["Decision", "Alternatives considered", "Reason"], [
      ["Shared network for all agents, per-agent memory", "Independent networks per agent", "1,000 independent networks do not fit in 16-18 GB; sharing is standard and keeps learning stable. Independent learners remain an ablation."],
      ["Counter-based hash RNG (seed, tick, index, stream)", "Global NumPy RNG", "Results identical regardless of thread count and order of evaluation"],
      ["One fused Numba call per tick", "Separate kernels called from Python", "Each extra Python-side dispatch costs ~5-10 us; fusing reached the throughput target"],
      ["Large prey need >= 2 attackers", "Single-agent hunting only", "Creates a stag-hunt affordance: cooperation can pay, but is never rewarded directly"],
      ["Held-out test worlds 0-4; training worlds 1000+", "Evaluate on training worlds", "Separates learning from memorising a map"],
    ], [0.28, 0.27, 0.45], { size: 17 }),
    H1("5. Feasibility"),
    table(["Question", "Estimate or measurement", "Verdict"], [
      ["Can the world run fast enough?", "Measured 112k env steps/min with observations at 1,000 agents (MacBook Air M4)", "Feasible"],
      ["Can agents learn on a laptop?", "20M agent-steps of PPO in 1.0-1.7 h on the Air with two runs sharing it; about 50 min on the M3 Pro", "Feasible for 20-100M steps"],
      ["Can we keep all the data?", "Full recording costs ~63 MB per simulated year at 256 agents; thinning to 1 rollout in 20 keeps a 100M-step run near 0.5 GB", "Feasible with thinning"],
      ["Can we show it live?", "75-90 fps at 1,000 agents in the browser (M3 Pro)", "Feasible"],
      ["Can results be trusted?", "Seeded, bit-identical runs; baselines reproduce bit-for-bit across both machines", "Feasible"],
    ], [0.3, 0.52, 0.18], { size: 17 }),
    callout("Main risks at concept stage", ["Learners might never discover cooperative strategies such as group hunting (the most likely negative result). The testbed is still valuable if it measures that absence rigorously, so the plan treats it as a reportable outcome, not a failure.", "Compute and heat on fanless laptops; mitigated by moving long runs to the M3 Pro and thinning recordings."]),
    H1("6. From concept to plan"),
    ...(await figure("docs/vip/diagrams/v04_roadmap.png", "Figure 5. Three-semester roadmap with current status.")),
    H1("References"),
    ...REFS.filter(([k]) => ["Barnes2014"].includes(k)).map(([k, t]) => P(`[${k}] ${t}`, { run: { size: 18 } })),
  ];
  await save("sem4/03_Initial_System_Concept_and_Feasibility.docx", "Initial System Concept & Feasibility", c);
}

async function team() {
  const c = [
    ...titlePage({ title: "Team Role Definition & Working Agreement", subtitle: "Who does what, how we collaborate, and how the team grows vertically", course: COURSE4 }),
    H1("1. Team"),
    callout("Fill before submission", ["Faculty mentor, any Semester VI or Semester IV members, and meeting dates are not recorded in the repository. They are marked [to fill]. Do not invent them: add only real names and real meetings."]),
    table(["Member", "Semester / level", "Role", "Primary responsibilities"], [
      ["[Faculty mentor, to fill]", "Faculty", "Mentor", "Research guidance, milestone reviews, connections to labs and venues"],
      ["Heet Ranpura", "Sem V (mid-level)", "Engine & learning lead; integrator", "World engine, PPO learner, experiments, statistics, paper; reviews and merges PRs; repository owner"],
      ["Advik Divekar", "Sem V (mid-level)", "Visualisation & measurement lead", "3-D viewer, flight recorder and replay, evaluation tooling, preregistration, pilot preprint and diagrams; long training runs on the M3 Pro"],
      ["[Sem VI senior, open role]", "Sem VI", "System integration lead", "Integration and mentoring (VIP model); filled by the current Sem V members next semester if no senior joins"],
      ["[Sem IV juniors, to onboard]", "Sem IV", "Research assistants", "Literature, documentation, tests, small analyses under guidance"],
    ], [0.22, 0.16, 0.22, 0.4], { size: 17 }),
    ...(await figure("docs/vip/diagrams/v03_team_structure.png", "Figure 1. The team in the VIP vertical structure.")),
    H1("2. Sub-teams"),
    table(["Sub-team", "Scope", "Lead", "Support"], [
      ["Engine & ecology", "`sim/`: world generation, kernels, fauna, climate, performance", "Heet", "Advik (survival-analysis fix, recorder hooks)"],
      ["Learning", "`agents/`: network, PPO trainer, baselines, checkpoints", "Heet", "Advik (original rppo design, now the trainer defaults)"],
      ["Visualisation", "`viewer/`: three.js client, server, dashboard", "Advik", "Heet (preview server, learner integration)"],
      ["Measurement & analysis", "`analysis/`, `experiments/`: recorder, evaluation, statistics, behaviour", "Advik (recorder), Heet (analysis)", "Both"],
      ["Research operations", "Preregistration, logs, handoffs, CI, papers", "Shared", "Both"],
    ], [0.2, 0.42, 0.14, 0.24], { size: 17 }),
    H1("3. Responsibility matrix (RACI)"),
    P("R = responsible (does the work), A = accountable (signs off), C = consulted, I = informed."),
    table(["Activity", "Heet", "Advik", "Mentor"], [
      ["World engine changes", "R/A", "C", "I"],
      ["Learner and training runs", "R/A", "R (long runs)", "I"],
      ["Viewer", "C", "R/A", "I"],
      ["Flight recorder, evaluation tooling", "C", "R/A", "I"],
      ["Preregistration entries", "R", "R", "C"],
      ["Pull-request review and merge", "R/A", "C", "I"],
      ["Papers and reports", "R", "R", "A"],
      ["Milestone reviews", "R", "R", "A"],
    ], [0.46, 0.18, 0.18, 0.18]),
    H1("4. Working agreement"),
    ...bullets(["**Code flow:** Heet commits to `HeetRanpura/Genesis` main; Advik works on the fork `advikdivekar/Genesis` and opens pull requests. `make test` must pass before a PR; CI runs on every push and PR.",
      "**Reviews check:** tests, append-only edits to shared logs, data provenance (which run produced which number), and the hard rules.",
      "**Handoffs:** whoever stops work updates `HANDOFF.md` (and a personal handoff such as `HANDOFF_ADVIK.md`) with state, results, next tasks and exact commands.",
      "**Records:** `logs/SESSION_LOG.md` (engineering sessions), `logs/AGENT_LOG.md` (how agents are doing, negative results kept), `docs/PREREGISTRATION.md` (hypotheses before results).",
      "**Compute:** long training on the M3 Pro; analysis and writing on the Air; one heavy job at a time on the fanless Air.",
      "**Meetings (proposed cadence):** a weekly sync with minutes in the meeting template, and a fortnightly mentor review. Record only meetings that actually happen."]),
    H1("5. Contribution evidence (from the repository)"),
    table(["Evidence", "Heet Ranpura", "Advik Divekar"], [
      ["Commits on main (27 Oct 2026)", "16", "8"],
      ["Pull requests", "reviewer / merger of #1-#4", "author of #1-#4 (#1 re-applied by cherry-pick, #2-#4 merged)"],
      ["Main areas (files touched)", "reports, experiments, paper, sim, tests, agents", "reports, viewer, experiments, tests, docs"],
    ], [0.34, 0.33, 0.33]),
    H1("6. Growing the team vertically"),
    ...bullets(["`TEACHER.md` is the RL study guide and FAQ for new members.",
      "Suggested first tasks for Sem IV juniors: verify a reference, add a world-rule test, reproduce a baseline number with `make baselines`, write up one figure.",
      "In Semester VI the current members take the senior role: integration, mentoring and the innovation output."]),
  ];
  await save("sem4/04_Team_Roles_and_Working_Agreement.docx", "Team Role Definition & Working Agreement", c);
}

async function templates() {
  const blankRows = (n, k) => Array.from({ length: n }, () => Array(k).fill(" "));
  const c = [
    ...titlePage({ title: "Design Notebook & Meeting Templates", subtitle: "Blank templates for the team's own records", course: COURSE4 }),
    callout("How to use", ["Copy a template for each entry. Fill it in on the day, with real dates and attendees. The repository's SESSION_LOG.md and AGENT_LOG.md already record engineering sessions and results; these templates add meetings, decisions and task tracking."]),
    H1("1. Design notebook entry"),
    table(["Field", "Entry"], [["Date", " "], ["Author", " "], ["Goal of the session", " "], ["What was tried", " "], ["Result (numbers, files, commit)", " "], ["Decision and why", " "], ["Open questions", " "], ["Next step", " "]], [0.3, 0.7]),
    H1("2. Meeting minutes"),
    table(["Field", "Entry"], [["Date / time / place", " "], ["Attendees", " "], ["Absent", " "], ["Agenda", " "], ["Discussion summary", " "], ["Decisions", " "], ["Feedback given / received", " "]], [0.3, 0.7]),
    H3("Action items"),
    table(["#", "Action", "Owner", "Due", "Status"], blankRows(5, 5), [0.06, 0.5, 0.16, 0.14, 0.14]),
    H1("3. Weekly task tracker"),
    table(["Task", "Sub-team", "Owner", "Started", "Due", "Status", "Link (commit / PR)"], blankRows(8, 7), [0.26, 0.13, 0.11, 0.1, 0.1, 0.1, 0.2], { size: 17 }),
    H1("4. Pull-request review checklist"),
    ...bullets(["`make test` passes; CI green.", "No reward term depends on other agents.", "Shared logs and preregistration only appended to.",
      "Every new number traces to a run, a file and a command.", "No co-author or AI attribution lines in commits.", "Handoff updated if the change affects the next person."]),
    H1("5. Independent-learning log"),
    table(["Date", "Topic", "Resource (course, paper, tutorial)", "What I can now do", "Evidence"], blankRows(6, 5), [0.12, 0.18, 0.3, 0.24, 0.16], { size: 17 }),
  ];
  await save("sem4/05_Design_Notebook_and_Meeting_Templates.docx", "Design Notebook & Meeting Templates", c);
}

(async () => {
  await problemDefinition();
  await literature();
  await concept();
  await team();
  await templates();
})().catch((e) => { console.error(e); process.exit(1); });
