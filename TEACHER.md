# TEACHER: reinforcement learning, explained through GENESIS

This is a study guide for the people working on GENESIS. It explains the reinforcement-learning (RL) ideas the
project uses, always with a concrete GENESIS example, and ends with answers to questions people have actually asked.
No prior ML knowledge is assumed. Where it says **(built)** the thing exists in the code today; **(planned)** means
it is a later stage in `PLAN.md`.

Contents:
1. The one-paragraph version
2. The loop: agent, environment, observation, action, reward
3. Reward, return and discounting, and our survival reward
4. Policies and value functions
5. How agents actually learn: policy gradients, actor-critic, PPO
6. Memory and partial observability (why a GRU)
7. Many agents at once: multi-agent RL
8. World models: learning to imagine (planned)
9. Learning for a whole lifetime: continual learning
10. Learning plus evolution: genes, generations, the Baldwin effect
11. Exploration vs exploitation
12. How we measure: baselines, survival statistics, seeds
13. Emergence: what counts and what doesn't
14. Glossary
15. FAQ: questions people have asked

---

## 1. The one-paragraph version

RL is learning by trial and error from a number called **reward**. An **agent** looks at its situation, picks an
**action**, and the world answers with a new situation and a reward. Repeating this millions of times, a learning
algorithm adjusts the agent's decision rule (its **policy**) so that actions which led to more reward become more likely.
In GENESIS, the reward is only "keep your own body healthy". Nobody tells an agent to drink, cooperate or talk.
The research question is what behaviours, and what *social* behaviours, appear anyway.

## 2. The loop

```
        observation o_t                       action a_t
 WORLD ─────────────────────▶ AGENT ─────────────────────▶ WORLD
   ▲                                                         │
   └──────────── reward r_t, next observation o_{t+1} ◀──────┘
```

| RL term | Meaning | In GENESIS |
|---|---|---|
| environment | everything outside the agent | `sim/` (`GenesisEnv`): terrain, weather, plants, animals, other agents |
| time step | one turn of the loop | one **tick**; every agent acts once per tick; 120 ticks = 1 day, 3,840 = 1 year |
| state s_t | the complete truth about the world | every array in `GenesisEnv`; no agent ever sees all of it |
| observation o_t | what the agent perceives | an 11×11 patch around it (16 channels: water, plants, animals, other agents, fire…), its own body (energy, hydration, health, warmth, temperature…), the nearest 4 neighbours, and heard signals |
| action a_t | what it can do | 11 actions: move N/E/S/W, rest, eat, drink, hunt, pick up, drop, no-op (+ build, strike and signals when those switches are on) |
| reward r_t | a number saying how good the step was | change in the body's "drive" (section 3) |
| episode | a run from start to end | an agent's **lifetime**, from birth to death |

**Markov decision process (MDP).** This is the textbook setting: the agent sees the full state.
**Partially observable MDP (POMDP).** The agent sees only part of the state. GENESIS is a POMDP: water outside your 11×11 view
is invisible, and at night or in fog the view shrinks to 5×5 or 7×7. This one fact drives much of the design (section 6).

## 3. Reward, return, discounting, and our survival reward

**Return.** Agents should not maximise the *next* reward, but the total reward to come:
`G_t = r_t + γ r_{t+1} + γ² r_{t+2} + …`.

**Discount γ (gamma)**, e.g. 0.99, makes distant rewards count less. With γ = 0.99 the agent effectively plans about
1/(1−γ) = 100 ticks ahead (under a day). With γ = 0.999 it plans about 1,000 ticks ahead (about 8 days). Walking to a river
the agent can't see needs a long horizon, so γ is an important knob in Stage 2.

**Our reward (built, `GenesisEnv._drive`, `kernels.tick_rewards`).** It is *homeostatic RL* (Keramati & Gutkin 2014, eLife).
The body has four variables h = (energy, hydration, health, warmth) with set-points (1, 1, 1, 0.8). The **drive** is the
distance from the set-point, counting only shortfalls:

```
D(h) = sqrt( Σ_k max(setpoint_k − h_k, 0)² )
r_t  = D(h_t) − D(h_{t+1})            (+ death_penalty = −1 if the agent died this tick)
```

So drinking when thirsty reduces the drive and gives positive reward. Starving makes the drive grow and gives negative reward.
Dying costs −1. That's all. **Hard rule 1:** reward never depends on anyone else. There is no bonus for sharing, grouping,
talking or helping, and a test (`test_reward_is_own_homeostatic_drive_only`) enforces this. An optional
`reproduction_bonus` exists but is 0 by default.

**Why reward drive *reduction* rather than the body state itself?** Reduction rewards *doing something about* a need,
which makes learning faster and more natural. Over a whole lifetime the two lead to the same optimal behaviour.

**Reward shaping** means adding extra hand-made rewards to speed learning, e.g. "+0.1 for walking toward water".
We avoid it, because every hint we add is something we would then wrongly call "emergent".

## 4. Policies and value functions

- **Policy π(a | o):** the decision rule, a probability for each action given what the agent sees. Today's agents
  use hand-written policies (`agents/baselines.py`). In Stage 2 the policy is a neural network.
- **Value function V(o):** "how much return do I expect from here?" A thirsty agent next to a river has high value;
  a thirsty agent in a desert at night has low value.
- **Action-value Q(o, a):** the expected return if I take action a now and then follow my policy.
- **Advantage A(o, a) = Q(o, a) − V(o):** how much better this action is than my *average* action here. Learning pushes up
  actions with positive advantage. This is the core of modern policy learning.

## 5. How agents learn: policy gradients, actor-critic, PPO

**Policy gradient.** The policy network outputs action probabilities. After acting, we nudge the network's weights so that
actions with positive advantage become more likely and actions with negative advantage less likely. It is gradient ascent on
expected return.

**Actor-critic.** One network with two heads. The **actor** outputs the policy π, and the **critic** estimates V. The critic's
estimates are used to compute advantages, which tells the actor whether an outcome was better or worse than expected. This is
much less noisy than judging actions by raw returns.

**GAE (generalised advantage estimation)** blends short-horizon estimates, which are less noisy but more biased, with
long-horizon ones, which are less biased but noisier, using a parameter λ (typically 0.95).

**PPO (proximal policy optimisation)**, planned for Stage 2:
1. Run the current policy for a batch of steps. With 256 agents × 128 ticks that is 32,768 experiences, a **rollout**.
2. Compute advantages with GAE.
3. Update the network for a few passes over that batch. Each update is **clipped**, so the new policy can't move too
   far from the one that collected the data (typically probabilities may change by at most ±20%). This clip is what makes PPO stable.
4. Throw the batch away and repeat. PPO is **on-policy**: it learns only from data produced by its current self.

The loss also has a value term (to train the critic) and an **entropy bonus** that rewards some randomness, so the policy
doesn't commit too early to one behaviour (see section 11).

**Why PPO and not DQN?** DQN learns Q-values from a replay buffer of old experience. It is good for single agents with small action
sets, but it struggles with memory (recurrent) policies and with many agents changing at once. PPO with a recurrent core is the
standard, robust choice for multi-agent, partially observed worlds like this one.

## 6. Memory and partial observability: why a GRU

The greedy baseline dies of thirst mostly because **water is usually out of view and it has no memory**. It forgets the river the
moment it walks away. A policy that only looks at the current observation (**memoryless** or **feed-forward**) has the same
problem.

A **recurrent neural network (RNN)** carries a **hidden state** h_t from tick to tick, a small vector updated every step:
`h_t = f(h_{t−1}, o_t)`. A **GRU** (gated recurrent unit) is an RNN with gates that learn what to keep and what to forget,
so the agent *can* learn to remember "the river was 20 cells east". Nothing tells it to. Memory only develops if it
improves survival.

GENESIS details:
- Each agent has its own hidden state, even when all agents share one network.
- When a slot gets a **new** agent, `info["born"]` is true and the hidden state must be reset to zero. A newborn must not
  inherit a dead agent's memories by accident.
- The **memory ablation** (Stage 2/7) compares memoryless, GRU, and external episodic memory. Before any training, the agent log
  recorded the prediction that memory should help most with water-finding. That is a falsifiable claim we can test.

## 7. Many agents at once: multi-agent RL

- **Parameter sharing:** all agents use *one* network (different hidden states and observations, same weights). It learns
  256× faster from the same compute. The catch is that agents start out identical in "brain", and differences come only
  from experience, position and genes.
- **Independent learners:** each agent (or lineage) has its own network. This is more realistic and allows genuine individual
  differences, but costs far more memory and compute. On this laptop it is feasible only for small populations (about 64 or fewer).
  It is a planned ablation.
- **Non-stationarity:** from one agent's point of view the other agents are part of the environment, and they keep
  changing as they learn. What worked yesterday may not work today. This is why multi-agent learning is harder and less
  stable than single-agent learning.
- **Credit assignment:** an agent eats a carcass that another agent killed. Was that a good action for the eater? Yes, for its
  own drive. Nothing in the reward credits the killer. Any reciprocity has to emerge from repeated interaction.

## 8. World models: learning to imagine (planned)

A **world model** is a network that learns to predict "if I see o_t and do a_t, what happens next (o_{t+1}, r_t)?"
**Dreamer-style** agents compress observations into a small latent state, learn its dynamics, and then **train the policy
inside imagined rollouts** ("dreams"), which costs almost nothing compared with running the real world.

In GENESIS we measure the **prediction error**: does it fall online? We also show imagined futures in the viewer as ghost paths.
Ablation: with vs without a world model. One shared model with per-agent latent states keeps this feasible on a laptop (`PLAN.md`).

## 9. Learning for a whole lifetime: continual learning

Standard RL trains, then freezes, then evaluates. GENESIS agents keep learning *while they live*, through seasons, famines and
fires. Two classic problems follow:
- **Catastrophic forgetting:** learning winter survival overwrites summer foraging skills.
- **Loss of plasticity:** after long training, networks get worse at learning anything new.

The countermeasures we plan to compare:
- **Replay buffer:** keep some old experience mixed in.
- **EWC (elastic weight consolidation):** penalise changing weights that mattered for old skills.
- **Plasticity injection:** periodically add fresh trainable capacity.

We measure forgetting on a fixed set of probe situations across seasons (the **stability-plasticity** trade-off).

## 10. Learning plus evolution: genes, generations, the Baldwin effect

**Built, off by default:** with `capabilities.reproduction: true`, a mature, well-fed agent gives birth automatically. The child
inherits the parent's 8 genes plus small random mutations. A quick check with heuristic agents showed the population
**booming in summer and crashing 60–85% every winter**. Random agents went extinct within 1–2 years (`logs/AGENT_LOG.md`).

- **Learning** happens within a lifetime, and the network weights change.
- **Evolution** happens across lifetimes, and the gene frequencies change because some genes leave more children.
- **Baldwin effect:** learning can *guide* evolution. If some individuals can *learn* a useful behaviour, selection can
  favour genes that make that learning easier or faster, until the behaviour becomes nearly innate. GENESIS can test this with
  heritable learning-related traits (planned).
- **Lamarckism**, inheriting learned weights directly, is **not** how biology works, and we don't do it by default.

## 11. Exploration vs exploitation

To learn, an agent must sometimes try actions it doesn't yet think are best (**explore**). To survive, it must mostly
do what works (**exploit**).

The tools are the entropy bonus (section 5), randomness in sampling actions, and, later, **curiosity** (intrinsic reward for
surprising observations). Curiosity would be an *added* reward term, so if we ever use it, it must be disclosed and kept
strictly non-social.

**Warning:** too little exploration and the agents never find the river. Too much and they wander into fires.

## 12. How we measure: baselines, survival statistics, seeds

- **Baselines (hard rule 5)** are non-learning comparison policies, present in every comparison:
  - *random*: picks actions uniformly.
  - *greedy*: walks to the nearest visible food or water for its lowest need.
  - *heuristic*: hand-written survival rules (flee danger, drink and eat with hysteresis, shelter when cold, rest at night).

  A learner is only interesting if it beats these, and **at the same population size**: at 1,000 agents the heuristic is
  worse than random.
- **Deaths per agent-year:** total deaths ÷ total years lived by all agents. Corrected Stage 1.1 values:
  random 3.83, greedy 3.31, heuristic **1.50** (the target for Stage 2).
- **Kaplan–Meier survival curve:** the fraction still alive after t days. It correctly handles agents that were still alive when
  the run ended ("censored" lifetimes).
- **Seeds:** a seed fixes all randomness, including the world map, weather and starting positions. Same seed, same code
  ⇒ identical run, bit for bit (tested). Results are reported over **several seeds**, because one world can be lucky.
- **Confidence intervals (CIs):** a range that likely contains the true value. We use bootstrap CIs over seeds. Narrow
  CIs need more seeds.
- **Effect size (Hedges' g):** how big a difference is in standard-deviation units. Around 0.2 is small, 0.8 large; the heuristic vs
  random is about −3.8, which is enormous.
- **Preregistration:** write the hypothesis *before* running the experiment, so results can't be retro-fitted. Every
  experiment script in `experiments/` starts with its hypotheses.

## 13. Emergence: what counts and what doesn't

"Emergent" means that the behaviour arose from learning under survival pressure, **without** being scripted, rewarded or
hinted. In GENESIS:
- We **can't** script it (hard rule 4) or reward it (hard rule 1).
- We **do** provide *affordances*, ecology that makes a behaviour possible or useful, and tag each one in `docs/WORLD.md`.
  Examples: large prey need two or more attackers, food can be dropped and picked up, and a kin-similarity signal exists.
  An affordance is not a reward. Results must be read knowing which affordances were on.
- **Measure, don't eyeball.** A cluster of agents in the viewer might be "a group", or just 20 thirsty agents at the only river.
  The Clark–Evans statistic on the dashboard shows aggregation, but baseline agents aggregate around water too. Claims need
  metrics with **null baselines**: the same metric on random or heuristic agents, where we know nothing social is happening.
- **"Ate others' kill"** is co-feeding, not sharing. Sharing would need evidence of *giving*: dropping food near
  another agent when you could have eaten it, more than chance predicts.
- **Negative results get reported** (hard rule 6). "No communication emerged under these conditions" is a valid finding.

## 14. Glossary

| term | short definition |
|---|---|
| action space | the set of actions available (11 by default) |
| actor / critic | policy head / value head of one network |
| advantage | how much better an action was than average in that situation |
| affordance | a world feature that makes a behaviour possible, never rewarded |
| agent-year | one agent living one simulated year (3,840 ticks) |
| baseline | non-learning comparison policy (random, greedy, heuristic) |
| batch / rollout | experience collected before one learning update |
| capability switch | a feature that is off by default (memory, communication, reproduction…) in `configs/default.yaml` |
| discount γ | how much future reward counts; sets the planning horizon |
| drive D(h) | distance of the body from its set-points; the reward is its decrease |
| entropy bonus | extra loss term rewarding randomness in the policy |
| episode | here, one agent's lifetime |
| flight recorder | `sim/recorder.py`: every agent's state, action and reward at every tick, replayable bit-exactly |
| GAE | generalised advantage estimation (bias/variance trade-off, λ) |
| GRU | a recurrent unit with gates; gives the agent memory |
| hidden state | the memory vector a recurrent policy carries between ticks |
| homeostasis | keeping body variables near set-points |
| on-policy / off-policy | learn only from current-policy data (PPO) / also from older data (DQN, replay) |
| parameter sharing | all agents use one network |
| policy | the mapping from observation to action probabilities |
| POMDP | a world the agent sees only partially |
| PPO | proximal policy optimisation; clipped, stable policy-gradient method |
| return | discounted sum of future rewards |
| seed | number fixing all randomness of a run |
| spin-up | 1 simulated year of ecology without agents before a run starts (`env.spinup_ticks`) |
| tick | one simulation step; the smallest unit of time |
| value function | expected return from a situation |
| world model | a learned predictor of what happens next; used for imagination |

## 15. FAQ: questions people have asked

**Q: Are the agents an LLM, like ChatGPT or Claude?**
A: No. Today each agent is a simulated body driven by a few dozen lines of hand-written rules (the heuristic baseline). It
doesn't learn, remember, or know other agents exist. In Stage 2 they become small neural networks (a few hundred thousand
weights) trained by RL from scratch, with no language or human knowledge. An LLM would bring human ideas about cooperation
and language with it, which would ruin the experiment. We want to see what arises *without* that.

**Q: How many agents are there?**
A: 256 by default, in 2,048 slots (`agents.n_initial`, `agents.max_agents`). The population stays at 256 because a dead
agent's slot is refilled immediately. There are also about 400 small prey, 48 large prey and 12 predators at the start; these change
with food. You can run more: `AGENTS=1000 make viewer`.

**Q: Then why do agents behave sensibly at all if nothing is learning?**
A: Because the heuristic baseline's rules are sensible (drink when thirsty, flee predators). Its failures are just as
informative. It dies of thirst when water is out of view because it has no memory, and it collapses at 1,000 agents.

**Q: Why not just reward cooperation and see what happens?**
A: Then cooperation would be *caused by us*, not emergent. The project's question is whether social behaviour arises
from individual survival pressure alone, as in the evolution of real animals. Hard rule 1 forbids social reward terms, and a
test enforces it.

**Q: What is a tick, and why not record "every second"?**
A: A tick is the simulation's atomic step. Every agent acts exactly once, and nothing happens in between. One tick is about 12
simulated minutes (120 per day). The flight recorder stores every agent at every tick, so it already has the finest
resolution that exists.

**Q: Is everything recorded?**
A: Yes, three ways:
1. `runs/…/metrics.csv`, `events_*.npy` and `controls.jsonl`: dashboard metrics, every event, every button you press.
2. **Flight recorder** (`runs/…/trajectory/`): every agent's full state, action, reward, death and cause each tick; every
   animal; world state; births with genes. About 63 MB per simulated year at 256 agents.
3. **Exact replay** (`python -m analysis.trajectories RUN --verify`): re-simulates the run bit-exactly. It can therefore
   reconstruct anything not stored, such as exactly what any agent *saw* at any tick.

**Q: Can the agents reproduce? Have sex? Use medicine?**
A: Reproduction is built in (asexual, with gene mutation) but off by default. Two-parent reproduction and disease with
medicinal plants are not built yet. Sex raises a real design question, because a survival-only learner gains nothing
personally from mating; evolution, not RL, would have to select for it. Medicine is a great test of discovery and
cultural transmission (chimpanzees self-medicate with bitter plants), but only learning agents could discover it.

**Q: Why did agents start dying faster when I added more of them?**
A: Measured: at 1,000 agents heuristic mortality jumps to about 4.9 deaths/agent-year, 96% thirst, worse than random. The
leading hypothesis, not yet tested, is crowding at the few drinkable water cells, since one body fits per cell. Random agents
don't seek water, so they aren't hurt by the queue.

**Q: What are the little icons above agents (🍖 💧 ❄️ ❤️)? Are they emotions?**
A: No. They show a body variable below 0.3 (hunger, thirst, cold, injury). Emotion-like *internal states* can only be
studied once agents have learned networks, and then only by **probing** the network's activations (planned, Stage 3).

**Q: What does "deaths per agent-year" mean? Isn't that more than one death per agent?**
A: It is a rate: deaths divided by total time lived. 1.5 means that if an agent could live a full year it would, on average,
"use up" 1.5 lives, i.e. most die well before a year. It is the cleanest way to compare runs with different population sizes and
lengths.

**Q: Why do we need several seeds? The viewer looks convincing.**
A: One world can be lucky: a river near the spawn area, a mild winter. Results become claims only when they hold
across seeds, with confidence intervals. We also found a real bug (founders credited with a year they never lived) only
because numbers looked odd across runs.

**Q: How long will training take on this laptop?**
A: The simulation runs about 1,250–1,780 ticks/s with 256 agents (≈ 320–450k agent-steps/s); the neural network will be the
bottleneck. A first learning curve is realistic within hours; a clean 5-seed comparison likely takes a day or more. This is an
estimate until Stage 2 is measured.

**Q: Will they develop language?**
A: Possibly something, but only if communication is switched on (`capabilities.communication`: 8 symbols, each with a small
energy cost) and signalling helps survival. We'd test for it with measures such as: do signals correlate with
context (mutual information)? Can they refer to absent things (displacement)? Do separated groups drift into different
"dialects"? Most likely early outcome: no meaningful signalling. That is reported honestly if so.

**Q: Why a GRU and not a Transformer?**
A: A GRU is cheap per step and naturally handles one agent acting once per tick for thousands of ticks. Transformers need
attention over long histories, which is expensive for 256 agents on a laptop. An external episodic memory is planned as the
"long memory" ablation instead.

**Q: If all agents share one network, aren't they clones?**
A: In weights, yes. In behaviour, no. Each has its own memory state, location, history and genes, so they act differently.
Independent networks per agent or lineage are a planned comparison.

**Q: What happens to what an agent learned when it dies?**
A: With a shared network, its experience has already updated the shared weights, so the *population* keeps it while the
individual's memory (hidden state) is lost. This is a known limitation of parameter sharing, and one reason independent
learners plus reproduction (and later social learning from elders) are planned comparisons.

**Q: Is the 3-D terrain real? Are mountains slowing agents down?**
A: The simulation is a flat 2-D grid. The viewer's relief, water depth and foliage are decoration derived from
elevation. Elevation does matter for temperature, and terrain *types* matter (rock is slow, mountains are impassable,
caves are warm). See `docs/VIEWER.md` for exactly what's real.

**Q: Can we trust the simulation's randomness?**
A: Every random number inside the world comes from a hash of (seed, tick, index, stream). Results don't depend on thread
count or timing, playback speed never changes outcomes, and identical seeds give identical runs (tests check this).

**Q: What if nothing social ever emerges?**
A: Then we report that, with the conditions under which it didn't, and use the switches (memory, communication, social learning,
reproduction) to find which ingredients are necessary. A well-measured negative result is a real scientific result
(hard rule 6).

---
*Keep this file current: when a concept is added to the code or a new question comes up, add it here.*
