# GENESIS agent minds (Stage 2)

## The learner (single implementation)

`agents/ppo.py: PPOTrainer` with the network in `agents/policy.py`. (Until 2026-10-26 a second, parallel
implementation `agents/rppo.py` existed; it produced the first Stage 2 results and was folded into PPOTrainer.
Its defaults are now PPOTrainer's defaults.)

**Network** (`ActorCritic`, about 1.1M parameters):
- **Grid branch:** the 11×11×16 uint8 grid (channel-last) goes through conv 3×3 (32), conv 3×3 stride 2 (64), then
  a 256-unit layer.
- **Vector branch:** self (13) + neighbours (4×9) + heard signals (8) go through a 64-unit layer.
- **Trunk and core:** the branches concatenate into a 256-unit layer, then the **core**, then policy and value heads.
  The core is a `GRUCell(256)` (`core="gru"`, memory) or a 256-unit feed-forward layer (`core="mlp"`, the memoryless
  ablation).
- **Initialisation:** orthogonal (gain √2; policy head 0.01, value head 1).
- One set of weights for all agents; each agent has its own recurrent state, reset when a newborn takes
  over its slot (`info["born"]`).

**Learning rule** (PPOConfig defaults):
- **Rollouts:** clipped PPO on 128-tick rollouts × 256 agents, BPTT through the whole rollout (`seq_len =
  rollout_len`; a shorter `seq_len` gives truncated BPTT from stored hidden states).
- **Advantages:** GAE with γ = 0.995 and λ = 0.95. A death cuts the bootstrap.
- **Update:** 4 epochs × 4 minibatches split by agent, Adam 3e-4, clip 0.2, entropy 0.01, value loss 0.5·(V−R)² × 0.5,
  gradient clip 0.5.
- **Sampling:** on the CPU with a seeded generator, so training is deterministic on CPU (tested). MPS arithmetic is
  not guaranteed bit-identical.

**Reward**: exactly the environment's homeostatic reward (drive reduction + death penalty, HARD RULE 1). Raw by
default; `normalize_returns=True` divides by a running std of discounted returns (scale only). Nothing else:
no social term, no shaping, no curiosity bonus.

## Protocol
- **Training worlds:** seeds 1000 + 100·seed + w, new world every 40 iterations, 256 agents, fixed population with
  respawn. Rollouts are flight-recorded (`--record-every`), and recording pauses when free disk < `--min-free-gb`.
- **Evaluation** (`experiments/eval_policy.py`): held-out seeds 0–4, never trained on, with the same protocol as the
  baselines (2 years, 256 agents, respawn, sampled actions). Runs are fully recorded including learner internals.
  The output reports deaths/agent-year and Kaplan-Meier survival with bootstrap CIs, plus Hedges' g vs each baseline.

## Commands

```bash
make train NAME=gru_s0
```

```bash
make train NAME=ff_s0 MEMORYLESS=1
```

```bash
make eval NAME=gru_s0
```

```bash
.venv/bin/python experiments/plot_training.py gru_s0 ff_s0
```

Resume: add `--resume` (continues to `--iters`). Logs: `runs/stage2/<name>/train.jsonl`.

## Not yet built (rest of Stage 2, see PLAN.md)
External episodic memory, Dreamer-style world model, continual-learning metrics (forgetting, plasticity,
EWC), lifetimes + evolution of heritable hyperparameters, independent learners, social extensions
(model-of-others probe, imitation channel, communication training).
