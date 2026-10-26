# GENESIS agent minds (Stage 2)

## First slice: parameter-shared recurrent PPO

**Policy** (`agents/policy.py: ActorCritic`):
- **Grid branch:** the 11×11×16 uint8 grid (channel-last) goes through conv 3×3 (32), then conv 3×3
  stride 2 (32), then flatten.
- **Vector branch:** self (13) + neighbours (4×9) + heard signals (8) go through a 64-unit linear layer.
- **Trunk and core:** both branches concatenate into a 256-unit trunk, then the core, then policy and
  value heads.
- **Core:** a `GRUCell(256)` when `capabilities.memory` is on, a 256-unit MLP (memoryless) when off.
  This is the memory ablation.

One set of weights is shared by all agents (parameter sharing). Independent learners are a later ablation.
Each agent keeps its own recurrent state, which resets when a new agent takes over its slot (`info["born"]`).

**Learning rule** (`agents/ppo.py: PPOTrainer`):
- **Streams:** standard clipped PPO. Every env slot is a continuing experience stream.
- **Rollouts:** 64 ticks × 256 agents per update.
- **Recurrent training:** BPTT over 32-tick chunks, starting from the hidden state recorded at
  rollout time. `test_unroll_matches_stepwise_forward` checks that the training unroll reproduces the
  acting-time computation exactly.
- **Advantages:** GAE (γ = 0.995, λ = 0.95). A death cuts the bootstrap, because the slot's next
  observation belongs to a newborn.
- **Update:** 3 epochs × 4 minibatches, Adam 3e-4, entropy 0.01, gradient clip 0.5.

**Reward**: exactly the environment's homeostatic reward (drive reduction + death penalty, HARD RULE 1),
divided by a running std of discounted returns (scale only; sign and ordering unchanged). The learner
receives no other signal: no social term, no shaping, no curiosity bonus.

**Compute**:
- **Devices:** the network runs on Apple MPS (~1 ms inference for 256 agents vs ~13 ms on CPU); the
  world runs on CPU (Numba).
- **Reproducibility:** MPS training is not bit-reproducible; the world is.
- **Throughput:** about 6–8k agent-steps/s of training per run (two runs at once, ~3 s per 16k-sample
  iteration).

## Protocol
- **Training worlds:** seeds 1000, 1001, … (new world every simulated year, `--world-years`), 256 agents,
  fixed population with respawn.
- **Evaluation** (`experiments/eval_agents.py`): held-out world seeds 0–4, never seen in training. Same
  protocol as the baselines (2 years, 256 agents, respawn, stochastic policy). Reports mortality and
  Kaplan-Meier survival with bootstrap CIs, plus Hedges' g vs each baseline.

## Commands

```bash
.venv/bin/python experiments/train_ppo.py --name gru --core gru --minutes 90
```

```bash
.venv/bin/python experiments/train_ppo.py --name mlp --core mlp --minutes 90
```

```bash
.venv/bin/python experiments/eval_agents.py --runs gru mlp --seeds 5 --years 2
```

Resume: `--resume runs/stage2/<name>/ckpt_latest.pt`. Logs: `runs/stage2/<name>/train.csv`.

## Not yet built (rest of Stage 2, see PLAN.md)
External episodic memory, Dreamer-style world model, continual-learning metrics (forgetting, plasticity,
EWC), lifetimes + evolution of heritable hyperparameters, independent learners, social extensions
(model-of-others probe, imitation channel, communication training).
