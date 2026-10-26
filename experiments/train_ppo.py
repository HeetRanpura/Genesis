"""Train the GENESIS learner (agents/ppo.py: PPOTrainer; parameter-shared, survival reward only).

    python experiments/train_ppo.py --name gru_s0 --iters 600                 # GRU memory
    python experiments/train_ppo.py --name ff_s0 --iters 600 --core mlp       # memoryless ablation
    python experiments/train_ppo.py --name gru_s0 --iters 900 --resume        # continue a run

Worlds: training uses seeds 1000 + 100*seed + w (a new world every --world-every iterations); evaluation
seeds 0-4 (the baseline seeds) are never trained on. Every --val-every iterations the current policy is run
on validation worlds (--val-seeds, default 500 and 501: neither training nor test) for --val-years, giving a
learning curve in val.jsonl that is not confounded by the training world's season and difficulty. Fixed population of --agents (respawn), all
capabilities off. Logs: runs/stage2/<name>/{config.json, train.jsonl, checkpoints/, world_*/}. Rollouts are
flight-recorded (1 of every --record-every, all ticks, with the learner's value / action probabilities /
log-prob; births and interventions always), unless free disk drops below --min-free-gb.
PPO hyperparameters: see agents/ppo.py PPOConfig (each is also a --flag).
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from agents.policy import PolicyAgent  # noqa: E402
from agents.ppo import PPOConfig, PPOTrainer  # noqa: E402
from sim import GenesisEnv, load_config  # noqa: E402
from sim.constants import DEATH_NAMES  # noqa: E402
from sim.logger import append_agent_log, git_commit  # noqa: E402
from sim.recorder import TrajectoryRecorder  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
_SKIP = ("n_agents", "core", "device", "extra")


def make_env(seed, n, memory, config=None):
    return GenesisEnv(load_config(config, overrides={"agents": {"n_initial": n, "max_agents": n},
                                                     "capabilities": {"memory": memory}}), seed=seed)


@torch.no_grad()
def validate(model, seeds, years, n, device, config=None, make=None):
    """Deaths per agent-year of the current policy on fixed validation worlds (sampled actions, fresh memory).
    Exposure = alive agent-ticks, i.e. the same measure as the baselines' lifetime-based rate."""
    out = {}
    for s in seeds:
        env = make(s) if make else make_env(s, n, model.core_type == "gru", config)
        pol = PolicyAgent(model, env.N, seed=s, device=device)
        obs, deaths, exposure = env.observe(), 0, 0
        causes0 = env.deaths_by_cause.copy()
        for _ in range(int(years * env.year_len)):
            exposure += int(env.alive.sum())
            obs, r, died, info = env.step(pol.act(obs, env.alive))
            deaths += int(died.sum())
            if info["born"].any():
                pol.reset(info["born"])
        out[s] = dict(deaths_per_agent_year=deaths / (exposure / env.year_len),
                      causes={n_: int(c) for n_, c in zip(DEATH_NAMES, env.deaths_by_cause - causes0) if c})
    model.train()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--seed", type=int, default=0, help="learner seed (network init, sampling, world sequence)")
    ap.add_argument("--iters", type=int, default=600, help="total iterations (resume continues up to this)")
    ap.add_argument("--agents", type=int, default=256)
    ap.add_argument("--config", default=None, help="world config YAML merged over the default")
    ap.add_argument("--world-every", type=int, default=40, help="iterations per training world")
    ap.add_argument("--core", default="gru", choices=["gru", "mlp"])
    ap.add_argument("--no-memory", action="store_true", help="same as --core mlp")
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    ap.add_argument("--record-every", type=int, default=1, help="record 1 of k rollouts in full (0 = off)")
    ap.add_argument("--min-free-gb", type=float, default=15.0, help="stop recording below this free disk")
    ap.add_argument("--ckpt-every", type=int, default=25)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--val-every", type=int, default=50, help="validation every k iterations (0 = off)")
    ap.add_argument("--val-seeds", default="500,501", help="validation worlds (not training, not test)")
    ap.add_argument("--val-years", type=float, default=1.0)
    for k, v in PPOConfig().__dict__.items():
        if k not in _SKIP:
            ap.add_argument(f"--{k.replace('_', '-')}", type=(lambda s: s.lower() in ("1", "true", "yes"))
                            if isinstance(v, bool) else type(v), default=v)
    a = ap.parse_args()
    core = "mlp" if a.no_memory else a.core
    out = ROOT / "runs" / "stage2" / a.name
    (out / "checkpoints").mkdir(parents=True, exist_ok=True)
    cfg = PPOConfig(n_agents=a.agents, core=core, device=a.device,
                    **{k: getattr(a, k) for k in PPOConfig().__dict__ if k not in _SKIP})
    torch.set_num_threads(5)
    world_seed = lambda w: 1000 + a.seed * 100 + w  # noqa: E731
    trainer = PPOTrainer(make_env(world_seed(0), a.agents, core == "gru", a.config), cfg, seed=a.seed)
    if a.resume and (out / "checkpoints" / "last.pt").exists():
        trainer.load(out / "checkpoints" / "last.pt")
        print(f"resumed at iteration {trainer.iteration}", flush=True)
    (out / "config.json").write_text(json.dumps(dict(args=vars(a), ppo={k: v for k, v in cfg.__dict__.items() if k != "extra"},
                                                     git_commit=git_commit(), torch=torch.__version__,
                                                     n_actions=trainer.env.n_actions,
                                                     action_names=trainer.env.action_names), indent=2))
    log = open(out / "train.jsonl", "a")
    world, rec = -1, None
    t_start = time.perf_counter()
    print(f"{a.name}: core={core} device={trainer.dev} agents={a.agents} iters {trainer.iteration}->{a.iters}", flush=True)
    while trainer.iteration < a.iters:
        it = trainer.iteration
        w = it // a.world_every
        if w != world:
            if rec:
                rec.close()
            if world != -1 or w != 0:
                trainer.set_env(make_env(world_seed(w), a.agents, core == "gru", a.config))
            world = w
            rec = (TrajectoryRecorder(trainer.env, out / f"world_{w:03d}",
                                      meta=dict(source="train_ppo", learner=a.name, world_index=w,
                                                note="sampled rollouts; policy changes every iteration"))
                   if a.record_every else None)
        recording = bool(rec) and it % a.record_every == 0 and shutil.disk_usage(ROOT).free / 1e9 > a.min_free_gb
        info = trainer.iterate(rec, recording)
        env = trainer.env
        row = dict(iter=it, world=w, tick=env.tick, agent_steps=trainer.agent_steps, mean_reward=info["mean_reward"],
                   deaths=info["deaths"], deaths_per_agent_year=info["deaths"] / max(info["agent_years"], 1e-9),
                   causes={n: int(c) for n, c in zip(DEATH_NAMES, info["causes"]) if c},
                   mean_energy=float(env.A[env.alive, 0].mean()), mean_hydration=float(env.A[env.alive, 1].mean()),
                   recorded=recording, collect_s=info["collect_s"], total_s=info["total_s"],
                   agent_steps_per_s=round(cfg.rollout_len * a.agents / info["total_s"]),
                   wall_s=round(time.perf_counter() - t_start, 1),
                   **{k: info[k] for k in ("pg", "vf", "ent", "kl", "clipfrac", "explained_var")})
        log.write(json.dumps(row, default=float) + "\n")
        log.flush()
        if it % 10 == 0:
            print(f"it {it:4d} w{w} deaths/agent-yr {row['deaths_per_agent_year']:.2f} r {row['mean_reward']:+.5f} "
                  f"ent {row['ent']:.2f} kl {row['kl']:.4f} {row['agent_steps_per_s']} sps", flush=True)
        if a.val_every and (trainer.iteration % a.val_every == 0 or trainer.iteration == a.iters):
            t0 = time.perf_counter()
            val = validate(trainer.model, [int(x) for x in a.val_seeds.split(",")], a.val_years, a.agents,
                           trainer.dev, a.config)
            d = [v["deaths_per_agent_year"] for v in val.values()]
            with open(out / "val.jsonl", "a") as vf:
                vf.write(json.dumps(dict(iter=trainer.iteration, agent_steps=trainer.agent_steps,
                                         deaths_per_agent_year=float(np.mean(d)), per_seed=val,
                                         seconds=round(time.perf_counter() - t0, 1))) + "\n")
            print(f"  validation @ {trainer.agent_steps / 1e6:.1f}M steps: deaths/agent-yr {np.mean(d):.2f} "
                  f"({', '.join(f'{x:.2f}' for x in d)}) [{time.perf_counter() - t0:.0f}s]", flush=True)
        if trainer.iteration % a.ckpt_every == 0 or trainer.iteration == a.iters:
            trainer.save(out / "checkpoints" / "last.pt", world=w)
            if trainer.iteration % 100 == 0 or trainer.iteration == a.iters:
                trainer.save(out / "checkpoints" / f"iter_{trainer.iteration:05d}.pt", world=w)
    if rec:
        rec.close()
    log.close()
    rows = [json.loads(line) for line in open(out / "train.jsonl")]
    k = max(1, len(rows) // 10)
    f = lambda rr: np.mean([x["deaths_per_agent_year"] for x in rr])  # noqa: E731
    append_agent_log(f"Stage 2 training: {a.name}", [
        f"PPOTrainer ({'GRU' if core == 'gru' else 'memoryless'}), {a.agents} agents, {len(rows)} iterations × "
        f"{cfg.rollout_len} ticks ({rows[-1]['agent_steps']:,} agent-steps), training worlds {world_seed(0)}+ "
        "(eval worlds 0-4 held out).",
        f"- deaths/agent-yr during training: first {k} iters {f(rows[:k]):.2f} → last {k} iters {f(rows[-k:]):.2f} "
        "(training-time rate with exploration; not an evaluation).",
        f"- last-iteration causes: {rows[-1]['causes']}; entropy {rows[-1]['ent']:.2f}.",
        f"- logs: `runs/stage2/{a.name}/train.jsonl`, checkpoints in `runs/stage2/{a.name}/checkpoints/`."])
    print("done", flush=True)


if __name__ == "__main__":
    main()
