"""Train parameter-shared recurrent PPO agents in GENESIS (Stage 2, first slice).

    python experiments/train_ppo.py --name gru --core gru --minutes 60
    python experiments/train_ppo.py --name mlp --core mlp --minutes 60        # memoryless ablation
    python experiments/train_ppo.py --resume runs/stage2/gru/ckpt_latest.pt --minutes 30

Training worlds use seeds 1000+ (a new world every --world-years); evaluation uses seeds 0-4, so test
worlds are never seen in training. Reward = the env's homeostatic reward (HARD RULE 1).
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from agents.ppo import PPOConfig, PPOTrainer  # noqa: E402
from sim import GenesisEnv, load_config  # noqa: E402
from sim.logger import append_agent_log, git_commit  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def make_env(args, world_idx: int):
    cfg = load_config(args.config, overrides={
        "agents": {"n_initial": args.agents, "max_agents": args.agents},
        "capabilities": {"memory": args.core == "gru"},
        "logging": {"events": False}})
    return GenesisEnv(cfg, seed=1000 + world_idx)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="gru")
    ap.add_argument("--core", default="gru", choices=["gru", "mlp"])
    ap.add_argument("--agents", type=int, default=256)
    ap.add_argument("--config", default=None, help="world config (YAML) merged over the default")
    ap.add_argument("--minutes", type=float, default=60)
    ap.add_argument("--max-agent-steps", type=float, default=float("inf"))
    ap.add_argument("--world-years", type=float, default=1.0, help="switch to a new training world this often")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ckpt-every", type=int, default=25)
    ap.add_argument("--resume", default=None)
    ap.add_argument("--device", default="mps")
    for k, v in PPOConfig().__dict__.items():
        if k not in ("n_agents", "core", "device", "extra"):
            ap.add_argument(f"--{k.replace('_', '-')}", type=type(v), default=v)
    args = ap.parse_args()

    out = ROOT / "runs" / "stage2" / args.name
    out.mkdir(parents=True, exist_ok=True)
    pcfg = PPOConfig(n_agents=args.agents, core=args.core, device=args.device,
                     **{k: getattr(args, k) for k in PPOConfig().__dict__ if k not in ("n_agents", "core", "device", "extra")})
    world_idx = 0
    trainer = PPOTrainer(make_env(args, world_idx), pcfg, seed=args.seed)
    if args.resume:
        ck = torch.load(args.resume, weights_only=False)
        trainer.load_state_dict(ck["trainer"])
        world_idx = ck["world_idx"]
        trainer.set_env(make_env(args, world_idx))
        print(f"resumed at iteration {trainer.iteration}, world {world_idx}")
    (out / "config.yaml").write_text(yaml.safe_dump(dict(args=vars(args), ppo=pcfg.__dict__, git=git_commit(),
                                                         env=trainer.env.cfg.to_dict()), sort_keys=False))
    log_path = out / "train.csv"
    new_log = not log_path.exists() or not args.resume
    f = open(log_path, "w" if new_log else "a", newline="")
    writer = None
    t_end = time.time() + 60 * args.minutes
    ticks_per_world = int(args.world_years * trainer.env.year_len)
    world_start_tick = trainer.env.tick
    print(f"device={trainer.dev} core={args.core} agents={args.agents} actions={trainer.env.n_actions}", flush=True)
    recent = []      # (deaths, agent-steps) of the last 20 iterations for a less noisy printout
    while time.time() < t_end and trainer.agent_steps < args.max_agent_steps:
        info = trainer.iterate()
        info["world"] = world_idx
        if writer is None:
            writer = csv.DictWriter(f, fieldnames=list(info.keys()))
            if new_log:
                writer.writeheader()
        writer.writerow(info)
        f.flush()
        recent = (recent + [(info["deaths"], args.agents * pcfg.rollout_len)])[-20:]
        rolling = sum(d for d, _ in recent) / (sum(n for _, n in recent) / trainer.env.year_len)
        if trainer.iteration % 10 == 0:
            print(f"it {trainer.iteration:5d} steps {trainer.agent_steps/1e6:6.2f}M world {world_idx} "
                  f"deaths/agent-yr(20it) {rolling:.2f} E/H {info['mean_energy']:.2f}/"
                  f"{info['mean_hydration']:.2f} ent {info['entropy']:.2f} kl {info['approx_kl']:.4f} "
                  f"ev {info['explained_var']:.2f} [{info['collect_s']:.1f}s+{info['update_s']:.1f}s]", flush=True)
        if trainer.iteration % args.ckpt_every == 0:
            ck = dict(trainer=trainer.state_dict(), world_idx=world_idx, obs_shapes=trainer.env.obs_shapes,
                      n_actions=trainer.env.n_actions, core=args.core, hidden=pcfg.hidden)
            torch.save(ck, out / "ckpt_latest.pt")
        if trainer.env.tick - world_start_tick >= ticks_per_world:
            world_idx += 1
            trainer.set_env(make_env(args, world_idx))
            world_start_tick = trainer.env.tick
    ck = dict(trainer=trainer.state_dict(), world_idx=world_idx, obs_shapes=trainer.env.obs_shapes,
              n_actions=trainer.env.n_actions, core=args.core, hidden=pcfg.hidden)
    torch.save(ck, out / "ckpt_latest.pt")
    torch.save(ck, out / f"ckpt_iter{trainer.iteration}.pt")
    f.close()
    summary = dict(name=args.name, core=args.core, iterations=trainer.iteration, agent_steps=trainer.agent_steps,
                   worlds_seen=world_idx + 1, last=info)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    append_agent_log(f"Stage 2 training run '{args.name}' ({args.core})", [
        f"PPO, parameter-shared {args.core.upper()} policy, {args.agents} agents, "
        f"{trainer.agent_steps/1e6:.1f}M agent-steps, {trainer.iteration} iterations, {world_idx + 1} training worlds "
        f"(seeds 1000+). Last iteration: deaths/agent-year {info['deaths_per_agent_year']:.2f} (training worlds, "
        f"stochastic policy), mean energy {info['mean_energy']:.2f}, hydration {info['mean_hydration']:.2f}. "
        f"Evaluation on held-out seeds is a separate entry."])
    print("done", json.dumps(summary["last"], default=float)[:300])


if __name__ == "__main__":
    main()
