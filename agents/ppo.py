"""PPOTrainer: the single GENESIS learner (recurrent, parameter-shared PPO on the slot-based GenesisEnv).

Each env slot is one stream of experience. A death ends that agent's return (done=1, no bootstrap: the
slot may be refilled the same tick by a newborn whose first observation follows). info["born"] marks slots
whose recurrent state must be reset before the next step. The reward is the environment's homeostatic
reward, unchanged (HARD RULE 1); `normalize_returns` (off by default) only rescales it.

Defaults are the configuration that produced the first held-out Stage 2 results (formerly agents/rppo.py):
128-tick rollouts trained through in full (seq_len == rollout_len), 4 epochs × 4 minibatches split by agent,
raw rewards. A shorter seq_len gives truncated BPTT with hidden states stored at chunk starts.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn

from .policy import CHECKPOINT_FORMAT, ActorCritic, flatten_vec, sample_actions


@dataclass
class PPOConfig:
    n_agents: int = 256
    core: str = "gru"                 # "gru" (memory) or "mlp" (memoryless ablation)
    hidden: int = 256
    rollout_len: int = 128            # ticks per rollout, per agent slot
    seq_len: int = 128                # BPTT length (== rollout_len: through the whole rollout)
    epochs: int = 4
    minibatches: int = 4
    lr: float = 3e-4
    gamma: float = 0.995              # ~200-tick horizon (1.7 sim days)
    gae_lambda: float = 0.95
    clip: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5              # value loss = vf_coef * 0.5 * (V - R)^2
    max_grad_norm: float = 0.5
    normalize_returns: bool = False   # divide rewards by a running std of discounted returns
    device: str = "mps"
    extra: dict = field(default_factory=dict)


def gae(rew, val, done, last_v, gamma, lam):
    """Generalised advantage estimation over (T, N) arrays. A death (done) cuts the bootstrap: the next
    value in that slot belongs to a newborn. Returns (advantages, returns)."""
    T = rew.shape[0]
    adv = np.zeros_like(rew, dtype=np.float32)
    g = np.zeros(rew.shape[1], np.float32)
    for t in reversed(range(T)):
        nv = last_v if t == T - 1 else val[t + 1]
        nonterm = 1.0 - done[t]
        delta = rew[t] + gamma * nv * nonterm - val[t]
        g = delta + gamma * lam * nonterm * g
        adv[t] = g
    return adv, adv + val


class ReturnNormalizer:
    """Scales rewards by the running std of discounted returns (per-slot return accumulators)."""

    def __init__(self, n, gamma):
        self.ret = np.zeros(n)
        self.gamma = gamma
        self.count, self.mean, self.m2 = 1e-4, 0.0, 0.0

    def __call__(self, r, done):
        self.ret = self.ret * self.gamma + r
        b_mean, b_var, b_n = self.ret.mean(), self.ret.var(), self.ret.size
        delta = b_mean - self.mean
        tot = self.count + b_n
        self.mean += delta * b_n / tot
        self.m2 += b_var * b_n + delta ** 2 * self.count * b_n / tot
        self.count = tot
        self.ret[done] = 0.0
        return r / np.sqrt(self.m2 / self.count + 1e-8)

    def state_dict(self):
        return dict(ret=self.ret, count=self.count, mean=self.mean, m2=self.m2)

    def load_state_dict(self, d):
        self.ret, self.count, self.mean, self.m2 = d["ret"], d["count"], d["mean"], d["m2"]


class PPOTrainer:
    def __init__(self, env, cfg: PPOConfig, seed: int = 0):
        assert cfg.rollout_len % cfg.seq_len == 0, "rollout_len must be a multiple of seq_len"
        self.env, self.cfg, self.seed = env, cfg, seed
        torch.manual_seed(seed)
        self.rng = np.random.default_rng(seed)                       # minibatch order
        self.gen = torch.Generator().manual_seed(seed)               # action sampling (CPU, deterministic)
        self.dev = torch.device(cfg.device if (cfg.device != "mps" or torch.backends.mps.is_available()) else "cpu")
        self.model = ActorCritic(env.obs_shapes, env.n_actions, cfg.hidden, cfg.core).to(self.dev)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=cfg.lr, eps=1e-5)
        self.N = env.N
        self.norm = ReturnNormalizer(self.N, cfg.gamma) if cfg.normalize_returns else None
        self.h = self.model.initial_state(self.N, self.dev)
        self.reset_next = np.ones(self.N, bool)
        self.obs = env.observe()
        self.iteration = 0
        self.agent_steps = 0

    # ------------------------------------------------------------------------------------
    def set_env(self, env):
        """Swap in a new world (e.g. the next training seed); all recurrent states reset."""
        self.env = env
        self.obs = env.observe()
        self.h = self.model.initial_state(self.N, self.dev)
        self.reset_next[:] = True
        if self.norm:
            self.norm.ret[:] = 0.0

    @torch.no_grad()
    def collect(self, recorder=None, record: bool = False):
        """One rollout. With a flight recorder, record=True logs every tick in full (including the learner's
        value, action probabilities and log-prob); record=False still logs births and interventions."""
        cfg, T, N, dev, env = self.cfg, self.cfg.rollout_len, self.N, self.dev, self.env
        S = env.obs_shapes["grid"]
        F = flatten_vec(self.obs).shape[1]
        buf = dict(grid=np.zeros((T, N) + S, np.uint8), vec=np.zeros((T, N, F), np.float32),
                   act=np.zeros((T, N), np.int64), logp=np.zeros((T, N), np.float32),
                   val=np.zeros((T, N), np.float32), rew=np.zeros((T, N), np.float32),
                   done=np.zeros((T, N), bool), reset=np.zeros((T, N), bool), valid=np.zeros((T, N), bool))
        h_chunks = []
        raw_rew_sum, deaths = 0.0, 0
        causes0 = env.deaths_by_cause.copy()
        for t in range(T):
            if t % cfg.seq_len == 0:
                h_chunks.append(self.h.clone())      # state before the reset mask; unroll applies the same mask
            grid, vec = self.obs["grid"], flatten_vec(self.obs)
            reset = self.reset_next.copy()
            logits, value, self.h = self.model.step(torch.from_numpy(grid).to(dev), torch.from_numpy(vec).to(dev),
                                                    self.h, torch.from_numpy(reset).to(dev))
            a, logp, probs = sample_actions(logits, self.gen)
            v = value.float().cpu().numpy()
            buf["grid"][t], buf["vec"][t], buf["reset"][t] = grid, vec, reset
            buf["act"][t], buf["logp"][t], buf["val"][t] = a, logp, v
            buf["valid"][t] = env.alive
            if recorder is not None and record:
                recorder.pre(self.obs)
            self.obs, r, died, info = env.step(a)
            if recorder is not None:
                if record:
                    recorder.post(a, r, info, extras=dict(value=v, logp=logp, probs=probs))
                else:
                    recorder.skip(info)
            buf["rew"][t] = self.norm(r.astype(np.float64), died) if self.norm else r
            buf["done"][t] = died
            self.reset_next = info["born"].copy()
            raw_rew_sum += float(r[buf["valid"][t]].sum())
            deaths += int(died.sum())
        _, last_v, _ = self.model.step(torch.from_numpy(self.obs["grid"]).to(dev),
                                       torch.from_numpy(flatten_vec(self.obs)).to(dev), self.h,
                                       torch.from_numpy(self.reset_next).to(dev))
        buf["adv"], buf["ret"] = gae(buf["rew"], buf["val"], buf["done"].astype(np.float32),
                                     last_v.float().cpu().numpy(), cfg.gamma, cfg.gae_lambda)
        buf["h0"] = torch.stack(h_chunks)            # (T/seq_len, N, H)
        n_valid = int(buf["valid"].sum())
        self.agent_steps += n_valid
        stats = dict(deaths=deaths, causes=env.deaths_by_cause - causes0, agent_years=n_valid / env.year_len,
                     mean_reward=raw_rew_sum / max(1, n_valid))
        return buf, stats

    def update(self, buf):
        cfg, dev = self.cfg, self.dev
        T, N = buf["act"].shape
        L = cfg.seq_len
        chunks = [(c, i) for c in range(T // L) for i in range(N)]
        tt = {k: torch.from_numpy(v).to(dev) for k, v in buf.items() if isinstance(v, np.ndarray)}
        h0_all = buf["h0"]
        info = dict(pg=0.0, vf=0.0, ent=0.0, kl=0.0, clipfrac=0.0, n=0)
        for _ in range(cfg.epochs):
            perm = self.rng.permutation(len(chunks))
            for sel in np.array_split(perm, cfg.minibatches):
                cs = np.array([chunks[k][0] for k in sel])
                slots = np.array([chunks[k][1] for k in sel])
                ti = torch.from_numpy(cs[None, :] * L + np.arange(L)[:, None]).to(dev)      # (L, B)
                si = torch.from_numpy(np.broadcast_to(slots, (L, len(sel))).copy()).to(dev)
                h0 = h0_all[torch.from_numpy(cs).to(dev), torch.from_numpy(slots).to(dev)]
                logits, values = self.model.unroll(tt["grid"][ti, si], tt["vec"][ti, si], h0, tt["reset"][ti, si])
                logp_all = torch.log_softmax(logits, -1)
                logp = logp_all.gather(2, tt["act"][ti, si].unsqueeze(2)).squeeze(2)
                old_logp, ret = tt["logp"][ti, si], tt["ret"][ti, si]
                mask = tt["valid"][ti, si].float()
                m = mask.sum().clamp(min=1.0)
                adv = tt["adv"][ti, si]
                mu = (adv * mask).sum() / m
                adv = (adv - mu) / (torch.sqrt((((adv - mu) ** 2) * mask).sum() / m) + 1e-8)
                ratio = torch.exp(logp - old_logp)
                pg = -torch.min(ratio * adv, ratio.clamp(1 - cfg.clip, 1 + cfg.clip) * adv)
                vf = 0.5 * (values - ret) ** 2
                ent = -(logp_all.exp() * logp_all).sum(-1)
                loss = ((pg + cfg.vf_coef * vf - cfg.ent_coef * ent) * mask).sum() / m
                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), cfg.max_grad_norm)
                self.opt.step()
                with torch.no_grad():
                    info["pg"] += float((pg * mask).sum() / m)
                    info["vf"] += float((vf * mask).sum() / m)
                    info["ent"] += float((ent * mask).sum() / m)
                    info["kl"] += float((((ratio - 1) - torch.log(ratio)) * mask).sum() / m)
                    info["clipfrac"] += float((((ratio - 1).abs() > cfg.clip).float() * mask).sum() / m)
                    info["n"] += 1
        n = max(1, info.pop("n"))
        out = {k: v / n for k, v in info.items()}
        y, p = buf["ret"][buf["valid"]], buf["val"][buf["valid"]]
        out["explained_var"] = float(1 - np.var(y - p) / (np.var(y) + 1e-8))
        return out

    def iterate(self, recorder=None, record: bool = False):
        t0 = time.perf_counter()
        buf, stats = self.collect(recorder, record)
        t1 = time.perf_counter()
        info = self.update(buf)
        self.iteration += 1
        info.update(stats, collect_s=round(t1 - t0, 2), total_s=round(time.perf_counter() - t0, 2))
        return info

    # ------------------------------------------------------------------------------------
    def state_dict(self):
        return dict(model=self.model.state_dict(), opt=self.opt.state_dict(),
                    norm=self.norm.state_dict() if self.norm else None, iteration=self.iteration,
                    agent_steps=self.agent_steps, rng=self.rng.bit_generator.state, gen=self.gen.get_state(),
                    torch_rng=torch.get_rng_state())

    def load_state_dict(self, d):
        self.model.load_state_dict(d["model"])
        self.opt.load_state_dict(d["opt"])
        if self.norm and d.get("norm"):
            self.norm.load_state_dict(d["norm"])
        self.iteration, self.agent_steps = d["iteration"], d["agent_steps"]
        self.rng.bit_generator.state = d["rng"]
        self.gen.set_state(d["gen"])
        torch.set_rng_state(d["torch_rng"])

    def save(self, path, **extra):
        """Self-describing checkpoint: everything needed to rebuild the network (agents.policy.load_model)."""
        cfg = {k: v for k, v in self.cfg.__dict__.items() if k != "extra"}
        torch.save(dict(format=CHECKPOINT_FORMAT, trainer=self.state_dict(), cfg=cfg, seed=self.seed,
                        obs_shapes=self.env.obs_shapes, n_actions=self.env.n_actions,
                        action_names=self.env.action_names, **extra), path)

    def load(self, path) -> dict:
        ck = torch.load(path, weights_only=False, map_location="cpu")
        if ck.get("format") != CHECKPOINT_FORMAT:
            raise ValueError(f"{path}: checkpoint format {ck.get('format', 1)} != {CHECKPOINT_FORMAT}")
        self.load_state_dict(ck["trainer"])
        return ck
