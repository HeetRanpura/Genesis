"""Recurrent PPO for the slot-based GenesisEnv (continuing, many-agent, shared parameters).

Each env slot is one stream of experience. A death ends that agent's return (done=1; the slot may
be refilled the same tick by a newborn whose first observation follows). info["born"] marks slots
whose recurrent state must be reset before the next step. The reward is the environment's
homeostatic reward, unchanged (HARD RULE 1); it is only rescaled by a running return std.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn

from .policy import ActorCritic, flatten_vec


@dataclass
class PPOConfig:
    n_agents: int = 256
    core: str = "gru"
    hidden: int = 256
    rollout_len: int = 64
    seq_len: int = 32
    epochs: int = 3
    minibatches: int = 4
    lr: float = 3e-4
    gamma: float = 0.995
    gae_lambda: float = 0.95
    clip: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    device: str = "mps"
    extra: dict = field(default_factory=dict)


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
        self.env, self.cfg = env, cfg
        torch.manual_seed(seed)
        self.rng = np.random.default_rng(seed)
        self.dev = torch.device(cfg.device if (cfg.device != "mps" or torch.backends.mps.is_available()) else "cpu")
        self.model = ActorCritic(env.obs_shapes, env.n_actions, cfg.hidden, cfg.core).to(self.dev)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=cfg.lr, eps=1e-5)
        self.norm = ReturnNormalizer(env.N, cfg.gamma)
        self.N = env.N
        self.h = self.model.initial_state(self.N, self.dev)
        self.reset_next = np.ones(self.N, bool)
        self.obs = env.observe()
        self.iteration = 0
        self.agent_steps = 0

    # ------------------------------------------------------------------------------------
    def set_env(self, env):
        """Swap in a new world (e.g. a fresh seed); all recurrent states reset."""
        self.env = env
        self.obs = env.observe()
        self.h = self.model.initial_state(self.N, self.dev)
        self.reset_next[:] = True
        self.norm.ret[:] = 0.0

    @torch.no_grad()
    def collect(self):
        cfg, T, N, dev = self.cfg, self.cfg.rollout_len, self.N, self.dev
        S = self.env.obs_shapes["grid"]
        F = flatten_vec(self.obs).shape[1]
        buf = dict(grid=np.zeros((T, N) + S, np.uint8), vec=np.zeros((T, N, F), np.float32),
                   act=np.zeros((T, N), np.int64), logp=np.zeros((T, N), np.float32),
                   val=np.zeros((T, N), np.float32), rew=np.zeros((T, N), np.float32),
                   done=np.zeros((T, N), bool), reset=np.zeros((T, N), bool), valid=np.zeros((T, N), bool))
        h_chunks = []
        stats = dict(raw_reward=0.0, deaths=0, births=0)
        for t in range(T):
            if t % cfg.seq_len == 0:
                # state *before* the reset mask is applied; unroll applies the same mask
                h_chunks.append(self.h.clone())
            grid, vec = self.obs["grid"], flatten_vec(self.obs)
            reset = self.reset_next.copy()
            logits, value, self.h = self.model.step(torch.from_numpy(grid).to(dev), torch.from_numpy(vec).to(dev),
                                                    self.h, torch.from_numpy(reset).to(dev))
            dist = torch.distributions.Categorical(logits=logits)
            a = dist.sample()
            buf["grid"][t], buf["vec"][t], buf["reset"][t] = grid, vec, reset
            buf["act"][t] = a.cpu().numpy()
            buf["logp"][t] = dist.log_prob(a).cpu().numpy()
            buf["val"][t] = value.cpu().numpy()
            buf["valid"][t] = self.env.alive
            self.obs, r, died, info = self.env.step(buf["act"][t])
            buf["rew"][t] = self.norm(r.astype(np.float64), died)
            buf["done"][t] = died
            self.reset_next = info["born"].copy()
            stats["raw_reward"] += float(r[buf["valid"][t]].sum())
            stats["deaths"] += int(died.sum())
            stats["births"] += int(info["born"].sum())
        _, last_v, _ = self.model.step(torch.from_numpy(self.obs["grid"]).to(dev),
                                       torch.from_numpy(flatten_vec(self.obs)).to(dev), self.h,
                                       torch.from_numpy(self.reset_next).to(dev))
        last_v = last_v.cpu().numpy()
        # GAE; a death cuts the bootstrap (the next obs in that slot belongs to a newborn)
        adv = np.zeros((T, N), np.float32)
        gae = np.zeros(N, np.float32)
        for t in reversed(range(T)):
            nv = last_v if t == T - 1 else buf["val"][t + 1]
            nonterm = 1.0 - buf["done"][t]
            delta = buf["rew"][t] + cfg.gamma * nv * nonterm - buf["val"][t]
            gae = delta + cfg.gamma * cfg.gae_lambda * nonterm * gae
            adv[t] = gae
        buf["adv"], buf["ret"] = adv, adv + buf["val"]
        buf["h0"] = torch.stack(h_chunks)            # (T/seq_len, N, H)
        self.agent_steps += int(buf["valid"].sum())
        return buf, stats

    def update(self, buf):
        cfg, dev = self.cfg, self.dev
        T, N = buf["act"].shape
        L = cfg.seq_len
        n_chunks = T // L
        chunks = [(c, i) for c in range(n_chunks) for i in range(N)]
        tt = {k: torch.from_numpy(v).to(dev) for k, v in buf.items() if isinstance(v, np.ndarray)}
        h0_all = buf["h0"]
        info = dict(pg_loss=0.0, v_loss=0.0, entropy=0.0, approx_kl=0.0, clipfrac=0.0, n=0)
        mb_size = len(chunks) // cfg.minibatches
        for _ in range(cfg.epochs):
            perm = self.rng.permutation(len(chunks))
            for m in range(cfg.minibatches):
                sel = perm[m * mb_size:(m + 1) * mb_size]
                cs = np.array([chunks[k][0] for k in sel])
                slots = np.array([chunks[k][1] for k in sel])
                tidx = (cs[None, :] * L + np.arange(L)[:, None])                  # (L, B)
                ti = torch.from_numpy(tidx).to(dev)
                si = torch.from_numpy(np.broadcast_to(slots, tidx.shape).copy()).to(dev)
                g = tt["grid"][ti, si]
                v = tt["vec"][ti, si]
                rs = tt["reset"][ti, si]
                h0 = h0_all[torch.from_numpy(cs).to(dev), torch.from_numpy(slots).to(dev)]
                logits, values = self.model.unroll(g, v, h0, rs)
                dist = torch.distributions.Categorical(logits=logits)
                act, old_logp = tt["act"][ti, si], tt["logp"][ti, si]
                adv, ret, valid = tt["adv"][ti, si], tt["ret"][ti, si], tt["valid"][ti, si].float()
                nv = valid.sum().clamp(min=1.0)
                adv = (adv - (adv * valid).sum() / nv) / (((adv ** 2) * valid).sum() / nv - ((adv * valid).sum() / nv) ** 2 + 1e-8).sqrt()
                logp = dist.log_prob(act)
                ratio = (logp - old_logp).exp()
                pg = -torch.min(ratio * adv, ratio.clamp(1 - cfg.clip, 1 + cfg.clip) * adv)
                pg_loss = (pg * valid).sum() / nv
                v_loss = (((values - ret) ** 2) * valid).sum() / nv
                ent = (dist.entropy() * valid).sum() / nv
                loss = pg_loss + cfg.vf_coef * v_loss - cfg.ent_coef * ent
                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), cfg.max_grad_norm)
                self.opt.step()
                with torch.no_grad():
                    info["pg_loss"] += pg_loss.item()
                    info["v_loss"] += v_loss.item()
                    info["entropy"] += ent.item()
                    info["approx_kl"] += (((ratio - 1) - (logp - old_logp)) * valid).sum().item() / nv.item()
                    info["clipfrac"] += (((ratio - 1).abs() > cfg.clip).float() * valid).sum().item() / nv.item()
                    info["n"] += 1
        n = info.pop("n")
        out = {k: v / n for k, v in info.items()}
        y, p = buf["ret"][buf["valid"]], buf["val"][buf["valid"]]
        out["explained_var"] = float(1 - np.var(y - p) / (np.var(y) + 1e-8))
        return out

    def iterate(self):
        t0 = time.perf_counter()
        buf, stats = self.collect()
        t1 = time.perf_counter()
        info = self.update(buf)
        t2 = time.perf_counter()
        self.iteration += 1
        al = self.env.alive
        A = self.env.A[al]
        info.update(stats, iteration=self.iteration, agent_steps=self.agent_steps, tick=self.env.tick,
                    collect_s=round(t1 - t0, 3), update_s=round(t2 - t1, 3),
                    mean_energy=float(A[:, 0].mean()), mean_hydration=float(A[:, 1].mean()),
                    mean_health=float(A[:, 2].mean()), mean_warmth=float(A[:, 3].mean()),
                    deaths_per_agent_year=stats["deaths"] / max(1e-9, buf["valid"].sum() / self.env.year_len))
        return info

    # ------------------------------------------------------------------------------------
    def state_dict(self):
        return dict(model=self.model.state_dict(), opt=self.opt.state_dict(), norm=self.norm.state_dict(),
                    iteration=self.iteration, agent_steps=self.agent_steps, rng=self.rng.bit_generator.state,
                    cfg=self.cfg.__dict__, torch_rng=torch.get_rng_state())

    def load_state_dict(self, d):
        self.model.load_state_dict(d["model"])
        self.opt.load_state_dict(d["opt"])
        self.norm.load_state_dict(d["norm"])
        self.iteration, self.agent_steps = d["iteration"], d["agent_steps"]
        self.rng.bit_generator.state = d["rng"]
        torch.set_rng_state(d["torch_rng"])
