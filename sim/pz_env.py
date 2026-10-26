"""PettingZoo ParallelEnv wrapper around the batched GenesisEnv.

Agents are named `agent_<uid>`; uids are never reused, so a dead agent never comes back.
Newborn/respawned agents join `env.agents` and appear in the returned dicts of the step in
which they are born (reward 0). `possible_agents` grows as uids are issued.
"""
from __future__ import annotations

import functools

import numpy as np
from gymnasium import spaces
from pettingzoo import ParallelEnv

from .config import load_config
from .constants import NOOP
from .env import GenesisEnv
from .layout import I_UID


class GenesisParallelEnv(ParallelEnv):
    metadata = {"name": "genesis_v0", "render_modes": []}

    def __init__(self, cfg=None, seed=None, max_cycles: int = 100_000):
        self.core = GenesisEnv(cfg or load_config(), seed=seed)
        self.max_cycles = max_cycles
        self._issued: list[str] = []
        self._sync_names()

    # --- naming -------------------------------------------------------------------------
    def _sync_names(self):
        alive = np.flatnonzero(self.core.alive)
        self._slot_of = {f"agent_{int(self.core.I[s, I_UID])}": int(s) for s in alive}
        known = set(self._issued)
        for name in self._slot_of:
            if name not in known:
                self._issued.append(name)
        self.agents = list(self._slot_of.keys())

    @property
    def possible_agents(self):
        return list(self._issued)

    # --- spaces -------------------------------------------------------------------------
    @functools.lru_cache(maxsize=None)
    def _obs_space(self):
        s = self.core.obs_shapes
        return spaces.Dict({
            "grid": spaces.Box(0, 255, s["grid"], np.uint8),
            "self": spaces.Box(-np.inf, np.inf, s["self"], np.float32),
            "neighbors": spaces.Box(-np.inf, np.inf, s["neighbors"], np.float32),
            "heard": spaces.Box(0, np.inf, s["heard"], np.float32),
        })

    def observation_space(self, agent):
        if not hasattr(self, "_obs_spaces"):
            self._obs_spaces = {}
        if agent not in self._obs_spaces:
            self._obs_spaces[agent] = self._obs_space()
        return self._obs_spaces[agent]

    def action_space(self, agent):
        if not hasattr(self, "_act_spaces"):
            self._act_spaces = {}
        if agent not in self._act_spaces:
            self._act_spaces[agent] = spaces.Discrete(self.core.n_actions)
        return self._act_spaces[agent]

    # --- API ------------------------------------------------------------------------------
    def _obs_for(self, slots):
        o = self.core._obs
        return {n: {k: o[k][s].copy() for k in o} for n, s in slots.items()}

    def reset(self, seed=None, options=None):
        self.core.reset(seed=seed)
        self._issued = []
        self.cycles = 0
        self._sync_names()
        self.core.observe()
        return self._obs_for(self._slot_of), {a: {} for a in self.agents}

    def step(self, actions):
        acts = np.full(self.core.N, NOOP, np.int64)
        for name, a in actions.items():
            s = self._slot_of.get(name)
            if s is not None:
                acts[s] = int(a)
        before = dict(self._slot_of)
        _, rew, died, info = self.core.step(acts)
        self.cycles += 1
        trunc = self.cycles >= self.max_cycles
        self._sync_names()
        obs, rewards, terms, truncs, infos = {}, {}, {}, {}, {}
        o = self.core._obs
        for name, s in before.items():
            gone = bool(died[s])
            obs[name] = {k: o[k][s].copy() for k in o}
            rewards[name] = float(rew[s])
            terms[name] = gone
            truncs[name] = trunc and not gone
            infos[name] = {}
        for name, s in self._slot_of.items():
            if name not in before:          # newborn
                obs[name] = {k: o[k][s].copy() for k in o}
                rewards[name] = 0.0
                terms[name] = False
                truncs[name] = trunc
                infos[name] = {"born": True}
        if trunc:
            self.agents = []
        return obs, rewards, terms, truncs, infos

    def render(self):
        return None

    def close(self):
        pass


def parallel_env(**kw):
    return GenesisParallelEnv(**kw)
