"""The pilot learners' network, vendored so the pilot checkpoints stay loadable.

The pilot runs (runs/stage2/{gru,ff}_s0, held-out results in reports/stage2/rppo_original/) were trained with the
original agents/rppo.py (commits c5ea9ff, 50143db) before it was folded into PPOTrainer (agents/ppo.py, 79d713a).
Their checkpoints predate checkpoint format 2, so agents.policy refuses them. This file keeps exactly the network
and sampling those checkpoints were trained and evaluated with, for the analyses in this folder only. New work
should use agents.policy / agents.ppo.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def _init(m, gain=np.sqrt(2)):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        nn.init.orthogonal_(m.weight, gain)
        nn.init.zeros_(m.bias)
    return m


class ActorCritic(nn.Module):
    def __init__(self, n_actions: int, obs_shapes: dict, hidden: int = 256, recurrent: bool = True):
        super().__init__()
        gh, gw, gc = obs_shapes["grid"]
        vec = int(np.prod(obs_shapes["self"]) + np.prod(obs_shapes["neighbors"]) + np.prod(obs_shapes["heard"]))
        self.conv = nn.Sequential(_init(nn.Conv2d(gc, 32, 3, padding=1)), nn.ReLU(),
                                  _init(nn.Conv2d(32, 64, 3, stride=2, padding=1)), nn.ReLU(), nn.Flatten())
        with torch.no_grad():
            n_flat = self.conv(torch.zeros(1, gc, gh, gw)).shape[1]
        self.fc_grid = _init(nn.Linear(n_flat, 256))
        self.fc_vec = _init(nn.Linear(vec, 64))
        self.fc = _init(nn.Linear(320, hidden))
        self.recurrent = recurrent
        self.core = nn.GRUCell(hidden, hidden) if recurrent else _init(nn.Linear(hidden, hidden))
        self.pi = _init(nn.Linear(hidden, n_actions), 0.01)
        self.v = _init(nn.Linear(hidden, 1), 1.0)
        self.hidden = hidden

    def encode(self, grid, selff, nb, heard):
        g = grid.float().permute(0, 3, 1, 2) / 255.0
        x = F.relu(self.fc_grid(self.conv(g)))
        v = F.relu(self.fc_vec(torch.cat([selff, nb.flatten(1), heard], 1)))
        return F.relu(self.fc(torch.cat([x, v], 1)))

    def forward_seq(self, grid, selff, nb, heard, h0, reset):
        """A whole rollout for B agents at once: inputs [T, B, ...]. The encoder runs as one batch; only the
        GRU is stepped through time (with memory resets for new agents). Returns logits [T,B,A], values [T,B]."""
        T, B = reset.shape
        x = self.encode(grid.flatten(0, 1), selff.flatten(0, 1), nb.flatten(0, 1), heard.flatten(0, 1)).view(T, B, -1)
        if self.recurrent:
            h, zs = h0, []
            for t in range(T):
                h = self.core(x[t], h * (1.0 - reset[t]).unsqueeze(1))
                zs.append(h)
            z = torch.stack(zs)
        else:
            z = F.relu(self.core(x))
        return self.pi(z), self.v(z).squeeze(-1)

    def step(self, grid, selff, nb, heard, h, reset):
        """One tick for a batch of agents. `reset` (float 0/1) zeroes the memory of newly born agents."""
        x = self.encode(grid, selff, nb, heard)
        if self.recurrent:
            h = self.core(x, h * (1.0 - reset).unsqueeze(1))
            z = h
        else:
            z = F.relu(self.core(x))
            h = torch.zeros_like(h)
        return self.pi(z), self.v(z).squeeze(1), h


class PilotPolicy:
    """Checkpoint-backed policy with the baselines' interface (act(obs, alive), reset(mask)); sampling as in rppo."""

    def __init__(self, checkpoint, env, device: str | None = None, greedy: bool = False, seed: int = 0):
        ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
        self.device = torch.device(device or ("mps" if torch.backends.mps.is_available() else "cpu"))
        torch.manual_seed(seed)
        self.net = ActorCritic(env.n_actions, env.obs_shapes, ck["cfg"]["hidden"], ck["cfg"]["recurrent"]).to(self.device)
        self.net.load_state_dict(ck["net"])
        self.net.eval()
        self.gen = torch.Generator(device="cpu").manual_seed(seed)
        self.greedy = greedy
        self.h = torch.zeros(env.N, ck["cfg"]["hidden"], device=self.device)
        self.pending_reset = np.ones(env.N, np.float32)

    def reset(self, mask=None):
        if mask is None:
            self.pending_reset[:] = 1.0
        else:
            self.pending_reset[np.asarray(mask, bool)] = 1.0

    @torch.no_grad()
    def act(self, obs, alive):
        t = lambda a: torch.from_numpy(np.ascontiguousarray(a)).to(self.device)  # noqa: E731
        logits, _, self.h = self.net.step(t(obs["grid"]), t(obs["self"]), t(obs["neighbors"]), t(obs["heard"]),
                                          self.h, t(self.pending_reset))
        probs = torch.softmax(logits.float().cpu(), 1)
        a = probs.argmax(1) if self.greedy else torch.multinomial(probs, 1, generator=self.gen).squeeze(1)
        self.pending_reset[:] = 0.0
        return a.numpy()
