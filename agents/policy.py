"""Recurrent actor-critic over GENESIS observations (parameter-shared across agents).

Inputs per agent: grid (11, 11, 16) uint8 channel-last, self (13), neighbors (4, 9), heard (n_symbols).
core = "gru" when capabilities.memory is on, "mlp" (memoryless) when off — the memory ablation.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn


class ActorCritic(nn.Module):
    def __init__(self, obs_shapes: dict, n_actions: int, hidden: int = 256, core: str = "gru"):
        super().__init__()
        S, _, C = obs_shapes["grid"]
        self.core_type = core
        self.hidden = hidden
        self.conv = nn.Sequential(
            nn.Conv2d(C, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Flatten())
        conv_out = 32 * ((S + 1) // 2) ** 2
        n_vec = obs_shapes["self"][0] + int(np.prod(obs_shapes["neighbors"])) + obs_shapes["heard"][0]
        self.vec = nn.Sequential(nn.Linear(n_vec, 64), nn.ReLU())
        self.trunk = nn.Sequential(nn.Linear(conv_out + 64, hidden), nn.ReLU())
        if core == "gru":
            self.rnn = nn.GRUCell(hidden, hidden)
        elif core == "mlp":
            self.rnn = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU())
        else:
            raise ValueError(core)
        self.pi = nn.Linear(hidden, n_actions)
        self.v = nn.Linear(hidden, 1)
        nn.init.orthogonal_(self.pi.weight, 0.01)
        nn.init.zeros_(self.pi.bias)

    def initial_state(self, n: int, device=None) -> torch.Tensor:
        return torch.zeros(n, self.hidden, device=device)

    def encode(self, grid, vec):
        x = grid.float().permute(0, 3, 1, 2) / 255.0
        return self.trunk(torch.cat([self.conv(x), self.vec(vec)], dim=1))

    def step(self, grid, vec, h, reset):
        """One timestep. reset: (B,) bool — zero the state where a new agent occupies the slot."""
        z = self.encode(grid, vec)
        if self.core_type == "gru":
            h = self.rnn(z, h * (~reset).unsqueeze(1).float())
            out = h
        else:
            out = self.rnn(z)
        return self.pi(out), self.v(out).squeeze(-1), h

    def unroll(self, grid, vec, h0, reset):
        """Sequence forward for training. grid (T, B, ...), vec (T, B, F), reset (T, B), h0 (B, H)."""
        T, B = reset.shape
        z = self.encode(grid.reshape((T * B,) + grid.shape[2:]), vec.reshape(T * B, -1)).reshape(T, B, -1)
        if self.core_type == "gru":
            outs, h = [], h0
            for t in range(T):
                h = self.rnn(z[t], h * (~reset[t]).unsqueeze(1).float())
                outs.append(h)
            out = torch.stack(outs)
        else:
            out = self.rnn(z)
        return self.pi(out), self.v(out).squeeze(-1)


def flatten_vec(obs: dict) -> np.ndarray:
    """self + neighbors + heard -> (N, F) float32."""
    n = obs["self"].shape[0]
    return np.concatenate([obs["self"], obs["neighbors"].reshape(n, -1), obs["heard"]], axis=1)


class PolicyAgent:
    """Wraps a trained ActorCritic with the baseline interface: act(obs, alive) -> actions."""
    name = "ppo"

    def __init__(self, model: ActorCritic, n_slots: int, seed: int = 0, greedy: bool = False, device: str = "cpu"):
        if device == "mps" and not torch.backends.mps.is_available():
            device = "cpu"
        self.dev = torch.device(device)
        self.model = model.to(self.dev).eval()
        self.h = model.initial_state(n_slots, self.dev)
        self.reset_mask = np.ones(n_slots, bool)
        self.gen = torch.Generator().manual_seed(seed)
        self.greedy = greedy

    def reset(self, mask=None):
        if mask is None:
            self.reset_mask[:] = True
        else:
            self.reset_mask |= mask

    @torch.no_grad()
    def act(self, obs, alive):
        grid = torch.from_numpy(obs["grid"]).to(self.dev)
        vec = torch.from_numpy(flatten_vec(obs)).to(self.dev)
        logits, _, self.h = self.model.step(grid, vec, self.h, torch.from_numpy(self.reset_mask).to(self.dev))
        self.reset_mask[:] = False
        logits = logits.cpu()
        if self.greedy:
            return logits.argmax(1).numpy()
        return torch.multinomial(torch.softmax(logits, 1), 1, generator=self.gen).squeeze(1).numpy()


def load_policy_agent(run: str, n_slots: int, seed: int = 0, device: str = "cpu",
                      root: "Path | None" = None) -> PolicyAgent:
    """PolicyAgent from runs/stage2/<run>/ckpt_latest.pt (written by experiments/train_ppo.py)."""
    from pathlib import Path
    root = root or Path(__file__).resolve().parent.parent
    ck = torch.load(root / "runs" / "stage2" / run / "ckpt_latest.pt", weights_only=False, map_location="cpu")
    model = ActorCritic(ck["obs_shapes"], ck["n_actions"], ck["hidden"], ck["core"])
    model.load_state_dict(ck["trainer"]["model"])
    agent = PolicyAgent(model, n_slots, seed=seed, device=device)
    agent.description = f"PPO {ck['core'].upper()} ({ck['trainer']['agent_steps'] / 1e6:.1f}M agent-steps)"
    return agent


def trained_runs(root: "Path | None" = None) -> list[str]:
    """Names of runs under runs/stage2 that have a checkpoint."""
    from pathlib import Path
    root = root or Path(__file__).resolve().parent.parent
    return sorted(p.parent.name for p in (root / "runs" / "stage2").glob("*/ckpt_latest.pt"))
