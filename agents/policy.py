"""The GENESIS learner's network: one parameter-shared recurrent actor-critic for every agent.

    grid 11x11x16 (uint8, channel-last) -> conv(32) -> conv(64, stride 2) -> fc 256 ┐
    self(13) + neighbours(4x9) + heard(n_symbols)       -> fc 64                    ├-> fc 256 -> core -> policy, value
                                                                                    ┘
core = "gru" (GRUCell 256; capabilities.memory on) or "mlp" (feed-forward layer; the memoryless ablation).
Each agent keeps its own recurrent state, zeroed when a newborn takes over the slot (info["born"]).
Orthogonal init (gain sqrt 2; policy head 0.01, value head 1). This is the architecture of the former
agents/rppo.py, which produced the first Stage 2 results; it is now the only learner network.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT_FORMAT = 2          # 1 = pre-unification PPOTrainer checkpoints (ckpt_latest.pt), not loadable


def _init(m, gain=np.sqrt(2)):
    nn.init.orthogonal_(m.weight, gain)
    nn.init.zeros_(m.bias)
    return m


class ActorCritic(nn.Module):
    def __init__(self, obs_shapes: dict, n_actions: int, hidden: int = 256, core: str = "gru"):
        super().__init__()
        gh, gw, gc = obs_shapes["grid"]
        n_vec = int(np.prod(obs_shapes["self"]) + np.prod(obs_shapes["neighbors"]) + np.prod(obs_shapes["heard"]))
        self.conv = nn.Sequential(_init(nn.Conv2d(gc, 32, 3, padding=1)), nn.ReLU(),
                                  _init(nn.Conv2d(32, 64, 3, stride=2, padding=1)), nn.ReLU(), nn.Flatten())
        with torch.no_grad():
            n_flat = self.conv(torch.zeros(1, gc, gh, gw)).shape[1]
        self.fc_grid = _init(nn.Linear(n_flat, 256))
        self.fc_vec = _init(nn.Linear(n_vec, 64))
        self.fc = _init(nn.Linear(256 + 64, hidden))
        if core == "gru":
            self.core = nn.GRUCell(hidden, hidden)
        elif core == "mlp":
            self.core = _init(nn.Linear(hidden, hidden))
        else:
            raise ValueError(core)
        self.core_type = core
        self.hidden = hidden
        self.pi = _init(nn.Linear(hidden, n_actions), 0.01)
        self.v = _init(nn.Linear(hidden, 1), 1.0)

    def initial_state(self, n: int, device=None) -> torch.Tensor:
        return torch.zeros(n, self.hidden, device=device)

    def encode(self, grid, vec):
        g = grid.float().permute(0, 3, 1, 2) / 255.0
        x = F.relu(self.fc_grid(self.conv(g)))
        return F.relu(self.fc(torch.cat([x, F.relu(self.fc_vec(vec))], 1)))

    def _core(self, z, h, reset):
        if self.core_type == "gru":
            h = self.core(z, h * (~reset).unsqueeze(1).float())
            return h, h
        return F.relu(self.core(z)), h

    def step(self, grid, vec, h, reset):
        """One tick for B agents. reset (B,) bool zeroes the memory of slots holding a new agent."""
        out, h = self._core(self.encode(grid, vec), h, reset)
        return self.pi(out), self.v(out).squeeze(-1), h

    def unroll(self, grid, vec, h0, reset):
        """A sequence for training: grid (T, B, ...), vec (T, B, F), reset (T, B), h0 (B, H).
        The encoder runs as one batch; only the core steps through time."""
        T, B = reset.shape
        z = self.encode(grid.reshape((T * B,) + grid.shape[2:]), vec.reshape(T * B, -1)).reshape(T, B, -1)
        if self.core_type == "gru":
            outs, h = [], h0
            for t in range(T):
                out, h = self._core(z[t], h, reset[t])
                outs.append(out)
            out = torch.stack(outs)
        else:
            out = F.relu(self.core(z))
        return self.pi(out), self.v(out).squeeze(-1)


def flatten_vec(obs: dict) -> np.ndarray:
    """self + neighbors + heard -> (N, F) float32."""
    n = obs["self"].shape[0]
    return np.concatenate([obs["self"], obs["neighbors"].reshape(n, -1), obs["heard"]], axis=1)


@torch.no_grad()
def sample_actions(logits: torch.Tensor, gen: torch.Generator, greedy: bool = False):
    """Actions from logits, sampled on the CPU with a seeded generator (deterministic on any device).
    Returns actions, log-probs and the action probabilities (all numpy)."""
    probs = torch.softmax(logits.float().cpu(), 1)
    a = probs.argmax(1) if greedy else torch.multinomial(probs, 1, generator=gen).squeeze(1)
    logp = torch.log(probs.gather(1, a.unsqueeze(1)).squeeze(1) + 1e-12)
    return a.numpy(), logp.numpy(), probs.numpy()


class PolicyAgent:
    """A trained ActorCritic with the baseline interface: act(obs, alive) -> actions, reset(mask).
    `.last` holds the value, action probabilities and log-prob of the last decision (recorder, viewer)."""
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
        self.mode = None
        self.last = {}

    def reset(self, mask=None):
        if mask is None:
            self.reset_mask[:] = True
        else:
            self.reset_mask |= np.asarray(mask, bool)

    @torch.no_grad()
    def act(self, obs, alive):
        grid = torch.from_numpy(obs["grid"]).to(self.dev)
        vec = torch.from_numpy(flatten_vec(obs)).to(self.dev)
        logits, value, self.h = self.model.step(grid, vec, self.h, torch.from_numpy(self.reset_mask).to(self.dev))
        self.reset_mask[:] = False
        a, logp, probs = sample_actions(logits, self.gen, self.greedy)
        self.last = dict(value=value.float().cpu().numpy(), probs=probs, logp=logp)
        return a


def checkpoint_path(run: str, root: Path | None = None) -> Path:
    return (root or ROOT) / "runs" / "stage2" / run / "checkpoints" / "last.pt"


def load_model(path) -> tuple[ActorCritic, dict]:
    """Network + checkpoint dict from a PPOTrainer checkpoint (experiments/train_ppo.py)."""
    ck = torch.load(path, weights_only=False, map_location="cpu")
    if ck.get("format") != CHECKPOINT_FORMAT:
        raise ValueError(f"{path}: checkpoint format {ck.get('format', 1)} predates the unified learner "
                         f"(format {CHECKPOINT_FORMAT}); retrain with experiments/train_ppo.py")
    model = ActorCritic(ck["obs_shapes"], ck["n_actions"], ck["cfg"]["hidden"], ck["cfg"]["core"])
    model.load_state_dict(ck["trainer"]["model"])
    return model, ck


def load_policy_agent(run_or_path, n_slots: int, seed: int = 0, device: str = "cpu", greedy: bool = False,
                      root: Path | None = None) -> PolicyAgent:
    """PolicyAgent from a run name (runs/stage2/<run>/checkpoints/last.pt) or a checkpoint path."""
    p = Path(run_or_path)
    path = p if p.suffix == ".pt" else checkpoint_path(str(run_or_path), root)
    model, ck = load_model(path)
    agent = PolicyAgent(model, n_slots, seed=seed, device=device, greedy=greedy)
    agent.description = f"PPO {ck['cfg']['core'].upper()} ({ck['trainer']['agent_steps'] / 1e6:.1f}M agent-steps)"
    agent.checkpoint = str(path)
    return agent


def trained_runs(root: Path | None = None) -> list[str]:
    """Runs under runs/stage2 with a loadable (current-format) checkpoint."""
    out = []
    for p in sorted(((root or ROOT) / "runs" / "stage2").glob("*/checkpoints/last.pt")):
        try:
            if torch.load(p, weights_only=False, map_location="cpu").get("format") == CHECKPOINT_FORMAT:
                out.append(p.parent.parent.name)
        except Exception:
            pass
    return out
