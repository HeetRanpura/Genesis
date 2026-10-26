"""Integration tests for the Stage 2 PPO learner (CPU, tiny world)."""
import numpy as np
import torch

from agents.policy import ActorCritic, PolicyAgent, flatten_vec
from agents.ppo import PPOConfig, PPOTrainer
from tests.helpers import make_env


def _trainer(core="gru", seed=0):
    env = make_env(seed=3, agents={"n_initial": 12, "max_agents": 12, "respawn": True})
    cfg = PPOConfig(n_agents=12, core=core, hidden=32, rollout_len=16, seq_len=8, epochs=2, minibatches=2,
                    device="cpu")
    return PPOTrainer(env, cfg, seed=seed)


def test_training_iterations_run_and_losses_finite():
    for core in ("gru", "mlp"):
        tr = _trainer(core)
        for _ in range(3):
            info = tr.iterate()
        for k in ("pg_loss", "v_loss", "entropy", "approx_kl"):
            assert np.isfinite(info[k]), k
        assert tr.agent_steps == 3 * 16 * 12


def test_unroll_matches_stepwise_forward():
    """The training unroll must reproduce the acting-time recurrent computation exactly."""
    env = make_env(seed=1, agents={"n_initial": 6, "max_agents": 6, "respawn": True})
    torch.manual_seed(0)
    m = ActorCritic(env.obs_shapes, env.n_actions, hidden=16, core="gru")
    rng = np.random.default_rng(0)
    grids, vecs, resets, logits = [], [], [], []
    h = m.initial_state(env.N)
    reset = np.ones(env.N, bool)
    obs = env.observe()
    with torch.no_grad():
        for t in range(10):
            g, v = torch.from_numpy(obs["grid"].copy()), torch.from_numpy(flatten_vec(obs))
            lo, _, h = m.step(g, v, h, torch.from_numpy(reset))
            grids.append(g), vecs.append(v), resets.append(torch.from_numpy(reset.copy())), logits.append(lo)
            obs, r, d, info = env.step(rng.integers(0, env.n_actions, env.N))
            reset = info["born"].copy()
        lo2, _ = m.unroll(torch.stack(grids), torch.stack(vecs), m.initial_state(env.N), torch.stack(resets))
    assert torch.allclose(torch.stack(logits), lo2, atol=1e-5)


def test_checkpoint_roundtrip():
    tr = _trainer()
    tr.iterate()
    sd = tr.state_dict()
    tr2 = _trainer(seed=99)
    tr2.load_state_dict(sd)
    for a, b in zip(tr.model.parameters(), tr2.model.parameters()):
        assert torch.equal(a, b)
    assert tr2.iteration == tr.iteration and tr2.agent_steps == tr.agent_steps


def test_trainer_rewards_finite():
    """The learner trains on the env reward divided by a positive running scale (HARD RULE 1)."""
    tr = _trainer()
    buf, stats = tr.collect()
    raw = stats["raw_reward"]
    assert np.isfinite(raw)
    scaled = buf["rew"][buf["valid"]]
    assert np.isfinite(scaled).all()


def test_policy_agent_interface():
    tr = _trainer()
    agent = PolicyAgent(tr.model.cpu(), tr.env.N, seed=0)
    a = agent.act(tr.env.observe(), tr.env.alive)
    assert a.shape == (tr.env.N,) and a.min() >= 0 and a.max() < tr.env.n_actions
