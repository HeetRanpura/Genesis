"""Stage 2 learner (agents/ppo.py PPOTrainer + agents/policy.py): training loop, recurrence, GAE,
determinism, checkpoints, flight-recorder integration. CPU, tiny world."""
import numpy as np
import pytest
import torch

from agents.policy import ActorCritic, PolicyAgent, flatten_vec, load_model, load_policy_agent
from agents.ppo import PPOConfig, PPOTrainer, gae
from tests.helpers import SMALL, make_env


def _env(seed=3):
    return make_env(seed=seed, agents=dict(SMALL["agents"], n_initial=12, max_agents=12, respawn=True))


def _trainer(core="gru", seed=0, **kw):
    cfg = dict(n_agents=12, core=core, hidden=32, rollout_len=16, seq_len=16, epochs=2, minibatches=2, device="cpu")
    cfg.update(kw)
    return PPOTrainer(_env(), PPOConfig(**cfg), seed=seed)


@pytest.mark.parametrize("core", ["gru", "mlp"])
def test_training_iterations_run_and_losses_finite(core):
    tr = _trainer(core)
    for _ in range(3):
        info = tr.iterate()
    for k in ("pg", "vf", "ent", "kl", "clipfrac", "explained_var"):
        assert np.isfinite(info[k]), k
    assert tr.agent_steps == 3 * 16 * 12 and tr.iteration == 3


def test_training_is_deterministic_on_cpu():
    """Same seed -> identical parameters after training (world, sampling and minibatch order are seeded)."""
    a, b = _trainer(seed=3), _trainer(seed=3)
    for _ in range(2):
        a.iterate()
        b.iterate()
    for p, q in zip(a.model.parameters(), b.model.parameters()):
        assert torch.equal(p, q)
    c = _trainer(seed=4)
    c.iterate()
    assert any(not torch.equal(p, q) for p, q in zip(a.model.parameters(), c.model.parameters()))


@pytest.mark.parametrize("core", ["gru", "mlp"])
def test_unroll_matches_stepwise_forward(core):
    """The training unroll must reproduce the acting-time recurrent computation exactly."""
    env = make_env(seed=1, agents={"n_initial": 6, "max_agents": 6, "respawn": True})
    torch.manual_seed(0)
    m = ActorCritic(env.obs_shapes, env.n_actions, hidden=16, core=core)
    rng = np.random.default_rng(0)
    grids, vecs, resets, logits = [], [], [], []
    h = m.initial_state(env.N)
    reset = np.ones(env.N, bool)
    obs = env.observe()
    with torch.no_grad():
        for _ in range(10):
            g, v = torch.from_numpy(obs["grid"].copy()), torch.from_numpy(flatten_vec(obs))
            lo, _, h = m.step(g, v, h, torch.from_numpy(reset))
            grids.append(g), vecs.append(v), resets.append(torch.from_numpy(reset.copy())), logits.append(lo)
            obs, r, d, info = env.step(rng.integers(0, env.n_actions, env.N))
            reset = info["born"].copy()
        lo2, _ = m.unroll(torch.stack(grids), torch.stack(vecs), m.initial_state(env.N), torch.stack(resets))
    assert torch.allclose(torch.stack(logits), lo2, atol=1e-5)


def test_memory_reset_for_new_agents():
    env = _env()
    m = ActorCritic(env.obs_shapes, env.n_actions, hidden=32, core="gru")
    obs = env.observe()
    g, v = torch.from_numpy(obs["grid"]), torch.from_numpy(flatten_vec(obs))
    h = torch.ones(env.N, 32)
    reset = torch.zeros(env.N, dtype=torch.bool)
    reset[0] = True
    with torch.no_grad():
        z = m.encode(g, v)
        _, _, h_new = m.step(g, v, h, reset)
        assert torch.allclose(h_new[0], m.core(z[:1], torch.zeros(1, 32))[0], atol=1e-6)   # slot 0 forgot
        assert torch.allclose(h_new[1], m.core(z[1:2], torch.ones(1, 32))[0], atol=1e-6)   # slot 1 remembered


def test_gae_does_not_bootstrap_through_death():
    rew = np.array([[0.0], [-1.0]], np.float32)
    done = np.array([[0.0], [1.0]], np.float32)
    val = np.array([[0.5], [0.5]], np.float32)
    adv, ret = gae(rew, val, done, np.array([100.0], np.float32), gamma=0.9, lam=1.0)
    assert np.isclose(ret[1, 0], -1.0)           # dead: no value from beyond
    assert np.isclose(ret[0, 0], 0.9 * -1.0)


def test_truncated_bptt_and_return_normalisation_options():
    tr = _trainer(rollout_len=16, seq_len=8, normalize_returns=True)
    info = tr.iterate()
    assert np.isfinite(info["pg"]) and tr.norm is not None
    with pytest.raises(AssertionError):
        _trainer(rollout_len=16, seq_len=5)


def test_checkpoint_roundtrip_and_policy_agent(tmp_path):
    tr = _trainer(seed=5)
    tr.iterate()
    tr.save(tmp_path / "c.pt", world=0)
    tr2 = _trainer(seed=99)
    tr2.load(tmp_path / "c.pt")
    for a, b in zip(tr.model.parameters(), tr2.model.parameters()):
        assert torch.equal(a, b)
    assert tr2.iteration == tr.iteration and tr2.agent_steps == tr.agent_steps
    agent = load_policy_agent(tmp_path / "c.pt", tr.env.N, seed=0, device="cpu")
    for a, b in zip(tr.model.parameters(), agent.model.parameters()):
        assert torch.equal(a, b)
    act = agent.act(tr.env.observe(), tr.env.alive)
    assert act.shape == (tr.env.N,) and act.min() >= 0 and act.max() < tr.env.n_actions
    assert set(agent.last) == {"value", "logp", "probs"}


def test_old_checkpoint_format_is_rejected(tmp_path):
    torch.save(dict(trainer={}, core="gru"), tmp_path / "old.pt")
    with pytest.raises(ValueError, match="predates"):
        load_model(tmp_path / "old.pt")


def test_policy_agent_sampling_is_seeded():
    tr = _trainer()
    obs = tr.env.observe()
    a1 = PolicyAgent(tr.model, tr.env.N, seed=7).act(obs, tr.env.alive)
    a2 = PolicyAgent(tr.model, tr.env.N, seed=7).act(obs, tr.env.alive)
    assert np.array_equal(a1, a2)


def test_training_rollouts_are_flight_recorded(tmp_path):
    from analysis.trajectories import iter_chunks
    from sim.recorder import TrajectoryRecorder
    tr = _trainer()
    rec = TrajectoryRecorder(tr.env, tmp_path / "w0", chunk_ticks=8)
    tr.iterate(rec, record=True)          # 16 recorded ticks
    tr.iterate(rec, record=False)         # 16 skipped ticks (births still logged)
    idx = rec.close()
    assert idx["ticks"] == 16 and idx["skipped_ticks"] == 16
    chunk = next(iter_chunks(tmp_path / "w0", ("x_value", "x_probs")))
    assert chunk["x_probs"].shape[1] == tr.env.n_actions
