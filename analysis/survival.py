"""Lifetimes from event logs, Kaplan-Meier curves, bootstrap confidence intervals."""
from __future__ import annotations

import numpy as np

from sim.constants import EV_BIRTH, EV_DEATH


def lifetimes(events: np.ndarray, initial_uids: np.ndarray, end_tick: int):
    """Per-agent (duration, observed_death, cause) measured from spawn (tick 0 for the initial cohort)."""
    start = {int(u): 0 for u in initial_uids}
    for t, typ, a, *_ in events[events[:, 1] == EV_BIRTH]:
        start[int(a)] = int(t)
    dur, obs, cause = [], [], []
    dead = set()
    for row in events[events[:, 1] == EV_DEATH]:
        uid = int(row[2])
        if uid in start:
            dur.append(int(row[0]) - start[uid] + 1)
            obs.append(True)
            cause.append(int(row[6]))
            dead.add(uid)
    for uid, s in start.items():
        if uid not in dead:
            dur.append(end_tick - s)
            obs.append(False)
            cause.append(-1)
    return np.array(dur), np.array(obs, bool), np.array(cause)


def kaplan_meier(durations: np.ndarray, observed: np.ndarray):
    """returns (times, survival) step function."""
    order = np.argsort(durations)
    d, o = durations[order], observed[order]
    times = np.unique(d[o])
    surv, s = [], 1.0
    for t in times:
        at_risk = (d >= t).sum()
        deaths = ((d == t) & o).sum()
        s *= 1.0 - deaths / at_risk
        surv.append(s)
    return np.concatenate([[0], times]), np.concatenate([[1.0], surv])


def km_at(durations, observed, t):
    times, surv = kaplan_meier(durations, observed)
    k = np.searchsorted(times, t, side="right") - 1
    return float(surv[max(k, 0)])


def bootstrap_ci(values, stat=np.mean, n=2000, alpha=0.05, seed=0):
    values = np.asarray(values, float)
    rng = np.random.default_rng(seed)
    boots = np.array([stat(rng.choice(values, values.size, replace=True)) for _ in range(n)])
    return float(stat(values)), float(np.quantile(boots, alpha / 2)), float(np.quantile(boots, 1 - alpha / 2))


def hedges_g(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    na, nb = a.size, b.size
    sp = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    g = (a.mean() - b.mean()) / sp if sp > 0 else float("inf")
    return float(g * (1 - 3 / (4 * (na + nb) - 9)))
