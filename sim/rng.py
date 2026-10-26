"""Counter-based RNG: u = hash(seed, tick, index, stream).

Every random draw inside a kernel is a pure function of its coordinates, so results are
identical regardless of thread count or iteration order (needed for parallel kernels and
for bit-exact replays).
"""
import numpy as np
from numba import njit, uint64

_M1 = uint64(0xBF58476D1CE4E5B9)
_M2 = uint64(0x94D049BB133111EB)
_G = uint64(0x9E3779B97F4A7C15)


@njit(cache=True, inline="always")
def splitmix(x):
    x = (x + _G)
    x = (x ^ (x >> uint64(30))) * _M1
    x = (x ^ (x >> uint64(27))) * _M2
    return x ^ (x >> uint64(31))


@njit(cache=True, inline="always")
def hash4(seed, tick, idx, stream):
    h = splitmix(uint64(seed))
    h = splitmix(h ^ uint64(tick))
    h = splitmix(h ^ uint64(idx))
    return splitmix(h ^ uint64(stream))


@njit(cache=True, inline="always")
def uniform(seed, tick, idx, stream):
    """float64 in [0, 1)."""
    return (hash4(seed, tick, idx, stream) >> uint64(11)) * (1.0 / 9007199254740992.0)


@njit(cache=True, inline="always")
def randint(seed, tick, idx, stream, n):
    return int(uniform(seed, tick, idx, stream) * n)


@njit(cache=True)
def permutation(seed, tick, stream, n):
    """Deterministic shuffle of range(n) (Fisher-Yates driven by the hash RNG)."""
    p = np.arange(n)
    for i in range(n - 1, 0, -1):
        j = randint(seed, tick, i, stream, i + 1)
        p[i], p[j] = p[j], p[i]
    return p
