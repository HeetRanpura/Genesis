"""Procedural world generation from a seed.

Pipeline: domain-warped fBm elevation with continental falloff -> rank-normalised heights ->
priority-flood depression filling (lakes = filled depressions) -> D8 flow accumulation
(rivers and streams) -> moisture from noise + distance to fresh water -> biome classes ->
caves on rock/mountain margins -> fertility, fruit trees, wood.
"""
from __future__ import annotations

import heapq

import numpy as np
from numba import njit
from scipy import ndimage

from .constants import (BARREN, CAVE, DEEP_WATER, FOREST, GRASS, MEADOW, MOUNTAIN, ROCK,
                        SAND, SHALLOW)

_D8 = np.array([(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)], dtype=np.int64)


def value_noise(rng: np.random.Generator, h: int, w: int, cell: float) -> np.ndarray:
    """Smooth value noise with lattice spacing `cell` pixels."""
    gh, gw = int(np.ceil(h / cell)) + 2, int(np.ceil(w / cell)) + 2
    lattice = rng.random((gh, gw))
    y = np.arange(h) / cell
    x = np.arange(w) / cell
    y0, x0 = np.floor(y).astype(int), np.floor(x).astype(int)
    ty, tx = y - y0, x - x0
    ty = ty * ty * (3 - 2 * ty)
    tx = tx * tx * (3 - 2 * tx)
    a = lattice[y0][:, x0]
    b = lattice[y0][:, x0 + 1]
    c = lattice[y0 + 1][:, x0]
    d = lattice[y0 + 1][:, x0 + 1]
    top = a + (b - a) * tx[None, :]
    bot = c + (d - c) * tx[None, :]
    return top + (bot - top) * ty[:, None]


def fbm(rng, h, w, cell=64.0, octaves=5, persistence=0.5):
    out = np.zeros((h, w))
    amp, total = 1.0, 0.0
    for _ in range(octaves):
        out += amp * value_noise(rng, h, w, max(cell, 1.5))
        total += amp
        amp *= persistence
        cell /= 2.0
    return out / total


def rank_normalise(a: np.ndarray) -> np.ndarray:
    flat = a.ravel()
    ranks = np.empty(flat.size)
    ranks[np.argsort(flat, kind="stable")] = np.arange(flat.size)
    return (ranks / (flat.size - 1)).reshape(a.shape)


@njit(cache=True)
def _flow_accumulation(filled, land):
    h, w = filled.shape
    order = np.argsort(-filled.ravel())
    acc = np.ones(h * w)
    recv = -np.ones(h * w, dtype=np.int64)
    for k in range(order.size):
        i = order[k]
        y, x = i // w, i % w
        if not land[y, x]:
            continue
        best, by, bx = filled[y, x], -1, -1
        for d in range(8):
            ny, nx = y + _D8[d, 0], x + _D8[d, 1]
            if 0 <= ny < h and 0 <= nx < w and filled[ny, nx] < best:
                best, by, bx = filled[ny, nx], ny, nx
        if by >= 0:
            recv[i] = by * w + bx
            acc[by * w + bx] += acc[i]
    return acc.reshape(h, w), recv


def priority_flood(elev: np.ndarray, ocean: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Barnes et al. (2014) priority-flood + epsilon: every land cell drains to the ocean."""
    h, w = elev.shape
    filled = elev.copy()
    closed = ocean.copy()
    heap = []
    # seeds: land cells adjacent to ocean, plus map border
    ys, xs = np.nonzero(~ocean)
    for y, x in zip(ys, xs):
        border = y == 0 or x == 0 or y == h - 1 or x == w - 1
        if border or ocean[max(y - 1, 0):y + 2, max(x - 1, 0):x + 2].any():
            heapq.heappush(heap, (filled[y, x], y, x))
            closed[y, x] = True
    while heap:
        e, y, x = heapq.heappop(heap)
        for dy, dx in _D8:
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w and not closed[ny, nx]:
                closed[ny, nx] = True
                filled[ny, nx] = max(filled[ny, nx], e + eps)
                heapq.heappush(heap, (filled[ny, nx], ny, nx))
    return filled


def generate_world(cfg, seed: int) -> dict:
    wc = cfg.world
    h, w = wc.height, wc.width
    rng = np.random.default_rng([seed, 0x5EED])

    # ---- elevation ------------------------------------------------------------------
    scale = max(h, w) / 256.0
    warp_y = (fbm(rng, h, w, 96 * scale, 3) - 0.5) * 40 * scale
    warp_x = (fbm(rng, h, w, 96 * scale, 3) - 0.5) * 40 * scale
    base = fbm(rng, h, w, 110 * scale, 6, 0.52)
    yy, xx = np.mgrid[0:h, 0:w].astype(float)
    base = ndimage.map_coordinates(base, [np.clip(yy + warp_y, 0, h - 1), np.clip(xx + warp_x, 0, w - 1)],
                                   order=1, mode="nearest")
    dy, dx = (yy - h / 2) / (h / 2), (xx - w / 2) / (w / 2)
    dist = np.sqrt(dy ** 2 + dx ** 2) / np.sqrt(2)
    raw = base - wc.continent_falloff * dist ** 2.2
    elev = rank_normalise(raw)
    ocean = elev < wc.sea_level
    ocean_lab, _ = ndimage.label(ocean)
    border_labels = np.unique(np.concatenate([ocean_lab[0], ocean_lab[-1], ocean_lab[:, 0], ocean_lab[:, -1]]))
    ocean = np.isin(ocean_lab, border_labels[border_labels > 0])
    inland_low = (elev < wc.sea_level) & ~ocean           # becomes lake
    # land-relative height quantile: rock/mountain thresholds are fractions of the land area
    land_q = np.zeros_like(elev)
    land_q[~ocean] = rank_normalise(elev[~ocean])

    # ---- hydrology ---------------------------------------------------------------------
    filled = priority_flood(elev, ocean)
    depth = filled - elev
    lake = ((depth > wc.lake_min_depth) | inland_low) & ~ocean
    lake_lab, n_lake = ndimage.label(lake)
    if n_lake:  # drop tiny ponds (< 6 cells)
        sizes = ndimage.sum(lake, lake_lab, index=np.arange(1, n_lake + 1))
        lake &= np.isin(lake_lab, np.nonzero(sizes >= 6)[0] + 1)
    flow, _ = _flow_accumulation(filled, ~ocean)
    river = (flow >= wc.river_flow_threshold) & ~ocean & ~lake
    stream = (flow >= wc.stream_flow_threshold) & ~river & ~ocean & ~lake
    # widen big rivers
    big = flow >= wc.river_flow_threshold * 6
    river |= ndimage.binary_dilation(big, iterations=1) & ~ocean & ~lake & (land_q < wc.rock_level)
    fresh = lake | river | stream

    # ---- moisture ------------------------------------------------------------------------
    dist_fresh = ndimage.distance_transform_edt(~fresh)
    dist_ocean = ndimage.distance_transform_edt(~ocean)
    moist_noise = fbm(rng, h, w, 72 * scale, 5, 0.55)
    moisture = 0.55 * moist_noise + 0.35 * np.exp(-dist_fresh / (10 * scale)) + 0.10 * np.exp(-dist_ocean / (14 * scale))
    moisture[~ocean] = rank_normalise(moisture[~ocean])   # land-relative quantile
    moisture[ocean] = 1.0

    # ---- biomes ---------------------------------------------------------------------------
    terrain = np.full((h, w), GRASS, dtype=np.uint8)
    land = ~ocean & ~lake
    terrain[land & (moisture < 0.22)] = BARREN
    terrain[land & (moisture > 0.62)] = FOREST
    meadow = land & (moisture > 0.45) & (moisture <= 0.62) & (dist_fresh < 6 * scale)
    terrain[meadow] = MEADOW
    terrain[land & (dist_ocean <= 2) & (elev < wc.sea_level + 0.04)] = SAND
    terrain[land & (land_q >= wc.rock_level)] = ROCK
    terrain[land & (land_q >= wc.mountain_level)] = MOUNTAIN
    terrain[ocean] = DEEP_WATER
    # lakes: deep core, shallow rim
    lake_core = ndimage.binary_erosion(lake, iterations=2)
    terrain[lake] = SHALLOW
    terrain[lake_core] = DEEP_WATER
    terrain[(river | stream) & (terrain != MOUNTAIN)] = SHALLOW
    # ocean shoreline: shallow but salty
    shore = ocean & ndimage.binary_dilation(land, iterations=1)
    terrain[shore] = SHALLOW
    # caves on rock cells touching mountains
    near_mtn = ndimage.binary_dilation(terrain == MOUNTAIN, iterations=1)
    cave_cand = (terrain == ROCK) & near_mtn
    cave = cave_cand & (rng.random((h, w)) < wc.cave_prob)
    if not cave.any() and cave_cand.any():   # guarantee at least a few caves
        idx = np.flatnonzero(cave_cand)
        cave.flat[rng.choice(idx, size=min(4, idx.size), replace=False)] = True
    terrain[cave] = CAVE

    fresh = fresh & (terrain != MOUNTAIN)        # drinkable water (ocean is salt)
    stream_only = stream & fresh                  # dries in drought

    # ---- fertility (plant carrying capacity) --------------------------------------------------
    base_fert = np.zeros(16)
    base_fert[[GRASS, MEADOW, FOREST, BARREN, SAND, ROCK, SHALLOW, CAVE]] = [0.55, 0.95, 0.45, 0.12, 0.06, 0.10, 0.0, 0.0]
    fertility = base_fert[terrain] * (0.55 + 0.45 * moisture)
    if wc.resource_layout == "uniform":
        veg = fertility > 0
        fertility[veg] = fertility[veg].mean()
    elif wc.resource_layout != "patchy":
        raise ValueError(f"unknown resource_layout {wc.resource_layout}")
    fertility = fertility.astype(np.float32)

    # ---- fruit trees -------------------------------------------------------------------------
    rc = cfg.resources
    tree_ok = (terrain == FOREST) | (terrain == MEADOW)
    if wc.resource_layout == "uniform":
        tree_ok = fertility > 0
    tmask = tree_ok & (rng.random((h, w)) < rc.fruit_tree_density * (1.5 if wc.resource_layout == "patchy" else 1.0))
    ty, tx = np.nonzero(tmask)
    tree_season = rng.choice(3, size=ty.size, p=[0.25, 0.4, 0.35]).astype(np.int8)  # spring/summer/autumn

    wood = np.where(terrain == FOREST, 1.0, 0.0).astype(np.float32)

    return dict(
        terrain=terrain, elevation=elev.astype(np.float32), moisture=moisture.astype(np.float32),
        fertility=fertility, fresh=fresh, stream=stream_only, river=river, lake=lake, ocean=ocean,
        flow=flow.astype(np.float32), dist_fresh=dist_fresh.astype(np.float32),
        tree_x=tx.astype(np.int32), tree_y=ty.astype(np.int32), tree_season=tree_season, wood=wood,
    )
