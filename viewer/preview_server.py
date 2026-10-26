"""Lightweight live preview of a GENESIS run (stdlib only; the full PixiJS viewer is Stage 5).

    .venv/bin/python viewer/preview_server.py --policy ppo:gru --seed 0 --port 8765

Serves http://localhost:PORT/ :
  /            viewer page (canvas)
  /world.png   world layer image (terrain tinted by plants, snow, fire, burn scars, floods, buildings)
  /state       JSON: clock, weather, stats, agents, animals
  /control     ?speed=<ticks per second>&paused=0|1&reset=1&fire=x,y&flood=x,y
  /agent?slot= JSON detail for one agent
"""
from __future__ import annotations

import argparse
import json
import struct
import sys
import threading
import time
import zlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from agents.baselines import make_baseline  # noqa: E402
from sim import GenesisEnv, load_config  # noqa: E402
from sim.constants import DEATH_NAMES  # noqa: E402
from sim.env import SEASON_NAMES  # noqa: E402
from sim.layout import (A_ENERGY, A_HEALTH, A_HYDRATION, A_WARMTH, F_ALIVE, F_SPECIES, F_X, F_Y, I_AGE,  # noqa: E402
                        I_ALIVE, I_CARRY_TYPE, I_GEN, I_LAST_ACTION, I_LIFESPAN, I_UID, I_X, I_Y, S_DISTANCE,
                        S_ENERGY_EATEN, S_KILL_LARGE, S_KILL_SMALL)

# terrain palette (painterly-ish): deep, shallow, sand, grass, meadow, forest, barren, rock, mountain, cave
PALETTE = np.array([[28, 58, 104], [64, 128, 176], [226, 211, 158], [150, 186, 98], [112, 176, 84],
                    [46, 104, 60], [196, 168, 112], [140, 134, 124], [86, 80, 92], [52, 44, 50]], np.float32)
BARE = np.array([182, 160, 112], np.float32)


def png_bytes(rgb: np.ndarray) -> bytes:
    h, w, _ = rgb.shape
    raw = b"".join(b"\x00" + rgb[y].tobytes() for y in range(h))

    def chunk(tag, data):
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw, 3)) + chunk(b"IEND", b""))


class Sim:
    def __init__(self, args):
        self.args = args
        self.lock = threading.Lock()
        self.speed = args.speed
        self.paused = False
        self.events_tail: list[str] = []
        self._build()

    def _build(self):
        a = self.args
        cfg = load_config(overrides={"agents": {"n_initial": a.agents, "max_agents": a.agents}})
        self.env = GenesisEnv(cfg, seed=a.seed)
        if a.policy.startswith("ppo:"):
            from agents.policy import load_policy_agent
            self.policy = load_policy_agent(a.policy[4:], self.env.N, seed=a.seed, device="cpu")
            self.policy_desc = self.policy.description
        else:
            self.policy = make_baseline(a.policy, self.env, a.seed)
            self.policy_desc = a.policy
        self.obs = self.env.observe()
        self.deaths_recent = []
        self._world_png = None
        self._world_tick = -1

    def reset(self):
        with self.lock:
            self._build()

    def step(self):
        with self.lock:
            env = self.env
            obs, r, died, info = env.step(self.policy.act(self.obs, env.alive))
            self.obs = obs
            if info["born"].any() and hasattr(self.policy, "reset"):
                self.policy.reset(info["born"])
            ev = env.last_events
            for row in ev[ev[:, 1] == 1][:3]:
                self.events_tail.append(f"day {env.day}: agent {int(row[2])} died ({DEATH_NAMES[int(row[6])]})")
            for typ, msg in ((6, "a large prey was brought down by a group hunt"), (13, "wildfire started"),
                             (14, "river flood"), (15, "famine began"), (16, "drought began"),
                             (17, "cold snap began")):
                if (ev[:, 1] == typ).any():
                    self.events_tail.append(f"day {env.day}: {msg}")
            self.events_tail = self.events_tail[-8:]

    def run_forever(self):
        last = time.perf_counter()
        while True:
            if self.paused:
                time.sleep(0.05)
                continue
            target = 1.0 / max(self.speed, 0.1)
            self.step()
            now = time.perf_counter()
            sleep = target - (now - last)
            if sleep > 0:
                time.sleep(sleep)
            last = time.perf_counter()

    # ---------------------------------------------------------------------------------
    def world_png(self) -> bytes:
        with self.lock:
            env = self.env
            if self._world_png is not None and env.tick - self._world_tick < 4:
                return self._world_png
            t = env.terrain
            base = PALETTE[t]
            veg = np.isin(t, [3, 4, 5, 6])
            p = np.clip(env.plant / np.maximum(env.fertility, 0.05), 0, 1)[..., None]
            rgb = np.where(veg[..., None], BARE * (1 - p) + base * p, base)
            shade = 0.82 + 0.3 * (env.elev[..., None] - 0.3)           # gentle relief shading
            rgb = rgb * np.clip(shade, 0.75, 1.1)
            fl = env.flood > 0
            rgb[fl] = rgb[fl] * 0.4 + np.array([70, 140, 210]) * 0.6
            sn = env.snow[..., None]
            rgb = rgb * (1 - sn) + np.array([240, 244, 250]) * sn
            scar = env.fire < 0
            rgb[scar] = rgb[scar] * 0.35 + np.array([40, 30, 25]) * 0.65
            burn = env.fire > 0
            rgb[burn] = np.array([255, 120, 30])
            b = env.built > 0
            rgb[b] = np.array([150, 100, 60])
            img = np.clip(rgb, 0, 255).astype(np.uint8)
            self._world_png = png_bytes(img)
            self._world_tick = env.tick
            return self._world_png

    def state(self) -> dict:
        with self.lock:
            env = self.env
            al = np.flatnonzero(env.alive)
            I, A, G = env.I, env.A, env.G
            agents = np.stack([I[al, I_X], I[al, I_Y], (A[al, A_ENERGY] * 100).astype(int),
                               (A[al, A_HYDRATION] * 100).astype(int), (A[al, A_HEALTH] * 100).astype(int),
                               (A[al, A_WARMTH] * 100).astype(int), I[al, I_LAST_ACTION],
                               ((G[al, 3] + 1) * 180).astype(int), al, I[al, I_CARRY_TYPE]], 1).tolist()
            fa = np.flatnonzero(env.FI[:, F_ALIVE] == 1)
            fauna = np.stack([env.FI[fa, F_X], env.FI[fa, F_Y], env.FI[fa, F_SPECIES]], 1).tolist()
            fruit = np.flatnonzero(env.tree_fruit >= 1.0)
            m = env.metrics()
            return dict(
                tick=env.tick, day=env.day, year=env.year, season=SEASON_NAMES[env.season],
                time_of_day=env.time_of_day, night=env.night_level(), weather=["clear", "rain", "fog"][env.weather],
                temp=round(env.global_temp(), 1), drought=env.drought_left > 0, cold_snap=env.cold_snap_left > 0,
                famine=env.famine_left > 0, w=env.W, h=env.H, policy=self.policy_desc, speed=self.speed,
                paused=self.paused, population=len(al), agents=agents, fauna=fauna,
                fruit=np.stack([env.tree_x[fruit], env.tree_y[fruit]], 1).tolist(),
                counts=dict(small_prey=int(env.fcount[0]), large_prey=int(env.fcount[1]), predators=int(env.fcount[2])),
                needs={k: round(m[f"mean_{k}"], 2) for k in ("energy", "hydration", "health", "warmth")},
                deaths={k: m[f"deaths_{k}"] for k in DEATH_NAMES if m[f"deaths_{k}"]},
                group_hunts=m["ev_kill_large"], food_drops=m["ev_drop_food"], events=self.events_tail,
                action_names=env.action_names)

    def agent(self, slot: int) -> dict:
        with self.lock:
            env = self.env
            if slot < 0 or slot >= env.N or not env.alive[slot]:
                return {}
            I, A, S = env.I, env.A, env.S
            return dict(slot=slot, uid=int(I[slot, I_UID]), generation=int(I[slot, I_GEN]),
                        age_days=round(I[slot, I_AGE] / env.tpd, 1),
                        lifespan_days=round(I[slot, I_LIFESPAN] / env.tpd, 1),
                        energy=float(A[slot, A_ENERGY]), hydration=float(A[slot, A_HYDRATION]),
                        health=float(A[slot, A_HEALTH]), warmth=float(A[slot, A_WARMTH]),
                        last_action=env.action_names[I[slot, I_LAST_ACTION]] if I[slot, I_LAST_ACTION] >= 0 else "-",
                        carrying=["nothing", "food", "wood", "stone"][I[slot, I_CARRY_TYPE]],
                        eaten=round(float(S[slot, S_ENERGY_EATEN]), 2), walked=int(S[slot, S_DISTANCE]),
                        small_kills=int(S[slot, S_KILL_SMALL]), large_kills=int(S[slot, S_KILL_LARGE]),
                        genes=[round(float(g), 2) for g in env.G[slot]])


def make_handler(sim: Sim):
    page = (Path(__file__).parent / "preview.html").read_bytes()

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, body: bytes, ctype: str):
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            u = urlparse(self.path)
            q = parse_qs(u.query)
            if u.path in ("/", "/index.html"):
                self._send(page, "text/html; charset=utf-8")
            elif u.path == "/world.png":
                self._send(sim.world_png(), "image/png")
            elif u.path == "/state":
                self._send(json.dumps(sim.state()).encode(), "application/json")
            elif u.path == "/agent":
                self._send(json.dumps(sim.agent(int(q.get("slot", ["-1"])[0]))).encode(), "application/json")
            elif u.path == "/control":
                if "speed" in q:
                    sim.speed = float(q["speed"][0])
                if "paused" in q:
                    sim.paused = q["paused"][0] == "1"
                if q.get("reset"):
                    sim.reset()
                for key, fn in (("fire", "trigger_fire"), ("flood", "trigger_flood")):
                    if key in q:
                        x, y = (int(v) for v in q[key][0].split(","))
                        with sim.lock:
                            getattr(sim.env, fn)(x, y)
                self._send(b"ok", "text/plain")
            else:
                self.send_error(404)

    return H


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default="ppo:gru_s0", help="ppo:<run> | heuristic | greedy | random")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--agents", type=int, default=256)
    ap.add_argument("--speed", type=float, default=20, help="ticks per second")
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()
    if args.policy.startswith("ppo:") and not (ROOT / "runs" / "stage2" / args.policy[4:] / "checkpoints" / "last.pt").exists():
        print(f"no checkpoint for {args.policy}; falling back to the heuristic baseline")
        args.policy = "heuristic"
    sim = Sim(args)
    threading.Thread(target=sim.run_forever, daemon=True).start()
    srv = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(sim))
    print(f"GENESIS preview on http://localhost:{args.port}  policy={sim.policy_desc}", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
