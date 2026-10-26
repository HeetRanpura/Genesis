"""Smoke test for the live viewer: headless sim thread + WebSocket server boot, a client receives the
static world, a field update and a frame; a command is applied and logged; closing writes the run files."""
import asyncio
import json

import numpy as np
from aiohttp.test_utils import TestClient, TestServer

from tests.helpers import SMALL
from viewer.server import Sim, make_app, start_sim_thread

TYPES = {"u8": np.uint8, "u16": np.uint16, "u32": np.uint32, "i32": np.int32, "f32": np.float32}


def decode(raw: bytes):
    hl = int.from_bytes(raw[:4], "little")
    hdr = json.loads(raw[4:4 + hl])
    base = 4 + hl
    data = {b["name"]: np.frombuffer(raw, TYPES[b["dtype"]], b["length"], base + b["offset"]) for b in hdr["bufs"]}
    return hdr, data


def test_viewer_server_streams_and_logs(tmp_path):
    over = dict(SMALL, agents=dict(SMALL["agents"], respawn=True))
    sim = Sim(0, "heuristic", None, None, overrides=over, log_root=tmp_path, speed=0, agent_log=False)
    th = start_sim_thread(sim)
    assert sim.ready.wait(60)

    async def run():
        client = TestClient(TestServer(make_app(sim)))
        await client.start_server()
        try:
            ws = await client.ws_connect("/ws")
            got = {}
            while len(got) < 3:
                msg = await asyncio.wait_for(ws.receive(), 20)
                hdr, data = decode(msg.data)
                got.setdefault(hdr["type"], (hdr, data))
            init, idata = got["init"]
            assert (init["W"], init["H"]) == (64, 64)
            assert idata["terrain"].size == 64 * 64 and idata["height"].size == 64 * 64
            _, fdata = got["fields"]
            assert fdata["field"].size == 64 * 64 * 4
            frame, adata = got["frame"]
            n_alive = adata["agents"].size // 8
            assert n_alive == sim.env.population() or n_alive > 0
            assert frame["tps"] >= 0
            await ws.send_str(json.dumps({"cmd": "fire", "x": 10, "y": 12}))
            await ws.send_str(json.dumps({"cmd": "speed", "value": 4}))
            for _ in range(200):
                if sim.speed == 4:
                    break
                await asyncio.sleep(0.05)
            await ws.close()
        finally:
            await client.close()

    asyncio.run(run())
    assert sim.speed == 4
    sim.stop_flag.set()
    th.join(timeout=30)
    run_dir = next(tmp_path.iterdir())
    controls = [json.loads(line) for line in (run_dir / "controls.jsonl").read_text().splitlines()]
    assert [c["cmd"] for c in controls] == ["fire", "speed"]
    assert sim.env.manual_log[0]["kind"] == "fire"
    for f in ("config.yaml", "manifest.json", "summary.json"):
        assert (run_dir / f).exists()
    summary = json.loads((run_dir / "summary.json").read_text())
    assert summary["manual_interventions"][0]["kind"] == "fire"
    assert summary["ticks"] > 0
    # the session's flight recording exists and replays bit-exactly (including the manual fire)
    from analysis.trajectories import load, replay
    idx = json.loads((run_dir / "trajectory" / "index.json").read_text())
    assert idx["ticks"] == summary["ticks"] and idx["agent_rows"] > 0
    assert [m["kind"] for m in load(run_dir, "manual")] == ["fire"]
    assert sum(1 for _ in replay(run_dir)) == idx["ticks"]


def test_available_policies_include_baselines():
    from viewer.server import POLICIES, available_policies
    pols = available_policies()
    assert list(POLICIES) == pols[:3]
    assert all(p.startswith(("ppo:", "rppo:")) for p in pols[3:])
