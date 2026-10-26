# GENESIS live viewer

```bash
make viewer-setup     # once: aiohttp + three.js (npm, pinned 0.170.0 in viewer/package.json)
make viewer           # http://localhost:8787   (SEED=3 AGENTS=1000 POLICY=greedy PORT=8790 make viewer)
```

Port 8765 is avoided on purpose, because another local service on the development machine uses it.

## Architecture
```
viewer/server.py  (Python, one process)
  sim thread ── GenesisEnv.step() + baseline policy, paced at speed × 10 ticks/s (MAX = unpaced)
     │           packs a frame at 30 Hz, a field snapshot at 2 Hz, a dashboard sample every 20 ticks
     ▼
  aiohttp ── GET /  (viewer/static)  · /vendor/three (viewer/node_modules/three) · /ws (WebSocket)
     │
browser (viewer/static/js, ES modules, no build step)
  net.js       binary decoder        world.js     terrain, water, flora, rocks, carcasses, fields
  sky.js       sun/moon, sky, fog    creatures.js agents + fauna, interpolation, animation, icons
  effects.js   weather, fire, FX     post.js      bloom → ACES tone map → tilt-shift/grade/grain
  hud.js       dashboard, inspector, feed, minimap, controls        main.js  loop, camera, input
```
The simulation runs on its own thread, and the browser only draws what it receives. Commands such as speed, pause,
disturbances and restarts go into a queue and are applied **between ticks**. Playback speed has no effect on results.

## Wire format
Every binary message is `[u32 header_len][header JSON][typed buffers, 4-byte aligned]`, and the header's `bufs`
lists each buffer's name, dtype, offset and length.
- `init`: once per world, and again after a restart. It carries `terrain` u8, `height` f32 (visual relief from
  `scene_heights`), `water` and `flood` surface levels f32 (NaN = none), `fertility`, tree positions and seasons,
  config facts, the server-side dashboard history (720 samples) and the recent event feed.
- `fields`: 2 Hz. RGBA u8 per cell: R plant, G snow, B fire, A flags (1 flood, 2 dried stream, 4 food item,
  8 burnt forest, 16 built). Also per-tree fruit and alive state.
- `frame`: 30 Hz. Agents as `u16×8` per live agent: x, y, uid lo/hi, action|carry, energy|hydration,
  health|warmth, gene hue|tone|size. Fauna as `u16×4`: x, y, species|fleeing|health, slot. The JSON
  header holds clock and weather, alerts, measured ticks/s, new dashboard samples and `fx` events
  (deaths with cause, births, kills, predator attacks, co-feeding, disasters).
- `detail` (text JSON): 10–30 Hz, only for the agent this client selected. It holds the full body state,
  lifetime stats, genes and the last 16 actions.

## Controls
Left drag or WASD pans. Right drag or Q/E rotates. The wheel zooms.
Click an agent to open the inspector. F follows it, Esc releases it.
Space pauses, `.` advances one tick, and 1–6 set the speed (¼× … 100×, MAX).
C toggles the cinematic director, O cycles overlays (plants, agent density, fertility), and H hides the UI.
P takes a PNG, R records a WebM, and Shift+Q changes quality.
The bottom bar triggers fire or flood (click the map), drought, cold snap and famine, and "New run" starts a new seed, policy or agent count.

## Logging (HARD RULE 3)
Each session writes `runs/viewer/<stamp>_s<seed>_<policy>/`. The directory holds:
- `config.yaml` and `manifest.json` (git commit, versions, world hash)
- `metrics.csv` (every `logging.metrics_every` ticks)
- `events_XXXX.npy` (the full event stream, in chunks)
- `controls.jsonl` (every command, with wall time and the tick it was applied at)
- `summary.json` (final metrics, manual interventions, deaths per agent-year)

A session that lasts at least one sim day also appends a summary to `logs/AGENT_LOG.md`; use `--no-agent-log` for test sessions.

## What is real and what is decoration
**From the simulation:** agent and animal positions, actions, body variables, genes, plant biomass,
snow, fire, floods, dried streams, carcasses and dropped food, fruit, burnt forest, weather, temperature,
the time of day and season, and every number on the dashboard.

**Visual only:**
- Terrain relief. The simulation is a flat grid, and elevation only affects temperature.
- Water depth and shading, tree and grass placement, and snow caps on the highest mountain peaks.
- Cloud shadows, fireflies, falling leaves and petals.
- The agent walk, eat, drink and rest animations, which are driven by the agent's last action.

**Need icons** (🍖 💧 ❄️ ❤️) show the agent's own body variable below 0.3. They are not probed internal
states; probing waits for learners in Stage 2/3. 💤 marks resting at night, and ⚔️ marks hunting.

**Clark–Evans R** is computed on the server over passable land. Baseline agents aggregate around water and
food, so R < 1 is not evidence of anything social.

**"Ate others' kill"** counts meals from a carcass that another agent killed or dropped. It is co-feeding,
not sharing. Large prey need two or more attackers by world rule (an [affordance], see WORLD.md).

## Performance (MacBook Pro M3 Pro, headless Chromium + Metal ANGLE, 1600×900, pixel ratio 1)
- 256 agents: about 90–110 render fps. At MAX speed the sim runs about 1,450 ticks/s while streaming.
- 1,000 agents: about 75–90 fps. At MAX the sim runs about 840 ticks/s.

The renderer is not the bottleneck. A real browser on a Retina display runs at a higher pixel ratio,
so the default quality is chosen from `devicePixelRatio`, and Shift+Q changes it.

## Not done yet (Stage 5 items still open)
These are still to build: replay scrubbing and bookmarks, side-by-side seeds, mid-run resource redistribution,
spawning and removing agents, procedural audio, 4K offline export, social-network and signal-flow overlays (they need
learners first), and imagined-future ghosts (they need world models).
