# Live viewer evidence (2026-10-26)

Captured by headless Chromium (Metal ANGLE) at 1600×900, pixel ratio 1, on a MacBook Pro M3 Pro. The server ran seed 0,
256 heuristic agents, `--no-agent-log`. Fire, flood and drought were triggered from the UI for these shots, and all of
them are logged in that session's `controls.jsonl`.

| file | shows |
|---|---|
| `01_day_closeup.jpg` | spring morning: forest, river delta, agents (gene-coloured), need icons, live dashboard |
| `02_overview.jpg` | whole island: mountains, lakes, rivers, ocean, minimap |
| `03_inspector_follow.jpg` | agent inspector (body state, stats, genes, last actions), follow camera, trail, perception radius |
| `04_flood.jpg` | a river flood covering the floodplain (surface driven by the sim's `flood` field) |
| `05_wildfire.jpg` | a wildfire front during a drought: flames, smoke, burnt scar, fire light at dusk |
| `06_night.jpg` | moonlit night, stars, fireflies |
| `07_autumn.jpg` | autumn foliage, snow-capped peaks, river network |
| `08_density_overlay.jpg` | agent-density heatmap overlay |
| `09_winter.jpg` | winter: the sim's snow cover on the uplands, snow on conifers |

Render rate in these runs: 85–115 fps with 256 agents, and 75–90 fps with 1,000 agents.
