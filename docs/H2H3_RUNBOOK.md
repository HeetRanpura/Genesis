# H2/H3 runbook: one command, about 13–16 hours on an M3 Pro

This trains and evaluates the preregistered H2/H3 experiment (`docs/PREREGISTRATION.md`, entry "H2/H3 protocol").
One script does everything: `experiments/run_h2h3.py`.

**What it runs.** 2 arms (GRU memory, memoryless) × learner seeds 1, 2, 3 = **6 runs**, each **3,000 iterations**
(98.3M agent-steps). Then each run's iteration-900 and iteration-3,000 checkpoints are evaluated on the held-out
worlds 0–4 × 2 years. Finally, the preregistered decision rules are applied.

**What it guarantees.**
- Progress bars (tqdm): one per running job, plus an overall bar with an ETA.
- Weights saved as `.pt` every 200 iterations and at iteration 900:
  `runs/stage2/h2h3_<arm>_s<seed>/checkpoints/iter_XXXXX.pt`. `last.pt` is refreshed every 25 iterations for resuming.
- It keeps going until all 6 runs (and their evaluations) are finished:
  - a run that crashes is restarted from its checkpoint, up to 3 times;
  - Ctrl-C, a power cut or a closed terminal loses at most 25 iterations per run: run the same command again and it
    resumes, skipping everything already finished;
  - on macOS it keeps the machine awake (`caffeinate`) while it runs.

## 1. One-time setup on a new laptop

macOS on Apple silicon (M1–M4) is the target: training runs on the GPU through MPS. Linux or Windows works on CPU,
but about 10× slower.

```bash
xcode-select --install                 # git and make, if missing (macOS)
brew install uv                        # or: curl -LsSf https://astral.sh/uv/install.sh | sh
git clone https://github.com/HeetRanpura/Genesis.git
cd Genesis
make setup                             # Python 3.11 venv in .venv + all dependencies (torch, tqdm, numba, ...)
make test                              # every test must pass before you start
```

Already have a clone, or the fork `advikdivekar/Genesis`? Bring it up to date instead of cloning:

```bash
git fetch upstream && git checkout main && git rebase upstream/main   # upstream = HeetRanpura/Genesis
make setup && make test
```

Check the GPU is visible (should print `True`):

```bash
.venv/bin/python -c "import torch; print(torch.backends.mps.is_available())"
```

## 2. Run it

Plug the laptop into power. Use a terminal you can leave open, ideally `tmux`, so closing the window doesn't stop
it (`brew install tmux`, then `tmux new -s h2h3`; detach with Ctrl-b d, reattach with `tmux attach -t h2h3`).

```bash
make h2h3                                          # = .venv/bin/python experiments/run_h2h3.py
```

Options (the defaults are the preregistered values; don't change seeds, iterations or worlds for the real run):

| flag | default | meaning |
|---|---|---|
| `--jobs` | 2 | runs at the same time. 2 suits an M3 Pro. Use 1 on a fanless MacBook Air |
| `--keep-every` | 200 | keep a `.pt` checkpoint every k iterations |
| `--skip-eval` | off | train only (evaluate later by running the command again without it) |
| `--retries` | 3 | automatic restarts per run after a crash |

**Disk:** about 12 GB in total (training recordings ~3 GB, checkpoints ~1.3 GB, evaluations ~7 GB). Start with
**25 GB free**. Training stops flight-recording below 15 GB free; the run itself continues.

**Time:** the M3 Pro trained 20M agent-steps in about 50 minutes, so one 3,000-iteration run takes roughly 4 h on
its own. With `--jobs 2` expect about 13–16 h for all six runs, plus about 1 h of evaluation. This is an estimate;
the overall bar shows the live ETA. The first start compiles the simulator (~30 s).

## 3. While it runs

```bash
tail -f runs/h2h3_logs/h2h3_gru_s1.log              # a job's full output (losses every 10 iterations)
tail -n 3 runs/stage2/h2h3_gru_s1/val.jsonl         # validation curve (worlds 500-501, every 50 iterations)
ls runs/stage2/h2h3_gru_s1/checkpoints/             # saved weights
```

- **To stop:** Ctrl-C. **To resume:** run `make h2h3` again.
- Don't edit code or hyperparameters between a stop and a resume: the six runs must all use the same code.
- Don't look at test worlds 0–4 before the script's own evaluation (no `make eval` on these runs midway).

## 4. When it finishes

The script prints the decision table and writes:
- `reports/stage2/h2h3_results.md` and `.json`: per-seed scores, means, Hedges' g, and H2/H3/H5 verdicts at iterations 900 and 3,000;
- `reports/stage2/eval_h2h3_<arm>_s<seed>_it<iter>.json`: one per evaluation (12 files);
- entries in `logs/AGENT_LOG.md`.

Commit those and open a PR to `HeetRanpura/Genesis` (no co-author trailers):

```bash
git checkout -b h2h3-results
git add reports/stage2/h2h3_results.md reports/stage2/h2h3_results.json reports/stage2/eval_h2h3_*.json logs/AGENT_LOG.md
git commit -m "H2/H3 results: 3 seeds x 2 arms x 3,000 iterations, held-out evaluation"
git push origin h2h3-results
```

The weights are not in git (`runs/` is ignored). Share the tested checkpoints and logs with Heet (about 170 MB):

```bash
tar czf h2h3_checkpoints.tgz runs/stage2/h2h3_*/checkpoints/iter_00900.pt runs/stage2/h2h3_*/checkpoints/iter_03000.pt \
    runs/stage2/h2h3_*/config.json runs/stage2/h2h3_*/train.jsonl runs/stage2/h2h3_*/val.jsonl
```

Report every outcome, including "not supported". If a run needed automatic restarts, mention it (the job's log
shows each attempt).

## 5. Troubleshooting

| message | fix |
|---|---|
| `docs/PREREGISTRATION.md has no H2/H3 protocol entry` | your checkout is old: `git pull` (or rebase on upstream) |
| `WARNING: no GPU backend found` | Python isn't the arm64 build: `.venv/bin/python -c "import platform; print(platform.machine())"` must print `arm64`; redo `make setup` with uv |
| `<job>: exited with code ...; restarting` | automatic; read `runs/h2h3_logs/<job>.log` if it repeats |
| `FAILED 4 times, giving up` | fix the cause shown at the end of that log, then run `make h2h3` again (it resumes) |
| disk warning | free space; the six final evaluations need ~1.1 GB each |
