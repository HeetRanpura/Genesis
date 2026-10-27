PY := .venv/bin/python

.PHONY: setup test bench baselines evidence stage1 viewer-setup viewer train eval h2h3

setup:            ## create .venv (Python 3.11) and install
	uv venv --python 3.11 .venv
	uv pip install --python $(PY) -e ".[dev,viewer,learn]"
	cd viewer && npm install --no-audit --no-fund

test:
	$(PY) -m pytest -q

bench:            ## headless throughput table -> reports/stage1/bench.json
	$(PY) experiments/bench_sim.py

baselines:        ## random / greedy / heuristic, 5 seeds x 2 years -> reports/stage1/baselines.json
	$(PY) experiments/run_baselines.py --seeds 5 --years 2

evidence:         ## Stage 1 figures -> reports/stage1/*.png
	$(PY) experiments/stage1_evidence.py

stage1: test baselines evidence bench

viewer-setup:     ## three.js for the live viewer (pinned in viewer/package.json)
	uv pip install --python $(PY) -e ".[viewer]"
	cd viewer && npm install --no-audit --no-fund

viewer:           ## live viewer at http://localhost:8787 (SEED=, POLICY=, AGENTS=, PORT=)
	$(PY) -m viewer.server --seed $(or $(SEED),0) --policy $(or $(POLICY),heuristic) --port $(or $(PORT),8787) $(if $(AGENTS),--agents $(AGENTS),)

train:            ## train the learner, agents/ppo.py (NAME=gru_s0 ITERS=600 SEED=0; MEMORYLESS=1 for the ablation)
	$(PY) experiments/train_ppo.py --name $(or $(NAME),gru_s0) --iters $(or $(ITERS),600) --seed $(or $(SEED),0) $(if $(MEMORYLESS),--no-memory,)

eval:             ## held-out evaluation of a trained learner, same protocol as the baselines (NAME=gru_s0)
	$(PY) experiments/eval_policy.py runs/stage2/$(or $(NAME),gru_s0)/checkpoints/last.pt --name $(or $(NAME),gru_s0)

h2h3:             ## preregistered H2/H3 experiment: 6 runs x 3,000 iterations + evaluation, resumable (docs/H2H3_RUNBOOK.md)
	$(PY) experiments/run_h2h3.py $(ARGS)
