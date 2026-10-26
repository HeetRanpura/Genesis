PY := .venv/bin/python

.PHONY: setup test bench baselines evidence stage1

setup:            ## create .venv (Python 3.11) and install
	uv venv --python 3.11 .venv
	uv pip install --python $(PY) -e ".[dev]"

test:
	$(PY) -m pytest -q

bench:            ## headless throughput table -> reports/stage1/bench.json
	$(PY) experiments/bench_sim.py

baselines:        ## random / greedy / heuristic, 5 seeds x 2 years -> reports/stage1/baselines.json
	$(PY) experiments/run_baselines.py --seeds 5 --years 2

evidence:         ## Stage 1 figures -> reports/stage1/*.png
	$(PY) experiments/stage1_evidence.py

stage1: test baselines evidence bench
