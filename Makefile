PY ?= python3

.PHONY: all test stats figures examples smoke audit

all: test stats figures examples

test:
	$(PY) -m pytest -q

stats:
	$(PY) scripts/reproduce_summary_stats.py

figures:
	$(PY) scripts/make_figures.py

examples:
	$(PY) examples/compute_lambda_off_from_blocks.py
	$(PY) examples/conservative_field_control.py

smoke:
	$(PY) scripts/run_smoke.py

audit:
	$(PY) scripts/release_audit.py
