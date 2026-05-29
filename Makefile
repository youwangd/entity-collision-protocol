# Engram developer make targets.
#
# Default `test` runs the fast tier (~140 s). Slow-tier coverage is
# preserved via `test-slow`, intended for nightly / pre-merge runs.

.PHONY: test test-slow test-all lint

PYTEST ?= python -m pytest

test:
	$(PYTEST) -q --tb=line

test-slow:
	$(PYTEST) -q --tb=line -m slow

test-all:
	$(PYTEST) -q --tb=line -m "slow or not slow"

lint:
	ruff check src tests evals
