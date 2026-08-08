# Local equivalents of the CI jobs. `make check` is what CI runs, so a green
# `make check` means a green pipeline — run it before pushing.
#
# The repo needs Python >= 3.10 (`X | None` annotations). macOS system Python is
# 3.9 and cannot even import src/, hence the venv.

PYTHON ?= .venv/bin/python
RUFF   ?= .venv/bin/ruff
VENV_PYTHON ?= python3.11

.DEFAULT_GOAL := help
.PHONY: help venv install test lint fmt health health-strict guards check clean

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

venv:  ## Create the local virtualenv
	$(VENV_PYTHON) -m venv .venv
	$(PYTHON) -m pip install --upgrade pip

install: venv  ## Install CPU-only dependencies for tests and analysis
	$(PYTHON) -m pip install torch --index-url https://download.pytorch.org/whl/cpu
	$(PYTHON) -m pip install pytest pyyaml numpy scipy transformers datasets ruff

test:  ## Run the test suite
	$(PYTHON) -m pytest tests/

lint:  ## Lint (correctness rules only) and byte-compile everything
	$(RUFF) check .
	$(PYTHON) -m compileall -q src scripts tests analyze.py smoke_test.py

fmt:  ## Apply the safe lint autofixes
	$(RUFF) check . --fix

health:  ## Repo integrity: contract, hash pins, cohorts, model pins, policy
	$(PYTHON) scripts/healthcheck.py

health-strict:  ## Health check with warnings escalated to failures
	$(PYTHON) scripts/healthcheck.py --strict

guards:  ## Just the SPEC 14 BUG-1/2/3 multi-hop regression guards
	$(PYTHON) -m pytest tests/test_retrieval.py::MultiHopRegressionGuards -v

check: lint test health  ## Everything CI runs
	@echo
	@echo "All CI checks passed locally."

clean:  ## Remove caches and build junk
	find . -type d -name __pycache__ -not -path "./.venv/*" -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache .ruff_cache dist
