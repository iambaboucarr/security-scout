# Security Scout — task runner (ADR-025). Run from repository root so `.env` and `repos.yaml` resolve.
# Requires: uv, Docker (for Redis via `make services`).

.PHONY: help install format lint typecheck test testcov testslow check all clean \
	services services-down run worker migrate db-upgrade

.DEFAULT_GOAL := help

# Resolve imports for the src/ layout (matches pytest pythonpath in pyproject.toml).
export PYTHONPATH := src

UVICORN_HOST ?= 127.0.0.1
UVICORN_PORT ?= 8000

help:
	@echo "Security Scout — common targets"
	@echo ""
	@echo "  make install          Sync deps (uv) and install pre-commit hooks"
	@echo "  make services         Start Redis (docker compose up -d)"
	@echo "  make services-down    Stop compose services"
	@echo "  make run              FastAPI: uvicorn with --reload ($(UVICORN_HOST):$(UVICORN_PORT))"
	@echo "  make worker           ARQ worker (process_advisory_workflow_job)"
	@echo "  make migrate          Alembic: upgrade database to head (set DATABASE_URL)"
	@echo "  make check            lint + typecheck + coverage tests"
	@echo ""
	@echo "Typical local run (two terminals):"
	@echo "  1. make services && make run"
	@echo "  2. make worker"

install:
	uv sync --dev
	uv run pre-commit install

format:
	uv run ruff format src/ tests/
	uv run ruff check --fix src/ tests/

lint:
	uv run ruff check src/ tests/
	uv run ruff format --check src/ tests/

typecheck:
	uv run mypy src/

test:
	uv run pytest -x -n auto

testcov:
	uv run pytest -x -n auto --cov=src --cov-report=term-missing

testslow:
	uv run pytest -m slow

check: lint typecheck testcov

all: format check

clean:
	rm -rf .mypy_cache .pytest_cache .ruff_cache htmlcov .coverage coverage.xml dist/
	find . -type d -name __pycache__ -exec rm -rf {} +

services:
	docker compose up -d redis

services-down:
	docker compose down

run:
	uv run uvicorn main:app --reload --host $(UVICORN_HOST) --port $(UVICORN_PORT)

worker:
	uv run arq worker.WorkerSettings

migrate: db-upgrade

db-upgrade:
	uv run alembic upgrade head
