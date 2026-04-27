.PHONY: help install format lint typecheck test testcov testslow testintegration check all clean \
	services services-down run worker migrate db-upgrade

.DEFAULT_GOAL := help

# Resolve imports for the src/ layout (matches pytest pythonpath in pyproject.toml).
export PYTHONPATH := src

UVICORN_HOST ?= 127.0.0.1
UVICORN_PORT ?= 8000

POSTGRES_TEST_URL ?= postgresql+asyncpg://postgres:postgres@127.0.0.1:5432/postgres
export POSTGRES_TEST_URL

help:
	@echo "Security Scout — common targets"
	@echo ""
	@echo "  make install          Sync deps (uv) and install pre-commit hooks"
	@echo "  make services         Start Redis + Postgres (docker compose up -d)"
	@echo "  make services-down    Stop compose services"
	@echo "  make run              FastAPI CLI (dev + reload) via src/main.py ($(UVICORN_HOST):$(UVICORN_PORT))"
	@echo "  make worker           ARQ worker (process_advisory_workflow_job)"
	@echo "  make migrate          Alembic: upgrade database to head (set DATABASE_URL)"
	@echo "  make check            lint + typecheck + coverage (SQLite + Postgres suites)"
	@echo "  make testpostgres     pytest -m postgres only (needs POSTGRES_TEST_URL / compose postgres)"
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
	uv run pytest -x -n auto -m "not postgres"

testcov:
	uv run pytest -x -n auto -m "not postgres" --cov=src --cov-report=term-missing
	uv run pytest -x -n auto -m postgres --cov=src --cov-append --cov-report=term-missing

testpostgres:
	uv run pytest -x -n auto -m postgres

testslow:
	uv run pytest -m slow

testintegration:
	uv run pytest -m integration

check: lint typecheck testcov

all: format check

clean:
	rm -rf .mypy_cache .pytest_cache .ruff_cache htmlcov .coverage coverage.xml dist/
	find . -type d -name __pycache__ -exec rm -rf {} +

services:
	docker compose up -d redis postgres

services-down:
	docker compose down

run:
	uv run fastapi dev src/main.py --host $(UVICORN_HOST) --port $(UVICORN_PORT)

worker:
	uv run python src/run_worker.py

migrate: db-upgrade

db-upgrade:
	uv run alembic upgrade head
