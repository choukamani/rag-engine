.DEFAULT_GOAL := help
# Container engine: `make up ENGINE=podman` works too.
ENGINE ?= docker
.PHONY: help install lint format typecheck test check eval eval-scifact bench up down docker-build

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-14s %s\n", $$1, $$2}'

install: ## Install dependencies (all extras) and git hooks
	uv sync --all-extras
	uv run pre-commit install

lint: ## Lint and check formatting
	uv run ruff check .
	uv run ruff format --check .

format: ## Auto-format code
	uv run ruff format .
	uv run ruff check --fix .

typecheck: ## Strict type checking
	uv run mypy

test: ## Unit + integration tests (set RAG_TEST_DATABASE_URL for the pgvector ones)
	uv run pytest

check: lint typecheck test ## Everything CI runs, except the Docker build

eval: ## Retrieval evaluation on the bundled Nimbus set (offline, seconds)
	uv run rag-eval --dataset nimbus --out evaluation/results/nimbus.json

eval-scifact: ## BEIR SciFact with BM25, hashing and bge-small (downloads data + model)
	uv run rag-eval --dataset scifact --embedders hashing,fastembed --out evaluation/results/scifact.json

bench: ## pgvector HNSW vs exact search (needs RAG_DATABASE_URL)
	uv run python benchmarks/hnsw_vs_exact.py --dataset scifact --embedder fastembed --out evaluation/results/hnsw-scifact-bge.json

docker-build: ## Build the API image
	$(ENGINE) build -t rag-engine:local .

up: ## Start PostgreSQL + pgvector and the API, then load the demo corpus
	$(ENGINE) build -t rag-engine:local .
	$(ENGINE) compose up -d --no-build
	@echo "API docs: http://localhost:$${API_PORT:-8000}/docs"

down: ## Stop the stack (add -v to drop the database volume)
	$(ENGINE) compose down
