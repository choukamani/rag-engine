# Contributing

Thanks for your interest! This is a demonstration project, but issues and pull requests are welcome.

## Setup

```bash
uv sync --all-extras       # Python 3.12, dev dependencies, pgvector + fastembed extras
uv run pre-commit install  # ruff + mypy on every commit
```

## Before opening a pull request

```bash
make check   # ruff, mypy --strict, pytest with coverage >= 80 %
make eval    # retrieval metrics must not regress on the Nimbus set
```

- Keep the public API typed; `mypy --strict` must pass.
- Add tests for every behaviour change. Integration tests live in `tests/integration`.
- Use [Conventional Commits](https://www.conventionalcommits.org/) (`feat:`, `fix:`, `docs:`, `test:`, `refactor:`, `ci:`, `chore:`).
- Significant design changes come with an ADR in `docs/adr/`.

## Adding a provider

1. Implement the port from `rag_engine/ports` in a new module under `rag_engine/adapters/`.
2. Register a factory in `rag_engine/container.py` and add the name to the matching `Literal` in `config.py`.
3. Add unit tests (HTTP providers: mock with `respx`) and, for storage adapters, an integration test.
4. Run `make eval` and report the metrics in the pull request.

To run the PostgreSQL tests locally, start the database with `docker compose up -d postgres` and set `RAG_TEST_DATABASE_URL`.
