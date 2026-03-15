.PHONY: notebook install lint

install:
	uv sync
	uv pip install -e .

lint:
	uv run ruff check .

notebook: install
	uv run marimo edit tests/mpi.py