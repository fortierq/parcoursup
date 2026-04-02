.PHONY: notebook install lint

NB = tests/nb_mpi.py

install:
	uv sync
	uv pip install -e .

lint:
	uv run ruff check .

nb:
	uv run marimo edit $(NB)

html:
	uv run marimo export html-wasm $(NB) -o html2 --mode run --no-show-code
