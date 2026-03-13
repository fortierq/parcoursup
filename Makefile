.PHONY: notebook install

install:
	uv sync
	uv pip install -e .

notebook: install
	uv run marimo edit nb/nb_parcoursup.py