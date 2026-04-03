.PHONY: format lint

format:
	uv run ruff check . --fix
	uv run black .

lint:
	uv run ruff check .
	uv run black --check .
