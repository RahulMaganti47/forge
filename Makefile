.PHONY: test verify lint reproduce
test:
	uv run pytest
verify:
	uv run forge artifacts verify --group paper-model-v1
	uv run forge artifacts verify --group submission19337-evidence-v1
lint:
	uv run ruff check forge examples tests
	uv run black --check forge examples tests
reproduce:
	uv run forge reproduce --target all --output results/reproduced-tables
