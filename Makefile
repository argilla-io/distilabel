sources = src/distilabel tests

.PHONY: format
format:
	ruff check --fix $(sources)
	ruff format $(sources)

.PHONY: lint
lint:
	ruff check $(sources)
	ruff format --check $(sources)

.PHONY: unit-tests
unit-tests:
	pytest tests/unit

.PHONY: integration-tests
integration-tests:
	pytest tests/integration
