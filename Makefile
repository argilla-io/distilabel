sources = src/distilabel tests

.PHONY: format
format:
	ruff --fix $(sources)
	ruff format $(sources)

.PHONY: lint
lint:
	ruff $(sources)
	ruff format --check $(sources)

.PHONY: unit-tests
unit-tests:
	pytest tests/unit

.PHONY: integration-tests
integration-tests:
	pytest tests/integration
