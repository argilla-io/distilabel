sources = src/distilabel tests

.PHONY: format
format:
	ruff --fix $(sources)
	ruff format $(sources)

.PHONY: lint
lint:
	ruff $(sources)
	ruff format --check $(sources)

.PHONY: test
test:
	pytest
