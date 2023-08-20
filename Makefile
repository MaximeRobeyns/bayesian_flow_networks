.PHONY: kernel lab docs install help test mypy run board

lab: ## To start a Jupyter Lab server
	jupyter lab --notebook-dir=notebooks --ip=0.0.0.0 --port 8888

docs:  ## For writing documentation
	@./docs/writedocs.sh

mypy:  ## Run MyPy static type checking
	@mypy

test:  ## Run the unit tests
	@python -m pytest -s tests -m "not slow"

alltest:  ## Run all the tests (including the slow ones)
	@python -m pytest -s tests

help:
	@grep -E '^[a-zA-Z0-9_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'
