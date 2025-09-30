SOURCE_DIRS = src/async_collab/ src/data_preparation/ src/evaluation/ src/experimentation/
TEST_DIRS = tests
SOURCE_AND_TEST_DIRS = $(SOURCE_DIRS) $(TEST_DIRS)

.PHONY: format lint-fix fix format-check lint pyright test backend frontend lint-frontend lint-fix-frontend test-ci-frontend help simulate-conv

all: format-check lint pyright test ## Run all checks, lints and tests

format: ## Format the code
	ruff -e --fix-only --select I001 $(SOURCE_AND_TEST_DIRS)
	black $(SOURCE_AND_TEST_DIRS)

lint-fix: ## Fix lints
	ruff -e --fix-only $(SOURCE_AND_TEST_DIRS)

fix: lint-fix ## Fix formatting and lints
	black $(SOURCE_AND_TEST_DIRS)

format-check: ## Check code formatting
	@(ruff --select I001 $(SOURCE_AND_TEST_DIRS)) && (black --check $(SOURCE_AND_TEST_DIRS)) || (echo "run \"make format\" to format the code"; exit 1)

lint: ## Lint the codebase
	@(ruff $(SOURCE_AND_TEST_DIRS)) || (echo "run \"make lint-fix\" to fix some lint errors automatically"; exit 1)

pyright: ## Run pyright
	pyright

test: ## Run tests
	python -m pytest $(TEST_DIRS)

AGENT_CONF := src/async_collab/scenarios/people_join_qa/agent_configs/spider_sample.json
backend: ## Start web server (use after `poetry shell`)
	echo "Using conf file: '$(AGENT_CONF)'"
	hypercorn "async_collab.api:app('$(AGENT_CONF)')"

# https://marmelab.com/blog/2016/02/29/auto-documented-makefile.html
help: ## Print this help text.
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	sort | \
	awk 'BEGIN {FS = ":.*?## "}; \
	{printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'
