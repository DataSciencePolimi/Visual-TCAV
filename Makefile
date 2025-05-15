SRC_DIR = src
CHECK_DIRS= $(SRC_DIR)
.PHONY: format
CONFIG ?= configs/config_minimal.yaml

format: ## Format repository code
	poetry run black $(CHECK_DIRS)
	poetry run isort $(CHECK_DIRS)

.PHONY: install
install: ## Install the dependencies from the lock file
	poetry install -v

.PHONY: run
run: ## Run the visual tcav analysis
	poetry run python -m src.visual_tcav.main $(CONFIG)

.PHONY: help
help: ## Show the available commands
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'
