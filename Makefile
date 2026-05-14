.PHONY: help install install-dev lint format type test test-fast cov clean download-data generate-data data-quality build-notebook notebook-server train train-baseline mlflow-up

help: ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# ---------- Setup ----------
install: ## Install runtime dependencies
	uv sync --no-dev

install-dev: ## Install all dev dependencies + pre-commit
	uv sync --extra dev
	uv run pre-commit install

# ---------- Quality ----------
lint: ## Lint with ruff
	uv run ruff check src tests

format: ## Format with ruff
	uv run ruff format src tests
	uv run ruff check --fix src tests

type: ## Type-check with mypy
	uv run mypy src

test: ## Run all tests
	uv run pytest

test-fast: ## Run only fast tests
	uv run pytest -m "not slow"

cov: ## Run tests with HTML coverage report
	uv run pytest --cov-report=html
	@echo "Open htmlcov/index.html"

# ---------- Data ----------
download-data: ## Download real PaySim from Kaggle
	uv run python scripts/download_data.py

generate-data: ## Generate synthetic PaySim-shaped demo data
	uv run python scripts/generate_synthetic_paysim.py --rows 500000

data-quality: ## Run data quality checks
	uv run python scripts/run_data_quality.py

# ---------- Notebook ----------
build-notebook: ## Rebuild the analysis notebook with executed outputs
	uv run python scripts/build_notebook.py

notebook-server: ## Start Jupyter to interact with the notebook
	uv run jupyter notebook notebooks/

# ---------- Training ----------
train: ## Run the Prefect training pipeline
	uv run python scripts/train.py

train-baseline: ## Train just the baseline logistic regression
	uv run python -m fraud_detection.models.baseline

# ---------- MLflow ----------
mlflow-up: ## Start standalone MLflow tracking server
	uv run mlflow server --host 0.0.0.0 --port 5000

# ---------- Cleanup ----------
clean: ## Remove caches and build artifacts
	rm -rf .pytest_cache .ruff_cache .mypy_cache htmlcov coverage.xml .coverage
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name "*.egg-info" -exec rm -rf {} +
