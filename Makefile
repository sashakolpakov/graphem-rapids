# GraphEm Rapids Test and Development Makefile

.PHONY: help test test-fast test-comprehensive test-gpu clean lint install dev-install

help:
	@echo "Available targets:"
	@echo "  test-fast           Run fast CI tests only"
	@echo "  test-comprehensive  Run comprehensive test suite"
	@echo "  test-gpu           Run GPU tests (requires CUDA)"
	@echo "  test               Run fast tests (alias for test-fast)"
	@echo "  lint               Run pylint on source code"
	@echo "  clean              Clean build artifacts and cache"
	@echo "  install            Install package"
	@echo "  dev-install        Install in development mode"

# Fast tests for CI/quick validation
test-fast:
	pytest -m "fast and not slow and not gpu" -v --maxfail=3

test: test-fast

# Comprehensive test suite (manual execution)
test-comprehensive:
	python scripts/run_comprehensive_tests.py --parallel --coverage

# GPU tests only
test-gpu:
	pytest -m "gpu" -v

# Comprehensive tests including GPU
test-all:
	python scripts/run_comprehensive_tests.py --include-gpu --parallel --coverage

# Code quality
lint:
	python -m pylint graphem_rapids/ --exit-zero

# Cleanup
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf htmlcov/
	rm -f coverage.xml
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

# Installation
install:
	pip install .

dev-install:
	pip install -e .