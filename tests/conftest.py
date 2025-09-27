"""Pytest configuration and fixtures for Graphem tests."""

import pytest
import numpy as np
import warnings


def pytest_configure(config):
    """Configure pytest settings."""
    # Register custom markers
    config.addinivalue_line("markers", "fast: Fast tests suitable for CI")
    config.addinivalue_line("markers", "slow: Slow tests for comprehensive testing")
    config.addinivalue_line("markers", "integration: Integration tests")
    config.addinivalue_line("markers", "gpu: Tests requiring GPU/CUDA")

    # Filter out common warnings that might clutter test output
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=PendingDeprecationWarning)
    warnings.filterwarnings("ignore", message=".*jax.*")


@pytest.fixture
def small_graph_edges():
    """Fixture providing edges for a small test graph."""
    return np.array([
        [0, 1], [1, 2], [2, 3], [3, 0],  # Square
        [0, 2], [1, 3]  # Diagonals (complete graph K4)
    ])


@pytest.fixture
def random_seed():
    """Fixture providing a consistent random seed for reproducible tests."""
    return 42