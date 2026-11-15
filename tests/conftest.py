import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@pytest.fixture(scope="session")
def small_data():
    """Small clean synthetic dataset shared by tests."""
    from data.generate_data import generate_all
    return generate_all(n_medicines=8, history_days=300, seed=7)
