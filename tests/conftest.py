"""Shared pytest fixtures and helpers for the CurbIQ test suite.

Tests use small synthetic inputs and never rebuild artifacts. A couple of fixtures
build a tiny known H3 lattice so the spatial-statistics assertions are exact and
do not depend on the full Bengaluru dataset.
"""
from __future__ import annotations

import sys
from pathlib import Path

import h3
import numpy as np
import pytest

# Make the repo importable regardless of how pytest is invoked.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


CENTER_LATLNG = (12.9716, 77.5946)   # Bengaluru city centre


@pytest.fixture(scope="session")
def hex_lattice():
    """A compact, deterministic res-9 H3 lattice (a grid disk of radius 3).

    Returns ``(cells, idx)`` where ``cells`` is the sorted cell list and ``idx``
    maps each cell id to its row index — the ordering used by the weight builders.
    """
    center = h3.latlng_to_cell(*CENTER_LATLNG, 9)
    cells = sorted(h3.grid_disk(center, 3))
    idx = {c: i for i, c in enumerate(cells)}
    return cells, idx


@pytest.fixture(scope="session")
def lattice_center():
    return h3.latlng_to_cell(*CENTER_LATLNG, 9)


@pytest.fixture
def clustered_field(hex_lattice, lattice_center):
    """High values on the centre + its 6 immediate neighbours, low elsewhere."""
    cells, idx = hex_lattice
    hot = {lattice_center} | set(h3.grid_disk(lattice_center, 1))
    return np.array([10.0 if c in hot else 1.0 for c in cells])


@pytest.fixture
def uniform_field(hex_lattice):
    cells, _ = hex_lattice
    return np.full(len(cells), 5.0)
