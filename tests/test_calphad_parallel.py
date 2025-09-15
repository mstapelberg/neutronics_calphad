"""Tests for the parallel CALPHAD runner."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest

if TYPE_CHECKING:  # pragma: no cover - typing-only imports for IDEs/static checkers
    from _pytest.capture import CaptureFixture  # noqa: F401
    from _pytest.fixtures import FixtureRequest  # noqa: F401
    from _pytest.logging import LogCaptureFixture  # noqa: F401
    from _pytest.monkeypatch import MonkeyPatch  # noqa: F401
    from pytest_mock.plugin import MockerFixture  # noqa: F401

from neutronics_calphad.workflows.calphad_runner import (
    run_calphad_batch,
    run_calphad_batch_parallel,
    ELEMENTS_ORDER,
)


def _random_simplex(n: int, d: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = rng.random((n, d))
    return x / x.sum(axis=1, keepdims=True)


@pytest.mark.parametrize("n_rows", [0, 1, 5])
def test_parallel_matches_schema(n_rows: int) -> None:
    """Parallel runner returns expected columns and row order.

    This test checks only structural properties and does not require
    Thermo-Calc to be available. If Thermo-Calc is missing, the
    implementation falls back to the sequential stub.
    """
    if n_rows == 0:
        arr = np.empty((0, len(ELEMENTS_ORDER)))
    else:
        arr = _random_simplex(n_rows, len(ELEMENTS_ORDER), seed=123)

    df = pd.DataFrame(arr, columns=list(ELEMENTS_ORDER))

    # Run both paths
    seq = run_calphad_batch(df, temperature_k=873.15, database="TCHEA8")
    par = run_calphad_batch_parallel(df, temperature_k=873.15, database="TCHEA8", num_workers=2)

    # Basic shape checks
    assert par.shape[0] == df.shape[0]
    for el in ELEMENTS_ORDER:
        assert f"x_{el}" in par.columns
    for col in ["phase_count", "dominant_phase", "single_phase", "phases"]:
        assert col in par.columns

    # Composition values should match inputs in x_ columns (subject to normalization)
    # Verify row order preserved
    if n_rows > 0:
        np.testing.assert_allclose(par[[f"x_{el}" for el in ELEMENTS_ORDER]].to_numpy(), df.to_numpy(), rtol=0, atol=1e-12)

    # Columns should be identical to sequential API
    assert set(seq.columns) == set(par.columns)


