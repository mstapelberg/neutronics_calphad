"""Tests for depletion composition building with impurities.

This module verifies that the depletion simulator composition builder
produces expected atomic compositions when including C, N, O impurities
via atomic fractions, weight fractions, and using a total CNO fraction.
"""

from __future__ import annotations

from typing import Dict

from neutronics_calphad.workflows.lightgbm_simulator import build_depletion_composition


def _almost_equal(a: float, b: float, tol: float = 1e-10) -> bool:
    """Return True if two floats are approximately equal within tolerance."""
    return abs(float(a) - float(b)) <= tol


def test_build_depletion_composition_no_impurities() -> None:
    """Main elements should renormalize to sum 1.0 when no impurities given."""
    comp = build_depletion_composition(cr=0.05, ti=0.04, w=0.03, zr=0.02)
    assert _almost_equal(sum(comp.values()), 1.0)
    assert set(comp.keys()) == {"V", "Cr", "Ti", "W", "Zr"}
    # Check that V balances 1 - sum(others)
    s = comp["V"] + comp["Cr"] + comp["Ti"] + comp["W"] + comp["Zr"]
    assert _almost_equal(s, 1.0)
    # Renormalization means exact V changes only if negative; here should match ratio
    assert comp["V"] > 0.0 and comp["Cr"] > 0.0


def test_build_depletion_composition_with_atomic_impurities() -> None:
    """Atomic impurities reduce main-element sum and are preserved exactly."""
    impurity_atomic: Dict[str, float] = {"C": 0.002, "N": 0.002, "O": 0.001}
    comp = build_depletion_composition(cr=0.05, ti=0.04, w=0.03, zr=0.02, impurity_atomic=impurity_atomic)
    assert _almost_equal(sum(comp.values()), 1.0)
    for k, v in impurity_atomic.items():
        assert _almost_equal(comp[k], v)
    # Main elements should fill the remainder
    imp_sum = sum(impurity_atomic.values())
    main_sum = sum(comp[k] for k in ["V", "Cr", "Ti", "W", "Zr"])
    assert _almost_equal(main_sum, 1.0 - imp_sum)


def test_build_depletion_composition_with_cno_frac_split() -> None:
    """Total CNO fraction splits equally among C, N, O when no explicit impurities."""
    comp = build_depletion_composition(cr=0.05, ti=0.04, w=0.03, zr=0.02, cno_frac=0.006)
    assert _almost_equal(sum(comp.values()), 1.0)
    assert _almost_equal(comp["C"], 0.006 / 3.0)
    assert _almost_equal(comp["N"], 0.006 / 3.0)
    assert _almost_equal(comp["O"], 0.006 / 3.0)


def test_build_depletion_composition_with_weight_impurities() -> None:
    """Weight impurities are converted to atomic fractions and normalized properly."""
    # Provide small wt% values; behavior: each is converted using atomic weights,
    # base alloy average weight is computed from the raw composition.
    imp_wt_percent = {"C": 0.006, "N": 0.012, "O": 0.015}  # wt%
    imp_wtfrac = {k: v / 100.0 for k, v in imp_wt_percent.items()}  # convert to fraction
    comp = build_depletion_composition(cr=0.05, ti=0.04, w=0.03, zr=0.02, impurity_wtfrac=imp_wtfrac)
    assert _almost_equal(sum(comp.values()), 1.0)
    # Sanity: impurities present and small
    assert comp["C"] > 0.0 and comp["N"] > 0.0 and comp["O"] > 0.0
    # Main elements fill the rest
    main_sum = sum(comp[k] for k in ["V", "Cr", "Ti", "W", "Zr"])
    assert 0.9 <= main_sum <= 1.0


