"""Filters for activation and ductility constraints.

This module provides compositional filters using linear mixing of
pure-element neutronics results and a ductility heuristic cap on total
alloying content.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Tuple

import numpy as np  # type: ignore
import pandas as pd  # type: ignore


GasKey = str  # e.g. "He_appm", "H_appm"
CoolingDays = int


@dataclass(frozen=True)
class ActivationLimits:
    """Activation limits for gas production and dose.

    Attributes:
        gas_appm: Limit per gas species in appm, e.g. {"He_appm": 586.1, "H_appm": 1200}.
        dose_at_days: Dose-rate limits in Sv/h at specific cooling days, e.g. {30: 1e3, 365: 1}.
    """

    gas_appm: Mapping[GasKey, float]
    dose_at_days: Mapping[CoolingDays, float]


def make_activation_filter(
    pure_results: Mapping[str, Mapping[str, object]],
    limits: ActivationLimits,
) -> Callable[[Mapping[str, float]], bool]:
    """Create a filter that checks linear-mixed activation vs limits.

    The linear model combines pure-element properties with atomic-fraction
    weights. For gas: sum_i x_i * gas_i. For dose at day d: sum_i x_i * dose_i(d).

    Args:
        pure_results: Mapping from element symbol to its results dict, expected keys:
            - "gas_production": {"He_appm": float, "H_appm": float, ...}
            - "dose_at_cooling_times": {days: Sv/h, ...}
        limits: ActivationLimits containing allowed maxima.

    Returns:
        Callable that returns True if composition satisfies all activation limits.
    """

    # Pre-extract arrays for fast evaluation
    elements: List[str] = list(pure_results.keys())
    gases: List[GasKey] = list(limits.gas_appm.keys())
    dose_days: List[CoolingDays] = list(limits.dose_at_days.keys())

    gas_matrix = np.zeros((len(gases), len(elements)))
    for g_idx, gas in enumerate(gases):
        for e_idx, el in enumerate(elements):
            gas_val = float(pure_results[el].get("gas_production", {}).get(gas, 0.0))
            gas_matrix[g_idx, e_idx] = gas_val

    dose_matrix = np.zeros((len(dose_days), len(elements)))
    for d_idx, day in enumerate(dose_days):
        for e_idx, el in enumerate(elements):
            dose_val = float(pure_results[el].get("dose_at_cooling_times", {}).get(day, 0.0))
            dose_matrix[d_idx, e_idx] = dose_val

    gas_limits = np.array([limits.gas_appm[g] for g in gases], dtype=float)
    dose_limits = np.array([limits.dose_at_days[d] for d in dose_days], dtype=float)

    element_index: Dict[str, int] = {el: i for i, el in enumerate(elements)}

    def _check(comp: Mapping[str, float]) -> bool:
        x = np.zeros(len(elements), dtype=float)
        for el, frac in comp.items():
            idx = element_index.get(el)
            if idx is not None:
                x[idx] = float(frac)
        # Gas constraints
        gas_vals = gas_matrix @ x
        if np.any(gas_vals > gas_limits + 1e-12):
            return False
        # Dose constraints
        dose_vals = dose_matrix @ x
        if np.any(dose_vals > dose_limits + 1e-12):
            return False
        return True

    return _check


def make_ductility_filter(
    max_total_alloy: float,
    base_element: str = "V",
) -> Callable[[Mapping[str, float]], bool]:
    """Create a filter for ductility heuristic based on total alloy fraction.

    Args:
        max_total_alloy: Maximum allowed sum of non-base elements (e.g., 0.2).
        base_element: Base element symbol considered ductile matrix (default: "V").

    Returns:
        Callable returning True if sum(1 - x_base) <= max_total_alloy.
    """

    if max_total_alloy < 0.0 or max_total_alloy > 1.0:
        raise ValueError("max_total_alloy must be within [0, 1]")

    def _check(comp: Mapping[str, float]) -> bool:
        x_base = float(comp.get(base_element, 0.0))
        total_alloy = 1.0 - x_base
        return total_alloy <= max_total_alloy + 1e-12

    return _check


def apply_filters(
    compositions: pd.DataFrame,
    filters: Iterable[Callable[[Mapping[str, float]], bool]],
) -> pd.Series:
    """Apply a list of filters to compositions and return boolean mask.

    Args:
        compositions: DataFrame with columns subset of ["V","Cr","Ti","W","Zr"].
        filters: Iterable of callables that accept a mapping of element->fraction.

    Returns:
        Boolean Series aligned with `compositions` index, True if all filters pass.
    """
    mask = []
    for _, row in compositions.iterrows():
        comp_map = row.to_dict()
        ok = True
        for flt in filters:
            if not flt(comp_map):
                ok = False
                break
        mask.append(ok)
    return pd.Series(mask, index=compositions.index)


