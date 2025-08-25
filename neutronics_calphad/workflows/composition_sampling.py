"""Composition sampling utilities under constraints.

This module provides functions to sample alloy compositions for the
`V-Cr-Ti-W-Zr` system subject to per-element maximums and a total
alloying fraction cap (ductility heuristic), ensuring atomic fractions
sum to one with `V` as the balance element.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np  # type: ignore
import pandas as pd  # type: ignore


ALLOY_ELEMENTS: Tuple[str, ...] = ("Cr", "Ti", "W", "Zr")
BASE_ELEMENT: str = "V"


@dataclass(frozen=True)
class SamplingConstraints:
    """Constraints for composition sampling.

    Attributes:
        per_element_max: Maximum atomic fraction for each alloying element.
            Keys must be a subset of {"Cr", "Ti", "W", "Zr"}.
        total_alloy_max: Maximum total alloying fraction (sum of Cr+Ti+W+Zr).
        base_element: Base element symbol, defaults to "V".
        alloy_elements: Ordered tuple of alloying element symbols.
    """

    per_element_max: Dict[str, float]
    total_alloy_max: float
    base_element: str = BASE_ELEMENT
    alloy_elements: Tuple[str, ...] = ALLOY_ELEMENTS


def _validate_constraints(constraints: SamplingConstraints) -> None:
    """Validate constraints values.

    Raises:
        ValueError: If inputs are inconsistent or outside [0, 1].
    """
    if constraints.total_alloy_max < 0.0 or constraints.total_alloy_max > 1.0:
        raise ValueError("total_alloy_max must be within [0, 1]")
    for el in constraints.alloy_elements:
        max_val = constraints.per_element_max.get(el, 1.0)
        if max_val < 0.0 or max_val > 1.0:
            raise ValueError(f"per_element_max[{el}] must be within [0, 1]")
    if constraints.total_alloy_max > 1.0 - 1e-12:
        raise ValueError("total_alloy_max leaves no room for base element")


def sample_compositions(
    constraints: SamplingConstraints,
    n_samples: int = 1000,
    alpha: float = 1.0,
    random_state: Optional[int] = None,
) -> pd.DataFrame:
    """Sample compositions satisfying per-element and total alloying limits.

    The sampler draws Dirichlet-distributed proportions for alloying elements,
    scales them by a random total alloying amount up to `total_alloy_max`, and
    rejects samples that violate any per-element caps. This typically achieves
    high acceptance rates if caps are not extremely tight or contradictory.

    Args:
        constraints: SamplingConstraints instance.
        n_samples: Number of valid compositions to return.
        alpha: Dirichlet concentration parameter for alloying proportions.
            Use alpha > 1 for more uniform mixtures, alpha < 1 for sparse.
        random_state: Optional seed for reproducibility.

    Returns:
        DataFrame with columns ["V", "Cr", "Ti", "W", "Zr"], length n_samples.
    """
    _validate_constraints(constraints)
    rng = np.random.default_rng(random_state)

    elements: List[str] = [constraints.base_element, *constraints.alloy_elements]
    per_cap = np.array([constraints.per_element_max.get(el, 1.0) for el in constraints.alloy_elements])
    total_cap = float(constraints.total_alloy_max)

    rows: List[np.ndarray] = []
    max_attempts = max(10_000, 20 * n_samples)
    attempts = 0

    # Vectorized acceptance within a loop for reliability
    while len(rows) < n_samples and attempts < max_attempts:
        batch = min(4096, n_samples - len(rows))
        # proportions among alloying elements
        props = rng.dirichlet(alpha=np.full(len(constraints.alloy_elements), alpha), size=batch)
        # total alloy fraction per sample
        totals = rng.uniform(low=0.0, high=total_cap, size=batch)
        alloys = props * totals[:, None]

        # per-element caps
        ok = (alloys <= per_cap[None, :] + 1e-12).all(axis=1)
        accepted = alloys[ok]

        if accepted.size == 0:
            attempts += batch
            continue

        base = 1.0 - accepted.sum(axis=1)
        # ensure base remains non-negative
        ok_base = base >= -1e-12
        accepted = accepted[ok_base]
        base = base[ok_base]
        # clip small numerical negatives to zero
        base = np.clip(base, 0.0, 1.0)

        for i in range(accepted.shape[0]):
            row = np.zeros(len(elements))
            row[0] = base[i]
            row[1:] = accepted[i]
            rows.append(row)

        attempts += batch

    if len(rows) < n_samples:
        raise RuntimeError(
            f"Failed to sample requested {n_samples} compositions within attempts={attempts}. "
            f"Consider relaxing caps or increasing attempts."
        )

    data = np.vstack(rows[:n_samples])
    df = pd.DataFrame(data, columns=elements)
    # robust normalization to address any epsilon drift
    total = df.sum(axis=1)
    df = df.div(total, axis=0)
    return df


def make_ternary_grid(
    fixed_element: str,
    fixed_value: float,
    varying_elements: Tuple[str, str, str],
    resolution: int = 51,
) -> pd.DataFrame:
    """Create a ternary grid slice at a fixed element fraction.

    Args:
        fixed_element: Element symbol fixed at `fixed_value`.
        fixed_value: Atomic fraction fixed for `fixed_element` within [0, 1].
        varying_elements: Three-element tuple forming the ternary plane.
        resolution: Number of grid points per axis.

    Returns:
        DataFrame of compositions spanning the ternary plane with the fixed
        element adjusted to keep the sum of fractions equal to one.
    """
    if fixed_value < 0.0 or fixed_value > 1.0:
        raise ValueError("fixed_value must be within [0, 1]")
    if len(set(varying_elements)) != 3:
        raise ValueError("varying_elements must be three distinct symbols")

    grid_rows: List[Dict[str, float]] = []
    # simple integer grid over two independent axes; third is balance on ternary
    for i in range(resolution):
        for j in range(resolution - i):
            a = i / (resolution - 1)
            b = j / (resolution - 1)
            # ensure a + b <= 1 in the ternary simplex
            if a + b > 1.0 + 1e-12:
                continue
            c = 1.0 - a - b
            # scale ternary slice to (1 - fixed_value)
            scale = max(0.0, 1.0 - fixed_value)
            va = a * scale
            vb = b * scale
            vc = c * scale

            row_map: Dict[str, float] = {"V": 0.0, "Cr": 0.0, "Ti": 0.0, "W": 0.0, "Zr": 0.0}
            row_map[fixed_element] = fixed_value
            row_map[varying_elements[0]] = va
            row_map[varying_elements[1]] = vb
            row_map[varying_elements[2]] = vc
            # base element (V) is already in map; ensure exact normalization
            total = sum(row_map.values())
            if abs(total - 1.0) > 1e-9:
                # distribute tiny residual onto base element to keep closed simplex
                row_map[BASE_ELEMENT] = row_map.get(BASE_ELEMENT, 0.0) + (1.0 - total)
            grid_rows.append(row_map)

    return pd.DataFrame(grid_rows, columns=["V", "Cr", "Ti", "W", "Zr"])


