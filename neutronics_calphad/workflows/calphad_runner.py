"""Wrappers to run CALPHAD single-point evaluations for sampled compositions."""

from __future__ import annotations

from typing import List, Tuple

import numpy as np  # type: ignore
import pandas as pd  # type: ignore

from neutronics_calphad.calphad.phase_calculator import CALPHADBatchCalculator


ELEMENTS_ORDER: Tuple[str, ...] = ("V", "Cr", "Ti", "W", "Zr")


def run_calphad_batch(
    compositions: pd.DataFrame,
    temperature_k: float = 873.15,
    database: str = "TCHEA8",
) -> pd.DataFrame:
    """Run CALPHAD batch for given compositions.

    Args:
        compositions: DataFrame with columns matching a subset or all of ELEMENTS_ORDER.
            Missing elements will be treated as zero. Rows will be normalized to sum 1.
        temperature_k: Equilibration temperature in Kelvin.
        database: Thermo-Calc database name.

    Returns:
        DataFrame with CALPHAD outputs merged with the input compositions.
    """
    # Align and normalize
    df = compositions.copy()
    for el in ELEMENTS_ORDER:
        if el not in df.columns:
            df[el] = 0.0
    df = df[list(ELEMENTS_ORDER)]
    total = df.sum(axis=1)
    df = df.div(total, axis=0)

    calc = CALPHADBatchCalculator(database=database, temperature=temperature_k)
    comp_array = df.values.astype(float)
    results = calc.calculate_batch(compositions=comp_array, elements=list(ELEMENTS_ORDER))

    # Merge: results already contain x_ columns; also attach original fractions
    merged = pd.concat([df.reset_index(drop=True), results.reset_index(drop=True)], axis=1)
    return merged


