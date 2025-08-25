"""Utilities to load pure-element neutronics results for activation filters.

This module reconstructs the `pure_results` mapping by reading existing
OpenMC depletion results produced by the `examples/pure_element_neutronics_run.py`.
It expects the per-element `depletion_results.h5` files in a directory layout:

    {results_dir}/depletion_results/{Element}/depletion_results.h5

and uses `parse_openmc_results` to extract gas production and dose at cooling times.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Mapping

import openmc  # type: ignore

from neutronics_calphad.optimizer.parsers import parse_openmc_results


def load_pure_results(
    results_dir: str,
    chain_file: str,
    abs_file: str,
    elements: tuple[str, ...] = ("V", "Cr", "Ti", "W", "Zr"),
    cooling_days: tuple[int, ...] = (30, 365, 5 * 365, 36500),
) -> Dict[str, Dict[str, object]]:
    """Load pure-element activation summaries from OpenMC depletion outputs.

    Args:
        results_dir: Base directory containing element subfolders with results.
        chain_file: OpenMC depletion chain file path.
        abs_file: FISPACT ABS file for dose calculation.
        elements: Elements to load; defaults to ("V","Cr","Ti","W","Zr").
        cooling_days: Cooling days to extract dose limits for.

    Returns:
        Mapping suitable for activation filters: {element: {gas_production: {...}, dose_at_cooling_times: {...}}}.
    """
    out: Dict[str, Dict[str, object]] = {}

    for el in elements:
        res_path = Path(results_dir) / "depletion_results" / el / "depletion_results.h5"
        if not res_path.exists():
            continue
        results = openmc.deplete.Results(str(res_path))
        summary = parse_openmc_results(
            results=results,
            chain_file=chain_file,
            abs_file=abs_file,
            cooling_days=sorted(cooling_days),
        )
        out[el] = summary  # expected keys used by filters

    return out


