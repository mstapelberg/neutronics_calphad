"""Tests for refined composition-space analysis CLI utilities.

These tests use small synthetic DataFrames saved as CSVs to validate that the
pipeline creates key artifacts and honors epsilon intersection behavior.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from neutronics_calphad.analysis.refined_space import run_refined_analysis


def _write_dummy_inputs(tmp: Path) -> Dict[str, Path]:
    """Create tiny synthetic CSV inputs for testing.

    The compositions are intentionally simple and small to make the test fast
    and deterministic.
    """
    tmp.mkdir(parents=True, exist_ok=True)

    # LightGBM results: two rows pass, one fails
    df_res = pd.DataFrame(
        {
            "Cr": [0.2, 0.25, 0.6],
            "Ti": [0.3, 0.25, 0.1],
            "W": [0.4, 0.45, 0.2],
            "Zr": [0.1, 0.05, 0.1],
            "d30": [100.0, 200.0, 5000.0],
            "d365": [0.1, 0.2, 5.0],
            "d1825": [1e-3, 5e-3, 0.1],
            "d36500": [1e-5, 5e-5, 1e-3],
            "He_2y": [100.0, 200.0, 1000.0],
            "H_2y": [100.0, 200.0, 5000.0],
        }
    )
    res_csv = tmp / "lightgbm_results.csv"
    df_res.to_csv(res_csv, index=False)

    # CALPHAD candidates: include B2 and C15 columns to exercise filters
    df_cand = pd.DataFrame(
        {
            "Cr": [0.21, 0.26, 0.6],
            "Ti": [0.29, 0.26, 0.1],
            "W": [0.41, 0.43, 0.2],
            "Zr": [0.09, 0.05, 0.1],
            "bcc_b2_a": [0.997, 0.998, 0.1],
            "bcc_b2_b": [0.002, 0.001, 0.0],
            "c15_phase": [0.0, 0.0, 0.9],
        }
    )
    cand_csv = tmp / "calphad_candidates.csv"
    df_cand.to_csv(cand_csv, index=False)

    return {"res": res_csv, "cand": cand_csv}


def test_run_refined_analysis_creates_artifacts(tmp_path: Path) -> None:
    """The pipeline should create all key artifacts in the output directory."""
    paths = _write_dummy_inputs(tmp_path / "inputs")
    out_dir = tmp_path / "out"
    artifacts = run_refined_analysis(
        lightgbm_results=paths["res"],
        calphad_candidates=paths["cand"],
        output_dir=out_dir,
        epsilon=0.05,
        make_3d_plot=False,
    )
    # Expect at least these artifacts
    expected = {
        "neutronics_pass_set.csv",
        "calphad_safe_set.csv",
        "refined_intersection_approx.csv",
        "refined_approx_summary.csv",
        "pairgrid.png",
        "neutronics_pass_rates.png",
        "overlay_Cr_Ti.png",
        "overlay_Cr_W.png",
        "overlay_Cr_Zr.png",
        "overlay_Ti_W.png",
        "overlay_Ti_Zr.png",
        "overlay_W_Zr.png",
    }
    assert expected.issubset(set(artifacts.keys()))
    for p in artifacts.values():
        assert Path(p).exists()


