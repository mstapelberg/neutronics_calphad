"""Batch Bayesian Optimization loop for CALPHAD feasibility.

This module proposes batches of compositions, runs CALPHAD in batches,
labels feasibility per the analysis criteria, and fits a simple classifier
to improve proposals iteratively.

Outputs a CSV compatible with `examples/calphad_lgbm_analysis_package/scripts/comp_space_viz.py`:
    V,Cr,Ti,W,Zr,label,neutronics_ok,calphad_ok

Notes:
- `neutronics_ok` is set equal to `label` as a placeholder unless a separate
  neutronics criterion is provided by the caller.
- `calphad_ok` follows the same rule as analyze_calphad: no C15/Laves, ≤1 B2,
  and HCP_A3+FCC_L12 ≤ 0.5 vol% (or 0.005 fraction).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .calphad_runner import run_calphad_batch


ELEMENTS: Tuple[str, ...] = ("V", "Cr", "Ti", "W", "Zr")


@dataclass
class BOConfig:
    """Configuration for CALPHAD BO loop."""

    n_init: int = 64
    batch_q: int = 64
    n_iters: int = 5
    seed: int = 0
    temperature_k: float = 873.15
    database: str = "TCHEA8"
    phase_threshold: float = 0.995
    output_dir: Path = Path("analysis_results/calphad_bo")


def _rand_simplex(n: int, d: int, rng: np.random.Generator) -> np.ndarray:
    x = rng.random((n, d))
    x = x / x.sum(axis=1, keepdims=True)
    return x


def _calphad_ok_mask(df: pd.DataFrame) -> np.ndarray:
    # Parse phases column
    def _parse_phases_cell(val):
        if isinstance(val, dict):
            return val
        try:
            import ast
            parsed = ast.literal_eval(val)
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            return {}

    def _detect_percent_scale(phases_list: List[Dict[str, float]]) -> float:
        sample_vals: List[float] = []
        for d in phases_list[:50]:
            for v in d.values():
                try:
                    sample_vals.append(float(v))
                except Exception:
                    pass
            if len(sample_vals) > 200:
                break
        if not sample_vals:
            return 100.0
        return 100.0 if float(np.nanmax(sample_vals)) > 1.0 else 1.0

    def _has_c15_laves(phases_dict: Dict[str, float]) -> bool:
        for k in phases_dict.keys():
            s = str(k).lower()
            if "c15" in s or ("laves" in s and "c15" in s):
                return True
        return False

    def _count_b2(phases_dict: Dict[str, float]) -> int:
        return sum(1 for k in phases_dict.keys() if "b2" in str(k).lower())

    def _hcp_fcc_sum(phases_dict: Dict[str, float]) -> float:
        total = 0.0
        for key in phases_dict.keys():
            s = str(key).strip().lower()
            if s in {"hcp_a3", "fcc_l12"} or s.endswith("hcp_a3") or s.endswith("fcc_l12"):
                try:
                    total += float(phases_dict[key])
                except Exception:
                    continue
        return total

    df = df.copy()
    df["phases_dict"] = df["phases"].apply(_parse_phases_cell)
    percent_scale = _detect_percent_scale(list(df["phases_dict"]))
    thr = 0.5 if percent_scale == 100.0 else 0.005
    c15 = df["phases_dict"].apply(_has_c15_laves)
    b2c = df["phases_dict"].apply(_count_b2)
    sum_hf = df["phases_dict"].apply(_hcp_fcc_sum)
    ok = (~c15) & (b2c <= 1) & (sum_hf <= thr)
    return ok.to_numpy(dtype=bool)


def run_calphad_bo(config: BOConfig, neutronics_ok_fn: Optional[Callable[[pd.DataFrame], np.ndarray]] = None) -> Path:
    """Run a simple batch BO loop to screen CALPHAD feasibility and export comps.csv.

    Args:
        config: BOConfig with loop settings and CALPHAD parameters.
        neutronics_ok_fn: Optional function to compute neutronics_ok mask on
            a DataFrame with V,Cr,Ti,W,Zr columns. If None, uses calphad_ok as label.

    Returns:
        Path to exported comps.csv
    """
    rng = np.random.default_rng(config.seed)
    outdir = config.output_dir
    outdir.mkdir(parents=True, exist_ok=True)

    # Initialize random compositions on 5-simplex
    X = _rand_simplex(config.n_init, len(ELEMENTS), rng)
    df = pd.DataFrame(X, columns=list(ELEMENTS))

    # Storage
    all_rows: List[pd.DataFrame] = []

    for it in range(config.n_iters):
        # Evaluate CALPHAD
        cal = run_calphad_batch(
            df,
            temperature_k=config.temperature_k,
            database=config.database,
            phase_threshold=config.phase_threshold,
        )
        ok = _calphad_ok_mask(cal)

        # Label: CALPHAD feasibility as proxy (or combine with neutronics_ok)
        label = ok.copy()
        if neutronics_ok_fn is not None:
            nmask = neutronics_ok_fn(df)
            label = label & nmask

        # Export chunk
        chunk = df.copy()
        chunk["label"] = label.astype(int)
        chunk["neutronics_ok"] = (nmask if neutronics_ok_fn is not None else label).astype(bool)
        chunk["calphad_ok"] = ok.astype(bool)
        all_rows.append(chunk)

        # Fit a simple classifier (logistic regression) to propose next batch
        try:
            from sklearn.linear_model import LogisticRegression
            clf = LogisticRegression(max_iter=1000)
            y = label.astype(int)
            clf.fit(df.values, y)
            # Sample candidates and pick top by predicted prob
            pool = _rand_simplex(8192, len(ELEMENTS), rng)
            p = clf.predict_proba(pool)[:, 1]
            next_X = pool[np.argsort(p)[-config.batch_q:]]
        except Exception:
            next_X = _rand_simplex(config.batch_q, len(ELEMENTS), rng)

        df = pd.DataFrame(next_X, columns=list(ELEMENTS))

    # Combine and export
    out = pd.concat(all_rows, axis=0, ignore_index=True)
    path = outdir / "comps.csv"
    out.to_csv(path, index=False)
    return path


