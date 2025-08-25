"""High-level workflow to sample, filter, embed, and evaluate CALPHAD."""

from __future__ import annotations

from typing import Dict, Mapping, Optional, Tuple

import numpy as np  # type: ignore
import pandas as pd  # type: ignore

from .composition_sampling import SamplingConstraints, sample_compositions, make_ternary_grid
from .filters import ActivationLimits, make_activation_filter, make_ductility_filter, apply_filters
from .calphad_runner import run_calphad_batch
from .embedding import embed_2d, cluster_labels


def sample_and_filter(
    per_element_max: Mapping[str, float],
    total_alloy_max: float,
    pure_results: Mapping[str, Mapping[str, object]],
    activation_limits: ActivationLimits,
    n_samples: int = 1000,
    random_state: Optional[int] = None,
) -> pd.DataFrame:
    """Sample compositions and apply activation + ductility filters.

    Returns a DataFrame with composition columns and a boolean column `is_valid`.
    """
    constraints = SamplingConstraints(per_element_max=dict(per_element_max), total_alloy_max=float(total_alloy_max))
    comps = sample_compositions(constraints=constraints, n_samples=n_samples, random_state=random_state)

    f_activation = make_activation_filter(pure_results=pure_results, limits=activation_limits)
    f_ductility = make_ductility_filter(max_total_alloy=total_alloy_max)
    mask = apply_filters(compositions=comps, filters=(f_activation, f_ductility))
    comps = comps.copy()
    comps["is_valid"] = mask.values
    return comps


def embed_and_cluster_compositions(comps: pd.DataFrame, valid_col: str = "is_valid", random_state: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Create 2D embedding and cluster labels for the composition manifold."""
    frac_cols = [c for c in ("V", "Cr", "Ti", "W", "Zr") if c in comps.columns]
    X = comps[frac_cols]
    emb = embed_2d(X, random_state=random_state)
    labels = cluster_labels(emb, random_state=random_state)
    return emb, labels


def evaluate_calphad_for_valid(comps: pd.DataFrame, valid_col: str = "is_valid", temperature_k: float = 823.5, database: str = "TCHEA7") -> pd.DataFrame:
    """Run CALPHAD only for valid compositions and merge results back."""
    if valid_col not in comps.columns:
        raise ValueError(f"Missing validity column: {valid_col}")
    valid_df = comps[comps[valid_col]].drop(columns=[valid_col])
    if valid_df.empty:
        return comps.assign(phase_count=np.nan, dominant_phase=np.nan, single_phase=np.nan)
    cal = run_calphad_batch(valid_df, temperature_k=temperature_k, database=database)
    # Merge on index alignment
    out = comps.copy()
    for col in cal.columns:
        if col in ("V", "Cr", "Ti", "W", "Zr"):
            continue
        # initialize with NaN then fill where valid
        out[col] = np.nan
        out.loc[valid_df.index, col] = cal[col].values
    return out


