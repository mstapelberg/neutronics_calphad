"""
Refined composition-space analysis utilities and CLI-facing entrypoint.

This module implements functionality similar to the example script
`examples/neutronics_calphad_analysis_two/pspp_refined_space_analysis.py`,
but exposes typed, reusable functions and a main entrypoint callable from the
package CLI.

Features:
- Load LightGBM results and CALPHAD candidate CSVs.
- Apply neutronics pass thresholds and CALPHAD safety criteria.
- Compute an approximate intersection via an L1 distance tolerance.
- Generate pairwise overlay plots and a 4x4 pair-grid with diagonal histograms
  showing the distribution of element fractions by category: both, CALPHAD-only,
  neutronics-only, neither.
- Save CSV artifacts and a short summary.

Notes:
- Matplotlib is used for plotting; seaborn is not required.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Keep consistent with the example script
THRESHOLDS: Dict[str, float] = {
    "d30": 1e3,
    "d365": 1.0,
    "d1825": 1e-2,
    "d36500": 1e-4,
    "He_2y": 586.0,
    "H_2y": 1200.0,
}

COMP_COLS: List[str] = ["Cr", "Ti", "W", "Zr"]


def ensure_comp(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure composition columns `Cr`, `Ti`, `W`, `Zr` exist (case-insensitive).

    Parameters
    ----------
    df:
        Input DataFrame.

    Returns
    -------
    pd.DataFrame
        Copy of the DataFrame with composition columns ensured/renamed.
    """
    out = df.copy()
    for el in COMP_COLS:
        if el not in out.columns:
            matches = [c for c in out.columns if c.lower() == el.lower()]
            if matches:
                out = out.rename(columns={matches[0]: el})
    return out


def compute_neutronics_pass(df: pd.DataFrame, thresholds: Mapping[str, float]) -> pd.DataFrame:
    """Compute neutronics pass/fail boolean columns given thresholds.

    Parameters
    ----------
    df:
        LightGBM results DataFrame.
    thresholds:
        Mapping of metric name to threshold (pass if value <= threshold).

    Returns
    -------
    pd.DataFrame
        Copy of input with `pass_<metric>` boolean columns and
        `neutronics_pass_all` if any metric columns exist.
    """
    out = df.copy()
    for metric, limit in thresholds.items():
        # Prefer direct column match; otherwise fuzzy find a numeric column containing the metric name
        col = metric if metric in out.columns else next(
            (c for c in out.columns if (metric.lower() in c.lower()) and ("pass" not in c.lower()) and ("q" not in c.lower())),
            None,
        )
        if col is not None:
            out[f"pass_{metric}"] = pd.to_numeric(out[col], errors="coerce") <= limit
    pass_cols = [c for c in out.columns if c.startswith("pass_")]
    if pass_cols:
        out["neutronics_pass_all"] = out[pass_cols].all(axis=1)
    return out


def filter_calphad_safe(df_cand: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Filter CALPHAD candidates that are BCC_B2 dominant with no C15/Laves.

    Criteria (matching the example script):
    - Parse phase data from JSON strings in 'phases' column
    - Require no C15/Laves phases present
    - Require ≤1 BCC_B2 phase
    - Require HCP_A3 + FCC_L12 total ≤ 0.5% (or 0.005 if in fraction format)

    Parameters
    ----------
    df_cand:
        CALPHAD candidates DataFrame with 'phases' column containing JSON strings.

    Returns
    -------
    Tuple[pd.DataFrame, np.ndarray, np.ndarray]
        The filtered DataFrame (safe set), a boolean array for the BCC_B2 criterion,
        and a boolean array for the C15/Laves criterion aligned to `df_cand` rows.
    """
    import json
    from typing import Any
    
    def _parse_phases_cell(phases_str: str) -> dict[str, float]:
        """Parse JSON string from phases column into dictionary."""
        try:
            if isinstance(phases_str, dict):
                return phases_str
            return json.loads(phases_str)
        except Exception:
            return {}
    
    def _has_c15_laves(phases_dict: dict[str, float]) -> bool:
        """Check if any phase resembles C15 Laves (case-insensitive)."""
        for k in phases_dict.keys():
            s = str(k).lower()
            if "c15" in s or ("laves" in s and "c15" in s):
                return True
        return False
    
    def _count_b2(phases_dict: dict[str, float]) -> int:
        """Count phases whose names include 'b2' (case-insensitive)."""
        return sum(1 for k in phases_dict.keys() if "b2" in str(k).lower())
    
    def _hcp_fcc_sum(phases_dict: dict[str, float]) -> float:
        """Sum of HCP_A3 and FCC_L12 values."""
        total = 0.0
        for key in phases_dict.keys():
            s = str(key).strip().lower()
            if s in {"hcp_a3", "fcc_l12"} or s.endswith("hcp_a3") or s.endswith("fcc_l12"):
                try:
                    total += float(phases_dict[key])
                except Exception:
                    continue
        return total
    
    def _detect_percent_scale(phases_list: list[dict[str, Any]]) -> float:
        """Detect if phase values are in percent (100.0) or fraction (1.0) scale."""
        sample_vals: list[float] = []
        for phases_dict in phases_list[:100]:  # Sample first 100 rows
            for v in phases_dict.values():
                try:
                    val = float(v)
                    if np.isfinite(val):
                        sample_vals.append(val)
                except Exception:
                    continue
                if len(sample_vals) >= 50:  # Enough samples
                    break
            if len(sample_vals) >= 50:
                break
        if not sample_vals:
            return 100.0
        return 100.0 if float(np.nanmax(sample_vals)) > 1.0 else 1.0
    
    # Parse phases data
    df_cand = df_cand.copy()
    df_cand["phases_dict"] = df_cand["phases"].apply(_parse_phases_cell)
    
    # Detect scale and set threshold
    percent_scale = _detect_percent_scale(list(df_cand["phases_dict"]))
    thr = 0.5 if percent_scale == 100.0 else 0.005
    
    # Apply CALPHAD safety criteria
    c15_present = df_cand["phases_dict"].apply(_has_c15_laves)
    b2_counts = df_cand["phases_dict"].apply(_count_b2)
    hcp_fcc_total = df_cand["phases_dict"].apply(_hcp_fcc_sum)
    
    # BCC_B2 criterion: ≤1 BCC_B2 phase
    pass_b2 = b2_counts <= 1
    
    # C15/Laves criterion: no C15/Laves phases
    pass_c15 = ~c15_present
    
    # Additional criterion: HCP_A3 + FCC_L12 ≤ threshold
    pass_hcp_fcc = hcp_fcc_total <= thr
    
    # Combined safety mask
    mask = pass_b2 & pass_c15 & pass_hcp_fcc
    safe = df_cand.loc[mask].copy()
    
    return safe, pass_b2, pass_c15


def compute_min_l1_to_B(A: np.ndarray, B: np.ndarray, chunk: int = 5000) -> np.ndarray:
    """Compute per-row minimal L1 distance from A to the set B in chunks.

    Parameters
    ----------
    A:
        Array of shape (n_a, d).
    B:
        Array of shape (n_b, d).
    chunk:
        Number of rows of B to process per iteration to control memory.

    Returns
    -------
    np.ndarray
        Array of shape (n_a,) with minimal L1 distances to any point in B.
    """
    if A.size == 0:
        return np.array([], dtype=float)
    if B.size == 0:
        return np.full(A.shape[0], np.inf, dtype=float)
    dmin = np.full(A.shape[0], np.inf, dtype=float)
    for start in range(0, B.shape[0], chunk):
        Bj = B[start : start + chunk]
        d = np.sum(np.abs(A[:, None, :] - Bj[None, :, :]), axis=2)
        dmin = np.minimum(dmin, d.min(axis=1))
    return dmin


def classify_candidates(
    df_cand: pd.DataFrame,
    df_neut_pass: pd.DataFrame,
    epsilon: float,
) -> pd.Series:
    """Classify CALPHAD candidates into four categories w.r.t neutronics pass set.

    Categories are defined on the CALPHAD candidate universe using an L1 proximity
    rule to the neutronics pass set:
    - both: candidate is in CALPHAD-safe set and within epsilon of any neutronics-pass composition
    - calphad_only: candidate is CALPHAD-safe but not near any neutronics-pass composition
    - neutronics_only: candidate is not CALPHAD-safe but is near a neutronics-pass composition
    - neither: candidate is neither CALPHAD-safe nor near a neutronics-pass composition

    Parameters
    ----------
    df_cand:
        Full CALPHAD candidates DataFrame.
    df_neut_pass:
        LightGBM results rows where `neutronics_pass_all` is True.
    epsilon:
        L1 distance tolerance to define "near".

    Returns
    -------
    pd.Series
        Categorical series (string labels) aligned to `df_cand` index.
    """
    safe_df, pass_b2, pass_c15 = filter_calphad_safe(df_cand)
    
    # Recompute the full CALPHAD safety mask including HCP/FCC criterion
    import json
    from typing import Any
    
    def _parse_phases_cell(phases_str: str) -> dict[str, float]:
        """Parse JSON string from phases column into dictionary."""
        try:
            if isinstance(phases_str, dict):
                return phases_str
            return json.loads(phases_str)
        except Exception:
            return {}
    
    def _has_c15_laves(phases_dict: dict[str, float]) -> bool:
        """Check if any phase resembles C15 Laves (case-insensitive)."""
        for k in phases_dict.keys():
            s = str(k).lower()
            if "c15" in s or ("laves" in s and "c15" in s):
                return True
        return False
    
    def _count_b2(phases_dict: dict[str, float]) -> int:
        """Count phases whose names include 'b2' (case-insensitive)."""
        return sum(1 for k in phases_dict.keys() if "b2" in str(k).lower())
    
    def _hcp_fcc_sum(phases_dict: dict[str, float]) -> float:
        """Sum of HCP_A3 and FCC_L12 values."""
        total = 0.0
        for key in phases_dict.keys():
            s = str(key).strip().lower()
            if s in {"hcp_a3", "fcc_l12"} or s.endswith("hcp_a3") or s.endswith("fcc_l12"):
                try:
                    total += float(phases_dict[key])
                except Exception:
                    continue
        return total
    
    def _detect_percent_scale(phases_list: list[dict[str, Any]]) -> float:
        """Detect if phase values are in percent (100.0) or fraction (1.0) scale."""
        sample_vals: list[float] = []
        for phases_dict in phases_list[:100]:  # Sample first 100 rows
            for v in phases_dict.values():
                try:
                    val = float(v)
                    if np.isfinite(val):
                        sample_vals.append(val)
                except Exception:
                    continue
                if len(sample_vals) >= 50:  # Enough samples
                    break
            if len(sample_vals) >= 50:
                break
        if not sample_vals:
            return 100.0
        return 100.0 if float(np.nanmax(sample_vals)) > 1.0 else 1.0
    
    # Parse phases data
    df_cand_copy = df_cand.copy()
    df_cand_copy["phases_dict"] = df_cand_copy["phases"].apply(_parse_phases_cell)
    
    # Detect scale and set threshold
    percent_scale = _detect_percent_scale(list(df_cand_copy["phases_dict"]))
    thr = 0.5 if percent_scale == 100.0 else 0.005
    
    # Apply CALPHAD safety criteria
    c15_present = df_cand_copy["phases_dict"].apply(_has_c15_laves)
    b2_counts = df_cand_copy["phases_dict"].apply(_count_b2)
    hcp_fcc_total = df_cand_copy["phases_dict"].apply(_hcp_fcc_sum)
    
    # Full CALPHAD safety mask
    calphad_safe_mask = (b2_counts <= 1) & (~c15_present) & (hcp_fcc_total <= thr)

    A = df_cand[COMP_COLS].to_numpy(dtype=float)
    B = df_neut_pass[COMP_COLS].to_numpy(dtype=float)
    dmin = compute_min_l1_to_B(A, B)
    near_neutronics = dmin <= float(epsilon)

    labels = np.empty(len(df_cand), dtype=object)
    both = calphad_safe_mask & near_neutronics
    calphad_only = calphad_safe_mask & (~near_neutronics)
    neutronics_only = (~calphad_safe_mask) & near_neutronics
    neither = (~calphad_safe_mask) & (~near_neutronics)

    labels[both] = "both"
    labels[calphad_only] = "calphad_only"
    labels[neutronics_only] = "neutronics_only"
    labels[neither] = "neither"
    return pd.Series(labels, index=df_cand.index, name="category")


def plot_pair_overlay(
    df_all: Optional[pd.DataFrame],
    df_safe: Optional[pd.DataFrame],
    df_neut: Optional[pd.DataFrame],
    df_ref: Optional[pd.DataFrame],
    x: str,
    y: str,
    ax: plt.Axes,
) -> None:
    """Scatter overlay for a given pair of composition columns on a given axes.

    Parameters
    ----------
    df_all:
        Full CALPHAD candidate set.
    df_safe:
        CALPHAD safe set.
    df_neut:
        Neutronics-pass set (from LightGBM results).
    df_ref:
        Refined intersection (approximate) of `df_safe` and `df_neut`.
    x, y:
        Column names to plot on x and y axes.
    ax:
        Matplotlib Axes to draw on.
    """
    if df_all is not None and not df_all.empty:
        ax.scatter(df_all[x], df_all[y], s=6, alpha=0.15, color="lightgray", label="CALPHAD: all", marker=".")
    if df_safe is not None and not df_safe.empty:
        ax.scatter(df_safe[x], df_safe[y], s=10, alpha=0.58, color="#2A33C3", label="CALPHAD: safe BCC_B2", marker="o")
    if df_neut is not None and not df_neut.empty:
        ax.scatter(df_neut[x], df_neut[y], s=14, alpha=0.58, color="#A35D00", label="Neutronics: pass", marker="^")
    if df_ref is not None and not df_ref.empty:
        ax.scatter(df_ref[x], df_ref[y], s=16, alpha=0.58, color="#0B7285", label="Refined (∩ approx.)", marker="s")
    ax.set_xlabel(f"{x} (at. frac)")
    ax.set_ylabel(f"{y} (at. frac)")


def plot_4x4_pairgrid_with_histograms(
    df_cand: pd.DataFrame,
    df_safe: pd.DataFrame,
    df_neut_pass: pd.DataFrame,
    df_refined: pd.DataFrame,
    categories: pd.Series,
    output_path: Path,
) -> Path:
    """Create a 4x4 pair-grid for `Cr, Ti, W, Zr` with diagonal hist overlays.

    Off-diagonals show the standard overlay scatter (all, safe, neutronics, refined).
    Diagonals show overlaid histograms for categories: both, calphad_only,
    neutronics_only, neither, based on classification over CALPHAD candidates.

    Parameters
    ----------
    df_cand:
        Full CALPHAD candidates.
    df_safe:
        CALPHAD safe subset.
    df_neut_pass:
        Neutronics pass subset from LightGBM results.
    df_refined:
        Approx intersection subset.
    categories:
        Per-row labels for `df_cand` from `classify_candidates`.
    output_path:
        Path to save the pair-grid figure.

    Returns
    -------
    Path
        The saved figure path.
    """
    elements = COMP_COLS
    n = len(elements)
    fig, axes = plt.subplots(n, n, figsize=(12, 12), dpi=200)

    # Precompute masks for categories on df_cand
    masks = {
        "both": categories == "both",
        "calphad_only": categories == "calphad_only",
        "neutronics_only": categories == "neutronics_only",
        "neither": categories == "neither",
    }

    for i, xi in enumerate(elements):
        for j, yj in enumerate(elements):
            ax = axes[i, j]
            if i == j:
                # Diagonal: hist overlays by categories on df_cand[xi]
                values = df_cand[xi].to_numpy(dtype=float)
                # Guard against degenerate data for binning
                bins = 20
                try:
                    vmin = float(np.nanmin(values))
                    vmax = float(np.nanmax(values))
                    if not np.isfinite(vmin) or not np.isfinite(vmax) or (vmax <= vmin):
                        raise ValueError
                    bin_edges = np.linspace(vmin, vmax, bins + 1)
                except Exception:
                    bin_edges = bins

                # Overlay in a consistent order with better colors
                colors = {
                    "both": "#0B7285",
                    "calphad_only": "#2A33C3", 
                    "neutronics_only": "#A35D00",
                    "neither": "#8F2D56"
                }
                
                for label, alpha, lw in [
                    ("both", 0.58, 2.0),
                    ("calphad_only", 0.58, 1.5),
                    ("neutronics_only", 0.58, 1.5),
                    ("neither", 0.58, 1.0),
                ]:
                    vals = df_cand.loc[masks[label], xi].to_numpy(dtype=float)
                    if vals.size:
                        count = len(vals)
                        ax.hist(
                            vals,
                            bins=bin_edges,
                            histtype="step",
                            linewidth=lw,
                            alpha=alpha,
                            color=colors[label],
                            label=f"{label} (n={count})",
                            density=True,
                        )

                ax.set_xlabel(f"{xi} (at. frac)")
                ax.set_ylabel("Density (prob/at.frac)")
                if i == 0 and j == 0:
                    ax.legend(frameon=False, fontsize=8)
            else:
                # Off-diagonal: overlay scatter
                plot_pair_overlay(df_cand, df_safe, df_neut_pass, df_refined, yj, xi, ax)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def plot_refined_3d(df: pd.DataFrame, output_path: Path) -> Optional[Path]:
    """Plot a simple 3D scatter of `Cr, Ti, W` with point size proportional to `Zr`.

    Parameters
    ----------
    df:
        DataFrame with columns `Cr, Ti, W, Zr`.
    output_path:
        File path to write the figure to.

    Returns
    -------
    Optional[Path]
        The saved path if plotting occurs; otherwise None.
    """
    if df.empty:
        return None
    fig = plt.figure(figsize=(5, 4), dpi=200)
    ax = fig.add_subplot(111, projection="3d")
    sizes = (df["Zr"].to_numpy(dtype=float) * 600.0) + 10.0
    ax.scatter(df["Cr"].to_numpy(dtype=float), df["Ti"].to_numpy(dtype=float), df["W"].to_numpy(dtype=float), s=sizes, alpha=0.58, color="#6E8B00", depthshade=True)
    ax.set_xlabel("Cr")
    ax.set_ylabel("Ti")
    ax.set_zlabel("W")
    ax.set_title("Refined Space (approx. Structure ∩ Properties)")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def compute_pass_rates(df_res: pd.DataFrame, df_res_passed: pd.DataFrame) -> Dict[str, float]:
    """Compute pass fractions by metric including the overall fraction.

    Parameters
    ----------
    df_res:
        Original LightGBM results DataFrame.
    df_res_passed:
        LightGBM results with pass columns (`pass_*`) and `neutronics_pass_all`.

    Returns
    -------
    Dict[str, float]
        Mapping of metric to pass fraction in [0, 1]. Includes key "ALL" if available.
    """
    pass_rates: Dict[str, float] = {}
    for m, lim in THRESHOLDS.items():
        if m in df_res.columns:
            vals = pd.to_numeric(df_res[m], errors="coerce")
            pass_rates[m] = float(np.nanmean(vals <= lim))
        elif f"pass_{m}" in df_res_passed.columns:
            pass_rates[m] = float(np.nanmean(df_res_passed[f"pass_{m}"]))
    if "neutronics_pass_all" in df_res_passed.columns:
        pass_rates["ALL"] = float(np.nanmean(df_res_passed["neutronics_pass_all"]))
    return pass_rates


def run_refined_analysis(
    lightgbm_results: Path | str,
    calphad_candidates: Path | str,
    output_dir: Path | str,
    epsilon: float = 0.02,
    make_3d_plot: bool = True,
) -> Dict[str, Path]:
    """Run the refined analysis pipeline and write figures and CSV artifacts.

    Parameters
    ----------
    lightgbm_results:
        Path to `lightgbm_results.csv`.
    calphad_candidates:
        Path to `calphad_candidates.csv`.
    output_dir:
        Directory to write outputs (figures and CSVs).
    epsilon:
        L1 distance tolerance for approximate intersection.
    make_3d_plot:
        Whether to save a 3D scatter of the refined set.

    Returns
    -------
    Dict[str, Path]
        Mapping of artifact names to saved paths.
    """
    logging.info("Loading inputs")
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    df_res = ensure_comp(pd.read_csv(lightgbm_results))
    df_cand = ensure_comp(pd.read_csv(calphad_candidates))

    # Neutronics pass set
    df_res_pass = compute_neutronics_pass(df_res, THRESHOLDS)
    neut = df_res_pass[df_res_pass.get("neutronics_pass_all", False)].copy()

    # CALPHAD safe set
    calphad_safe, _, _ = filter_calphad_safe(df_cand)

    # Approximate intersection via L1 distance
    A = calphad_safe[COMP_COLS].to_numpy(dtype=float)
    B = neut[COMP_COLS].to_numpy(dtype=float)
    refined = calphad_safe.iloc[[]].copy()
    if A.size and B.size:
        dmin = compute_min_l1_to_B(A, B)
        mask = dmin <= float(epsilon)
        refined = calphad_safe.loc[mask].copy()

    # CSV artifacts
    artifacts: Dict[str, Path] = {}
    neut_csv = out_dir / "neutronics_pass_set.csv"
    calphad_safe_csv = out_dir / "calphad_safe_set.csv"
    refined_csv = out_dir / "refined_intersection_approx.csv"
    summary_csv = out_dir / "refined_approx_summary.csv"
    neut.to_csv(neut_csv, index=False)
    calphad_safe.to_csv(calphad_safe_csv, index=False)
    refined.to_csv(refined_csv, index=False)
    pd.DataFrame(
        [
            {
                "epsilon_L1_for_intersection": float(epsilon),
                "n_lgbm_results": int(df_res.shape[0]),
                "n_neutronics_pass_all": int(neut.shape[0]),
                "n_calphad_all": int(df_cand.shape[0]),
                "n_calphad_safe": int(calphad_safe.shape[0]),
                "n_refined_intersection_approx": int(refined.shape[0]),
            }
        ]
    ).to_csv(summary_csv, index=False)

    artifacts.update(
        {
            "neutronics_pass_set.csv": neut_csv,
            "calphad_safe_set.csv": calphad_safe_csv,
            "refined_intersection_approx.csv": refined_csv,
            "refined_approx_summary.csv": summary_csv,
        }
    )

    # Pair overlays (individual figures like example)
    pairs: List[Tuple[str, str]] = [("Cr", "Ti"), ("Cr", "W"), ("Cr", "Zr"), ("Ti", "W"), ("Ti", "Zr"), ("W", "Zr")]
    for x, y in pairs:
        fig = plt.figure(figsize=(4, 4), dpi=200)
        ax = fig.add_subplot(111)
        plot_pair_overlay(df_cand[COMP_COLS], calphad_safe[COMP_COLS], neut[COMP_COLS], refined[COMP_COLS], x, y, ax)
        ax.set_title(f"Composition overlay: {x} vs {y}")
        ax.legend(frameon=False, fontsize=7)
        fig.tight_layout()
        fpath = fig_dir / f"overlay_{x}_{y}.png"
        fig.savefig(fpath)
        plt.close(fig)
        artifacts[f"overlay_{x}_{y}.png"] = fpath

    # 3D plot for refined set
    if make_3d_plot:
        p3d = plot_refined_3d(refined[COMP_COLS], fig_dir / "refined_approx_3D.png")
        if p3d is not None:
            artifacts["refined_approx_3D.png"] = p3d

    # Pass-rate bar chart
    pass_rates = compute_pass_rates(df_res, df_res_pass)
    if pass_rates:
        fig = plt.figure(figsize=(8, 4), dpi=200)
        ax = fig.add_subplot(111)
        keys = list(pass_rates.keys())
        vals = [pass_rates[k] for k in keys]
        # Use a color palette from the specified colors
        colors = ["#2A33C3", "#A35D00", "#0B7285", "#8F2D56", "#6E8B00"]
        bar_colors = [colors[i % len(colors)] for i in range(len(keys))]
        ax.bar(keys, vals, color=bar_colors, alpha=0.58)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Pass fraction")
        ax.set_title("Neutronics pass rates")
        for i, v in enumerate(vals):
            y = min(1.02, v + 0.02)
            ax.text(i, y, f"{v:.2f}", ha="center", va="bottom", fontsize=9)
        fig.tight_layout()
        out = fig_dir / "neutronics_pass_rates.png"
        fig.savefig(out)
        plt.close(fig)
        artifacts["neutronics_pass_rates.png"] = out

    # 4x4 pair-grid with diagonal hist overlays
    cats = classify_candidates(df_cand, neut, epsilon)
    pairgrid_path = fig_dir / "pairgrid.png"
    plot_4x4_pairgrid_with_histograms(df_cand[COMP_COLS], calphad_safe[COMP_COLS], neut[COMP_COLS], refined[COMP_COLS], cats, pairgrid_path)
    artifacts["pairgrid.png"] = pairgrid_path

    return artifacts


__all__ = [
    "THRESHOLDS",
    "COMP_COLS",
    "ensure_comp",
    "compute_neutronics_pass",
    "filter_calphad_safe",
    "compute_min_l1_to_B",
    "classify_candidates",
    "plot_pair_overlay",
    "plot_4x4_pairgrid_with_histograms",
    "plot_refined_3d",
    "compute_pass_rates",
    "run_refined_analysis",
]


