"""Generate ternary phase diagrams for Cr–Ti–W at fixed V and Zr.

This example script creates three side-by-side ternary plots over the
Cr–Ti–W simplex with the following fixed compositions (atomic fractions):

- Panel 1: V = 0.92, Zr = 0.005  → (Cr + Ti + W) = 0.075
- Panel 2: V = 0.85, Zr = 0.005  → (Cr + Ti + W) = 0.145
- Panel 3: V = 0.80, Zr = 0.005  → (Cr + Ti + W) = 0.195

For each panel, a grid of (Cr, Ti, W) points is evaluated using a trained
LightGBM quantile ensemble to compute a conservative feasibility margin
based on the 0.9-quantile upper bound versus target limits. CALPHAD
candidate points are overlaid for the corresponding slice (within a small
tolerance of the fixed V and Zr values).

Inputs (CLI):
- Path to a directory containing a pickled LightGBM model (lgbm_model.pkl).
- Path to CALPHAD candidate CSV (e.g., calphad_candidates.csv or an
  aggregated CSV of iterative runs).
- Optionally a CSV of LightGBM results to estimate the C+N+O atomic fraction
  (CNO_FRAC) used during training. If available, the median is used to align
  the feature pipeline with the model's training configuration.

Outputs:
- A PNG figure with three ternary plots saved to the specified output path.

Notes:
- This script uses a simple barycentric transformation and Matplotlib
  directly to avoid dependencies on external ternary-plot libraries.
"""

from __future__ import annotations

import argparse
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np  # type: ignore
import pandas as pd  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
from matplotlib.colors import TwoSlopeNorm  # type: ignore

# Optional: high-quality ternary plotting
try:
    import mpltern  # type: ignore  # noqa: F401
    _HAS_MPLTERN = True
except Exception:
    _HAS_MPLTERN = False

# Internal workflow utilities for features/limits
from neutronics_calphad.calphad.feasibility import compute_calphad_ok
from neutronics_calphad.workflows import lightgbm_workflow as lgbm_wf
from neutronics_calphad.workflows.lightgbm_workflow import _quantize_raw_floor, GRID_STEP
from neutronics_calphad.workflows.calphad_runner import (
    run_calphad_batch,
    run_calphad_batch_parallel,
)


# =====================================================================================
# Data structures and helpers
# =====================================================================================


@dataclass(frozen=True)
class SliceConfig:
    """Configuration for a ternary slice with fixed V and Zr.

    Attributes:
        v_fixed: Fixed atomic fraction of V.
        zr_fixed: Fixed atomic fraction of Zr.
        label: Human-readable label for the subplot title.
    """

    v_fixed: float
    zr_fixed: float
    label: str


def load_lgbm_model(path: Path) -> Any:
    """Load a pickled LightGBM quantile ensemble.

    Parameters
    ----------
    path : Path
        Path to the pickled model (e.g., ``lgbm_model.pkl``).

    Returns
    -------
    Any
        Deserialized model object with ``predict_quantiles`` and
        ``target_names`` attributes.
    """

    with open(path, "rb") as f:
        model = pickle.load(f)
    return model


def estimate_cno_frac_from_csv(csv_path: Path) -> Optional[float]:
    """Estimate ``CNO_FRAC`` from a results CSV if available.

    The estimate uses the closure relation ``cno_frac = 1 - (Cr+Ti+W+Zr) - V``.

    Parameters
    ----------
    csv_path : Path
        Path to ``lightgbm_results.csv``.

    Returns
    -------
    Optional[float]
        Median CNO fraction if the columns are present; otherwise ``None``.
    """

    try:
        df = pd.read_csv(csv_path)
        req = ["V", "Cr", "Ti", "W", "Zr"]
        if all(c in df.columns for c in req):
            s_raw = (df[["Cr", "Ti", "W", "Zr"]].astype(float)).sum(axis=1)
            v = df["V"].astype(float)
            cno = (1.0 - s_raw - v).clip(lower=0.0)
            if len(cno) > 0:
                return float(np.median(cno.values))
    except Exception:
        pass
    return None


def make_ternary_grid_two_fixed_step(
    *,
    v_fixed: float,
    zr_fixed: float,
    step_frac: float,
) -> pd.DataFrame:
    """Construct a grid over the Cr–Ti–W ternary with fixed V and Zr and step size.

    The grid enumerates points (Cr, Ti, W) such that Cr + Ti + W = T where
    T = max(0, 1 - v_fixed - zr_fixed) using a lattice with spacing
    ``step_frac`` (e.g., 0.005 for 0.5 at%). This ensures exact closure
    without interpolation artifacts when T is a multiple of step size.

    Parameters
    ----------
    v_fixed : float
        Fixed atomic fraction for V.
    zr_fixed : float
        Fixed atomic fraction for Zr.
    step_frac : float
        Lattice step for Cr, Ti, W in atomic fraction units (e.g., 0.005).

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ["V","Cr","Ti","W","Zr"].
    """

    if v_fixed < 0.0 or v_fixed > 1.0:
        raise ValueError("v_fixed must be within [0, 1]")
    if zr_fixed < 0.0 or zr_fixed > 1.0:
        raise ValueError("zr_fixed must be within [0, 1]")
    if step_frac <= 0.0 or step_frac > 1.0:
        raise ValueError("step_frac must be within (0, 1]")

    total_ctw = max(0.0, 1.0 - v_fixed - zr_fixed)
    if total_ctw <= 0.0:
        return pd.DataFrame([{ "V": float(v_fixed), "Cr": 0.0, "Ti": 0.0, "W": 0.0, "Zr": float(zr_fixed) }])

    n = int(round(total_ctw / step_frac))
    if abs(total_ctw - n * step_frac) > 1e-9:
        # Guard against accidental mismatches due to rounding; re-derive step
        step = total_ctw / max(n, 1)
    else:
        step = step_frac

    rows: List[Dict[str, float]] = []
    for i in range(n + 1):
        for j in range(n + 1 - i):
            k = n - i - j
            cr = i * step
            ti = j * step
            w = k * step
            row = {"V": float(v_fixed), "Cr": float(cr), "Ti": float(ti), "W": float(w), "Zr": float(zr_fixed)}
            # Exact closure by construction, but ensure V is balance
            row["V"] = max(0.0, 1.0 - (row["Cr"] + row["Ti"] + row["W"] + row["Zr"]))
            rows.append(row)

    df = pd.DataFrame(rows, columns=["V", "Cr", "Ti", "W", "Zr"])
    
    # Quantize the raw alloying elements to match the workflow's grid step (0.1 at%)
    if df.empty:
        return df
    X_raw = df[["Cr", "Ti", "W", "Zr"]].to_numpy(dtype=float)
    X_quantized = _quantize_raw_floor(X_raw, GRID_STEP)
    
    # Rebuild the DataFrame with quantized values and recompute V as balance
    quantized_rows: List[Dict[str, float]] = []
    for i in range(len(df)):
        cr, ti, w, zr = X_quantized[i]
        v = max(0.0, 1.0 - (cr + ti + w + zr))
        quantized_rows.append({"V": float(v), "Cr": float(cr), "Ti": float(ti), "W": float(w), "Zr": float(zr)})
    
    return pd.DataFrame(quantized_rows, columns=["V", "Cr", "Ti", "W", "Zr"])


def _ternary_xy(cr: np.ndarray, ti: np.ndarray, w: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Map (Cr, Ti, W) barycentric coordinates to 2D Cartesian.

    The mapping uses an equilateral triangle with vertices at (0,0), (1,0),
    and (0.5, sqrt(3)/2). Input values are normalized internally so that
    cr + ti + w = 1 for positioning.

    Parameters
    ----------
    cr : np.ndarray
        Chromium atomic fractions.
    ti : np.ndarray
        Titanium atomic fractions.
    w : np.ndarray
        Tungsten atomic fractions.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        2D Cartesian X and Y arrays.
    """

    pts = np.stack([cr, ti, w], axis=1).astype(float)
    sums = pts.sum(axis=1, keepdims=True)
    sums[sums == 0.0] = 1.0
    abc = pts / sums
    # Using vertices: A=Cr -> (0,0), B=Ti -> (1,0), C=W -> (0.5, sqrt(3)/2)
    x = abc[:, 1] + 0.5 * abc[:, 2]
    y = (np.sqrt(3.0) / 2.0) * abc[:, 2]
    return x, y


def compute_margin_and_mask(
    *,
    lgbm_model: Any,
    X_raw: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute conservative feasibility margin and pass mask using the model.

    For each target, the 0.9-quantile upper bound is compared to the 
    configured limit. The minimum relative margin across targets is used as a
    scalar score per composition; positive values indicate predicted pass.

    Parameters
    ----------
    lgbm_model : Any
        Trained LightGBM quantile ensemble with ``predict_quantiles``.
    X_raw : np.ndarray
        Raw feature matrix of shape (n, 4) with columns [Cr, Ti, W, Zr].

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        A tuple of (margins, pass_mask), where margins are floats and
        pass_mask is a boolean array.
    """

    X_ilr = lgbm_wf.build_feature_matrix_ilr(X_raw)
    preds = lgbm_model.predict_quantiles(X_ilr)

    # Compute per-target margins and aggregate conservatively
    margins_min = np.full(X_raw.shape[0], np.inf, dtype=float)
    pass_mask = np.ones(X_raw.shape[0], dtype=bool)

    for name in getattr(lgbm_model, "target_names", []):
        if name not in lgbm_wf.LIMITS:
            # If the limit is unknown, skip this target
            continue
        ub_log = preds[name][0.9]
        ub_nat = lgbm_wf.inv_log10_transform(ub_log)
        limit = float(lgbm_wf.LIMITS[name])
        # Relative margin (positive is good)
        m = (limit - ub_nat) / max(limit, 1e-30)
        margins_min = np.minimum(margins_min, m)
        pass_mask &= (ub_nat <= limit)

    # Replace inf (if all targets skipped) with zeros
    margins_min[~np.isfinite(margins_min)] = 0.0
    return margins_min, pass_mask


def compute_neutronics_ok(
    Y_nat: np.ndarray,
) -> np.ndarray:
    """Compute neutronics pass mask from natural-scale outputs.

    A composition passes if all targets are below or equal to their limits.

    Parameters
    ----------
    Y_nat : np.ndarray
        Array of shape (n, M) with natural-scale predictions per target
        ordered according to ``lgbm_wf.TARGET_NAMES``.

    Returns
    -------
    np.ndarray
        Boolean mask of shape (n,) indicating compositions that pass all
        neutronics limits.
    """

    if Y_nat.ndim != 2 or Y_nat.shape[1] != len(lgbm_wf.TARGET_NAMES):
        raise ValueError("Y_nat must have shape (n, M) with M = len(TARGET_NAMES)")
    ok = np.ones(Y_nat.shape[0], dtype=bool)
    for j, name in enumerate(lgbm_wf.TARGET_NAMES):
        limit = float(lgbm_wf.LIMITS[name])
        ok &= (Y_nat[:, j] <= limit)
    return ok


def categorize(neutronics_ok: np.ndarray, calphad_ok: np.ndarray) -> np.ndarray:
    """Assign category labels from two boolean masks.

    Categories:
    - 'both': True & True
    - 'neutronics_only': True & False
    - 'calphad_only': False & True
    - 'neither': False & False

    Parameters
    ----------
    neutronics_ok : np.ndarray
        Boolean mask of neutronics-passing compositions.
    calphad_ok : np.ndarray
        Boolean mask of CALPHAD-passing compositions.

    Returns
    -------
    np.ndarray
        Array of string category labels.
    """

    if neutronics_ok.shape != calphad_ok.shape:
        raise ValueError("Mask shapes must match")
    out = np.empty_like(neutronics_ok, dtype=object)
    both = neutronics_ok & calphad_ok
    neut_only = neutronics_ok & (~calphad_ok)
    cal_only = (~neutronics_ok) & calphad_ok
    neither = (~neutronics_ok) & (~calphad_ok)
    out[both] = "both"
    out[neut_only] = "neutronics_only"
    out[cal_only] = "calphad_only"
    out[neither] = "neither"
    return out


def filter_calphad_slice(
    df: pd.DataFrame,
    v_target: float,
    zr_target: float,
    tol_v: float,
    tol_zr: float,
) -> pd.DataFrame:
    """Filter CALPHAD candidates to those near the specified V/Zr slice.

    Parameters
    ----------
    df : pd.DataFrame
        CALPHAD candidates with columns at least ["V","Cr","Ti","W","Zr"].
    v_target : float
        Target V fraction.
    zr_target : float
        Target Zr fraction.
    tol_v : float
        Absolute tolerance for V.
    tol_zr : float
        Absolute tolerance for Zr.

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame containing only near-slice candidates.
    """

    cols = ["V", "Cr", "Ti", "W", "Zr"]
    for c in cols:
        if c not in df.columns:
            raise ValueError(f"CALPHAD CSV missing required column: {c}")
    dv = (df["V"].astype(float) - float(v_target)).abs()
    dz = (df["Zr"].astype(float) - float(zr_target)).abs()
    mask = (dv <= float(tol_v)) & (dz <= float(tol_zr))
    return df.loc[mask].copy()


def render_three_slices(
    *,
    lgbm_model: Any,
    calphad_df: pd.DataFrame,
    slices: Sequence[SliceConfig],
    resolution: int,
    tol_v: float,
    tol_zr: float,
    out_path: Path,
    mode: str = "continuous",
) -> None:
    """Render three side-by-side ternary plots for specified slices.

    Parameters
    ----------
    lgbm_model : Any
        Trained LightGBM model used for predictions.
    calphad_df : pd.DataFrame
        CALPHAD candidates data frame.
    slices : Sequence[SliceConfig]
        Three slice configurations with fixed V and Zr.
    resolution : int
        Ternary grid resolution per axis.
    tol_v : float
        Absolute tolerance for V when selecting CALPHAD points for a slice.
    tol_zr : float
        Absolute tolerance for Zr for the same selection.
    out_path : Path
        Destination path for the saved PNG figure.
    mode : str
        "continuous" to color by conservative feasibility margin using the
        LightGBM model; "categorical" to color by four-way classification
        requires precomputed columns '_neutronics_ok' and '_calphad_ok' on
        the grid DataFrames passed in via 'calphad_df' argument (see real
        evaluation flow).
    """

    # Build all grids and compute margins to set a shared color scale centered at 0
    all_margins: List[np.ndarray] = []
    grids: List[pd.DataFrame] = []
    masks: List[np.ndarray] = []

    for cfg in slices:
        # Continuous mode still uses a resolution-based grid (surrogate visualization)
        grid = make_ternary_grid_two_fixed_step(
            v_fixed=cfg.v_fixed,
            zr_fixed=cfg.zr_fixed,
            step_frac=max(1.0 / max(resolution - 1, 1), 1e-6),
        )
        if mode == "continuous":
            X_raw = grid[["Cr", "Ti", "W", "Zr"]].to_numpy(dtype=float)
            margins, pass_mask = compute_margin_and_mask(lgbm_model=lgbm_model, X_raw=X_raw)
            grid = grid.assign(_margin=margins, _pass=pass_mask)
            all_margins.append(margins)
            masks.append(pass_mask)
        grids.append(grid)

    # Shared color normalization with center at 0 (continuous mode only)
    if mode == "continuous":
        all_vals = np.concatenate(all_margins) if all_margins else np.array([0.0])
        vmin = float(np.nanmin(all_vals))
        vmax = float(np.nanmax(all_vals))
        bound = max(abs(vmin), abs(vmax))
        norm = TwoSlopeNorm(vmin=-bound, vcenter=0.0, vmax=bound if bound > 0 else 1.0)
    else:
        norm = None  # type: ignore

    # Figure
    # Prefer mpltern for publication-quality ternaries when available
    if _HAS_MPLTERN:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True, subplot_kw={"projection": "ternary"})
    else:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)

    for ax, cfg, grid in zip(axes, slices, grids):
        cr = grid["Cr"].to_numpy(dtype=float)
        ti = grid["Ti"].to_numpy(dtype=float)
        w = grid["W"].to_numpy(dtype=float)

        if _HAS_MPLTERN:
            # Normalize to the ternary plane for plotting clarity
            s = np.maximum(cr + ti + w, 1e-12)
            a = cr / s
            b = ti / s
            c = w / s
            if mode == "continuous":
                sc = ax.scatter(a, b, c, c=grid["_margin"].to_numpy(), s=10, cmap="RdYlGn", edgecolors="none")
            elif mode == "phases":
                # Color by phase regions (multi-phase support); fill via nearest assignment
                if "phases" not in grid.columns:
                    raise ValueError("Phases mode requires 'phases' column. Rerun with --redo-calphad or categorical compute.")
                import ast
                # Build per-point phase list filtered by threshold
                phases_list: List[List[str]] = []
                all_phases: set[str] = set()
                for val in grid["phases"].tolist():
                    d = val if isinstance(val, dict) else (ast.literal_eval(val) if isinstance(val, str) else {})
                    keep = [k for k, v in d.items() if float(v) >= 1e-3]
                    phases_list.append(keep if keep else ["<none>"])
                    for k in keep:
                        all_phases.add(str(k))
                if not all_phases:
                    all_phases = {"<none>"}
                uniq = sorted(all_phases)
                base = plt.get_cmap("tab20")(np.linspace(0, 1, 20))
                color_map: Dict[str, str] = {phase: plt.matplotlib.colors.to_hex(base[i % len(base)]) for i, phase in enumerate(uniq)}
                # Assign each point a representative color (first kept phase), for filling
                point_color = [color_map[plist[0]] for plist in phases_list]
                # Render as filled scatter for density; outlines added below by contours
                ax.scatter(a, b, c, c=point_color, s=20, marker="s", edgecolors="none", alpha=0.95)
                # Phase annotations disabled - add manually as needed
            else:
                if ("_neutronics_ok" not in grid.columns) or ("_calphad_ok" not in grid.columns):
                    raise ValueError("Categorical mode requires '_neutronics_ok' and '_calphad_ok' columns")
                cats = categorize(grid["_neutronics_ok"].to_numpy(dtype=bool), grid["_calphad_ok"].to_numpy(dtype=bool))
                cat_to_style = {
                    "both": {"c": "#2ca02c", "marker": "o", "s": 14, "label": "Both"},
                    "neutronics_only": {"c": "#ff7f0e", "marker": "s", "s": 12, "label": "Neutronics only"},
                    "calphad_only": {"c": "#17becf", "marker": "^", "s": 12, "label": "CALPHAD only"},
                    "neither": {"c": "#9e9e9e", "marker": ".", "s": 8, "label": "Neither"},
                }
                draw_order = ["neither", "calphad_only", "neutronics_only", "both"]
                for key in draw_order:
                    m = (cats == key)
                    if not np.any(m):
                        continue
                    style = cat_to_style[key]
                    ax.scatter(a[m], b[m], c[m], c=style["c"], s=style["s"], marker=style["marker"], edgecolors="none", alpha=0.95, label=style["label"])  # type: ignore[arg-type]

            # Overlay neutronics-feasible region as contour/halo when available
            if mode in {"categorical", "phases"} and ("_neutronics_ok" in grid.columns):
                ok = grid["_neutronics_ok"].to_numpy(dtype=bool)
                if np.any(ok):
                    # draw translucent halo
                    ax.scatter(a[ok], b[ok], c[ok], s=36, marker='o', edgecolors='none', alpha=0.15, c="#2ca02c")

            ax.grid(True, alpha=0.6, linewidth=0.8, color='black')
            ax.set_tlabel("Cr")
            ax.set_llabel("Ti")
            ax.set_rlabel("W")
            
            # Make internal grid lines more visible
            ax.taxis.grid(True, alpha=0.3, linewidth=0.5, color='black')
            ax.laxis.grid(True, alpha=0.3, linewidth=0.5, color='black')
            ax.raxis.grid(True, alpha=0.3, linewidth=0.5, color='black')
            
            # Set tick labels to show actual amounts instead of normalized fractions
            total_ctw = max(0.0, 1.0 - cfg.v_fixed - cfg.zr_fixed)
            if total_ctw > 0:
                # Create tick positions and labels for actual amounts
                n_ticks = 6
                tick_positions = np.linspace(0, 1, n_ticks)
                tick_labels = [f"{pos * total_ctw * 100:.1f}" for pos in tick_positions]
                
                ax.set_ticks(tick_positions)
                ax.set_ticklabels(tick_labels)
                ax.set_llimits(0, 1)
                ax.set_rlimits(0, 1)
                ax.set_tlimits(0, 1)
        else:
            # Fallback: barycentric transform in 2D
            x, y = _ternary_xy(cr, ti, w)
            if mode == "continuous":
                sc = ax.scatter(x, y, c=grid["_margin"].to_numpy(), s=10, cmap="RdYlGn", norm=norm, edgecolors="none")
            else:
                if ("_neutronics_ok" not in grid.columns) or ("_calphad_ok" not in grid.columns):
                    raise ValueError("Categorical mode requires '_neutronics_ok' and '_calphad_ok' columns")
                cats = categorize(grid["_neutronics_ok"].to_numpy(dtype=bool), grid["_calphad_ok"].to_numpy(dtype=bool))
                cat_to_style = {
                    "both": {"c": "#2ca02c", "marker": "o", "s": 14, "label": "Both"},
                    "neutronics_only": {"c": "#ff7f0e", "marker": "s", "s": 12, "label": "Neutronics only"},
                    "calphad_only": {"c": "#17becf", "marker": "^", "s": 12, "label": "CALPHAD only"},
                    "neither": {"c": "#9e9e9e", "marker": ".", "s": 8, "label": "Neither"},
                }
                draw_order = ["neither", "calphad_only", "neutronics_only", "both"]
                for key in draw_order:
                    m = (cats == key)
                    if not np.any(m):
                        continue
                    style = cat_to_style[key]
                    ax.scatter(x[m], y[m], c=style["c"], s=style["s"], marker=style["marker"], edgecolors="none", alpha=0.95, label=style["label"])  # type: ignore[arg-type]

            # Draw reference triangle
            tri = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, np.sqrt(3.0) / 2.0], [0.0, 0.0]])
            ax.plot(tri[:, 0], tri[:, 1], "k-", lw=1)
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])

        if (mode == "continuous") and (not calphad_df.empty):
            sub = filter_calphad_slice(calphad_df, v_target=cfg.v_fixed, zr_target=cfg.zr_fixed, tol_v=tol_v, tol_zr=tol_zr)
            if not sub.empty:
                cr_c = sub["Cr"].to_numpy(dtype=float)
                ti_c = sub["Ti"].to_numpy(dtype=float)
                w_c = sub["W"].to_numpy(dtype=float)
                if _HAS_MPLTERN:
                    s_c = np.maximum(cr_c + ti_c + w_c, 1e-12)
                    a_c = cr_c / s_c
                    b_c = ti_c / s_c
                    c_c = w_c / s_c
                    ax.scatter(a_c, b_c, c_c, s=18, c="#1f77b4", marker="o", edgecolors="white", linewidths=0.3, alpha=0.9, label="CALPHAD candidates")
                else:
                    x_c, y_c = _ternary_xy(cr_c, ti_c, w_c)
                    ax.scatter(x_c, y_c, s=18, c="#1f77b4", marker="o", edgecolors="white", linewidths=0.3, alpha=0.9, label="CALPHAD candidates")

        # Title and annotations (no legend)
        total_ctw = max(0.0, 1.0 - cfg.v_fixed - cfg.zr_fixed)
        ax.set_title(f"{cfg.label}\nV={cfg.v_fixed*100:.1f} at%, Zr={cfg.zr_fixed*100:.2f} at% (Cr+Ti+W={total_ctw*100:.1f} at%)")
        
        # Category annotations disabled - add manually as needed

    # Shared colorbar for continuous mode
    if mode == "continuous":
        cbar = fig.colorbar(sc, ax=axes.ravel().tolist(), shrink=0.85, pad=0.02)
        cbar.set_label("Conservative feasibility margin (min over targets)")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def evaluate_slice_real(
    *,
    v_fixed: float,
    zr_fixed: float,
    resolution: int,
    simulator,
    calphad_temperature_k: float,
    calphad_database: str,
    impurity_wtfrac: Optional[Dict[str, float]] = None,
    calphad_workers: int = 32,
) -> pd.DataFrame:
    """Evaluate neutronics and CALPHAD on a slice grid (real computations).

    Parameters
    ----------
    v_fixed : float
        Fixed atomic fraction of V.
    zr_fixed : float
        Fixed atomic fraction of Zr.
    resolution : int
        Ternary grid resolution per axis.
    simulator : Callable[[np.ndarray], np.ndarray]
        Configured depletion simulator function returning Y_nat for X_raw.
    calphad_temperature_k : float
        Temperature in Kelvin for CALPHAD.
    calphad_database : str
        CALPHAD database name (e.g., 'TCHEA8').

    Returns
    -------
    pd.DataFrame
        Grid DataFrame with neutronics and calphad booleans added.
    """

    # Use a step-based grid derived from the ternary closure to ensure clean labeling
    # Note: step_frac is used for initial grid generation, but final compositions
    # are quantized to GRID_STEP (0.1 at%) to match the workflow
    step = 0.005  # 0.5 at% for initial grid density
    grid = make_ternary_grid_two_fixed_step(v_fixed=v_fixed, zr_fixed=zr_fixed, step_frac=step)
    X_raw = grid[["Cr", "Ti", "W", "Zr"]].to_numpy(dtype=float)
    # Run neutronics
    Y_nat = simulator(X_raw)
    
    # Debug: print some statistics about the neutronics results
    print(f"Neutronics results shape: {Y_nat.shape}")
    print(f"Target names: {lgbm_wf.TARGET_NAMES}")
    print(f"Limits: {lgbm_wf.LIMITS}")
    print("Sample Y_nat values (first 5 rows):")
    for i in range(min(5, Y_nat.shape[0])):
        print(f"  Row {i}: {dict(zip(lgbm_wf.TARGET_NAMES, Y_nat[i]))}")
    
    neut_ok = compute_neutronics_ok(Y_nat)
    print(f"Neutronics pass rate: {neut_ok.sum()}/{len(neut_ok)} ({100*neut_ok.mean():.1f}%)")
    # Run CALPHAD across the entire grid (parallel when possible) and keep phases dict
    if int(calphad_workers) > 1:
        cal_df = run_calphad_batch_parallel(
            compositions=grid[["V", "Cr", "Ti", "W", "Zr"]],
            temperature_k=calphad_temperature_k,
            database=calphad_database,
            num_workers=int(calphad_workers),
            chunksize=1,
            impurity_wtfrac=impurity_wtfrac,
        )
    else:
        cal_df = run_calphad_batch(
            grid[["V", "Cr", "Ti", "W", "Zr"]],
            temperature_k=calphad_temperature_k,
            database=calphad_database,
            impurity_wtfrac=impurity_wtfrac,
        )
    cal_ok = compute_calphad_ok(cal_df)
    # Propagate phases column for later plotting; ensure it's present
    if "phases" in cal_df.columns:
        grid = grid.assign(phases=cal_df["phases"].values)
    grid = grid.assign(_neutronics_ok=neut_ok, _calphad_ok=cal_ok)
    return grid


# =====================================================================================
# CLI
# =====================================================================================


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Parameters
    ----------
    argv : Optional[Sequence[str]]
        Optional argument vector (defaults to ``sys.argv``).

    Returns
    -------
    argparse.Namespace
        Parsed arguments.
    """

    p = argparse.ArgumentParser(description="Generate ternary phase diagrams for Cr–Ti–W at fixed V and Zr.")
    p.add_argument("--results-dir", type=str, required=True, help="Directory containing lgbm_model.pkl and (optionally) lightgbm_results.csv")
    p.add_argument("--calphad-csv", type=str, required=False, help="Path to CALPHAD candidates CSV (for continuous mode overlay)")
    p.add_argument("--output-dir", type=str, required=True, help="Output directory to store depletion runs, CALPHAD results, and plots")
    p.add_argument("--plot-name", type=str, default="ternary_three_slices.png", help="Filename for the output PNG (default: ternary_three_slices.png)")
    p.add_argument("--step-at-percent", type=float, default=0.5, help="Grid step in at% for Cr/Ti/W (default: 0.5)")
    p.add_argument("--zr", type=float, default=0.005, help="Fixed Zr atomic fraction for slices (default: 0.005)")
    p.add_argument("--v-values", type=str, default="0.92,0.85,0.80", help="Comma-separated V values for the three slices (default: 0.92,0.85,0.80)")
    p.add_argument("--tol-v", type=float, default=0.003, help="Absolute tolerance for V when selecting CALPHAD points (default: 0.003)")
    p.add_argument("--tol-zr", type=float, default=0.001, help="Absolute tolerance for Zr for CALPHAD selection (default: 0.001)")
    p.add_argument("--no-estimate-cno", action="store_true", help="Disable estimating CNO_FRAC from lightgbm_results.csv")
    # Mode: continuous (LGBM scoring) vs categorical (real depletion+CALPHAD)
    p.add_argument("--mode", type=str, choices=["continuous", "categorical", "phases", "phases_grid"], default="categorical", help="Plot mode: continuous, categorical, phases, or phases_grid")
    # Real-depletion configuration (required for categorical mode)
    p.add_argument("--flux-microxs-dir", type=str, default = '/home/myless/Packages/neutronics_calphad/examples/analysis_results/impulse_library/library/flux_microxs', help="Path to flux_microxs directory")
    p.add_argument("--chain-file", type=str, default = '/home/myless/nuclear_data/tendl21-fispact20-gefy61-chain.xml', help="OpenMC chain XML file path")
    p.add_argument("--abs-file", type=str, default = '/home/myless/Packages/fispact/nuclear_data/decay/abs_2012', help="FISPACT/ABS decay data file path")
    # All outputs are organized under --output-dir
    p.add_argument("--max-parallel-jobs", type=int, default=32, help="Max parallel depletion jobs (default: 32)")
    p.add_argument("--threads-per-process", type=int, default=1, help="OMP threads per depletion process (default: 1)")
    # CALPHAD config
    p.add_argument("--calphad-temperature-k", type=float, default=873.15, help="CALPHAD temperature in K (default: 873.15)")
    p.add_argument("--calphad-database", type=str, default="TCHEA8", help="CALPHAD database (default: TCHEA8)")
    p.add_argument("--calphad-workers", type=int, default=32, help="Parallel workers for CALPHAD (default: 32)")
    p.add_argument("--redo-calphad", action="store_true", help="Re-run CALPHAD for an existing evaluations.csv and update phases/_calphad_ok")
    # Grid/phase-map config (for phases_grid)
    p.add_argument("--grid-config", type=str, help="Path to JSON defining a grid of panels for phases_grid mode")
    # Plot-only mode (skip computations)
    p.add_argument("--plot-only", action="store_true", help="Skip computations and plot from a previously saved evaluations CSV")
    p.add_argument("--evaluations-csv", type=str, help="Path to evaluations CSV to plot (defaults to <output-dir>/evaluations.csv)")
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Entrypoint: load model and data, then render ternary slices.

    Parameters
    ----------
    argv : Optional[Sequence[str]]
        Optional argument vector (defaults to ``sys.argv``).
    """

    args = parse_args(argv)

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    out_path = plots_dir / args.plot_name
    calphad_csv = Path(args.calphad_csv) if getattr(args, "calphad_csv", None) else None

    # Load LightGBM model
    model_path = results_dir / "lgbm_model.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"Missing model file: {model_path}")
    lgbm_model = load_lgbm_model(model_path)

    # Optionally estimate and set CNO_FRAC to match training
    if not bool(args.no_estimate_cno):
        cno_csv = results_dir / "lightgbm_results.csv"
        cno_est = estimate_cno_frac_from_csv(cno_csv)
        if cno_est is not None:
            lgbm_wf.CNO_FRAC = float(cno_est)

    # Load CALPHAD candidates if provided (for continuous overlay)
    calphad_df = pd.read_csv(calphad_csv) if (calphad_csv is not None and calphad_csv.exists()) else pd.DataFrame()

    # Build slice configurations
    try:
        v_values = [float(x.strip()) for x in str(args.v_values).split(",")]
    except Exception:
        v_values = [0.92, 0.85, 0.80]
    # Ensure exactly three slices
    if len(v_values) != 3:
        raise ValueError("--v-values must provide exactly three comma-separated values")

    zr = float(args.zr)
    slices: List[SliceConfig] = [
        SliceConfig(v_fixed=v_values[0], zr_fixed=zr, label=""),
        SliceConfig(v_fixed=v_values[1], zr_fixed=zr, label=""),
        SliceConfig(v_fixed=v_values[2], zr_fixed=zr, label=""),
    ]

    if args.plot_only and (args.mode not in {"phases", "phases_grid"}):
        # Optional: only redo CALPHAD for an existing evaluations.csv
        if args.redo_calphad:
            eval_csv = Path(args.evaluations_csv) if args.evaluations_csv else (output_dir / "evaluations.csv")
            if not eval_csv.exists():
                raise FileNotFoundError(f"Evaluations CSV not found: {eval_csv}")
            df = pd.read_csv(eval_csv)
            # Re-run CALPHAD in parallel for all rows; preserve order
            impurity_wt_percent: Dict[str, float] = {"C": 0.006, "N": 0.012, "O": 0.015}
            impurity_wtfrac: Dict[str, float] = {k: (v / 100.0) for k, v in impurity_wt_percent.items() if float(v) != 0.0}
            compositions = df[["V", "Cr", "Ti", "W", "Zr"]]
            cal_df = run_calphad_batch_parallel(
                compositions=compositions,
                temperature_k=float(args.calphad_temperature_k),
                database=str(args.calphad_database),
                num_workers=int(args.calphad_workers),
                chunksize=1,
                impurity_wtfrac=impurity_wtfrac,
            ) if int(args.calphad_workers) > 1 else run_calphad_batch(
                compositions,
                temperature_k=float(args.calphad_temperature_k),
                database=str(args.calphad_database),
                impurity_wtfrac=impurity_wtfrac,
            )
            # Update phases and _calphad_ok
            if "phases" in cal_df.columns:
                df["phases"] = cal_df["phases"].values
            df["_calphad_ok"] = compute_calphad_ok(cal_df)
            df.to_csv(eval_csv, index=False)
            print(f"Updated CALPHAD results written back to: {eval_csv}")
            # Fall through to plotting from updated CSV
        # Plot from existing evaluations CSV
        eval_csv = Path(args.evaluations_csv) if args.evaluations_csv else (output_dir / "evaluations.csv")
        if not eval_csv.exists():
            raise FileNotFoundError(f"Evaluations CSV not found: {eval_csv}")
        df = pd.read_csv(eval_csv)
        # Rebuild per-slice grids and plot (categorical)
        evaluated_grids: List[pd.DataFrame] = []
        for cfg in slices:
            sub = df[(np.isclose(df["slice_V"], cfg.v_fixed)) & (np.isclose(df["slice_Zr"], cfg.zr_fixed))].copy()
            if sub.empty:
                continue
            evaluated_grids.append(sub)
        fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True, subplot_kw={"projection": "ternary"} if _HAS_MPLTERN else None)
        for ax, cfg, grid in zip(axes, slices, evaluated_grids):
            cr = grid["Cr"].to_numpy(dtype=float)
            ti = grid["Ti"].to_numpy(dtype=float)
            w = grid["W"].to_numpy(dtype=float)
            if _HAS_MPLTERN:
                s = np.maximum(cr + ti + w, 1e-12)
                a, b, c = cr / s, ti / s, w / s
                cats = categorize(grid["_neutronics_ok"].to_numpy(dtype=bool), grid["_calphad_ok"].to_numpy(dtype=bool))
                cat_to_style = {
                    "both": {"c": "#2ca02c", "marker": "o", "s": 14, "label": "Both"},
                    "neutronics_only": {"c": "#ff7f0e", "marker": "s", "s": 12, "label": "Neutronics only"},
                    "calphad_only": {"c": "#17becf", "marker": "^", "s": 12, "label": "CALPHAD only"},
                    "neither": {"c": "#9e9e9e", "marker": ".", "s": 8, "label": "Neither"},
                }
                for key in ["neither", "calphad_only", "neutronics_only", "both"]:
                    m = (cats == key)
                    if not np.any(m):
                        continue
                    st = cat_to_style[key]
                    ax.scatter(a[m], b[m], c[m], c=st["c"], s=st["s"], marker=st["marker"], edgecolors="none", alpha=0.95, label=st["label"])  # type: ignore[arg-type]
                ax.grid(True, alpha=0.6, linewidth=0.8, color='black')
                ax.set_tlabel("Cr")
                ax.set_llabel("Ti")
                ax.set_rlabel("W")
                
                # Make internal grid lines more visible
                ax.taxis.grid(True, alpha=0.6, linewidth=0.8)
                ax.laxis.grid(True, alpha=0.6, linewidth=0.8)
                ax.raxis.grid(True, alpha=0.6, linewidth=0.8)
                
                # Set tick labels to show actual amounts instead of normalized fractions
                total_ctw = max(0.0, 1.0 - cfg.v_fixed - cfg.zr_fixed)
                if total_ctw > 0:
                    # Create tick positions and labels for actual amounts
                    n_ticks = 6
                    tick_positions = np.linspace(0, 1, n_ticks)
                    tick_labels = [f"{pos * total_ctw * 100:.1f}" for pos in tick_positions]
                    
                    ax.taxis.set_ticks(tick_positions)
                    ax.taxis.set_ticklabels(tick_labels)
                    ax.laxis.set_ticks(tick_positions)
                    ax.laxis.set_ticklabels(tick_labels)
                    ax.raxis.set_ticks(tick_positions)
                    ax.raxis.set_ticklabels(tick_labels)
                    ax.set_llim(0, 1)
                    ax.set_rlim(0, 1)
                    ax.set_tlim(0, 1)
            else:
                x, y = _ternary_xy(cr, ti, w)
                cats = categorize(grid["_neutronics_ok"].to_numpy(dtype=bool), grid["_calphad_ok"].to_numpy(dtype=bool))
                cat_to_style = {
                    "both": {"c": "#2ca02c", "marker": "o", "s": 14, "label": "Both"},
                    "neutronics_only": {"c": "#ff7f0e", "marker": "s", "s": 12, "label": "Neutronics only"},
                    "calphad_only": {"c": "#17becf", "marker": "^", "s": 12, "label": "CALPHAD only"},
                    "neither": {"c": "#9e9e9e", "marker": ".", "s": 8, "label": "Neither"},
                }
                for key in ["neither", "calphad_only", "neutronics_only", "both"]:
                    m = (cats == key)
                    if not np.any(m):
                        continue
                    st = cat_to_style[key]
                    ax.scatter(x[m], y[m], c=st["c"], s=st["s"], marker=st["marker"], edgecolors="none", alpha=0.95, label=st["label"])  # type: ignore[arg-type]
                tri = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, np.sqrt(3.0) / 2.0], [0.0, 0.0]])
                ax.plot(tri[:, 0], tri[:, 1], "k-", lw=1)
                ax.set_aspect("equal")
                ax.set_xticks([])
                ax.set_yticks([])
            total_ctw = max(0.0, 1.0 - cfg.v_fixed - cfg.zr_fixed)
            ax.set_title(f"{cfg.label}\nV={cfg.v_fixed*100:.1f} at%, Zr={cfg.zr_fixed*100:.2f} at% (Cr+Ti+W={total_ctw*100:.1f} at%)")
            
            # Category annotations disabled - add manually as needed
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close(fig)
    elif args.mode == "continuous":
        render_three_slices(
            lgbm_model=lgbm_model,
            calphad_df=calphad_df,
            slices=slices,
            resolution=31,
            tol_v=float(args.tol_v),
            tol_zr=float(args.tol_zr),
            out_path=out_path,
            mode="continuous",
        )
        print(f"saved ternary plots to: {out_path}")

    elif args.mode in {"phases", "phases_grid"} and args.plot_only:
        import json as _json
        import ast
        import numpy as _np
        import pandas as _pd
        import matplotlib.pyplot as _plt

                # ---------- phases_grid: plot-only (smooth B2/C15 fills, bigger spacing, ticks on all sides) ----------
        if args.mode == "phases_grid":
            import json as _json
            import ast
            import numpy as _np
            import pandas as _pd
            import matplotlib.pyplot as _plt
            import matplotlib.tri as mtri

            if not args.grid_config:
                raise ValueError("--grid-config is required for phases_grid mode")
            cfg_path = Path(args.grid_config)
            with open(cfg_path, "r") as f:
                grid_cfg = _json.load(f)

            panels = grid_cfg.get("panels", [])
            nrows = int(grid_cfg.get("nrows", 3))
            ncols = int(grid_cfg.get("ncols", 3))

            # ~10% more spacing and a bit larger figure; explicit margins
            subplot_kw = {"projection": "ternary"} if _HAS_MPLTERN else {}
            fig, axes = _plt.subplots(
                nrows, ncols,
                figsize=(7.0 * ncols, 6 * nrows),
                gridspec_kw=dict(
                    left=0.16, right=0.985, bottom=0.16, top=0.9,
                    wspace=0.2, hspace=0.35
                ),
                subplot_kw=subplot_kw
            )
            axes = _np.array(axes).reshape(nrows, ncols)

            # Appearance
            AXIS_LABEL_FONTSIZE = 14
            TICK_FONTSIZE = 12
            _plt.rcParams.update({"font.size": TICK_FONTSIZE})

            # Phase thresholds & colors
            PHASE_THR_PERCENT = 0.5  # presence threshold in %
            #B2_COLOR   = "#88aadd"   # base fill  (B2-only region)
            B2_COLOR   = "#6E8D00"
            #C15_COLOR  = "#1f77b4"   # overlay    (B2 + C15 region)
            C15_COLOR  = "#8F2D56"
            BOUNDARY   = "#222222"
            ALPHA_B2   = 0.58
            ALPHA_C15  = 0.58

            # Helpers ----------------------------------------------------------
            def _parse_phases_cell(val) -> dict[str, float]:
                if isinstance(val, dict):
                    return {str(k): float(v) for k, v in val.items()}
                try:
                    d = ast.literal_eval(val)
                    return {str(k): float(v) for k, v in d.items()} if isinstance(d, dict) else {}
                except Exception:
                    return {}

            def _detect_scale(dicts: list[dict[str, float]]) -> float:
                # Return 100.0 if values look like percents, else 1.0 (fractions)
                sample = []
                for d in dicts:
                    sample += [float(v) for v in d.values() if _np.isfinite(float(v))]
                    if len(sample) >= 300:
                        break
                if not sample:
                    return 100.0
                return 100.0 if float(_np.nanmax(sample)) > 1.0 else 1.0

            # Pass 1: read each panel, pre-compute barycentric coords & z fields
            parsed_panels: list[dict] = []
            c15_keys: set[str] = set()
            b2_keys: set[str]  = set()

            for p in panels:
                r, c = int(p["row"]), int(p["col"])
                ax = axes[r][c]

                df = _pd.read_csv(p["eval_csv"])
                # Optional exact-slice filter if present in CSV
                if "v" in p and "zr" in p and "slice_V" in df.columns and "slice_Zr" in df.columns:
                    v_target, zr_target = float(p["v"]), float(p["zr"])
                    df = df[_np.isclose(df["slice_V"], v_target) & _np.isclose(df["slice_Zr"], zr_target)]

                if df.empty:
                    parsed_panels.append({"ax": ax, "df": df, "r": r, "c": c})
                    continue

                cr = df["Cr"].to_numpy(float)
                ti = df["Ti"].to_numpy(float)
                w = df["W"].to_numpy(float)
                s  = _np.maximum(cr + ti + w, 1e-12)
                a = cr/s
                b = ti/s
                c3 = w/s  # barycentric for mpltern

                # Parse phases and collect keys
                ph_list = df["phases"].apply(_parse_phases_cell).tolist()
                scale = _detect_scale(ph_list)
                thr_raw = PHASE_THR_PERCENT if scale == 100.0 else PHASE_THR_PERCENT / 100.0

                for d in ph_list:
                    for nm in d.keys():
                        sname = str(nm).lower()
                        if "c15" in sname or "laves" in sname:
                            c15_keys.add(str(nm))
                        if "b2" in sname:
                            b2_keys.add(str(nm))

                # Build continuous fields: z_b2, z_c15
                z_b2  = _np.array([sum(float(d.get(k, 0.0)) for k in b2_keys ) for d in ph_list], dtype=float)
                z_c15 = _np.array([sum(float(d.get(k, 0.0)) for k in c15_keys) for d in ph_list], dtype=float)

                # 2D triangulation (for safe triangle selection / clipping)
                x = b + 0.5*c3
                y = (_np.sqrt(3.0)/2.0) * c3
                tri2d = mtri.Triangulation(x, y)

                parsed_panels.append({
                    "ax": ax, "df": df, "r": r, "c": c,
                    "a": a, "b": b, "c3": c3, "tri2d": tri2d,
                    "z_b2": z_b2, "z_c15": z_c15, "thr_raw": thr_raw
                })

            # Pass 2: draw each panel ------------------------------------------
            for pp in parsed_panels:
                ax, df = pp["ax"], pp["df"]
                if df.empty:
                    ax.set_axis_off()
                    continue

                a, b, c3    = pp["a"], pp["b"], pp["c3"]
                tri2d       = pp["tri2d"]
                z_b2, z_c15 = pp["z_b2"], pp["z_c15"]
                thr_raw     = pp["thr_raw"]

                inside_b2 = (z_b2 >= thr_raw)
                if _np.any(inside_b2):
                    # Work on the subset inside B2 to avoid masked values within triangles
                    idx = _np.nonzero(inside_b2)[0]
                    a_in, b_in, c_in = a[idx], b[idx], c3[idx]
                    z_b2_in = z_b2[idx]
                    z_c15_in = z_c15[idx]

                    # Base fill: entire B2≥thr region
                    zmax_b2 = float(_np.nanmax(z_b2_in)) + 1e-12
                    ax.tricontourf(
                        a_in, b_in, c_in, z_b2_in,
                        levels=[thr_raw, zmax_b2],
                        colors=[B2_COLOR], alpha=ALPHA_B2,
                        antialiased=True, zorder=1
                    )
                    # Boundary for B2 threshold
                    ax.tricontour(
                        a_in, b_in, c_in, z_b2_in,
                        levels=[thr_raw], colors=[BOUNDARY],
                        linewidths=1.0,                         zorder=3
                    )

                    # Overlay: C15≥thr, computed within the B2 region only
                    inside_c15 = (z_c15_in >= thr_raw)
                    if _np.any(inside_c15):
                        zmax_c15 = float(_np.nanmax(z_c15_in)) + 1e-12
                        ax.tricontourf(
                            a_in, b_in, c_in, z_c15_in,
                            levels=[thr_raw, zmax_c15],
                            colors=[C15_COLOR], alpha=ALPHA_C15,
                            antialiased=True, zorder=2
                        )
                        # Boundary for C15 threshold
                        ax.tricontour(
                            a_in, b_in, c_in, z_c15_in,
                            levels=[thr_raw], colors=[BOUNDARY],
                            linewidths=1.2, zorder=4
                        )

                # Axes cosmetics, ticks on all sides, bigger fonts
                ax.grid(True, alpha=0.6, linewidth=0.8, color='black')
                ax.set_tlabel("Cr")
                ax.set_llabel("Ti")
                ax.set_rlabel("W")
                
                # Make internal grid lines more visible
                ax.taxis.grid(True, alpha=0.6, linewidth=0.8)
                ax.laxis.grid(True, alpha=0.6, linewidth=0.8)
                ax.raxis.grid(True, alpha=0.6, linewidth=0.8)
                
                ax.taxis.label.set_fontsize(AXIS_LABEL_FONTSIZE)
                ax.laxis.label.set_fontsize(AXIS_LABEL_FONTSIZE)
                ax.raxis.label.set_fontsize(AXIS_LABEL_FONTSIZE)

                total_ctw = float(_np.nanmean(_np.maximum(1.0 - df["V"].astype(float) - df["Zr"].astype(float), 0.0)))
                tick_pos = _np.linspace(0, 1, 6)
                tick_lbl = [f"{p * total_ctw * 100:.1f}" for p in tick_pos]

                ax.taxis.set_ticks(tick_pos)
                ax.taxis.set_ticklabels(tick_lbl)
                ax.laxis.set_ticks(tick_pos)
                ax.laxis.set_ticklabels(tick_lbl)
                ax.raxis.set_ticks(tick_pos)
                ax.raxis.set_ticklabels(tick_lbl)
                ax.set_tlim(0, 1)
                ax.set_llim(0, 1)
                ax.set_rlim(0, 1)

                # Ensure tick label font size on mpltern axes
                for axis in (ax.taxis, ax.laxis, ax.raxis):
                    for txt in axis.get_ticklabels():
                        txt.set_fontsize(TICK_FONTSIZE)

            # Phase annotations disabled - add manually as needed

            # Column headers (V) and row headers (Zr) from your JSON
            v_by_col = {int(p["col"]): float(p["v"]) * 100.0 for p in panels if "v" in p}
            zr_by_row = {int(p["row"]): float(p["zr"]) * 100.0 for p in panels if "zr" in p}

            for c in range(ncols):
                bbox = axes[nrows - 1][c].get_position()
                if c in v_by_col:
                    fig.text(bbox.x0 + bbox.width / 2.0, bbox.y0 - 0.04,
                             f"V = {v_by_col[c]:.0f} at%", ha="center", va="top",
                             fontsize=16, fontweight="bold")
            for r in range(nrows):
                bbox = axes[r][0].get_position()
                if r in zr_by_row:
                    fig.text(bbox.x0 - 0.065, bbox.y0 + bbox.height / 2.0,
                             f"Zr = {zr_by_row[r]:.1f} at%", ha="right", va="center",
                             rotation=90, fontsize=16, fontweight="bold")

            out_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
            _plt.close(fig)
            print(f"Saved ternary plots to: {out_path}")




if __name__ == "__main__":
    main()


