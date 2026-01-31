"""Generate ternary phase diagrams for Cr–Ti–W at fixed V and varying temperature.

This example script creates three side-by-side ternary plots over the
Cr–Ti–W simplex with the following fixed compositions (atomic fractions):

- Panel 1: V = 0.92, Zr = 0.005, T = 773K  → (Cr + Ti + W) = 0.075
- Panel 2: V = 0.85, Zr = 0.005, T = 823K  → (Cr + Ti + W) = 0.145  
- Panel 3: V = 0.80, Zr = 0.005, T = 873K  → (Cr + Ti + W) = 0.195

For each panel, a grid of (Cr, Ti, W) points is evaluated using real neutronics
and CALPHAD calculations at the specified temperature to show temperature
sensitivity of the results.

Inputs (CLI):
- Path to a directory containing a pickled LightGBM model (lgbm_model.pkl).
- Path to CALPHAD candidate CSV (optional, for overlay).
- Output directory for results and plots.

Outputs:
- A PNG figure with three ternary plots saved to the specified output path.
- Evaluations CSV with neutronics and CALPHAD results for each temperature.

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
class TemperatureSliceConfig:
    """Configuration for a ternary slice with fixed V, Zr, and temperature.

    Attributes:
        v_fixed: Fixed atomic fraction of V.
        zr_fixed: Fixed atomic fraction of Zr.
        temperature_k: Temperature in Kelvin for CALPHAD calculations.
        label: Human-readable label for the subplot title.
    """

    v_fixed: float
    zr_fixed: float
    temperature_k: float
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


def evaluate_slice_real(
    *,
    v_fixed: float,
    zr_fixed: float,
    temperature_k: float,
    resolution: int,
    simulator,
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
    temperature_k : float
        Temperature in Kelvin for CALPHAD.
    resolution : int
        Ternary grid resolution per axis.
    simulator : Callable[[np.ndarray], np.ndarray]
        Configured depletion simulator function returning Y_nat for X_raw.
    calphad_database : str
        CALPHAD database name (e.g., 'TCHEA8').

    Returns
    -------
    pd.DataFrame
        Grid DataFrame with neutronics and calphad booleans added.
    """

    # Use a step-based grid derived from the ternary closure to ensure clean labeling
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
            temperature_k=temperature_k,
            database=calphad_database,
            num_workers=int(calphad_workers),
            chunksize=1,
            impurity_wtfrac=impurity_wtfrac,
        )
    else:
        cal_df = run_calphad_batch(
            grid[["V", "Cr", "Ti", "W", "Zr"]],
            temperature_k=temperature_k,
            database=calphad_database,
            impurity_wtfrac=impurity_wtfrac,
        )
    cal_ok = compute_calphad_ok(cal_df)
    
    # Propagate phases column for later plotting; ensure it's present
    if "phases" in cal_df.columns:
        grid = grid.assign(phases=cal_df["phases"].values)
    grid = grid.assign(_neutronics_ok=neut_ok, _calphad_ok=cal_ok)
    
    # Add temperature and slice information
    grid = grid.assign(
        slice_V=v_fixed,
        slice_Zr=zr_fixed,
        temperature_k=temperature_k
    )
    
    return grid


def evaluate_v_temp_grid_calphad_only(
    *,
    v_values: Sequence[float],
    zr_fixed: float,
    temperatures_k: Sequence[float],
    step_at_percent: float,
    calphad_database: str,
    calphad_workers: int,
    impurity_wtfrac: Optional[Dict[str, float]] = None,
) -> pd.DataFrame:
    """Evaluate CALPHAD only for a V×Temperature grid of ternary slices.

    This function builds a quantized Cr–Ti–W grid for each ``v`` in ``v_values``
    at a fixed ``zr_fixed``. It then runs CALPHAD for each specified temperature
    over the same compositions and aggregates results into a single DataFrame.

    Parameters
    ----------
    v_values : Sequence[float]
        Sequence of fixed V atomic fractions (e.g., [0.92, 0.85, 0.80]).
    zr_fixed : float
        Fixed atomic fraction of Zr shared by all panels.
    temperatures_k : Sequence[float]
        Sequence of temperatures in Kelvin (e.g., [773.0, 823.0, 873.0]).
    step_at_percent : float
        Grid step for Cr/Ti/W in at% (e.g., 0.5 for 0.5 at%).
    calphad_database : str
        CALPHAD database name (e.g., 'TCHEA8').
    calphad_workers : int
        Number of parallel workers for CALPHAD.
    impurity_wtfrac : Optional[Dict[str, float]]
        Optional impurity mass fractions.

    Returns
    -------
    pd.DataFrame
        Combined DataFrame containing columns ["V","Cr","Ti","W","Zr"],
        a dict-like 'phases' column, and metadata columns 'slice_V',
        'slice_Zr', and 'temperature_k'.
    """

    all_rows: List[pd.DataFrame] = []
    step_frac = float(step_at_percent) / 100.0

    for v in v_values:
        grid = make_ternary_grid_two_fixed_step(
            v_fixed=float(v), zr_fixed=float(zr_fixed), step_frac=step_frac
        )
        compositions = grid[["V", "Cr", "Ti", "W", "Zr"]]

        for T in temperatures_k:
            if int(calphad_workers) > 1:
                cal_df = run_calphad_batch_parallel(
                    compositions=compositions,
                    temperature_k=float(T),
                    database=str(calphad_database),
                    num_workers=int(calphad_workers),
                    chunksize=1,
                    impurity_wtfrac=impurity_wtfrac,
                )
            else:
                cal_df = run_calphad_batch(
                    compositions,
                    temperature_k=float(T),
                    database=str(calphad_database),
                    impurity_wtfrac=impurity_wtfrac,
                )

            out = grid.copy()
            if "phases" in cal_df.columns:
                out = out.assign(phases=cal_df["phases"].values)
            out = out.assign(
                slice_V=float(v),
                slice_Zr=float(zr_fixed),
                temperature_k=float(T),
            )
            all_rows.append(out)

    if not all_rows:
        return pd.DataFrame(columns=["V", "Cr", "Ti", "W", "Zr", "phases", "slice_V", "slice_Zr", "temperature_k"])

    return pd.concat(all_rows, ignore_index=True)


def render_v_temp_phases_grid(
    *,
    df: pd.DataFrame,
    v_values: Sequence[float],
    temperatures_k: Sequence[float],
    zr_fixed: float,
    out_path: Path,
) -> None:
    """Render a 3×3 mpltern grid: rows=temperatures, cols=V values, phases fill.

    The plotting style matches the 'phases_grid' mode in the original
    ``ternary_phase_diagrams.py``: smooth fills for B2≥0.5% (base) and
    C15≥0.5% (overlay), with threshold boundaries, ticks on all sides,
    and consistent fonts/spacing.

    Parameters
    ----------
    df : pd.DataFrame
        Combined evaluations with columns ["V","Cr","Ti","W","Zr","phases",
        "slice_V","slice_Zr","temperature_k"].
    v_values : Sequence[float]
        Column categories (V) in display order.
    temperatures_k : Sequence[float]
        Row categories (temperatures) in display order.
    zr_fixed : float
        Fixed Zr atomic fraction used in the evaluations.
    out_path : Path
        Where to save the rendered PNG.
    """

    import ast as _ast
    import numpy as _np
    import matplotlib.pyplot as _plt
    import matplotlib.tri as mtri  # noqa: F401  # for potential 2D fallback or triangle ops

    nrows = len(temperatures_k)
    ncols = len(v_values)

    subplot_kw = {"projection": "ternary"} if _HAS_MPLTERN else {}
    fig, axes = _plt.subplots(
        nrows,
        ncols,
        figsize=(7.0 * ncols, 6 * nrows),
        gridspec_kw=dict(left=0.16, right=0.985, bottom=0.16, top=0.9, wspace=0.2, hspace=0.35),
        subplot_kw=subplot_kw,
    )
    axes = _np.array(axes).reshape(nrows, ncols)

    AXIS_LABEL_FONTSIZE = 14
    TICK_FONTSIZE = 12
    _plt.rcParams.update({"font.size": TICK_FONTSIZE})

    PHASE_THR_PERCENT = 0.5  # presence threshold in %
    B2_COLOR = "#6E8D00"
    C15_COLOR = "#8F2D56"
    BOUNDARY = "#222222"
    ALPHA_B2 = 0.58
    ALPHA_C15 = 0.58

    def _parse_phases_cell(val: Any) -> Dict[str, float]:
        if isinstance(val, dict):
            return {str(k): float(v) for k, v in val.items()}
        try:
            d = _ast.literal_eval(val)
            return {str(k): float(v) for k, v in d.items()} if isinstance(d, dict) else {}
        except Exception:
            return {}

    def _detect_scale(dicts: List[Dict[str, float]]) -> float:
        sample: List[float] = []
        for d in dicts:
            for v in d.values():
                try:
                    sample.append(float(v))
                except Exception:
                    continue
            if len(sample) >= 300:
                break
        if not sample:
            return 100.0
        return 100.0 if float(_np.nanmax(sample)) > 1.0 else 1.0

    # Draw each panel
    for r, T in enumerate(temperatures_k):
        for c, v in enumerate(v_values):
            ax = axes[r][c]
            sub = df[
                _np.isclose(df["slice_V"].astype(float), float(v))
                & _np.isclose(df["slice_Zr"].astype(float), float(zr_fixed))
                & _np.isclose(df["temperature_k"].astype(float), float(T))
            ].copy()

            if sub.empty:
                ax.set_axis_off()
                continue

            cr = sub["Cr"].to_numpy(float)
            ti = sub["Ti"].to_numpy(float)
            w = sub["W"].to_numpy(float)
            s = _np.maximum(cr + ti + w, 1e-12)
            a = cr / s
            b = ti / s
            c3 = w / s

            ph_list = sub["phases"].apply(_parse_phases_cell).tolist()
            scale = _detect_scale(ph_list)
            thr_raw = PHASE_THR_PERCENT if scale == 100.0 else PHASE_THR_PERCENT / 100.0

            c15_keys: set[str] = set()
            b2_keys: set[str] = set()
            for d in ph_list:
                for nm in d.keys():
                    sname = str(nm).lower()
                    if "c15" in sname or "laves" in sname:
                        c15_keys.add(str(nm))
                    if "b2" in sname:
                        b2_keys.add(str(nm))

            z_b2 = _np.array([sum(float(d.get(k, 0.0)) for k in b2_keys) for d in ph_list], dtype=float)
            z_c15 = _np.array([sum(float(d.get(k, 0.0)) for k in c15_keys) for d in ph_list], dtype=float)

            inside_b2 = z_b2 >= thr_raw
            if _np.any(inside_b2):
                idx = _np.nonzero(inside_b2)[0]
                a_in, b_in, c_in = a[idx], b[idx], c3[idx]
                z_b2_in = z_b2[idx]
                z_c15_in = z_c15[idx]

                zmax_b2 = float(_np.nanmax(z_b2_in)) + 1e-12
                ax.tricontourf(
                    a_in,
                    b_in,
                    c_in,
                    z_b2_in,
                    levels=[thr_raw, zmax_b2],
                    colors=[B2_COLOR],
                    alpha=ALPHA_B2,
                    antialiased=True,
                    zorder=1,
                )
                ax.tricontour(
                    a_in,
                    b_in,
                    c_in,
                    z_b2_in,
                    levels=[thr_raw],
                    colors=[BOUNDARY],
                    linewidths=1.0,
                    zorder=3,
                )

                inside_c15 = z_c15_in >= thr_raw
                if _np.any(inside_c15):
                    zmax_c15 = float(_np.nanmax(z_c15_in)) + 1e-12
                    ax.tricontourf(
                        a_in,
                        b_in,
                        c_in,
                        z_c15_in,
                        levels=[thr_raw, zmax_c15],
                        colors=[C15_COLOR],
                        alpha=ALPHA_C15,
                        antialiased=True,
                        zorder=2,
                    )
                    ax.tricontour(
                        a_in,
                        b_in,
                        c_in,
                        z_c15_in,
                        levels=[thr_raw],
                        colors=[BOUNDARY],
                        linewidths=1.2,
                        zorder=4,
                    )

            # Axes cosmetics
            ax.grid(True, alpha=0.6, linewidth=0.8, color="black")
            ax.set_tlabel("Cr")
            ax.set_llabel("Ti")
            ax.set_rlabel("W")

            ax.taxis.grid(True, alpha=0.6, linewidth=0.8)
            ax.laxis.grid(True, alpha=0.6, linewidth=0.8)
            ax.raxis.grid(True, alpha=0.6, linewidth=0.8)

            ax.taxis.label.set_fontsize(AXIS_LABEL_FONTSIZE)
            ax.laxis.label.set_fontsize(AXIS_LABEL_FONTSIZE)
            ax.raxis.label.set_fontsize(AXIS_LABEL_FONTSIZE)

            total_ctw = float(
                _np.nanmean(_np.maximum(1.0 - sub["V"].astype(float) - sub["Zr"].astype(float), 0.0))
            )
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

            for axis in (ax.taxis, ax.laxis, ax.raxis):
                for txt in axis.get_ticklabels():
                    txt.set_fontsize(TICK_FONTSIZE)

    # Column headers (V) and row headers (Temperature)
    v_by_col = {c: float(v) * 100.0 for c, v in enumerate(v_values)}
    t_by_row = {r: float(T) for r, T in enumerate(temperatures_k)}

    for c in range(ncols):
        bbox = axes[nrows - 1][c].get_position()
        if c in v_by_col:
            fig.text(
                bbox.x0 + bbox.width / 2.0,
                bbox.y0 - 0.04,
                f"V = {v_by_col[c]:.0f} at%",
                ha="center",
                va="top",
                fontsize=16,
                fontweight="bold",
            )
    for r in range(nrows):
        bbox = axes[r][0].get_position()
        if r in t_by_row:
            fig.text(
                bbox.x0 - 0.065,
                bbox.y0 + bbox.height / 2.0,
                f"T = {t_by_row[r]:.0f} K",
                ha="right",
                va="center",
                rotation=90,
                fontsize=16,
                fontweight="bold",
            )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    _plt.close(fig)


def render_three_temperature_slices(
    *,
    evaluated_grids: List[pd.DataFrame],
    slices: Sequence[TemperatureSliceConfig],
    out_path: Path,
) -> None:
    """Render three side-by-side ternary plots for temperature sensitivity.

    Parameters
    ----------
    evaluated_grids : List[pd.DataFrame]
        List of evaluated grid DataFrames with neutronics and CALPHAD results.
    slices : Sequence[TemperatureSliceConfig]
        Three slice configurations with fixed V, Zr, and temperature.
    out_path : Path
        Destination path for the saved PNG figure.
    """

    # Figure
    # Prefer mpltern for publication-quality ternaries when available
    if _HAS_MPLTERN:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True, subplot_kw={"projection": "ternary"})
    else:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)

    for ax, cfg, grid in zip(axes, slices, evaluated_grids):
        cr = grid["Cr"].to_numpy(dtype=float)
        ti = grid["Ti"].to_numpy(dtype=float)
        w = grid["W"].to_numpy(dtype=float)

        if _HAS_MPLTERN:
            # Normalize to the ternary plane for plotting clarity
            s = np.maximum(cr + ti + w, 1e-12)
            a = cr / s
            b = ti / s
            c = w / s
            
            # Color by four-way classification
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
                
                ax.taxis.set_ticks(tick_positions)
                ax.taxis.set_ticklabels(tick_labels)
                ax.laxis.set_ticks(tick_positions)
                ax.laxis.set_ticklabels(tick_labels)
                ax.raxis.set_ticks(tick_positions)
                ax.raxis.set_ticklabels(tick_labels)
                ax.set_tlim(0, 1)
                ax.set_llim(0, 1)
                ax.set_rlim(0, 1)
        else:
            # Fallback: barycentric transform in 2D
            x, y = _ternary_xy(cr, ti, w)
            
            # Color by four-way classification
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

        # Title and annotations
        total_ctw = max(0.0, 1.0 - cfg.v_fixed - cfg.zr_fixed)
        ax.set_title(f"T = {cfg.temperature_k:.0f}K\nV={cfg.v_fixed*100:.1f} at%, Zr={cfg.zr_fixed*100:.2f} at% (Cr+Ti+W={total_ctw*100:.1f} at%)")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


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

    p = argparse.ArgumentParser(description="Generate ternary phase diagrams for Cr–Ti–W at fixed V and varying temperature.")
    p.add_argument("--results-dir", type=str, required=True, help="Directory containing lgbm_model.pkl and (optionally) lightgbm_results.csv")
    p.add_argument("--calphad-csv", type=str, required=False, help="Path to CALPHAD candidates CSV (for overlay)")
    p.add_argument("--output-dir", type=str, required=True, help="Output directory to store depletion runs, CALPHAD results, and plots")
    p.add_argument("--plot-name", type=str, default="ternary_temperature_sensitivity.png", help="Filename for the output PNG (default: ternary_temperature_sensitivity.png)")
    p.add_argument("--step-at-percent", type=float, default=0.5, help="Grid step in at% for Cr/Ti/W (default: 0.5)")
    p.add_argument("--zr", type=float, default=0.005, help="Fixed Zr atomic fraction for slices (default: 0.005)")
    p.add_argument("--v-values", type=str, default="0.92,0.85,0.80", help="Comma-separated V values for the three slices (default: 0.92,0.85,0.80)")
    p.add_argument("--temperatures", type=str, default="773,823,873", help="Comma-separated temperatures in K for the three slices (default: 773,823,873)")
    p.add_argument("--no-estimate-cno", action="store_true", help="Disable estimating CNO_FRAC from lightgbm_results.csv")
    p.add_argument("--mode", type=str, choices=["categorical", "phases_grid"], default="phases_grid", help="Plot mode: categorical (1x3 classification) or phases_grid (3x3 B2/C15 fills)")
    
    # Real-depletion configuration
    p.add_argument("--flux-microxs-dir", type=str, default = '/home/myless/Packages/neutronics_calphad/examples/analysis_results/impulse_library/library/flux_microxs', help="Path to flux_microxs directory")
    p.add_argument("--chain-file", type=str, default = '/home/myless/nuclear_data/tendl21-fispact20-gefy61-chain.xml', help="OpenMC chain XML file path")
    p.add_argument("--abs-file", type=str, default = '/home/myless/Packages/fispact/nuclear_data/decay/abs_2012', help="FISPACT/ABS decay data file path")
    
    # All outputs are organized under --output-dir
    p.add_argument("--max-parallel-jobs", type=int, default=32, help="Max parallel depletion jobs (default: 32)")
    p.add_argument("--threads-per-process", type=int, default=1, help="OMP threads per depletion process (default: 1)")
    
    # CALPHAD config
    p.add_argument("--calphad-database", type=str, default="TCHEA8", help="CALPHAD database (default: TCHEA8)")
    p.add_argument("--calphad-workers", type=int, default=32, help="Parallel workers for CALPHAD (default: 32)")
    
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
    # Note: CALPHAD CSV overlay is not used in this script; argument retained for parity

    # Load LightGBM model
    model_path = results_dir / "lgbm_model.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"Missing model file: {model_path}")
    _lgbm_model = load_lgbm_model(model_path)

    # Optionally estimate and set CNO_FRAC to match training
    if not bool(args.no_estimate_cno):
        cno_csv = results_dir / "lightgbm_results.csv"
        cno_est = estimate_cno_frac_from_csv(cno_csv)
        if cno_est is not None:
            lgbm_wf.CNO_FRAC = float(cno_est)

    # Overlay not used; skip reading CALPHAD CSV to avoid unused variable

    # Build slice configurations
    try:
        v_values = [float(x.strip()) for x in str(args.v_values).split(",")]
    except Exception:
        v_values = [0.92, 0.85, 0.80]
    
    try:
        temperatures = [float(x.strip()) for x in str(args.temperatures).split(",")]
    except Exception:
        temperatures = [773.0, 823.0, 873.0]
    
    # Ensure exactly three slices
    if len(v_values) != 3:
        raise ValueError("--v-values must provide exactly three comma-separated values")
    if len(temperatures) != 3:
        raise ValueError("--temperatures must provide exactly three comma-separated values")

    zr = float(args.zr)
    slices: List[TemperatureSliceConfig] = [
        TemperatureSliceConfig(v_fixed=v_values[0], zr_fixed=zr, temperature_k=temperatures[0], label=""),
        TemperatureSliceConfig(v_fixed=v_values[1], zr_fixed=zr, temperature_k=temperatures[1], label=""),
        TemperatureSliceConfig(v_fixed=v_values[2], zr_fixed=zr, temperature_k=temperatures[2], label=""),
    ]

    if args.mode == "phases_grid":
        # 3x3 grid: rows=temperatures, cols=V values. CALPHAD-only per temperature.
        if args.plot_only:
            eval_csv = Path(args.evaluations_csv) if args.evaluations_csv else (output_dir / "evaluations.csv")
            if not eval_csv.exists():
                raise FileNotFoundError(f"Evaluations CSV not found: {eval_csv}")
            df = pd.read_csv(eval_csv)
            render_v_temp_phases_grid(
                df=df,
                v_values=v_values,
                temperatures_k=temperatures,
                zr_fixed=zr,
                out_path=out_path,
            )
            print(f"Saved ternary plots to: {out_path}")
        else:
            # Impurity configuration (reused defaults)
            impurity_wt_percent: Dict[str, float] = {"C": 0.006, "N": 0.012, "O": 0.015}
            impurity_wtfrac: Dict[str, float] = {k: (v / 100.0) for k, v in impurity_wt_percent.items() if float(v) != 0.0}

            combined = evaluate_v_temp_grid_calphad_only(
                v_values=v_values,
                zr_fixed=zr,
                temperatures_k=temperatures,
                step_at_percent=float(args.step_at_percent),
                calphad_database=str(args.calphad_database),
                calphad_workers=int(args.calphad_workers),
                impurity_wtfrac=impurity_wtfrac,
            )
            eval_csv = output_dir / "evaluations.csv"
            combined.to_csv(eval_csv, index=False)
            print(f"Saved evaluations to: {eval_csv}")
            render_v_temp_phases_grid(
                df=combined,
                v_values=v_values,
                temperatures_k=temperatures,
                zr_fixed=zr,
                out_path=out_path,
            )
            print(f"Saved ternary plots to: {out_path}")

    elif args.plot_only:
        # Plot from existing evaluations CSV
        eval_csv = Path(args.evaluations_csv) if args.evaluations_csv else (output_dir / "evaluations.csv")
        if not eval_csv.exists():
            raise FileNotFoundError(f"Evaluations CSV not found: {eval_csv}")
        df = pd.read_csv(eval_csv)
        
        # Rebuild per-slice grids and plot
        evaluated_grids: List[pd.DataFrame] = []
        for cfg in slices:
            sub = df[
                (np.isclose(df["slice_V"], cfg.v_fixed)) & 
                (np.isclose(df["slice_Zr"], cfg.zr_fixed)) &
                (np.isclose(df["temperature_k"], cfg.temperature_k))
            ].copy()
            if sub.empty:
                continue
            evaluated_grids.append(sub)
        
        render_three_temperature_slices(
            evaluated_grids=evaluated_grids,
            slices=slices,
            out_path=out_path,
        )
        print(f"Saved ternary plots to: {out_path}")
    else:
        # Run real computations for each temperature slice
        from neutronics_calphad.workflows.lightgbm_simulator import create_simulator_from_config
        
        # Impurity configuration
        impurity_wt_percent: Dict[str, float] = {"C": 0.006, "N": 0.012, "O": 0.015}
        impurity_wtfrac: Dict[str, float] = {k: (v / 100.0) for k, v in impurity_wt_percent.items() if float(v) != 0.0}
        
        # Create depletion simulator (matches original workflow wiring)
        simulator = create_simulator_from_config(
            flux_microxs_dir=Path(args.flux_microxs_dir),
            chain_file=str(args.chain_file),
            abs_file=str(args.abs_file),
            output_dir=output_dir / "depletion_runs",
            dose_times_h=lgbm_wf.DOSE_TIMES_H,
            max_parallel_jobs=int(args.max_parallel_jobs),
            threads_per_process=int(args.threads_per_process),
            impurity_wtfrac=impurity_wtfrac,
        )
        
        # Evaluate each slice
        all_evaluations: List[pd.DataFrame] = []
        for cfg in slices:
            print(f"\nEvaluating slice: V={cfg.v_fixed:.2f}, Zr={cfg.zr_fixed:.3f}, T={cfg.temperature_k:.0f}K")
            grid = evaluate_slice_real(
                v_fixed=cfg.v_fixed,
                zr_fixed=cfg.zr_fixed,
                temperature_k=cfg.temperature_k,
                resolution=31,
                simulator=simulator,
                calphad_database=str(args.calphad_database),
                impurity_wtfrac=impurity_wtfrac,
                calphad_workers=int(args.calphad_workers),
            )
            all_evaluations.append(grid)
        
        # Combine all evaluations
        combined_df = pd.concat(all_evaluations, ignore_index=True)
        eval_csv = output_dir / "evaluations.csv"
        combined_df.to_csv(eval_csv, index=False)
        print(f"Saved evaluations to: {eval_csv}")
        
        # Render plots
        render_three_temperature_slices(
            evaluated_grids=all_evaluations,
            slices=slices,
            out_path=out_path,
        )
        print(f"Saved ternary plots to: {out_path}")


if __name__ == "__main__":
    main()
