#!/usr/bin/env python3
"""Generate ternary feasibility diagrams varying temperature and V.

Creates a 3×3 grid of ternary phase diagrams with shaded feasibility regions:
- Columns: V = 92, 85, 80 at% (left to right, decreasing)
- Rows: Temperature increasing bottom to top (773K, 823K, 873K)

Layout places V=92% / lowest temperature at bottom-left.
Arrows: UP for increasing temperature, RIGHT for decreasing V.

Feasibility is determined from CALPHAD phase rules using the ``phases`` column.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch

from neutronics_calphad.calphad.feasibility import compute_calphad_ok

try:
    import mpltern  # noqa: F401
    HAS_MPLTERN = True
except ImportError:
    HAS_MPLTERN = False
    print("ERROR: mpltern required. Install with: pip install mpltern")


# =============================================================================
# Colors (matching publication palette)
# =============================================================================

COLORS = {
    "olive": "#6E8D00",  # Feasible
    "rose": "#8F2D56",   # Unfeasible
}

FEASIBLE_COLOR = COLORS["olive"]
UNFEASIBLE_COLOR = COLORS["rose"]
BOUNDARY_COLOR = "#222222"

FILL_ALPHA = 0.75

# =============================================================================
# Style controls (tune here)
# =============================================================================

BASE_FONT_SIZE = 16
AXIS_LABEL_FONTSIZE = 18
TICK_FONTSIZE = 18
PANEL_LABEL_FONTSIZE = 20
HEADER_FONTSIZE = 20
ROW_LABEL_FONTSIZE = 20
ARROW_LABEL_FONTSIZE = 20
LEGEND_FONTSIZE = 22

BOUNDARY_LINEWIDTH = 2.2
GRID_LINEWIDTH = 1.0
GRID_ALPHA = 0.6

ARROW_LINEWIDTH = 3.2
ARROW_HEAD_SCALE = 18

TICK_COUNT = 4

FIG_W_PER_COL = 6.6
FIG_H_PER_ROW = 5.6

LEFT_MARGIN = 0.10
RIGHT_MARGIN = 0.94
BOTTOM_MARGIN = 0.10
TOP_MARGIN = 0.88
WSPACE = 0.16
HSPACE = 0.34

TEMPERATURE_VALUES = [773.0, 823.0, 873.0]


# =============================================================================
# Data handling
# =============================================================================

def load_evaluations(csv_path: Path) -> pd.DataFrame:
    """Load evaluations CSV and compute feasibility mask."""
    df = pd.read_csv(csv_path)
    df = df.copy()
    df["feasible"] = compute_calphad_ok(df)
    return df


def get_slice(df: pd.DataFrame, v: float, temperature_k: float) -> pd.DataFrame:
    """Extract data for a specific V and temperature slice."""
    return df[
        (np.abs(df["slice_V"] - v) < 0.001)
        & (np.abs(df["temperature_k"] - temperature_k) < 1e-6)
    ].copy()


# =============================================================================
# Plotting
# =============================================================================

def plot_panel(
    ax: plt.Axes,
    data: pd.DataFrame,
    v_fixed: float,
    zr_fixed: float,
) -> None:
    """Plot single ternary panel with filled feasibility regions."""
    if data.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        return

    cr = data["Cr"].to_numpy(dtype=float)
    ti = data["Ti"].to_numpy(dtype=float)
    w = data["W"].to_numpy(dtype=float)
    feasible = data["feasible"].to_numpy(dtype=bool)

    # Normalize to ternary simplex
    s = np.maximum(cr + ti + w, 1e-12)
    a = cr / s  # Cr corner
    b = ti / s  # Ti corner
    c = w / s   # W corner

    # Continuous field for feasibility
    z_feas = feasible.astype(float)
    thr = 0.5

    if len(z_feas) < 3:
        colors = np.where(feasible, FEASIBLE_COLOR, UNFEASIBLE_COLOR)
        ax.scatter(
            a,
            b,
            c,
            c=colors,
            s=55,
            marker="o",
            edgecolors=BOUNDARY_COLOR,
            linewidths=1.0,
            alpha=0.85,
            zorder=2,
        )
        return

    ax.tricontourf(
        a,
        b,
        c,
        z_feas,
        levels=[-0.5, thr, 1.5],
        colors=[UNFEASIBLE_COLOR, FEASIBLE_COLOR],
        alpha=FILL_ALPHA,
        zorder=1,
    )

    if np.any(z_feas > thr) and np.any(z_feas < thr):
        ax.tricontour(
            a,
            b,
            c,
            z_feas,
            levels=[thr],
            colors=[BOUNDARY_COLOR],
            linewidths=BOUNDARY_LINEWIDTH,
            zorder=3,
        )

    ax.grid(True, alpha=GRID_ALPHA, linewidth=GRID_LINEWIDTH, color="black")
    ax.taxis.grid(True, alpha=GRID_ALPHA, linewidth=GRID_LINEWIDTH)
    ax.laxis.grid(True, alpha=GRID_ALPHA, linewidth=GRID_LINEWIDTH)
    ax.raxis.grid(True, alpha=GRID_ALPHA, linewidth=GRID_LINEWIDTH)

    total_ctw = max(0.0, 1.0 - v_fixed - zr_fixed)
    max_at = total_ctw * 100

    ax.set_tlabel("Cr", fontsize=AXIS_LABEL_FONTSIZE, fontweight="bold")
    ax.set_llabel("Ti", fontsize=AXIS_LABEL_FONTSIZE, fontweight="bold")
    ax.set_rlabel("W", fontsize=AXIS_LABEL_FONTSIZE, fontweight="bold")

    tick_pos = np.linspace(0.0, 1.0, TICK_COUNT)
    tick_lbl = [f"{p * max_at:.0f}" for p in tick_pos]

    for axis in (ax.taxis, ax.laxis, ax.raxis):
        axis.set_ticks(tick_pos)
        axis.set_ticklabels(tick_lbl)
        for txt in axis.get_ticklabels():
            txt.set_fontsize(TICK_FONTSIZE)

    ax.set_tlim(0, 1)
    ax.set_llim(0, 1)
    ax.set_rlim(0, 1)


def create_figure(
    data: pd.DataFrame,
    output_dir: Path,
    v_values: List[float],
    temperature_values: List[float],
) -> None:
    """Create 3×3 ternary figure varying temperature and V."""
    if not HAS_MPLTERN:
        raise ImportError("mpltern required")

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
            "font.size": BASE_FONT_SIZE,
        }
    )

    nrows = len(temperature_values)
    ncols = len(v_values)

    # Top row should be highest temperature
    temp_by_row = list(reversed(temperature_values))

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(FIG_W_PER_COL * ncols, FIG_H_PER_ROW * nrows),
        subplot_kw={"projection": "ternary"},
        gridspec_kw=dict(
            left=LEFT_MARGIN,
            right=RIGHT_MARGIN,
            bottom=BOTTOM_MARGIN,
            top=TOP_MARGIN,
            wspace=WSPACE,
            hspace=HSPACE,
        ),
    )
    axes = np.array(axes).reshape(nrows, ncols)

    labels = [
        ["(a)", "(b)", "(c)"],
        ["(d)", "(e)", "(f)"],
        ["(g)", "(h)", "(i)"],
    ]

    for row_idx in range(nrows):
        temp_k = temp_by_row[row_idx]
        for col_idx, v in enumerate(v_values):
            ax = axes[row_idx, col_idx]

            slice_data = get_slice(data, v, temp_k)
            zr_val = float(np.median(slice_data["Zr"])) if not slice_data.empty else 0.005

            plot_panel(ax, slice_data, v, zr_val)
            ax.text(
                -0.10,
                1.08,
                labels[row_idx][col_idx],
                transform=ax.transAxes,
                fontsize=PANEL_LABEL_FONTSIZE,
                fontweight="bold",
                va="top",
                ha="left",
            )

    # Column headers (V values) at bottom
    for col_idx, v in enumerate(v_values):
        bbox = axes[nrows - 1, col_idx].get_position()
        fig.text(
            bbox.x0 + bbox.width / 2,
            bbox.y0 - 0.045,
            f"V = {v*100:.0f} at%",
            ha="center",
            va="top",
            fontsize=HEADER_FONTSIZE,
            fontweight="bold",
        )

    # Row labels (Temperature) on left
    for row_idx in range(nrows):
        temp_k = temp_by_row[row_idx]
        bbox = axes[row_idx, 0].get_position()
        fig.text(
            bbox.x0 - 0.055,
            bbox.y0 + bbox.height / 2,
            f"T = {temp_k:.0f} K",
            ha="right",
            va="center",
            fontsize=ROW_LABEL_FONTSIZE,
            fontweight="bold",
            rotation=90,
        )

    arrow_props = dict(
        arrowstyle="-|>,head_width=0.45,head_length=0.35",
        color="black",
        lw=ARROW_LINEWIDTH,
        mutation_scale=ARROW_HEAD_SCALE,
    )

    # V arrow: horizontal at top, pointing RIGHT (decreasing V)
    fig.patches.append(
        FancyArrowPatch((0.20, 0.935), (0.80, 0.935), transform=fig.transFigure, **arrow_props)
    )
    fig.text(
        0.50,
        0.96,
        "Decreasing V content",
        ha="center",
        va="bottom",
        fontsize=ARROW_LABEL_FONTSIZE,
        fontweight="bold",
    )

    # Temperature arrow: vertical on right, pointing UP
    fig.patches.append(
        FancyArrowPatch((0.955, 0.20), (0.955, 0.80), transform=fig.transFigure, **arrow_props)
    )
    fig.text(
        0.975,
        0.50,
        "Increasing Temperature",
        ha="left",
        va="center",
        fontsize=ARROW_LABEL_FONTSIZE,
        fontweight="bold",
        rotation=90,
    )

    legend_elements = [
        mpatches.Patch(
            facecolor=FEASIBLE_COLOR,
            edgecolor=BOUNDARY_COLOR,
            linewidth=BOUNDARY_LINEWIDTH,
            alpha=FILL_ALPHA,
            label="Feasible",
        ),
        mpatches.Patch(
            facecolor=UNFEASIBLE_COLOR,
            edgecolor="none",
            alpha=FILL_ALPHA,
            label="Unfeasible",
        ),
    ]

    fig.legend(
        handles=legend_elements,
        loc="upper left",
        bbox_to_anchor=(0.06, 1.02),
        fontsize=LEGEND_FONTSIZE,
        frameon=True,
        fancybox=True,
        framealpha=0.95,
        edgecolor="0.5",
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    for fmt in ["pdf", "png"]:
        path = output_dir / f"temperature_ternary_feasibility.{fmt}"
        fig.savefig(path, format=fmt, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"Saved: {path}")

    plt.close(fig)


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    """Entry point."""
    parser = argparse.ArgumentParser(description="Generate temperature ternary feasibility figure.")

    script_dir = Path(__file__).parent
    default_csv = script_dir.parent / "data" / "ternary-plot-data" / "v-vs-temperature" / "evaluations.csv"

    parser.add_argument("--csv-path", type=str, default=str(default_csv))
    parser.add_argument("--output-dir", type=str, default=str(script_dir / "plots"))
    parser.add_argument("--v-values", type=str, default="0.92,0.85,0.80")

    args = parser.parse_args()

    csv_path = Path(args.csv_path)
    output_dir = Path(args.output_dir)
    v_values = [float(x.strip()) for x in args.v_values.split(",")]

    data = load_evaluations(csv_path)
    create_figure(data, output_dir, v_values, TEMPERATURE_VALUES)


if __name__ == "__main__":
    main()
