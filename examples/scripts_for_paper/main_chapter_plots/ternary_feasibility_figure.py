#!/usr/bin/env python3
"""Generate publication-quality ternary feasibility diagrams for V-Cr-Ti-W-Zr.

Creates a 3×3 grid of ternary phase diagrams with shaded feasibility regions:
- Columns: V = 92, 85, 80 at% (left to right, decreasing)
- Rows: Zr = 0.5, 1.0, 1.5 at% (bottom to top, increasing)

Layout places V=92%/Zr=0.5% at bottom-left (most feasible region).
Arrows: UP for increasing Zr, RIGHT for decreasing V.

Feasibility = (_neutronics_ok AND _calphad_ok) from pre-computed evaluations.
Key finding: Feasible region shrinks dramatically as Zr increases.
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
    'olive': '#6E8D00',   # Feasible
    'rose': '#8F2D56',    # Unfeasible
}

FEASIBLE_COLOR = COLORS['olive']
UNFEASIBLE_COLOR = COLORS['rose']
BOUNDARY_COLOR = '#222222'

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


# =============================================================================
# Data handling
# =============================================================================

def load_evaluations(data_dir: Path) -> pd.DataFrame:
    """Load all evaluation CSVs and combine."""
    dfs = []
    for csv_file in sorted(data_dir.glob('evaluations_Zr*.csv')):
        df = pd.read_csv(csv_file)
        dfs.append(df)
        print(f"  Loaded {csv_file.name}: {len(df)} points")
    
    if not dfs:
        raise FileNotFoundError(f"No evaluation CSVs in {data_dir}")
    
    combined = pd.concat(dfs, ignore_index=True)
    combined['feasible'] = combined['_neutronics_ok'] & combined['_calphad_ok']
    return combined


def get_slice(df: pd.DataFrame, v: float, zr: float) -> pd.DataFrame:
    """Extract data for a specific V/Zr slice."""
    return df[
        (np.abs(df['slice_V'] - v) < 0.001) &
        (np.abs(df['slice_Zr'] - zr) < 0.001)
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
    """Plot single ternary panel with filled feasibility regions.
    
    Uses the same approach as ternary_phase_diagrams.py:
    subset filtering + tricontourf for clean shading.
    """
    if data.empty:
        ax.text(0.5, 0.5, "No data", ha='center', va='center', transform=ax.transAxes)
        return
    
    cr = data['Cr'].to_numpy(dtype=float)
    ti = data['Ti'].to_numpy(dtype=float)
    w = data['W'].to_numpy(dtype=float)
    feasible = data['feasible'].to_numpy(dtype=bool)
    
    # Normalize to ternary simplex
    s = np.maximum(cr + ti + w, 1e-12)
    a = cr / s  # Cr corner
    b = ti / s  # Ti corner
    c = w / s   # W corner
    
    # Create continuous field for feasibility (1.0 = feasible, 0.0 = not)
    z_feas = feasible.astype(float)
    
    # Threshold for contour fill
    THR = 0.5
    
    # Minimum points needed for triangulation
    n_points = len(z_feas)
    if n_points < 3:
        # Too few points for contour, fall back to scatter
        colors = np.where(feasible, FEASIBLE_COLOR, UNFEASIBLE_COLOR)
        ax.scatter(a, b, c, c=colors, s=55, marker='o',
                   edgecolors=BOUNDARY_COLOR, linewidths=1.0,
                   alpha=0.85, zorder=2)
        return
    
    # Filled regions from a single field to avoid gaps
    ax.tricontourf(
        a, b, c, z_feas,
        levels=[-0.5, THR, 1.5],
        colors=[UNFEASIBLE_COLOR, FEASIBLE_COLOR],
        alpha=FILL_ALPHA,
        zorder=1,
    )
    
    # Boundary line at interface (only if both regions exist)
    if np.any(z_feas > THR) and np.any(z_feas < THR):
        ax.tricontour(
            a, b, c, z_feas,
            levels=[THR],
            colors=[BOUNDARY_COLOR],
            linewidths=BOUNDARY_LINEWIDTH,
            zorder=3,
        )
    
    # Grid styling (matching ternary_phase_diagrams.py)
    ax.grid(True, alpha=GRID_ALPHA, linewidth=GRID_LINEWIDTH, color='black')
    ax.taxis.grid(True, alpha=GRID_ALPHA, linewidth=GRID_LINEWIDTH)
    ax.laxis.grid(True, alpha=GRID_ALPHA, linewidth=GRID_LINEWIDTH)
    ax.raxis.grid(True, alpha=GRID_ALPHA, linewidth=GRID_LINEWIDTH)
    
    # Corner labels with actual at% range
    total_ctw = max(0.0, 1.0 - v_fixed - zr_fixed)
    max_at = total_ctw * 100
    
    ax.set_tlabel("Cr", fontsize=AXIS_LABEL_FONTSIZE, fontweight='bold')
    ax.set_llabel("Ti", fontsize=AXIS_LABEL_FONTSIZE, fontweight='bold')
    ax.set_rlabel("W", fontsize=AXIS_LABEL_FONTSIZE, fontweight='bold')
    
    # Simplified ticks: 4 ticks across the range
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
    zr_values: List[float],
) -> None:
    """Create 3×3 ternary figure.
    
    Layout:
    - Columns (left to right): V decreasing (92% → 85% → 80%)
    - Rows (bottom to top): Zr increasing (0.5% → 1.0% → 1.5%)
    - Bottom-left = V=92%, Zr=0.5% (most feasible)
    """
    if not HAS_MPLTERN:
        raise ImportError("mpltern required")
    
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
        'font.size': BASE_FONT_SIZE,
    })
    
    nrows = len(zr_values)
    ncols = len(v_values)
    
    # Top row should be highest Zr; bottom row lowest Zr
    zr_by_row = list(reversed(zr_values))
    
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(FIG_W_PER_COL * ncols, FIG_H_PER_ROW * nrows),
        subplot_kw={'projection': 'ternary'},
        gridspec_kw=dict(
            left=LEFT_MARGIN, right=RIGHT_MARGIN,
            bottom=BOTTOM_MARGIN, top=TOP_MARGIN,
            wspace=WSPACE, hspace=HSPACE
        ),
    )
    axes = np.array(axes).reshape(nrows, ncols)
    
    # Panel labels (a-i), row 1 to row 3
    labels = [
        ['(a)', '(b)', '(c)'],  # top row
        ['(d)', '(e)', '(f)'],  # middle row
        ['(g)', '(h)', '(i)'],  # bottom row
    ]
    
    # Plot each panel
    for row_idx in range(nrows):
        # row_idx=0 is top of figure, row_idx=2 is bottom
        zr = zr_by_row[row_idx]
        
        for col_idx, v in enumerate(v_values):
            ax = axes[row_idx, col_idx]
            
            slice_data = get_slice(data, v, zr)
            n_feas = slice_data['feasible'].sum() if not slice_data.empty else 0
            n_total = len(slice_data)
            pct = 100 * n_feas / n_total if n_total > 0 else 0
            
            print(f"  [{labels[row_idx][col_idx]}] V={v*100:.0f}%, Zr={zr*100:.1f}%: "
                  f"{n_feas}/{n_total} feasible ({pct:.1f}%)")
            
            plot_panel(ax, slice_data, v, zr)
            
            # Panel label
            ax.text(-0.10, 1.08, labels[row_idx][col_idx], transform=ax.transAxes,
                    fontsize=PANEL_LABEL_FONTSIZE, fontweight='bold', va='top', ha='left')
    
    # Column headers (V values) - at BOTTOM
    for col_idx, v in enumerate(v_values):
        bbox = axes[nrows-1, col_idx].get_position()
        fig.text(
            bbox.x0 + bbox.width / 2, bbox.y0 - 0.045,
            f"V = {v*100:.0f} at%",
            ha='center', va='top',
            fontsize=HEADER_FONTSIZE, fontweight='bold'
        )
    
    # Row labels (Zr values) - on LEFT, reversed for bottom-to-top
    for row_idx in range(nrows):
        zr = zr_by_row[row_idx]
        bbox = axes[row_idx, 0].get_position()
        fig.text(
            bbox.x0 - 0.055, bbox.y0 + bbox.height / 2,
            f"Zr = {zr*100:.1f} at%",
            ha='right', va='center',
            fontsize=ROW_LABEL_FONTSIZE, fontweight='bold',
            rotation=90
        )
    
    # Arrows
    arrow_props = dict(
        arrowstyle='-|>,head_width=0.45,head_length=0.35',
        color='black',
        lw=ARROW_LINEWIDTH,
        mutation_scale=ARROW_HEAD_SCALE,
    )
    
    # V arrow: horizontal at top, pointing RIGHT (decreasing V)
    fig.patches.append(FancyArrowPatch(
        (0.20, 0.935), (0.80, 0.935),
        transform=fig.transFigure,
        **arrow_props,
    ))
    fig.text(0.50, 0.96, "Decreasing V content", ha='center', va='bottom',
             fontsize=ARROW_LABEL_FONTSIZE, fontweight='bold')
    
    # Zr arrow: vertical on right, pointing UP (increasing Zr)
    fig.patches.append(FancyArrowPatch(
        (0.955, 0.20), (0.955, 0.80),
        transform=fig.transFigure,
        **arrow_props,
    ))
    fig.text(0.975, 0.50, "Increasing Zr", ha='left', va='center',
             fontsize=ARROW_LABEL_FONTSIZE, fontweight='bold', rotation=90)
    
    # Legend
    legend_elements = [
        mpatches.Patch(facecolor=FEASIBLE_COLOR, edgecolor=BOUNDARY_COLOR,
                      linewidth=BOUNDARY_LINEWIDTH, alpha=FILL_ALPHA, label='Feasible'),
        mpatches.Patch(facecolor=UNFEASIBLE_COLOR, edgecolor='none',
                      alpha=FILL_ALPHA, label='Unfeasible'),
    ]
    
    fig.legend(
        handles=legend_elements,
        loc='upper left',
        bbox_to_anchor=(0.06, 1.02),
        fontsize=LEGEND_FONTSIZE,
        frameon=True,
        fancybox=True,
        framealpha=0.95,
        edgecolor='0.5',
    )
    
    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for fmt in ['pdf', 'png']:
        path = output_dir / f'ternary_feasibility_figure.{fmt}'
        fig.savefig(path, format=fmt, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved: {path}")
    
    plt.close(fig)


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate ternary feasibility figure.")
    
    script_dir = Path(__file__).parent
    default_data = script_dir.parent / 'data' / 'ternary-plot-data' / 'v-vs-zr'
    
    parser.add_argument('--data-dir', type=str, default=str(default_data))
    parser.add_argument('--output-dir', type=str, default=str(script_dir / 'plots'))
    parser.add_argument('--v-values', type=str, default='0.92,0.85,0.80')
    parser.add_argument('--zr-values', type=str, default='0.005,0.010,0.015')
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    v_values = [float(x.strip()) for x in args.v_values.split(',')]
    zr_values = [float(x.strip()) for x in args.zr_values.split(',')]
    
    print(f"Loading data from: {data_dir}")
    data = load_evaluations(data_dir)
    print(f"Total: {len(data)} points, {data['feasible'].sum()} feasible "
          f"({100*data['feasible'].mean():.1f}%)\n")
    
    create_figure(data, output_dir, v_values, zr_values)
    print("\nDone!")


if __name__ == '__main__':
    main()
