#!/usr/bin/env python3
"""Plot geometry sweep convergence metrics from precomputed CSV data.

This script visualizes the sensitivity of neutronics outputs (flux, activity,
gas production) to geometry parameters (first wall and vessel thickness).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_CSV = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "sweep-data"
    / "geometry"
    / "relative_deviation_metrics.csv"
)
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "plots"

# Reference geometry used in the sweep
REFERENCE_FW_CM = 0.2
REFERENCE_VESSEL_CM = 1.0

# Standard color scheme
COLORS = ["#2A33C3", "#A35D00", "#0B7285", "#8F2D56", "#6E8B00"]


def load_metrics(csv_path: Path) -> pd.DataFrame:
    """Load geometry sweep metrics from CSV.

    Args:
        csv_path: Path to the geometry sweep metrics CSV.

    Returns:
        DataFrame with geometry sweep metrics.
    """
    df = pd.read_csv(csv_path)
    df["relative_deviation"] = pd.to_numeric(df["relative_deviation"], errors="coerce")
    df["first_wall_cm"] = pd.to_numeric(df["first_wall_cm"], errors="coerce")
    df["vessel_cm"] = pd.to_numeric(df["vessel_cm"], errors="coerce")
    return df


def get_unique_geometries(
    df: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract unique first wall and vessel thicknesses.

    Args:
        df: Metrics DataFrame.

    Returns:
        Tuple of (first_wall_thicknesses, vessel_thicknesses) as sorted arrays.
    """
    fw_vals = np.sort(df["first_wall_cm"].dropna().unique())
    vessel_vals = np.sort(df["vessel_cm"].dropna().unique())
    return fw_vals, vessel_vals


def pivot_to_grid(
    df: pd.DataFrame,
    metric: str,
    species: str | None = None,
    fw_vals: np.ndarray | None = None,
    vessel_vals: np.ndarray | None = None,
) -> np.ndarray:
    """Pivot data into a 2D grid (vessel x first_wall).

    Args:
        df: Metrics DataFrame.
        metric: Metric name to filter.
        species: Optional species filter (for gas_production).
        fw_vals: First wall thickness values (determines column order).
        vessel_vals: Vessel thickness values (determines row order).

    Returns:
        2D numpy array with shape (len(vessel_vals), len(fw_vals)).
    """
    subset = df[df["metric"] == metric].copy()
    if species is not None:
        subset = subset[subset["species"] == species]

    if fw_vals is None or vessel_vals is None:
        fw_vals, vessel_vals = get_unique_geometries(df)

    grid = np.full((len(vessel_vals), len(fw_vals)), np.nan)

    for i, vessel_cm in enumerate(vessel_vals):
        for j, fw_cm in enumerate(fw_vals):
            mask = (subset["first_wall_cm"] == fw_cm) & (subset["vessel_cm"] == vessel_cm)
            vals = subset.loc[mask, "relative_deviation"]
            if not vals.empty:
                grid[i, j] = vals.iloc[0]

    return grid


def plot_heatmap(
    ax: plt.Axes,
    grid: np.ndarray,
    fw_vals: np.ndarray,
    vessel_vals: np.ndarray,
    title: str,
    cmap: str = "RdYlGn_r",
    vmin: float | None = None,
    vmax: float | None = None,
) -> plt.cm.ScalarMappable:
    """Plot a heatmap of relative deviation on given axes.

    Args:
        ax: Matplotlib axes.
        grid: 2D array of values (vessel x first_wall).
        fw_vals: First wall thickness values.
        vessel_vals: Vessel thickness values.
        title: Title for the subplot.
        cmap: Colormap name.
        vmin: Minimum value for color scale.
        vmax: Maximum value for color scale.

    Returns:
        ScalarMappable for colorbar.
    """
    im = ax.imshow(grid, aspect="auto", cmap=cmap, origin="lower", vmin=vmin, vmax=vmax)

    ax.set_xticks(range(len(fw_vals)))
    ax.set_yticks(range(len(vessel_vals)))
    ax.set_xticklabels([f"{t:.1f}" for t in fw_vals], fontsize=10)
    ax.set_yticklabels([f"{t:.1f}" for t in vessel_vals], fontsize=10)
    ax.set_xlabel("First Wall Thickness (cm)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Vessel Thickness (cm)", fontsize=11, fontweight="bold")
    ax.set_title(title, fontsize=12, fontweight="bold")

    # Add text annotations
    for i in range(len(vessel_vals)):
        for j in range(len(fw_vals)):
            val = grid[i, j]
            if np.isfinite(val):
                # Highlight reference geometry
                is_ref = (
                    fw_vals[j] == REFERENCE_FW_CM and vessel_vals[i] == REFERENCE_VESSEL_CM
                )
                text_color = "white" if is_ref else "black"
                fontweight = "bold" if is_ref else "normal"
                ax.text(
                    j,
                    i,
                    f"{val:.3f}",
                    ha="center",
                    va="center",
                    color=text_color,
                    fontsize=9,
                    fontweight=fontweight,
                )

    # Mark reference geometry with a border
    ref_j = np.where(fw_vals == REFERENCE_FW_CM)[0]
    ref_i = np.where(vessel_vals == REFERENCE_VESSEL_CM)[0]
    if len(ref_j) > 0 and len(ref_i) > 0:
        rect = plt.Rectangle(
            (ref_j[0] - 0.5, ref_i[0] - 0.5),
            1,
            1,
            linewidth=3,
            edgecolor="black",
            facecolor="none",
        )
        ax.add_patch(rect)

    return im


def plot_geometry_sweep_heatmaps(
    df: pd.DataFrame,
    output_dir: Path,
) -> Tuple[Path, Path]:
    """Create 2x2 heatmap showing geometry sensitivity for all metrics.

    Args:
        df: Metrics DataFrame.
        output_dir: Directory to save plots.

    Returns:
        Tuple of (png_path, pdf_path).
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    fw_vals, vessel_vals = get_unique_geometries(df)

    # Create grids for each metric
    flux_grid = pivot_to_grid(df, "flux", fw_vals=fw_vals, vessel_vals=vessel_vals)
    activity_grid = pivot_to_grid(
        df, "activity_total", fw_vals=fw_vals, vessel_vals=vessel_vals
    )
    he_grid = pivot_to_grid(
        df, "gas_production", species="He", fw_vals=fw_vals, vessel_vals=vessel_vals
    )
    h_grid = pivot_to_grid(
        df, "gas_production", species="H", fw_vals=fw_vals, vessel_vals=vessel_vals
    )

    # Determine common color scale bounds (excluding flux which has different scale)
    all_dev_vals = np.concatenate(
        [activity_grid.ravel(), he_grid.ravel(), h_grid.ravel()]
    )
    all_dev_vals = all_dev_vals[np.isfinite(all_dev_vals)]
    if len(all_dev_vals) > 0:
        dev_vmax = np.max(all_dev_vals)
    else:
        dev_vmax = 0.3

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    ax1, ax2, ax3, ax4 = axes.flatten()

    # Flux heatmap (separate scale since flux deviation can be larger)
    im1 = plot_heatmap(
        ax1,
        flux_grid,
        fw_vals,
        vessel_vals,
        "Flux Spectrum Relative Deviation",
        vmin=0,
        vmax=np.nanmax(flux_grid) if np.any(np.isfinite(flux_grid)) else 4.0,
    )
    plt.colorbar(im1, ax=ax1, label="Relative Deviation", shrink=0.8)

    # Activity heatmap
    im2 = plot_heatmap(
        ax2,
        activity_grid,
        fw_vals,
        vessel_vals,
        "Total Activity Relative Deviation",
        vmin=0,
        vmax=dev_vmax,
    )
    plt.colorbar(im2, ax=ax2, label="Relative Deviation", shrink=0.8)

    # He production heatmap
    im3 = plot_heatmap(
        ax3,
        he_grid,
        fw_vals,
        vessel_vals,
        "He Production Relative Deviation",
        vmin=0,
        vmax=dev_vmax,
    )
    plt.colorbar(im3, ax=ax3, label="Relative Deviation", shrink=0.8)

    # H production heatmap
    im4 = plot_heatmap(
        ax4,
        h_grid,
        fw_vals,
        vessel_vals,
        "H Production Relative Deviation",
        vmin=0,
        vmax=dev_vmax,
    )
    plt.colorbar(im4, ax=ax4, label="Relative Deviation", shrink=0.8)

    fig.suptitle(
        f"Geometry Sensitivity (Reference: {REFERENCE_FW_CM} cm FW, "
        f"{REFERENCE_VESSEL_CM} cm Vessel)",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout()

    png_path = output_dir / "geometry_sweep_heatmap.png"
    pdf_path = output_dir / "geometry_sweep_heatmap.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    return png_path, pdf_path


def build_line_series(
    df: pd.DataFrame,
) -> Dict[str, Dict[float, Tuple[np.ndarray, np.ndarray]]]:
    """Build series for line plots: deviation vs vessel thickness for each FW.

    Args:
        df: Metrics DataFrame.

    Returns:
        Nested dict: metric -> first_wall_cm -> (vessel_vals, deviation_vals).
    """
    fw_vals, vessel_vals = get_unique_geometries(df)
    series: Dict[str, Dict[float, Tuple[np.ndarray, np.ndarray]]] = {}

    for metric in ["flux", "activity_total"]:
        series[metric] = {}
        subset = df[df["metric"] == metric]
        for fw_cm in fw_vals:
            fw_subset = subset[subset["first_wall_cm"] == fw_cm].sort_values("vessel_cm")
            if not fw_subset.empty:
                series[metric][fw_cm] = (
                    fw_subset["vessel_cm"].to_numpy(),
                    fw_subset["relative_deviation"].to_numpy(),
                )

    # Gas production (combined He and H)
    for species in ["He", "H"]:
        key = f"gas_{species}"
        series[key] = {}
        subset = df[(df["metric"] == "gas_production") & (df["species"] == species)]
        for fw_cm in fw_vals:
            fw_subset = subset[subset["first_wall_cm"] == fw_cm].sort_values("vessel_cm")
            if not fw_subset.empty:
                series[key][fw_cm] = (
                    fw_subset["vessel_cm"].to_numpy(),
                    fw_subset["relative_deviation"].to_numpy(),
                )

    return series


def plot_geometry_sweep_lines(
    df: pd.DataFrame,
    output_dir: Path,
) -> Tuple[Path, Path]:
    """Create line plots showing deviation vs vessel thickness for each FW.

    Args:
        df: Metrics DataFrame.
        output_dir: Directory to save plots.

    Returns:
        Tuple of (png_path, pdf_path).
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    series = build_line_series(df)
    fw_vals, _ = get_unique_geometries(df)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    ax1, ax2, ax3, ax4 = axes.flatten()

    metric_axes = [
        ("flux", ax1, "Flux Spectrum"),
        ("activity_total", ax2, "Total Activity"),
        ("gas_He", ax3, "He Production"),
        ("gas_H", ax4, "H Production"),
    ]

    for metric, ax, title in metric_axes:
        if metric not in series:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", fontsize=12)
            ax.set_title(title, fontsize=12, fontweight="bold")
            continue

        for idx, fw_cm in enumerate(fw_vals):
            if fw_cm not in series[metric]:
                continue
            vessel_vals, dev_vals = series[metric][fw_cm]
            color = COLORS[idx % len(COLORS)]
            marker = "o" if fw_cm == REFERENCE_FW_CM else "s"
            linewidth = 2.5 if fw_cm == REFERENCE_FW_CM else 1.8
            ax.plot(
                vessel_vals,
                dev_vals,
                marker=marker,
                linewidth=linewidth,
                color=color,
                label=f"FW = {fw_cm:.1f} cm",
                alpha=0.9,
            )

        # Reference lines for thresholds
        ax.axhline(0.02, color="red", linestyle="--", alpha=0.6, linewidth=1.5, label="2%")
        ax.axhline(0.05, color="orange", linestyle="--", alpha=0.6, linewidth=1.5, label="5%")

        # Mark reference vessel thickness
        ax.axvline(
            REFERENCE_VESSEL_CM,
            color="black",
            linestyle=":",
            alpha=0.7,
            linewidth=1.5,
        )

        ax.set_xlabel("Vessel Thickness (cm)", fontsize=11, fontweight="bold")
        ax.set_ylabel("Relative Deviation", fontsize=11, fontweight="bold")
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.legend(loc="best", fontsize=9)

    fig.suptitle(
        "Geometry Sweep: Relative Deviation vs Vessel Thickness",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout()

    png_path = output_dir / "geometry_sweep_lines.png"
    pdf_path = output_dir / "geometry_sweep_lines.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    return png_path, pdf_path


def plot_summary_bar(
    df: pd.DataFrame,
    output_dir: Path,
) -> Tuple[Path, Path]:
    """Create a summary bar chart of max deviations by metric.

    Args:
        df: Metrics DataFrame.
        output_dir: Directory to save plots.

    Returns:
        Tuple of (png_path, pdf_path).
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics = {
        "Flux": df[df["metric"] == "flux"]["relative_deviation"].max(),
        "Activity": df[df["metric"] == "activity_total"]["relative_deviation"].max(),
        "He (gas)": df[(df["metric"] == "gas_production") & (df["species"] == "He")][
            "relative_deviation"
        ].max(),
        "H (gas)": df[(df["metric"] == "gas_production") & (df["species"] == "H")][
            "relative_deviation"
        ].max(),
    }

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(metrics))
    bars = ax.bar(x, list(metrics.values()), color=COLORS[: len(metrics)], alpha=0.85)

    # Add value labels on bars
    for bar, val in zip(bars, metrics.values()):
        if np.isfinite(val):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{val:.3f}",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
            )

    ax.axhline(0.02, color="red", linestyle="--", alpha=0.7, linewidth=2, label="2% threshold")
    ax.axhline(0.05, color="orange", linestyle="--", alpha=0.7, linewidth=2, label="5% threshold")

    ax.set_xticks(x)
    ax.set_xticklabels(list(metrics.keys()), fontsize=11)
    ax.set_ylabel("Maximum Relative Deviation", fontsize=12, fontweight="bold")
    ax.set_title("Geometry Sensitivity Summary", fontsize=14, fontweight="bold")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()

    png_path = output_dir / "geometry_sweep_summary.png"
    pdf_path = output_dir / "geometry_sweep_summary.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    return png_path, pdf_path


def main() -> None:
    """Run the geometry sweep plotter."""
    parser = argparse.ArgumentParser(description="Plot geometry sweep convergence metrics.")
    parser.add_argument("--csv-path", type=str, default=str(DEFAULT_CSV))
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument(
        "--plot-type",
        type=str,
        choices=["heatmap", "lines", "summary", "all"],
        default="all",
        help="Type of plot to generate.",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv_path)
    output_dir = Path(args.output_dir)

    df = load_metrics(csv_path)

    if args.plot_type in ("heatmap", "all"):
        png1, pdf1 = plot_geometry_sweep_heatmaps(df, output_dir)
        print(f"Heatmap saved to: {png1}")

    if args.plot_type in ("lines", "all"):
        png2, pdf2 = plot_geometry_sweep_lines(df, output_dir)
        print(f"Line plots saved to: {png2}")

    if args.plot_type in ("summary", "all"):
        png3, pdf3 = plot_summary_bar(df, output_dir)
        print(f"Summary bar chart saved to: {png3}")


if __name__ == "__main__":
    main()
