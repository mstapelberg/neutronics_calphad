#!/usr/bin/env python3
"""Plot particle sweep convergence metrics from precomputed CSV data."""

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
    / "particles"
    / "relative_deviation_metrics.csv"
)
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "plots"


def load_metrics(csv_path: Path) -> pd.DataFrame:
    """Load particle sweep metrics from CSV.

    Args:
        csv_path: Path to the particle sweep metrics CSV.

    Returns:
        DataFrame with particle sweep metrics.
    """
    df = pd.read_csv(csv_path)
    df["histories"] = df["particles"] * df["batches"]
    return df


def fit_loglog(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float]:
    """Fit a log-log line and compute R^2.

    Args:
        x: Positive x values.
        y: Positive y values.

    Returns:
        Tuple of (slope, intercept, r2).
    """
    mask = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    if not np.any(mask):
        return 0.0, 0.0, float("nan")
    xlog = np.log10(x[mask])
    ylog = np.log10(y[mask])
    slope, intercept = np.polyfit(xlog, ylog, 1)
    yhat = slope * xlog + intercept
    ss_res = float(np.sum((ylog - yhat) ** 2))
    ss_tot = float(np.sum((ylog - np.mean(ylog)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return float(slope), float(intercept), r2


def build_series(df: pd.DataFrame) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Build series mapping labels to (histories, deviation) arrays.

    Args:
        df: Metrics DataFrame.

    Returns:
        Mapping of label to (histories, deviation) arrays.
    """
    series: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    flux_df = df[df["metric"] == "flux"].copy()
    if not flux_df.empty:
        series["Flux"] = (flux_df["histories"].to_numpy(), flux_df["relative_deviation"].to_numpy())

    activity_df = df[df["metric"] == "activity_total"].copy()
    for material in sorted(activity_df["material"].dropna().unique()):
        mat_df = activity_df[activity_df["material"] == material]
        label = f"Activity {material}"
        series[label] = (mat_df["histories"].to_numpy(), mat_df["relative_deviation"].to_numpy())

    return series


def plot_particle_sweep(
    df: pd.DataFrame,
    output_dir: Path,
) -> Tuple[Path, Path]:
    """Create log-log particle sweep plot with fitted trend lines.

    Args:
        df: Metrics DataFrame.
        output_dir: Output directory for plots.

    Returns:
        Tuple of (png_path, pdf_path).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    series = build_series(df)

    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    colors = ["#2A33C3", "#A35D00", "#0B7285", "#8F2D56", "#6E8B00"]
    fit_lines: List[str] = []

    for idx, (label, (x, y)) in enumerate(series.items()):
        color = colors[idx % len(colors)]
        ax.scatter(x, y, s=35, color=color, label=label, alpha=0.8)
        slope, intercept, r2 = fit_loglog(x, y)
        if np.isfinite(r2):
            xs = np.linspace(np.nanmin(x[x > 0]), np.nanmax(x[x > 0]), 200)
            ys = 10 ** (intercept + slope * np.log10(xs))
            ax.plot(xs, ys, linewidth=1.6, color=color, alpha=0.9)
            fit_lines.append(f"{label}: slope={slope:.3f}, R^2={r2:.3f}")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Total histories (particles x batches)", fontweight="bold")
    ax.set_ylabel("Relative L2 deviation vs reference", fontweight="bold")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="lower left")

    if fit_lines:
        ax.text(
            0.98,
            0.98,
            "\n".join(fit_lines),
            transform=ax.transAxes,
            va="top",
            ha="right",
            fontsize=9,
            bbox=dict(boxstyle="round", alpha=0.2),
        )

    fig.tight_layout()
    png_path = output_dir / "particle_sweep_convergence.png"
    pdf_path = output_dir / "particle_sweep_convergence.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    return png_path, pdf_path


def main() -> None:
    """Run the particle sweep plotter."""
    parser = argparse.ArgumentParser(description="Plot particle sweep convergence metrics.")
    parser.add_argument("--csv-path", type=str, default=str(DEFAULT_CSV))
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()

    df = load_metrics(Path(args.csv_path))
    plot_particle_sweep(df, Path(args.output_dir))


if __name__ == "__main__":
    main()
