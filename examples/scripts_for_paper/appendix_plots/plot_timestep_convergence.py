#!/usr/bin/env python3
"""Plot timestep convergence metrics from precomputed CSV data."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import pandas as pd


DEFAULT_CSV = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "sweep-data"
    / "timesteps"
    / "timestep_deviation_metrics.csv"
)
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "plots"
CHOSEN_TIMESTEP_DAYS = 30.0


def load_metrics(csv_path: Path) -> pd.DataFrame:
    """Load timestep deviation metrics from CSV.

    Uses only the first block of data when the CSV contains repeated headers
    (e.g. from concatenated runs), so timestep_days and values stay correctly
    associated.

    Args:
        csv_path: Path to the timestep deviation metrics CSV.

    Returns:
        DataFrame with timestep deviation metrics.
    """
    df = pd.read_csv(csv_path)
    # If CSV was concatenated (repeated header), keep only the first block
    header_dup = (df["timestep_days"].astype(str).str.strip() == "timestep_days") & (
        df.index > 0
    )
    if header_dup.any():
        first_dup_idx = header_dup.idxmax()
        df = df.iloc[:first_dup_idx].copy()
    # Coerce to numeric so malformed values (e.g. concatenated numbers) become NaN
    df["relative_deviation"] = pd.to_numeric(df["relative_deviation"], errors="coerce")
    df["timestep_days"] = pd.to_numeric(df["timestep_days"], errors="coerce")
    df = df.dropna(subset=["timestep_days"]).copy()
    df["timestep_days"] = df["timestep_days"].astype(float)
    # One row per (timestep_days, metric, target, material); keep first if duplicates
    key_cols = ["timestep_days", "metric", "target", "material"]
    if df.duplicated(subset=key_cols).any():
        df = df.drop_duplicates(subset=key_cols, keep="first").copy()
    return df


def _pivot_metric(
    df: pd.DataFrame,
    metric: str,
    target_order: List[str],
    max_valid_deviation: float = 1.0,
) -> pd.DataFrame:
    """Pivot metrics by timestep for plotting.

    Args:
        df: Metrics DataFrame.
        metric: Metric name to filter (e.g., "dose_rate", "gas_production").
        target_order: Desired order of targets for columns.
        max_valid_deviation: Maximum valid relative deviation. Values above this
            are treated as invalid (e.g., comparing data at wrong cooling times)
            and replaced with NaN.

    Returns:
        Pivoted DataFrame indexed by timestep with target columns.
    """
    metric_df = df[df["metric"] == metric].copy()
    if metric_df.empty:
        return pd.DataFrame()
    # Filter out clearly invalid deviations (likely from missing/mismatched cooling times)
    metric_df.loc[
        metric_df["relative_deviation"] > max_valid_deviation, "relative_deviation"
    ] = float("nan")
    pivoted = metric_df.pivot_table(
        index="timestep_days",
        columns="target",
        values="relative_deviation",
        aggfunc="mean",
    ).sort_index()
    ordered_cols = [c for c in target_order if c in pivoted.columns]
    ordered_cols += [c for c in pivoted.columns if c not in ordered_cols]
    return pivoted[ordered_cols]


def plot_timestep_convergence(
    df: pd.DataFrame,
    output_dir: Path,
) -> Tuple[Path, Path]:
    """Create a convergence plot for dose and gas deviation metrics.

    Args:
        df: DataFrame containing the timestep deviation metrics.
        output_dir: Directory to save the plots.

    Returns:
        Tuple of (png_path, pdf_path).
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    colors = ["#2A33C3", "#A35D00", "#0B7285", "#8F2D56", "#6E8B00"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    gas_pivot = _pivot_metric(df, "gas_production", ["He", "H"])
    if gas_pivot.empty:
        axes[0].text(0.5, 0.5, "No gas data", ha="center", va="center")
    else:
        for idx, col in enumerate(gas_pivot.columns):
            axes[0].plot(
                gas_pivot.index,
                gas_pivot[col],
                marker="o",
                linewidth=2.2,
                color=colors[idx % len(colors)],
                label=str(col),
            )
        axes[0].axhline(0.02, color="red", linestyle="--", alpha=0.6, linewidth=1.8)
        axes[0].axhline(0.05, color="orange", linestyle="--", alpha=0.6, linewidth=1.8)
        axes[0].axvline(CHOSEN_TIMESTEP_DAYS, color="black", linestyle=":", alpha=0.7)
        axes[0].set_xlabel("Timestep (days)")
        axes[0].set_ylabel("Relative deviation")
        axes[0].set_title("Gas production deviation")
        axes[0].grid(True, alpha=0.3, linestyle="--")
        axes[0].legend(loc="best")

    dose_pivot = _pivot_metric(df, "dose_rate", ["30d", "1y", "5y", "100y"])
    if dose_pivot.empty:
        axes[1].text(0.5, 0.5, "No dose data", ha="center", va="center")
    else:
        for idx, col in enumerate(dose_pivot.columns):
            axes[1].plot(
                dose_pivot.index,
                dose_pivot[col],
                marker="o",
                linewidth=2.2,
                color=colors[idx % len(colors)],
                label=str(col),
            )
        axes[1].axhline(0.02, color="red", linestyle="--", alpha=0.6, linewidth=1.8)
        axes[1].axhline(0.05, color="orange", linestyle="--", alpha=0.6, linewidth=1.8)
        axes[1].axvline(CHOSEN_TIMESTEP_DAYS, color="black", linestyle=":", alpha=0.7)
        axes[1].set_xlabel("Timestep (days)")
        axes[1].set_ylabel("Relative deviation")
        axes[1].set_title("Dose rate deviation")
        axes[1].set_yscale("symlog", linthresh=1e-6)
        axes[1].grid(True, alpha=0.3, linestyle="--")
        axes[1].legend(loc="best")

    fig.suptitle("Timestep convergence metrics", fontsize=14, fontweight="bold")
    fig.tight_layout()

    png_path = output_dir / "timestep_convergence.png"
    pdf_path = output_dir / "timestep_convergence.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    return png_path, pdf_path


def main() -> None:
    """Run the timestep convergence plotter."""
    parser = argparse.ArgumentParser(description="Plot timestep convergence metrics.")
    parser.add_argument("--csv-path", type=str, default=str(DEFAULT_CSV))
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()

    csv_path = Path(args.csv_path)
    output_dir = Path(args.output_dir)
    df = load_metrics(csv_path)
    plot_timestep_convergence(df, output_dir)


if __name__ == "__main__":
    main()
