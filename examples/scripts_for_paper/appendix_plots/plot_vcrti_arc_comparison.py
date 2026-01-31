#!/usr/bin/env python3
"""Plot ARC comparison for V-15Cr-5Ti activity results."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import pandas as pd


DEFAULT_ARC_CSV = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "neutronics-data"
    / "vcrti_arc_results.csv"
)
DEFAULT_REF_CSV = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "neutronics-data"
    / "v15cr5ti_reference_data.csv"
)
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "plots"

# From comparison_results.txt
MODEL_FLUX = 1.93e15
REFERENCE_FLUX = 7.54e14


def load_arc_results(csv_path: Path) -> pd.DataFrame:
    """Load ARC model results from CSV.

    Args:
        csv_path: Path to the ARC results CSV.

    Returns:
        DataFrame with columns: time_years, activity_bqkg, dose_sv_hr.
    """
    return pd.read_csv(csv_path)


def load_reference_results(csv_path: Path) -> pd.DataFrame:
    """Load ARC reference activity data from CSV.

    Args:
        csv_path: Path to the reference CSV (time_years, activity_bqkg).

    Returns:
        DataFrame with columns: time_years, activity_bqkg.
    """
    return pd.read_csv(csv_path, header=None, names=["time_years", "activity_bqkg"])


def plot_arc_comparison(
    arc_df: pd.DataFrame,
    ref_df: pd.DataFrame,
    output_dir: Path,
) -> Tuple[Path, Path]:
    """Create ARC comparison plot for activity.

    Args:
        arc_df: DataFrame with model activity results.
        ref_df: DataFrame with reference activity data.
        output_dir: Output directory for plots.

    Returns:
        Tuple of (png_path, pdf_path).
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.loglog(
        arc_df["time_years"],
        arc_df["activity_bqkg"],
        marker="o",
        linewidth=2.0,
        color="#2A33C3",
        label="Model activity",
    )
    ax.loglog(
        ref_df["time_years"],
        ref_df["activity_bqkg"],
        linestyle="--",
        linewidth=2.0,
        color="#A35D00",
        label="Reference activity",
    )
    ax.set_xlabel("Cooling time (years)")
    ax.set_ylabel("Activity (Bq/kg)")
    ax.set_title("V-15Cr-5Ti activity comparison")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="best")

    flux_ratio = MODEL_FLUX / REFERENCE_FLUX
    ax.text(
        0.02,
        0.02,
        f"Flux ratio (model/ref): {flux_ratio:.3f}",
        transform=ax.transAxes,
        fontsize=9,
        bbox=dict(boxstyle="round", alpha=0.2),
    )

    fig.tight_layout()
    png_path = output_dir / "vcrti_arc_comparison.png"
    pdf_path = output_dir / "vcrti_arc_comparison.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    return png_path, pdf_path


def main() -> None:
    """Run the ARC comparison plotter."""
    parser = argparse.ArgumentParser(description="Plot V-15Cr-5Ti ARC comparison.")
    parser.add_argument("--arc-csv", type=str, default=str(DEFAULT_ARC_CSV))
    parser.add_argument("--ref-csv", type=str, default=str(DEFAULT_REF_CSV))
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()

    arc_df = load_arc_results(Path(args.arc_csv))
    ref_df = load_reference_results(Path(args.ref_csv))
    plot_arc_comparison(arc_df, ref_df, Path(args.output_dir))


if __name__ == "__main__":
    main()
