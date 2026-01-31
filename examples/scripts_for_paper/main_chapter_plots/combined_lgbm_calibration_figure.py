#!/usr/bin/env python3
"""Combined LightGBM Calibration and Performance Figure.

This script generates a publication-quality combined figure showing:
- (a) Coverage before/after CQR calibration for Q90
- (b) Cumulative discovery curve for screening performance
- (c-h) Parity plots for all 6 activation/transmutation limits

The script can operate in two modes:
1. Recompute from raw predictions CSV (for full reproducibility)
2. Load pre-computed metrics (for users without model access)

Usage:
    # Mode 1: Recompute from raw data
    python combined_lgbm_calibration_figure.py --predictions_csv path/to/predictions.csv

    # Mode 2: Use pre-computed data
    python combined_lgbm_calibration_figure.py --precomputed_dir path/to/analysis_outputs/

Author: Generated for publication
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from sklearn.metrics import mean_absolute_error, r2_score

# =============================================================================
# Publication Style Configuration
# =============================================================================

# Color palette (consistent with other publication figures)
COLORS = {
    'blue': '#2A33C3',
    'rose': '#8F2D56',
    'amber': '#A35D00',
    'teal': '#0B7285',
    'olive': '#6E8D00',
}

# Task display names for clean axis labels
TASK_DISPLAY_NAMES = {
    'dose_d30': 'Dose (30 days)',
    'dose_d365': 'Dose (1 year)',
    'dose_d1825': 'Dose (5 years)',
    'dose_d36500': 'Dose (100 years)',
    'He_2y': 'He (2 years)',
    'H_2y': 'H (2 years)',
}

# Ordered list of tasks for consistent plotting
TASKS = ['dose_d30', 'dose_d365', 'dose_d1825', 'dose_d36500', 'He_2y', 'H_2y']

# Default limits for pass/fail screening
DEFAULT_LIMITS = {
    'dose_d30': 1000,
    'dose_d365': 1,
    'dose_d1825': 1e-2,
    'dose_d36500': 1e-4,
    'He_2y': 586,
    'H_2y': 1200,
}


def setup_publication_style() -> None:
    """Configure matplotlib for publication-quality figures.

    Sets Helvetica font, appropriate sizes, and clean styling.
    """
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
        'font.size': 12,
        'axes.labelsize': 13,
        'axes.labelweight': 'bold',  # Bold axes labels
        'axes.titlesize': 12,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'legend.fontsize': 11,
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.1,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.linewidth': 1.0,
        'grid.alpha': 0.3,
    })


# =============================================================================
# Conformal Calibration Functions
# =============================================================================

def quantile_higher(values: np.ndarray, q: float) -> float:
    """Compute quantile using 'higher' method with compatibility fallback.

    Args:
        values: Array of numeric values.
        q: Quantile in [0, 1].

    Returns:
        Quantile value computed with the 'higher' rule.
    """
    try:
        return float(np.quantile(values, q, method='higher'))
    except TypeError:
        return float(np.quantile(values, q, interpolation='higher'))


def conformalize_upper(
    y_cal: np.ndarray,
    q90_cal: np.ndarray,
    alpha: float = 0.90
) -> float:
    """Compute one-sided conformal adjustment for upper bound q90.

    Args:
        y_cal: Calibration targets.
        q90_cal: Calibration upper quantile predictions.
        alpha: Desired coverage (e.g., 0.90).

    Returns:
        Tau value to add to q90.
    """
    residual = y_cal - q90_cal
    tau = quantile_higher(residual, alpha)
    return float(tau)


def empirical_coverage(y_true: np.ndarray, q: np.ndarray) -> float:
    """Compute empirical coverage P(y_true <= q).

    Args:
        y_true: True target values.
        q: Quantile predictions.

    Returns:
        Fraction of samples where y_true <= q.
    """
    return float(np.mean(y_true <= q))


# =============================================================================
# Data Loading Functions
# =============================================================================

def load_predictions_csv(csv_path: str) -> pd.DataFrame:
    """Load raw predictions CSV.

    Args:
        csv_path: Path to predictions CSV file.

    Returns:
        DataFrame with predictions and true values.
    """
    return pd.read_csv(csv_path)


def compute_coverage_metrics(
    df: pd.DataFrame,
    calibration_frac: float = 0.3,
    random_state: int = 42
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, np.ndarray]]]:
    """Compute coverage metrics before and after CQR calibration.

    Args:
        df: DataFrame with predictions and true values.
        calibration_frac: Fraction of data to use for calibration.
        random_state: Random seed for reproducibility.

    Returns:
        Tuple of (coverage_df, calibrated_quantiles_dict).
    """
    n = len(df)
    rng = np.random.RandomState(random_state)
    mask_cal = rng.rand(n) < calibration_frac
    mask_eval = ~mask_cal

    coverage_rows = []
    calibrated_quantiles: Dict[str, Dict[str, np.ndarray]] = {}

    for task in TASKS:
        y_col = f'{task}_true'
        q90_col = f'{task}_q90'
        q50_col = f'{task}_q50'
        q10_col = f'{task}_q10'

        if y_col not in df.columns or q90_col not in df.columns:
            continue

        y_eval = df.loc[mask_eval, y_col].astype(float).values
        q90_eval = df.loc[mask_eval, q90_col].astype(float).values
        q50_eval = df.loc[mask_eval, q50_col].astype(float).values
        q10_eval = df.loc[mask_eval, q10_col].astype(float).values if q10_col in df.columns else None

        # Before calibration
        cov90_before = empirical_coverage(y_eval, q90_eval)

        # Calibrate using calibration set
        y_cal = df.loc[mask_cal, y_col].astype(float).values
        q90_cal = df.loc[mask_cal, q90_col].astype(float).values
        tau = conformalize_upper(y_cal, q90_cal, alpha=0.90)

        # After calibration
        q90_after = q90_eval + tau
        cov90_after = empirical_coverage(y_eval, q90_after)

        coverage_rows.append({
            'task': task,
            'cov90_before': cov90_before,
            'cov90_after': cov90_after,
        })

        calibrated_quantiles[task] = {
            'y_eval': y_eval,
            'q10_eval': q10_eval,
            'q50_eval': q50_eval,
            'q90_before': q90_eval,
            'q90_after': q90_after,
        }

    return pd.DataFrame(coverage_rows), calibrated_quantiles


def load_precomputed_coverage(precomputed_dir: str) -> pd.DataFrame:
    """Load pre-computed coverage metrics.

    Args:
        precomputed_dir: Directory containing analysis outputs.

    Returns:
        DataFrame with coverage before/after values.
    """
    csv_path = os.path.join(precomputed_dir, 'coverage_before_after.csv')
    return pd.read_csv(csv_path)


def load_precomputed_metrics(precomputed_dir: str) -> pd.DataFrame:
    """Load pre-computed regression metrics.

    Args:
        precomputed_dir: Directory containing analysis outputs.

    Returns:
        DataFrame with MAE, R2, and coverage metrics.
    """
    csv_path = os.path.join(precomputed_dir, 'lgbm_regression_metrics.csv')
    return pd.read_csv(csv_path)


# =============================================================================
# Plotting Functions
# =============================================================================

def plot_coverage_comparison(
    ax: plt.Axes,
    coverage_df: pd.DataFrame,
    panel_label: str = '(a)'
) -> None:
    """Plot coverage before/after CQR calibration.

    Args:
        ax: Matplotlib axes to plot on.
        coverage_df: DataFrame with coverage metrics.
        panel_label: Label for the panel (e.g., '(a)').
    """
    tasks = coverage_df['task'].values
    display_names = [TASK_DISPLAY_NAMES.get(t, t) for t in tasks]

    x = np.arange(len(tasks))
    width = 0.35

    before_vals = coverage_df['cov90_before'].values
    after_vals = coverage_df['cov90_after'].values

    ax.bar(
        x - width / 2, before_vals, width,
        label='Before', color=COLORS['rose'], alpha=0.7
    )
    ax.bar(
        x + width / 2, after_vals, width,
        label='After', color=COLORS['blue'], alpha=0.7
    )

    # Target coverage line
    ax.axhline(0.90, color=COLORS['amber'], linestyle=':', linewidth=2,
               alpha=0.8, label='Target (90%)')

    ax.set_xticks(x)
    ax.set_xticklabels(display_names, rotation=45, ha='right', fontsize=10, fontweight='bold')
    ax.set_ylim(0, 1.05)
    ax.set_ylabel('Empirical Coverage', fontweight='bold')
    ax.legend(loc='lower right', fontsize=11, framealpha=0.95)
    ax.grid(True, axis='y', alpha=0.3)

    # Panel label - positioned higher
    ax.text(-0.10, 1.12, panel_label, transform=ax.transAxes,
            fontsize=16, fontweight='bold', va='top')


def plot_cumulative_discovery(
    ax: plt.Axes,
    df: pd.DataFrame,
    calibrated_quantiles: Optional[Dict[str, Dict[str, np.ndarray]]] = None,
    limits: Optional[Dict[str, float]] = None,
    panel_label: str = '(b)'
) -> None:
    """Plot cumulative discovery curve for screening.

    Args:
        ax: Matplotlib axes to plot on.
        df: DataFrame with predictions (used if calibrated_quantiles not provided).
        calibrated_quantiles: Pre-computed calibrated quantiles dict.
        limits: Task-to-limit mapping for screening.
        panel_label: Label for the panel.
    """
    if limits is None:
        limits = DEFAULT_LIMITS

    # Get true labels and predictions
    if 'all_pass_true' in df.columns and 'all_pass_pred' in df.columns:
        y_true = df['all_pass_true'].astype(int).values
        y_pred = df['all_pass_pred'].astype(int).values
    else:
        # Compute from individual task pass columns
        y_true = np.ones(len(df), dtype=bool)
        y_pred = np.ones(len(df), dtype=bool)
        for task in TASKS:
            if f'{task}_pass_true' in df.columns:
                y_true &= df[f'{task}_pass_true'].values
            if f'{task}_pass_pred' in df.columns:
                y_pred &= df[f'{task}_pass_pred'].values
        y_true = y_true.astype(int)
        y_pred = y_pred.astype(int)

    # Sort by prediction (passes first)
    order = np.argsort(-y_pred)
    cum_tp = np.cumsum(y_true[order])
    total_tp = np.sum(y_true)

    x = np.arange(1, len(y_true) + 1) / len(y_true)
    y_curve = cum_tp / max(total_tp, 1)

    # Compute AUC
    auc = np.trapz(y_curve, x)

    # Plot curves
    ax.plot(x, y_curve, color=COLORS['blue'], linewidth=2.5, alpha=0.8,
            label=f'Model (AUC = {auc:.3f})')
    ax.plot([0, 1], [0, 1], color=COLORS['rose'], linewidth=2,
            linestyle='--', alpha=0.7, label='Random')

    # Annotations for key milestones - bigger font with white background
    if total_tp > 0:
        idx_50 = np.where(cum_tp >= total_tp * 0.5)[0]
        idx_90 = np.where(cum_tp >= total_tp * 0.9)[0]

        if len(idx_50) > 0:
            x_50 = x[idx_50[0]]
            ax.axvline(x=x_50, color=COLORS['amber'], linestyle=':', alpha=0.7, linewidth=1.5)
            ax.annotate(f'50% at {x_50:.0%}',
                        xy=(x_50, 0.5), xytext=(x_50 + 0.08, 0.35),
                        fontsize=12, fontweight='bold', color='black',
                        arrowprops=dict(arrowstyle='->', color=COLORS['amber'],
                                        alpha=0.7, linewidth=1.5),
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                                  alpha=0.85, edgecolor='0.7'))

        if len(idx_90) > 0:
            x_90 = x[idx_90[0]]
            ax.axvline(x=x_90, color=COLORS['teal'], linestyle=':', alpha=0.7, linewidth=1.5)
            ax.annotate(f'90% at {x_90:.0%}',
                        xy=(x_90, 0.9), xytext=(x_90 + 0.08, 0.72),
                        fontsize=12, fontweight='bold', color='black',
                        arrowprops=dict(arrowstyle='->', color=COLORS['teal'],
                                        alpha=0.7, linewidth=1.5),
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                                  alpha=0.85, edgecolor='0.7'))

    # Summary statistics - bigger font
    tp = ((y_true == 1) & (y_pred == 1)).sum()
    fp = ((y_true == 0) & (y_pred == 1)).sum()
    fn = ((y_true == 1) & (y_pred == 0)).sum()
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)

    stats_text = (
        f'Precision: {precision:.3f}\n'
        f'Recall: {recall:.3f}\n'
        f'F1: {f1:.3f}'
    )
    ax.text(0.03, 0.97, stats_text, transform=ax.transAxes,
            verticalalignment='top', fontsize=11, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                      alpha=0.9, edgecolor='0.7'))

    ax.set_xlabel('Fraction Evaluated', fontweight='bold')
    ax.set_ylabel('Fraction of Passes Found', fontweight='bold')
    ax.legend(loc='lower right', fontsize=11, framealpha=0.95)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)

    # Panel label - positioned higher
    ax.text(-0.10, 1.12, panel_label, transform=ax.transAxes,
            fontsize=16, fontweight='bold', va='top')


def plot_parity_panel(
    ax: plt.Axes,
    task: str,
    y_true: np.ndarray,
    q10: Optional[np.ndarray],
    q50: np.ndarray,
    q90: np.ndarray,
    r2: float,
    mae: float,
    cov90: float,
    panel_label: str,
    show_legend: bool = False
) -> None:
    """Plot a single parity subplot.

    Args:
        ax: Matplotlib axes to plot on.
        task: Task name for labeling.
        y_true: True values.
        q10: Lower quantile predictions (can be None).
        q50: Median predictions.
        q90: Upper quantile predictions.
        r2: R-squared value.
        mae: Mean absolute error.
        cov90: Coverage at q90.
        panel_label: Label for the panel.
        show_legend: Whether to show the legend on this panel.
    """
    display_name = TASK_DISPLAY_NAMES.get(task, task)

    # Plot quantiles
    ax.scatter(y_true, q50, s=18, alpha=0.6, color=COLORS['blue'],
               label='q50', zorder=3)
    if q10 is not None and not np.all(np.isnan(q10)):
        ax.scatter(y_true, q10, s=12, alpha=0.5, color=COLORS['amber'],
                   marker='^', label='q10', zorder=2)
    ax.scatter(y_true, q90, s=12, alpha=0.5, color=COLORS['teal'],
               marker='v', label='q90', zorder=2)

    # Y=X line
    all_vals = np.concatenate([y_true, q50, q90])
    lims = [np.nanmin(all_vals), np.nanmax(all_vals)]
    margin = (lims[1] - lims[0]) * 0.05
    lims = [lims[0] - margin, lims[1] + margin]
    ax.plot(lims, lims, color=COLORS['rose'], linewidth=1.5,
            linestyle='--', alpha=0.7, zorder=1)

    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_aspect('equal', adjustable='box')

    # Labels - bold, with extra padding to avoid overlap with scientific notation offset
    ax.set_xlabel(f'{display_name} (True)', fontsize=11, fontweight='bold', labelpad=8)
    ax.set_ylabel(f'{display_name} (Pred)', fontsize=11, fontweight='bold')

    # Metrics annotation - bigger font and box
    metrics_text = f'R²={r2:.3f}\nMAE={mae:.2g}\nCov@90={cov90:.2f}'
    ax.text(0.04, 0.96, metrics_text, transform=ax.transAxes,
            verticalalignment='top', fontsize=12, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                      alpha=0.95, edgecolor='0.7', linewidth=1.2))

    ax.grid(False)

    # Panel label - positioned higher
    ax.text(-0.18, 1.12, panel_label, transform=ax.transAxes,
            fontsize=16, fontweight='bold', va='top')

    if show_legend:
        ax.legend(loc='lower right', fontsize=11, framealpha=0.95,
                  markerscale=1.5)


def create_combined_figure(
    coverage_df: pd.DataFrame,
    predictions_df: pd.DataFrame,
    calibrated_quantiles: Dict[str, Dict[str, np.ndarray]],
    metrics_df: Optional[pd.DataFrame] = None,
    output_dir: str = '.',
    output_basename: str = 'combined_lgbm_calibration_figure'
) -> None:
    """Create the combined publication figure.

    Args:
        coverage_df: DataFrame with coverage before/after values.
        predictions_df: DataFrame with raw predictions.
        calibrated_quantiles: Dict of calibrated quantile arrays per task.
        metrics_df: Optional pre-computed metrics DataFrame.
        output_dir: Directory to save outputs.
        output_basename: Base filename for outputs.
    """
    setup_publication_style()

    # Create figure with custom layout - larger overall size
    fig = plt.figure(figsize=(15, 17))

    # GridSpec: 3 rows
    # Row 0: 2 panels (a, b) - slightly taller for panel labels
    # Row 1: 3 panels (c, d, e) - first 3 parity plots
    # Row 2: 3 panels (f, g, h) - last 3 parity plots
    # Using nested GridSpec to have different spacing between top row and parity rows
    from matplotlib.gridspec import GridSpecFromSubplotSpec
    
    # Main GridSpec with 2 sections: top row and bottom 2 rows
    # Increased bottom ratio to make parity plots larger
    gs_main = GridSpec(2, 1, figure=fig,
                       height_ratios=[0.9, 2.2],
                       hspace=0.28)  # Spacing between a/b and c-h
    
    # Top row: panels (a) and (b)
    gs_top = GridSpecFromSubplotSpec(1, 2, subplot_spec=gs_main[0], wspace=0.30)
    
    # Bottom rows: panels (c)-(h) in 2x3 grid - reduced spacing, larger plots
    gs_bottom = GridSpecFromSubplotSpec(2, 3, subplot_spec=gs_main[1],
                                        hspace=0.18, wspace=0.45)

    # Panel (a): Coverage comparison
    ax_a = fig.add_subplot(gs_top[0, 0])
    plot_coverage_comparison(ax_a, coverage_df, panel_label='(a)')

    # Panel (b): Cumulative discovery
    ax_b = fig.add_subplot(gs_top[0, 1])
    plot_cumulative_discovery(ax_b, predictions_df, calibrated_quantiles,
                              panel_label='(b)')

    # Parity plots (c)-(h)
    panel_labels = ['(c)', '(d)', '(e)', '(f)', '(g)', '(h)']

    for idx, task in enumerate(TASKS):
        row = idx // 3
        col = idx % 3
        ax = fig.add_subplot(gs_bottom[row, col])

        if task in calibrated_quantiles:
            cq = calibrated_quantiles[task]
            y_true = cq['y_eval']
            q10 = cq.get('q10_eval')
            q50 = cq['q50_eval']
            q90 = cq['q90_after']

            # Compute metrics
            r2 = r2_score(y_true, q50)
            mae = mean_absolute_error(y_true, q50)
            cov90 = empirical_coverage(y_true, q90)
        elif metrics_df is not None:
            # Fall back to pre-computed metrics
            task_row = metrics_df[metrics_df['task'] == task]
            if len(task_row) > 0:
                r2 = task_row['R2(q50)'].values[0]
                mae = task_row['MAE(q50)'].values[0]
                cov90 = task_row['coverage@q90'].values[0]
            else:
                r2, mae, cov90 = np.nan, np.nan, np.nan

            # Get data from predictions_df
            y_true = predictions_df[f'{task}_true'].values
            q10 = predictions_df[f'{task}_q10'].values if f'{task}_q10' in predictions_df.columns else None
            q50 = predictions_df[f'{task}_q50'].values
            q90 = predictions_df[f'{task}_q90'].values
        else:
            continue

        # Show legend only on first parity plot (c)
        show_legend = (idx == 0)

        plot_parity_panel(
            ax, task, y_true, q10, q50, q90,
            r2, mae, cov90, panel_labels[idx],
            show_legend=show_legend
        )

    # Save outputs
    os.makedirs(output_dir, exist_ok=True)

    pdf_path = os.path.join(output_dir, f'{output_basename}.pdf')
    png_path = os.path.join(output_dir, f'{output_basename}.png')

    fig.savefig(pdf_path, format='pdf', bbox_inches='tight', facecolor='white')
    fig.savefig(png_path, format='png', dpi=300, bbox_inches='tight', facecolor='white')

    plt.close(fig)

    print(f"Saved: {pdf_path}")
    print(f"Saved: {png_path}")


# =============================================================================
# Main Entry Point
# =============================================================================

def main() -> None:
    """Main function to generate the combined figure."""
    parser = argparse.ArgumentParser(
        description='Generate combined LightGBM calibration figure.'
    )
    parser.add_argument(
        '--predictions_csv',
        type=str,
        help='Path to raw predictions CSV (for recomputing from scratch).'
    )
    parser.add_argument(
        '--precomputed_dir',
        type=str,
        help='Path to directory with pre-computed analysis outputs.'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='plots',
        help='Output directory for figures.'
    )
    parser.add_argument(
        '--output_basename',
        type=str,
        default='combined_lgbm_calibration_figure',
        help='Base filename for output files.'
    )
    parser.add_argument(
        '--calibration_frac',
        type=float,
        default=0.3,
        help='Fraction of data to use for CQR calibration.'
    )

    args = parser.parse_args()

    # Determine data source
    if args.predictions_csv:
        print(f"Loading predictions from: {args.predictions_csv}")
        predictions_df = load_predictions_csv(args.predictions_csv)

        print("Computing coverage metrics with CQR calibration...")
        coverage_df, calibrated_quantiles = compute_coverage_metrics(
            predictions_df,
            calibration_frac=args.calibration_frac
        )
        metrics_df = None

    elif args.precomputed_dir:
        print(f"Loading pre-computed data from: {args.precomputed_dir}")
        coverage_df = load_precomputed_coverage(args.precomputed_dir)
        metrics_df = load_precomputed_metrics(args.precomputed_dir)

        # Try to load predictions for parity plots
        pred_csv = os.path.join(args.precomputed_dir, '..', 'lightgbm_predictions.csv')
        if os.path.exists(pred_csv):
            predictions_df = load_predictions_csv(pred_csv)
            _, calibrated_quantiles = compute_coverage_metrics(predictions_df)
        else:
            # Look in parent directory
            pred_csv_alt = os.path.join(
                os.path.dirname(args.precomputed_dir.rstrip('/')),
                'lightgbm_predictions.csv'
            )
            if os.path.exists(pred_csv_alt):
                predictions_df = load_predictions_csv(pred_csv_alt)
                _, calibrated_quantiles = compute_coverage_metrics(predictions_df)
            else:
                raise FileNotFoundError(
                    f"Could not find predictions CSV. Tried: {pred_csv} and {pred_csv_alt}"
                )
    else:
        # Default paths for self-contained script (relative to scripts_for_paper/data/)
        script_dir = Path(__file__).parent
        data_dir = script_dir.parent / 'data'
        
        # Primary: use data from scripts_for_paper/data/ (self-contained)
        default_predictions = data_dir / 'surrogate-model-data' / 'lightgbm_predictions.csv'
        default_precomputed = data_dir / 'test_combined_fixed_calibrated'
        
        # Fallback: use data from analysis_results/ (original location)
        fallback_predictions = script_dir.parent.parent / (
            'analysis_results/lightgbm_workflow_production_w_impurities_quantized/'
            'lightgbm_predictions.csv'
        )
        fallback_precomputed = script_dir.parent.parent / (
            'analysis_results/lightgbm_workflow_production_w_impurities_quantized/'
            'test_combined_fixed_calibrated'
        )
        
        # Use primary if exists, else fallback
        if not default_predictions.exists() and fallback_predictions.exists():
            default_predictions = fallback_predictions
        if not default_precomputed.exists() and fallback_precomputed.exists():
            default_precomputed = fallback_precomputed

        if default_predictions.exists():
            print(f"Using default predictions: {default_predictions}")
            predictions_df = load_predictions_csv(str(default_predictions))
            coverage_df, calibrated_quantiles = compute_coverage_metrics(predictions_df)
            metrics_df = None
        elif default_precomputed.exists():
            print(f"Using default pre-computed data: {default_precomputed}")
            coverage_df = load_precomputed_coverage(str(default_precomputed))
            metrics_df = load_precomputed_metrics(str(default_precomputed))
            predictions_df = pd.DataFrame()
            calibrated_quantiles = {}
        else:
            parser.print_help()
            print("\nError: No data source specified and defaults not found.")
            print("Please provide --predictions_csv or --precomputed_dir")
            return

    # Resolve output directory
    output_dir = args.output_dir
    if not os.path.isabs(output_dir):
        script_dir = Path(__file__).parent
        output_dir = str(script_dir / output_dir)

    print(f"Output directory: {output_dir}")

    # Generate the figure
    create_combined_figure(
        coverage_df=coverage_df,
        predictions_df=predictions_df,
        calibrated_quantiles=calibrated_quantiles,
        metrics_df=metrics_df,
        output_dir=output_dir,
        output_basename=args.output_basename
    )

    print("\nFigure generation complete!")

if __name__ == '__main__':
    main()
