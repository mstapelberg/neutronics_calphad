#!/usr/bin/env python3
"""
Publication-quality combined figure showing composition space constraints for V-based alloys.

Panel (a): DBTT vs V content (at%) - Linear regression showing how DBTT decreases with V content,
           establishing the minimum V requirement (~80 at%) for ductility (DBTT ≈ 0°C).
           V content converted from wt% to at% assuming Cr-Ti dominated balance.

Panel (b): Maximum alloying fractions in V-X binary alloys from neutronics constraints
           (dose rate limits at 30d, 1y, 5y, 100y + He/H gas production at 2y).

Panel (c): Allowed composition ranges synthesizing DBTT + neutronics constraints,
           showing the dramatic reduction in search space for V-Cr-Ti-W-Zr alloys.

Output:
    combined_composition_space_figure.pdf  (vector, publication-ready)
    combined_composition_space_figure.png  (600 dpi raster)

Usage:
    python combined_composition_space_figure.py

Author: Generated for publication
"""

from typing import Dict, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error
from scipy.stats import pearsonr
import os

# =============================================================================
# Color Palette (from analyze_lightgbm_predictions.py)
# =============================================================================
COLORS = {
    'blue': '#2A33C3',
    'rose': '#8F2D56',
    'amber': '#A35D00',
    'teal': '#0B7285',
    'olive': '#6E8D00',
}

# Element-specific colors for consistency across panels
ELEMENT_COLORS = {
    'V': COLORS['blue'],
    'Cr': COLORS['amber'],
    'Ti': COLORS['teal'],
    'W': COLORS['olive'],
    'Zr': COLORS['rose'],
}

# =============================================================================
# Atomic Masses (g/mol)
# =============================================================================
ATOMIC_MASSES = {
    'V': 50.9415,
    'Cr': 51.9961,
    'Ti': 47.867,
    'W': 183.84,
    'Zr': 91.224,
}

# Effective atomic mass for "balance" in V-alloys (assume Cr-Ti dominated)
# Cr and Ti have similar atomic masses to V, so wt% ≈ at% for high-V alloys
EFFECTIVE_BALANCE_MASS = 51.0  # g/mol (average of Cr and Ti)


def wt_to_at_percent(wt_percent_v: np.ndarray, 
                     m_v: float = ATOMIC_MASSES['V'],
                     m_balance: float = EFFECTIVE_BALANCE_MASS) -> np.ndarray:
    """Convert V content from weight percent to atomic percent.
    
    Assumes binary V + "balance" system where balance has effective atomic mass.
    
    Args:
        wt_percent_v: V content in weight percent (0-100 scale).
        m_v: Atomic mass of V in g/mol.
        m_balance: Effective atomic mass of balance elements in g/mol.
        
    Returns:
        V content in atomic percent (0-100 scale).
    """
    wt_v = wt_percent_v
    wt_balance = 100.0 - wt_v
    
    # Moles of each component (per 100g total)
    mol_v = wt_v / m_v
    mol_balance = wt_balance / m_balance
    
    # Atomic percent
    at_percent_v = (mol_v / (mol_v + mol_balance)) * 100.0
    return at_percent_v


# =============================================================================
# Data Constants
# =============================================================================

# Maximum alloying fractions in V-X binary alloys (from neutronics analysis)
# These represent the maximum X content where V-X binary alloy passes ALL constraints:
# - Dose rate limits: 30d ≤ 1000 Sv/h, 365d ≤ 1 Sv/h, 1825d ≤ 0.01 Sv/h, 36500d ≤ 0.0001 Sv/h
# - Gas production limits (2y): He ≤ 396 appm, H ≤ 1200 appm
MAX_ALLOYING_FRACTIONS = {
    'Cr': 0.4204,
    'Ti': 0.4005,
    'W': 0.9500,
    'Zr': 0.0493,
}

# Neutronics limits
DOSE_LIMITS = {
    '30 days': 1e3,      # Sv/h - maintenance access
    '365 days': 1.0,     # Sv/h - hands-on maintenance
    '1825 days': 1e-2,   # Sv/h - 5 year cooling
    '36500 days': 1e-4,  # Sv/h - 100 year disposal
}

GAS_LIMITS = {
    'He': 396,   # appm - 2 year He production limit
    'H': 1200,   # appm - 2 year H production limit
}

# DBTT constraint: V ≥ 0.80 (from DBTT vs V linear regression)
V_MIN_DBTT = 0.80  # Minimum V content for DBTT ≈ 0°C (ductile)

# Coupled constraint for feasible region
DBTT_ALLOYING_LIMIT = 0.20  # Cr + Ti + W + Zr ≤ 0.20

# Individual element limits from neutronics (capped by binary analysis)
ELEMENT_LIMITS = {
    'V': (V_MIN_DBTT, 1.0),           # V must be at least 80%
    'Cr': (0.0, min(0.20, MAX_ALLOYING_FRACTIONS['Cr'])),  # capped by coupled constraint
    'Ti': (0.0, min(0.20, MAX_ALLOYING_FRACTIONS['Ti'])),  # capped by coupled constraint
    'W': (0.0, min(0.20, MAX_ALLOYING_FRACTIONS['W'])),    # capped by coupled constraint
    'Zr': (0.0, MAX_ALLOYING_FRACTIONS['Zr']),             # limited by neutronics
}


def setup_publication_style() -> None:
    """Configure matplotlib for publication-quality figures with Helvetica font."""
    mpl.rcParams.update({
        # Figure settings
        'figure.dpi': 150,
        'savefig.dpi': 600,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.1,
        
        # Font settings - Helvetica (or fallback to sans-serif)
        'font.family': 'sans-serif',
        'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
        'font.size': 12,
        
        # Axes settings
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'axes.labelweight': 'bold',
        'axes.linewidth': 1.0,
        'axes.spines.top': False,
        'axes.spines.right': False,
        
        # Tick settings
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'xtick.major.width': 1.0,
        'ytick.major.width': 1.0,
        'xtick.major.size': 5,
        'ytick.major.size': 5,
        
        # Legend settings
        'legend.fontsize': 11,
        'legend.frameon': True,
        'legend.framealpha': 0.9,
        'legend.edgecolor': '0.8',
        
        # Grid settings
        'grid.alpha': 0.3,
        'grid.linewidth': 0.5,
        
        # PDF/vector output
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
    })


def load_dbtt_data(csv_path: str, convert_to_at: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """Load DBTT data from CSV file.
    
    Args:
        csv_path: Path to CSV file with 'V' and 'DBTT(C)' columns.
        convert_to_at: If True, convert V from wt% to at%.
        
    Returns:
        Tuple of (V_values, DBTT_values) as numpy arrays.
        V_values are in at% if convert_to_at=True, else wt%.
    """
    df = pd.read_csv(csv_path)
    y = pd.to_numeric(df["DBTT(C)"], errors="coerce")
    V_wt = pd.to_numeric(df["V"], errors="coerce")
    mask = y.notna() & V_wt.notna()
    
    V_wt_clean = V_wt[mask].to_numpy()
    DBTT_clean = y[mask].to_numpy()
    
    if convert_to_at:
        V_at = wt_to_at_percent(V_wt_clean)
        return V_at, DBTT_clean
    
    return V_wt_clean, DBTT_clean


def fit_dbtt_model(V: np.ndarray, DBTT: np.ndarray) -> Dict:
    """Fit linear regression model for DBTT vs V content.
    
    Args:
        V: Vanadium content (wt%).
        DBTT: Ductile-to-brittle transition temperature (°C).
        
    Returns:
        Dictionary with model parameters and statistics.
    """
    X = V.reshape(-1, 1)
    lm = LinearRegression()
    lm.fit(X, DBTT)
    y_pred = lm.predict(X)
    
    r, p = pearsonr(V, DBTT)
    r2 = r2_score(DBTT, y_pred)
    mae = mean_absolute_error(DBTT, y_pred)
    
    # Calculate V threshold where DBTT = 0°C (room temperature ductility)
    dbtt_threshold = 0.0
    v_at_threshold = (dbtt_threshold - lm.intercept_) / lm.coef_[0]
    
    return {
        'model': lm,
        'intercept': lm.intercept_,
        'slope': lm.coef_[0],
        'r2': r2,
        'mae': mae,
        'pearson_r': r,
        'pearson_p': p,
        'n': len(V),
        'v_at_dbtt_0': v_at_threshold,
    }


def plot_panel_a(ax: plt.Axes, V: np.ndarray, DBTT: np.ndarray, stats: Dict) -> None:
    """Plot panel (a): DBTT vs V content with linear regression.
    
    Args:
        ax: Matplotlib axes to plot on.
        V: Vanadium content (at%).
        DBTT: DBTT values (°C).
        stats: Dictionary with model statistics from fit_dbtt_model().
    """
    # Scatter plot of data
    ax.scatter(V, DBTT, s=25, color=COLORS['rose'], alpha=0.7, 
               label='Experimental data', zorder=3, edgecolors='white', linewidths=0.5)
    
    # Fitted line
    x_span = np.linspace(V.min(), V.max(), 200).reshape(-1, 1)
    y_fit = stats['model'].predict(x_span)
    ax.plot(x_span, y_fit, linewidth=2.5, color=COLORS['blue'], 
            label='Linear fit', zorder=2)
    
    # 95% prediction interval
    X1 = np.c_[np.ones_like(V.reshape(-1, 1)), V.reshape(-1, 1)]
    XtX_inv = np.linalg.inv(X1.T @ X1)
    x1_span = np.c_[np.ones_like(x_span), x_span]
    residuals = DBTT - stats['model'].predict(V.reshape(-1, 1))
    s2 = (residuals**2).sum() / (len(DBTT) - 2)
    se_pred = np.sqrt(s2 * (1 + np.einsum("ij,jk,ik->i", x1_span, XtX_inv, x1_span)))
    ax.fill_between(x_span.ravel(), y_fit.ravel() - 1.96*se_pred, y_fit.ravel() + 1.96*se_pred,
                    alpha=0.15, color=COLORS['blue'], label='95% PI', zorder=1)
    
    # Horizontal line at DBTT = 0°C
    ax.axhline(y=0, color=COLORS['amber'], linestyle='--', linewidth=2.0,
               label='DBTT = 0°C', zorder=2, alpha=0.8)
    
    # Vertical line at V threshold (where DBTT = 0)
    v_threshold = stats['v_at_dbtt_0']
    ax.axvline(x=v_threshold, color=COLORS['amber'], linestyle=':', linewidth=1.5, 
               alpha=0.6, zorder=1)
    
    # Shade the "ductile" region (V ≥ threshold, DBTT ≤ 0)
    ax.axvspan(v_threshold, V.max() + 2, alpha=0.08, color=COLORS['olive'], zorder=0)
    
    # Labels and formatting
    ax.set_xlabel('V content (at%)')
    ax.set_ylabel('DBTT (°C)')
    ax.grid(True, alpha=0.3)
    
    # Simplified annotation: just show V threshold
    ax.text(0.97, 0.03, f"V at DBTT = 0°C: {v_threshold:.1f} at%", 
            ha='right', va='bottom', transform=ax.transAxes,
            fontsize=11, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white', 
                      alpha=0.9, edgecolor='0.8'))
    
    # Legend
    ax.legend(loc='upper right', fontsize=10, framealpha=0.9)


def plot_panel_b(ax: plt.Axes) -> None:
    """Plot panel (b): Maximum alloying fractions in V-X binary alloys.
    
    Args:
        ax: Matplotlib axes to plot on.
    """
    elements = list(MAX_ALLOYING_FRACTIONS.keys())
    fractions = list(MAX_ALLOYING_FRACTIONS.values())
    colors = [ELEMENT_COLORS[e] for e in elements]
    
    # Create bar chart
    x_pos = np.arange(len(elements))
    bars = ax.bar(x_pos, fractions, color=colors, edgecolor='black', 
                  alpha=0.75, linewidth=1.0, zorder=2)
    
    # Add data labels on bars
    for bar, value in zip(bars, fractions):
        height = bar.get_height()
        ax.annotate(
            f'{value:.3f}',
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 5),
            textcoords='offset points',
            ha='center', va='bottom',
            fontsize=11, fontweight='bold', color='black'
        )
    
    # Horizontal reference lines
    ax.axhline(y=DBTT_ALLOYING_LIMIT, color=COLORS['rose'], linestyle='--', 
               linewidth=2.0, alpha=0.8, label=f'DBTT constraint ({DBTT_ALLOYING_LIMIT:.0%})', zorder=1)
    
    # Labels and formatting
    ax.set_xlabel('Alloying element X (in V-X binary)')
    ax.set_ylabel('Maximum atomic fraction')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(elements)
    ax.set_ylim(0, 1.05)
    ax.grid(True, axis='y', alpha=0.3)
    
    # Legend positioned at center, near the 0.2 line (y ~ 0.19 in axes coords)
    ax.legend(loc='center', bbox_to_anchor=(0.5, 0.20), fontsize=10, framealpha=0.9)


def plot_panel_c(ax: plt.Axes) -> None:
    """Plot panel (c): Allowed composition ranges (synthesis of constraints).
    
    Shows horizontal bars for each element indicating the feasible range,
    combining DBTT constraint (V ≥ 0.80) and neutronics binary limits.
    
    Args:
        ax: Matplotlib axes to plot on.
    """
    elements = ['Zr', 'W', 'Ti', 'Cr', 'V']  # Bottom to top order
    y_positions = np.arange(len(elements))
    bar_height = 0.6
    
    for i, elem in enumerate(elements):
        low, high = ELEMENT_LIMITS[elem]
        color = ELEMENT_COLORS[elem]
        
        # Draw range bar
        ax.barh(i, high - low, left=low, height=bar_height, 
                color=color, alpha=0.75, edgecolor='black', linewidth=1.0, zorder=2)
        
        # Add range annotation
        if elem == 'V':
            label = f'{low:.3f}' if low < 1.0 else f'{low:.2f}'
            ax.text(low + 0.01, i, label, ha='left', va='center', 
                    fontsize=10, fontweight='bold', color='white')
            ax.text(high - 0.01, i, f'{high:.3f}', ha='right', va='center',
                    fontsize=10, fontweight='bold', color='white')
        else:
            ax.text(high + 0.01, i, f'{high:.3f}', ha='left', va='center',
                    fontsize=10, fontweight='bold', color='black')
    
    # Vertical line at V = 0.80 (DBTT constraint)
    ax.axvline(x=V_MIN_DBTT, color=COLORS['rose'], linestyle='--', linewidth=2.0,
               alpha=0.8, zorder=3, label=f'V ≥ {V_MIN_DBTT:.0%}')
    
    # Labels and formatting
    ax.set_xlabel('Atomic fraction')
    ax.set_yticks(y_positions)
    ax.set_yticklabels(elements, fontweight='bold')
    ax.set_xlim(-0.02, 1.05)
    ax.grid(True, axis='x', alpha=0.3)
    
    # Add coupled constraint annotation (larger font)
    constraint_text = (
        'Coupled constraint:\n'
        f'Cr + Ti + W + Zr ≤ {DBTT_ALLOYING_LIMIT:.2f}\n'
        '(and V = 1 − sum)'
    )
    ax.text(0.38, 0.08, constraint_text, ha='left', va='bottom', transform=ax.transAxes,
            fontsize=12, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', 
                      alpha=0.95, edgecolor='0.7', linewidth=1.5))
    
    # Legend in bottom right
    ax.legend(loc='lower right', fontsize=11, framealpha=0.9)


def create_combined_figure(csv_path: str, output_dir: str) -> None:
    """Create the combined 3-panel publication figure.
    
    Args:
        csv_path: Path to DBTT data CSV file.
        output_dir: Directory to save output figures.
    """
    setup_publication_style()
    
    # Load DBTT data and fit model
    V, DBTT = load_dbtt_data(csv_path)
    stats = fit_dbtt_model(V, DBTT)
    
    print("DBTT Model Statistics (V in at%):")
    print(f"  n = {stats['n']}")
    print(f"  slope = {stats['slope']:.4f} °C/(at% V)")
    print(f"  intercept = {stats['intercept']:.2f} °C")
    print(f"  R² = {stats['r2']:.4f}")
    print(f"  MAE = {stats['mae']:.2f} °C")
    print(f"  V at DBTT = 0°C: {stats['v_at_dbtt_0']:.2f} at%")
    print()
    
    # Create figure with 3 panels (2 rows: top has 2, bottom has 1 wide)
    # Adjusted aspect ratio: panels (a) and (b) are now shorter
    fig = plt.figure(figsize=(10, 8.5))  # Reduced height for better aspect ratio
    
    # Define grid: 2 columns, 2 rows
    # Top row: panels (a) and (b), each taking half width - reduced height
    # Bottom row: panel (c) spanning full width
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 0.65], 
                          hspace=0.30, wspace=0.28,
                          left=0.08, right=0.95, top=0.96, bottom=0.07)
    
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, :])  # Span both columns
    
    # Plot each panel
    plot_panel_a(ax_a, V, DBTT, stats)
    plot_panel_b(ax_b)
    plot_panel_c(ax_c)
    
    # Add panel labels (no titles, just labels)
    label_props = dict(fontsize=16, fontweight='bold', va='top', ha='left')
    ax_a.text(-0.12, 1.05, '(a)', transform=ax_a.transAxes, **label_props)
    ax_b.text(-0.12, 1.05, '(b)', transform=ax_b.transAxes, **label_props)
    ax_c.text(-0.05, 1.10, '(c)', transform=ax_c.transAxes, **label_props)
    
    # Save figures
    os.makedirs(output_dir, exist_ok=True)
    
    pdf_path = os.path.join(output_dir, 'combined_composition_space_figure.pdf')
    png_path = os.path.join(output_dir, 'combined_composition_space_figure.png')
    
    fig.savefig(pdf_path, format='pdf', bbox_inches='tight')
    fig.savefig(png_path, format='png', dpi=600, bbox_inches='tight')
    
    print(f"Saved: {pdf_path}")
    print(f"Saved: {png_path}")
    
    plt.close(fig)


def main() -> int:
    """Main entry point."""
    # Determine paths relative to this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    scripts_for_paper_dir = os.path.dirname(script_dir)
    examples_dir = os.path.dirname(scripts_for_paper_dir)
    
    # DBTT data CSV - check multiple possible locations
    csv_filename = 'DBTT_Data_V-Main_Data.csv'
    possible_paths = [
        os.path.join(scripts_for_paper_dir, 'data', 'thermo-physical-data', csv_filename),
        os.path.join(script_dir, csv_filename),
        os.path.join(scripts_for_paper_dir, 'data', csv_filename),
        os.path.join(examples_dir, csv_filename),
    ]
    
    csv_path = None
    for path in possible_paths:
        if os.path.exists(path):
            csv_path = path
            break
    
    if csv_path is None:
        print("ERROR: DBTT data file not found at expected locations.")
        for path in possible_paths:
            print(f"  Tried: {path}")
        print(f"\nPlease ensure '{csv_filename}' is in one of the above locations.")
        return 1
    
    print(f"Using DBTT data from: {csv_path}")
    print()
    
    # Output to same directory as script
    output_dir = os.path.join(script_dir, 'plots')
    os.makedirs(output_dir, exist_ok=True)
    
    create_combined_figure(csv_path, output_dir)
    
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
