#!/usr/bin/env python3
"""Compare linear weighted predictions to neutronics simulation results."""
import os
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score


# Data for Chart 2: Gas production and Dose rate
materials = ['V', 'Cr', 'Ti', 'W', 'Zr']

# Gas production data (He and H in appm)
gas_production = {
    'V': {'He': 244.52, 'H': 635.76},
    'Cr': {'He': 477.58, 'H': 1770.79},
    'Ti': {'He': 501.38, 'H': 1445.11},
    'W': {'He': 9.56, 'H': 47.57},
    'Zr': {'He': 164.49, 'H': 514.18}
}

# Dose rate data (Sv/h) at different cooling times (in days)
dose_rates = {
    'V': {'30 days': 0.013919599152391583, '365 days': 2.4011751643367956e-05, '1825 days': 3.029438835700662e-08, '36500 days': 4.075920006956689e-09},
    'Cr': {'30 days': 0.8636562634934948, '365 days': 0.00024009839328002942, '1825 days': 1.388597687968871e-06, '36500 days': 5.0686617338504686e-11},
    'Ti': {'30 days': 6.459089435152961, '365 days': 0.4014972333205736, '1825 days': 0.00023419569056964365, '36500 days': 3.141376917685182e-05},
    'W': {'30 days': 9.434409465139566, '365 days': 0.3866656570602694, '1825 days': 5.5820319441895954e-05, '36500 days': 4.00860744093665e-11},
    'Zr': {'30 days': 20268.468062418488, '365 days': 0.6047189151576632, '1825 days': 4.624079130919152e-06, '36500 days': 8.400192950675306e-09}
}

# User-defined limit thresholds for gas production (appm) and dose rates (Sv/h)
limits = {
    '30 days': 1e3,
    '365 days': 1.0,
    '1825 days': 1e-2,
    '36500 days': 1e-4,
    'He': 396,
    'H': 1200
}


def compute_linear_predictions(
    compositions: pd.DataFrame,
    elements: List[str],
    dose_rates: Dict[str, Dict[str, float]],
    gas_production: Dict[str, Dict[str, float]]
) -> pd.DataFrame:
    """Compute linear weighted predictions for dose rates and gas production.
    
    Args:
        compositions: DataFrame with element fraction columns
        elements: List of element symbols (column names)
        dose_rates: Dictionary mapping element to cooling time to dose rate
        gas_production: Dictionary mapping element to gas type to production
        
    Returns:
        DataFrame with predicted values for all 6 targets
    """
    n_samples = len(compositions)
    predictions = pd.DataFrame(index=compositions.index)
    
    # Dose rate predictions for each cooling time
    dose_times = {
        'dose_d30': '30 days',
        'dose_d365': '365 days',
        'dose_d1825': '1825 days',
        'dose_d36500': '36500 days'
    }
    
    for target_col, time_key in dose_times.items():
        pred = np.zeros(n_samples)
        for elem in elements:
            if elem in dose_rates and time_key in dose_rates[elem]:
                pred += compositions[elem].values * dose_rates[elem][time_key]
        predictions[target_col] = pred
    
    # Gas production predictions
    gas_types = {
        'He_2y': 'He',
        'H_2y': 'H'
    }
    
    for target_col, gas_type in gas_types.items():
        pred = np.zeros(n_samples)
        for elem in elements:
            if elem in gas_production and gas_type in gas_production[elem]:
                pred += compositions[elem].values * gas_production[elem][gas_type]
        predictions[target_col] = pred
    
    return predictions


def create_parity_plots(
    y_true: pd.DataFrame,
    y_pred: pd.DataFrame,
    output_dir: str,
    output_filename: str = "linear_vs_neutronics_parity.png"
) -> None:
    """Create publication-quality parity plots comparing predictions to true values.
    
    Args:
        y_true: DataFrame with true target values
        y_pred: DataFrame with predicted target values
        output_dir: Directory to save the plot
        output_filename: Name of the output file
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up publication-quality styling
    plt.style.use('default')
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    fig.suptitle('Linear Model vs. Neutronics Simulation', fontsize=16, fontweight='bold')
    
    # Consistent color scheme matching analyze_lightgbm_predictions.py
    data_color = '#2A33C3'  # Blue for data points
    line_color = '#8F2D56'  # Red for Y=X line
    
    tasks = ['dose_d30', 'dose_d365', 'dose_d1825', 'dose_d36500', 'He_2y', 'H_2y']
    
    metrics_rows = []
    
    for idx, task in enumerate(tasks):
        row = idx // 3
        col = idx % 3
        ax = axes[row, col]
        
        y = y_true[task].values
        y_pred_vals = y_pred[task].values
        
        # Calculate metrics
        mae = mean_absolute_error(y, y_pred_vals)
        r2 = r2_score(y, y_pred_vals)
        
        metrics_rows.append({
            'task': task,
            'MAE': mae,
            'R2': r2
        })
        
        # Plot data points
        ax.scatter(y, y_pred_vals, s=20, alpha=0.58, color=data_color, label='Linear Model', zorder=3)
        
        # Y=X line
        lims = [np.nanmin([y.min(), y_pred_vals.min()]), 
                np.nanmax([y.max(), y_pred_vals.max()])]
        ax.plot(lims, lims, color=line_color, linewidth=2, linestyle='--', alpha=0.58, zorder=1)
        
        # Set labels and title
        ax.set_xlabel(f'{task} True', fontsize=10, fontweight='bold')
        ax.set_ylabel(f'{task} Predicted', fontsize=10, fontweight='bold')
        ax.set_title(f'{task}', fontsize=12, fontweight='bold')
        
        # Add metrics to legend
        legend_text = f'R² = {r2:.3f}\nMAE = {mae:.3g}'
        ax.text(0.05, 0.95, legend_text, transform=ax.transAxes, 
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
               fontsize=12, fontweight='bold')
        
        # Set equal aspect ratio and limits
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        
        # Remove grid and set clean styling
        ax.grid(False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    # Adjust layout and save
    plt.tight_layout()
    
    # Save as high-resolution PNG
    output_path_png = os.path.join(output_dir, output_filename)
    plt.savefig(output_path_png, dpi=300, bbox_inches="tight", facecolor='white')
    print(f"Parity plot (PNG) saved to: {output_path_png}")
    
    # Save as PDF
    output_path_pdf = os.path.join(output_dir, output_filename.replace('.png', '.pdf'))
    plt.savefig(output_path_pdf, format='pdf', bbox_inches="tight", facecolor='white')
    print(f"Parity plot (PDF) saved to: {output_path_pdf}")
    
    plt.close()
    
    # Save metrics to CSV
    metrics_df = pd.DataFrame(metrics_rows)
    metrics_path = os.path.join(output_dir, "linear_model_metrics.csv")
    metrics_df.to_csv(metrics_path, index=False)
    print(f"Metrics saved to: {metrics_path}")
    
    return metrics_df


def main(
    csv_path: str,
    output_dir: str,
    elements: List[str] = ['V', 'Cr', 'Ti', 'W', 'Zr']
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Main function to compare linear predictions to neutronics results.
    
    Args:
        csv_path: Path to the lightgbm_results.csv file
        output_dir: Directory to save output plots and metrics
        elements: List of element symbols used in compositions
        
    Returns:
        Tuple of (true_values, predicted_values, metrics)
    """
    # Read the data
    print(f"Reading data from: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} samples")
    
    # Extract compositions
    compositions = df[elements].copy()
    
    # Extract true target values
    targets = ['dose_d30', 'dose_d365', 'dose_d1825', 'dose_d36500', 'He_2y', 'H_2y']
    y_true = df[targets].copy()
    
    # Compute linear predictions
    print("Computing linear weighted predictions...")
    y_pred = compute_linear_predictions(compositions, elements, dose_rates, gas_production)
    
    # Create parity plots
    print("Creating parity plots...")
    metrics = create_parity_plots(y_true, y_pred, output_dir)
    
    print("\nMetrics Summary:")
    print(metrics.to_string(index=False))
    
    return y_true, y_pred, metrics


if __name__ == "__main__":
    # Set paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(
        os.path.dirname(script_dir),
        "data",
        "surrogate-model-data",
        "lightgbm_results.csv",
    )
    output_dir = os.path.join(script_dir, "plots")
    
    # Run analysis
    y_true, y_pred, metrics = main(csv_path, output_dir)