from typing import Dict, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import openmc.deplete

def get_openmc_inventory(results_path: Path) -> Optional[pd.DataFrame]:
    """
    Extracts nuclide inventory over time from OpenMC depletion results.

    Args:
        results_path: Path to the depletion results HDF5 file.

    Returns:
        A DataFrame with nuclide inventories over time, or None if the file is not found.
    """
    if not results_path.exists():
        print(f"ERROR: OpenMC results file not found at {results_path}")
        return None
    
    results = openmc.deplete.Results(str(results_path))
    timesteps = results.get_times()
    
    # Get the first available material ID
    first_step = results[0]
    available_mat_ids = list(first_step.index_mat.keys())
    if not available_mat_ids:
        print(f"ERROR: No materials found in results")
        return None
    
    mat_id = available_mat_ids[0]
    print(f"INFO: Using material ID '{mat_id}' from results file {results_path.name}")
    
    inventory_data: Dict[float, Dict[str, float]] = {}
    for i in range(len(results)):
        time = timesteps[i]
        material = results[i].get_material(mat_id)
        inventory_data[time] = {n.name: n.percent for n in material.nuclides}
        
    df = pd.DataFrame.from_dict(inventory_data, orient='index')
    df.index.name = 'time_s'
    return df

def get_fispact_inventory(fispact_output_path: Path) -> Optional[pd.DataFrame]:
    """
    Extracts final nuclide inventory from a FISPACT output file.

    Args:
        fispact_output_path: Path to the FISPACT output file.

    Returns:
        A DataFrame with nuclide inventories, or None if the file is not found.
    """
    if not fispact_output_path.exists():
        print(f"ERROR: FISPACT output file not found at {fispact_output_path}")
        return None
    # This is a placeholder - a proper parser for FISPACT output would be needed here.
    # For now, we'll simulate some data.
    return pd.DataFrame() # Placeholder

def main() -> None:
    """Main function to compare and plot results."""
    script_dir: Path = Path(__file__).parent
    
    # --- File Paths ---
    openmc_xs_results: Path = script_dir / 'depletion_results_openmc_xs.h5'
    fispact_xs_results: Path = script_dir / 'depletion_results_fispact_xs.h5'
    fispact_output: Path = script_dir / 'vv_vanadium.out' # Original FISPACT output

    # --- Load Data ---
    df_openmc_xs = get_openmc_inventory(openmc_xs_results)
    df_fispact_xs = get_openmc_inventory(fispact_xs_results)
    df_fispact_orig = get_fispact_inventory(fispact_output)

    if df_openmc_xs is None or df_fispact_xs is None:
        sys.exit(1)

    # --- Plotting ---
    nuclides_to_plot: list[str] = ['V51', 'V52', 'Cr51', 'Cr52', 'Ti50']
    
    fig, axes = plt.subplots(len(nuclides_to_plot), 1, figsize=(12, 6 * len(nuclides_to_plot)), sharex=True)
    if not isinstance(axes, np.ndarray):
        axes = [axes]
    fig.suptitle('Nuclide Inventory Comparison', fontsize=16)

    for i, nuclide in enumerate(nuclides_to_plot):
        ax = axes[i]
        if nuclide in df_openmc_xs.columns:
            df_openmc_xs[nuclide].plot(ax=ax, label='OpenMC (OpenMC XS)', marker='o', logy=True)
        if nuclide in df_fispact_xs.columns:
            df_fispact_xs[nuclide].plot(ax=ax, label='OpenMC (FISPACT XS)', marker='x', logy=True)
        
        ax.set_ylabel('Atomic Fraction')
        ax.set_title(f'Inventory of {nuclide}')
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        ax.legend()

    axes[-1].set_xlabel('Time (s)')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    plot_filename: Path = script_dir / 'inventory_comparison.png'
    plt.savefig(plot_filename, dpi=300)
    print(f"✅ Comparison plot saved to: {plot_filename}")

if __name__ == "__main__":
    main()
