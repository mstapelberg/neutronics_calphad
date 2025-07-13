import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# It's good practice to check for dependencies and guide the user.
try:
    import openmc.deplete
    import pypact as pp
except ImportError as e:
    print(f"❌ A required library is not installed: {e}")
    print("➡️ Please install dependencies with 'pip install openmc pypact pandas matplotlib'")
    sys.exit(1)

# --- Configuration ---
# Threshold for including nuclides in the comparison, in atomic parts per million (appm)
APPM_THRESHOLD = 10.0

# --- File Paths ---
SCRIPT_DIR = Path(__file__).parent
# Assumes the OpenMC results are in the 'debugging' directory relative to project root
OPENMC_RESULTS_FILE = SCRIPT_DIR.parent / 'debugging' / 'depletion_results.h5'
FISPACT_OUTPUT_FILE = SCRIPT_DIR / 'vv_vanadium.out'

# --- Constants ---
AVOGADRO = 6.022e23

def get_fispact_inventory(fispact_output, nuclides_to_get):
    """Extracts nuclide inventory over time from FISPACT output."""
    data = {n: [] for n in nuclides_to_get}
    all_nuclides_data = []
    times_years = []

    with pp.Output(str(fispact_output)) as output:
        times_years = [t.irradiation_time / (3600*24*365.25) for t in output.inventory_data]
        
        for t_step in output.inventory_data:
            step_inventory = {n.name: (n.grams / n.mass) * AVOGADRO for n in t_step.nuclides if n.mass > 0}
            all_nuclides_data.append(step_inventory)
            for n_name in nuclides_to_get:
                data[n_name].append(step_inventory.get(n_name, 0.0))

    if not nuclides_to_get:
        # If no specific nuclides are requested, return the full inventory at each step
        return pd.DataFrame(all_nuclides_data, index=times_years)

    return pd.DataFrame(data, index=times_years)

def get_openmc_inventory(openmc_results_path, nuclides_to_get):
    """Extracts nuclide inventory over time from OpenMC depletion results."""
    if not openmc_results_path.exists():
        return None, None
    results = openmc.deplete.Results(str(openmc_results_path))
    
    material_id = '1'
    materials = results.get_materials(0)
    for key, mat in materials.items():
        if mat.name == 'vv_material_V':
            material_id = key
            break
            
    timesteps_years = results.get_times() / (24 * 3600) / 365.25
    data = {}
    if not nuclides_to_get:
        # Get all nuclides at all timesteps
        full_results = {}
        for i in range(len(timesteps_years)):
            nucs = results.get_atoms(material_id, "all", i)
            full_results[timesteps_years[i]] = dict(zip(nucs[0], nucs[1]))
        df = pd.DataFrame.from_dict(full_results, orient='index')
        return df.reindex(timesteps_years), results.get_materials(0)[material_id]

    # Get specific nuclides
    for n_name in nuclides_to_get:
        _, atoms = results.get_atoms(material_id, n_name)
        data[n_name] = atoms

    return pd.DataFrame(data, index=timesteps_years), results.get_materials(0)[material_id]

def main():
    """Main function to compare and plot results."""
    print("--- Starting OpenMC vs. FISPACT Comparison ---")

    if not all([FISPACT_OUTPUT_FILE.exists(), OPENMC_RESULTS_FILE.exists()]):
        print(f"❌ ERROR: Missing '{FISPACT_OUTPUT_FILE.name}' or '{OPENMC_RESULTS_FILE.name}'.")
        print("➡️ Please run previous steps and ensure result files are present.")
        sys.exit(1)

    print("✅ Found all necessary input files.")
    print("📖 Loading full data from FISPACT and OpenMC to determine nuclides to compare...")

    fispact_full_df = get_fispact_inventory(FISPACT_OUTPUT_FILE, [])
    openmc_full_df, initial_material = get_openmc_inventory(OPENMC_RESULTS_FILE, [])
    
    if openmc_full_df is None:
        print(f"❌ ERROR: Could not load OpenMC results from {OPENMC_RESULTS_FILE}")
        sys.exit(1)

    total_initial_atoms = sum(initial_material.get_nuclide_atoms().values())
    appm_atom_threshold = APPM_THRESHOLD * total_initial_atoms / 1e6
    
    fispact_final = fispact_full_df.iloc[-1]
    openmc_final = openmc_full_df.iloc[-1]
    
    nuclides_to_compare = set()
    for n, atoms in fispact_final.items():
        if atoms > appm_atom_threshold:
            nuclides_to_compare.add(n)
    for n, atoms in openmc_final.items():
        if atoms > appm_atom_threshold:
            nuclides_to_compare.add(n)
    
    nuclides_to_compare = sorted(list(nuclides_to_compare))
    print(f"ℹ️  Found {len(nuclides_to_compare)} nuclides exceeding {APPM_THRESHOLD} appm threshold.")

    fispact_df = fispact_full_df[nuclides_to_compare]
    openmc_df = openmc_full_df[nuclides_to_compare]

    print("\n--- Final Nuclide Inventory Comparison (atoms at 10 years) ---")
    
    # Find the closest matching time point to 10 years for comparison
    openmc_10y = openmc_df.iloc[np.abs(openmc_df.index - 10.0).argmin()]
    fispact_10y = fispact_df.iloc[np.abs(fispact_df.index - 10.0).argmin()]

    comparison_df = pd.DataFrame({'OpenMC': openmc_10y, 'FISPACT': fispact_10y})
    comparison_df.dropna(inplace=True)
    comparison_df['Difference (%)'] = 100 * (comparison_df['FISPACT'] - comparison_df['OpenMC']) / comparison_df['OpenMC'].replace(0, np.nan)
    print(comparison_df.to_string(float_format="%.3e"))

    print("\n📊 Generating comparison plots...")
    n_nuclides = len(nuclides_to_compare)
    fig, axes = plt.subplots(n_nuclides, 1, figsize=(12, 5 * n_nuclides), sharex=True)
    if n_nuclides == 1:
        axes = [axes]

    for i, nuclide in enumerate(nuclides_to_compare):
        ax = axes[i]
        openmc_df[nuclide].plot(ax=ax, label='OpenMC', marker='o', linestyle='-', logy=True)
        fispact_df[nuclide].plot(ax=ax, label='FISPACT', marker='x', linestyle='--', logy=True)
        
        ax.set_title(f'Inventory of {nuclide} vs. Time')
        ax.set_ylabel('Number of Atoms (log scale)')
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        ax.legend()

    axes[-1].set_xlabel('Irradiation Time (years)')
    plt.tight_layout()
    
    plot_filename = SCRIPT_DIR / 'fispact_vs_openmc_comparison_detailed.png'
    plt.savefig(plot_filename)
    print(f"\n✅ Comparison plot saved to: {plot_filename}")

if __name__ == "__main__":
    main() 