#!/usr/bin/env python3
"""Compare the effect of C, N, O impurities on gas production across timesteps.

This script compares gas production (He and H) from two timestep sweep runs:
1. With C=60wppm, N=120wppm, O=150wppm impurities
2. Without impurities (pure V-Cr-Ti-W-Zr system)
"""

import os
from pathlib import Path
from typing import Dict, List
import matplotlib.pyplot as plt
import numpy as np
import openmc.deplete

from neutronics_calphad.neutronics.depletion import extract_gas_production

# Directories for comparison
WITH_IMPURITIES_DIR = "analysis_results/timestep_sweep_depletion"
WITHOUT_IMPURITIES_DIR = "analysis_results/timestep_sweep_depletion_no_impurities"

# Reference and chosen timesteps
REFERENCE_TIMESTEP = 5.0
CHOSEN_TIMESTEP = 30.0


def load_gas_production_from_results(
    base_dir: str,
    timesteps: List[float],
) -> Dict[float, Dict[str, Dict[str, float]]]:
    """Load gas production data from depletion results.
    
    Args:
        base_dir: Base results directory containing timestep_XXdays subdirs.
        timesteps: List of timesteps to load.
        
    Returns:
        Mapping of timestep -> material -> {'He_appm', 'H_appm'}.
    """
    base_path = Path(base_dir)
    gas_by_timestep: Dict[float, Dict[str, Dict[str, float]]] = {}
    
    for timestep in timesteps:
        timestep_dir = base_path / f"timestep_{int(timestep)}days"
        if not timestep_dir.exists():
            print(f"Warning: Directory not found: {timestep_dir}")
            continue
        
        # Look for depletion results
        depletion_dir = timestep_dir / "depletion_results"
        if not depletion_dir.exists():
            print(f"Warning: Depletion directory not found: {depletion_dir}")
            continue
        
        for material_dir in depletion_dir.iterdir():
            if not material_dir.is_dir():
                continue
            
            material_name = material_dir.name
            
            # Try HDF5 file first (primary format)
            result_h5 = material_dir / "depletion_results.h5"
            if result_h5.exists():
                try:
                    results = openmc.deplete.Results(str(result_h5))
                    gas_prod = extract_gas_production(results)
                    
                    if gas_prod:
                        gas_by_timestep.setdefault(timestep, {})[material_name] = {
                            'He_appm': float(gas_prod.get('He_appm', 0.0)),
                            'H_appm': float(gas_prod.get('H_appm', 0.0)),
                        }
                except Exception as e:
                    print(f"Warning: Failed to load {result_h5}: {e}")
                continue
            
    
    return gas_by_timestep


def plot_impurity_comparison(
    gas_with: Dict[float, Dict[str, Dict[str, float]]],
    gas_without: Dict[float, Dict[str, Dict[str, float]]],
    outdir: str,
) -> str:
    """Create comparison plots showing impurity effect on gas production.
    
    Args:
        gas_with: Gas production with impurities (timestep -> material -> gas data).
        gas_without: Gas production without impurities.
        outdir: Output directory.
        
    Returns:
        Path to saved plot.
    """
    os.makedirs(outdir, exist_ok=True)
    
    # Get timesteps and materials
    all_timesteps = sorted(set(gas_with.keys()) | set(gas_without.keys()))
    
    # Get material names
    material_names = set()
    for per_t in gas_with.values():
        material_names.update(per_t.keys())
    for per_t in gas_without.values():
        material_names.update(per_t.keys())
    material_names = sorted(material_names)
    
    if not material_names:
        print("No materials found to compare")
        return ""
    
    # Create figure: 2 rows (He, H) x 2 cols (absolute values, relative difference)
    fig, axes = plt.subplots(2, 2, figsize=(18, 10))
    
    # Color palette
    colors = [
        '#2A33C3',  # Blue
        '#A35D00',  # Orange
        '#0B7285',  # Teal
        '#8F2D56',  # Magenta
    ]
    
    # Plot 1: He production absolute values
    ax_he_abs = axes[0, 0]
    for mat_idx, material in enumerate(material_names):
        color = colors[mat_idx % len(colors)]
        
        # With impurities
        he_with = [gas_with.get(t, {}).get(material, {}).get('He_appm', np.nan) 
                   for t in all_timesteps]
        ax_he_abs.plot(all_timesteps, he_with, marker='o', linewidth=2.5,
                      label=f'{material} (with C,N,O)', color=color, linestyle='-',
                      markersize=8)
        
        # Without impurities
        he_without = [gas_without.get(t, {}).get(material, {}).get('He_appm', np.nan) 
                      for t in all_timesteps]
        ax_he_abs.plot(all_timesteps, he_without, marker='s', linewidth=2.5,
                      label=f'{material} (pure)', color=color, linestyle='--',
                      markersize=8, alpha=0.7)
        
        # Highlight chosen timestep
        if CHOSEN_TIMESTEP in all_timesteps:
            idx = all_timesteps.index(CHOSEN_TIMESTEP)
            if idx < len(he_with) and np.isfinite(he_with[idx]):
                ax_he_abs.plot(CHOSEN_TIMESTEP, he_with[idx], marker='*',
                             markersize=20, color=color, markeredgecolor='black',
                             markeredgewidth=2, zorder=10)
    
    ax_he_abs.axhline(y=396, color='red', linestyle='--', alpha=0.6, linewidth=2,
                     label='Target (396 appm)')
    ax_he_abs.set_xlabel('Timestep Size (days)', fontsize=14, fontweight='bold')
    ax_he_abs.set_ylabel('He Production (appm)', fontsize=14, fontweight='bold')
    ax_he_abs.set_title('He Production @ 2y: With vs Without Impurities\n(* indicates 30-day choice)',
                       fontsize=15, fontweight='bold')
    ax_he_abs.tick_params(axis='both', which='major', labelsize=12)
    ax_he_abs.grid(True, alpha=0.3, linestyle='--')
    ax_he_abs.legend(fontsize=9, loc='best', ncol=2)
    
    # Plot 2: He relative difference due to impurities
    ax_he_rel = axes[0, 1]
    for mat_idx, material in enumerate(material_names):
        color = colors[mat_idx % len(colors)]
        
        he_with = np.array([gas_with.get(t, {}).get(material, {}).get('He_appm', np.nan) 
                            for t in all_timesteps])
        he_without = np.array([gas_without.get(t, {}).get(material, {}).get('He_appm', np.nan) 
                               for t in all_timesteps])
        
        # Relative difference: (with - without) / without
        with np.errstate(divide='ignore', invalid='ignore'):
            rel_diff = (he_with - he_without) / he_without
        
        ax_he_rel.plot(all_timesteps, rel_diff, marker='o', linewidth=2.5,
                      label=material, color=color, markersize=8)
        
        # Highlight chosen
        if CHOSEN_TIMESTEP in all_timesteps:
            idx = all_timesteps.index(CHOSEN_TIMESTEP)
            if idx < len(rel_diff) and np.isfinite(rel_diff[idx]):
                ax_he_rel.plot(CHOSEN_TIMESTEP, rel_diff[idx], marker='*',
                             markersize=20, color=color, markeredgecolor='black',
                             markeredgewidth=2, zorder=10)
    
    ax_he_rel.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=1)
    ax_he_rel.set_xlabel('Timestep Size (days)', fontsize=14, fontweight='bold')
    ax_he_rel.set_ylabel('Relative Change', fontsize=14, fontweight='bold')
    ax_he_rel.set_title('He Production: Impurity Effect\n(with - without) / without',
                       fontsize=15, fontweight='bold')
    ax_he_rel.tick_params(axis='both', which='major', labelsize=12)
    ax_he_rel.grid(True, alpha=0.3, linestyle='--')
    ax_he_rel.legend(fontsize=11, loc='best')
    
    # Plot 3: H production absolute values
    ax_h_abs = axes[1, 0]
    for mat_idx, material in enumerate(material_names):
        color = colors[mat_idx % len(colors)]
        
        # With impurities
        h_with = [gas_with.get(t, {}).get(material, {}).get('H_appm', np.nan) 
                  for t in all_timesteps]
        ax_h_abs.plot(all_timesteps, h_with, marker='o', linewidth=2.5,
                     label=f'{material} (with C,N,O)', color=color, linestyle='-',
                     markersize=8)
        
        # Without impurities
        h_without = [gas_without.get(t, {}).get(material, {}).get('H_appm', np.nan) 
                     for t in all_timesteps]
        ax_h_abs.plot(all_timesteps, h_without, marker='s', linewidth=2.5,
                     label=f'{material} (pure)', color=color, linestyle='--',
                     markersize=8, alpha=0.7)
        
        # Highlight chosen
        if CHOSEN_TIMESTEP in all_timesteps:
            idx = all_timesteps.index(CHOSEN_TIMESTEP)
            if idx < len(h_with) and np.isfinite(h_with[idx]):
                ax_h_abs.plot(CHOSEN_TIMESTEP, h_with[idx], marker='*',
                            markersize=20, color=color, markeredgecolor='black',
                            markeredgewidth=2, zorder=10)
    
    ax_h_abs.axhline(y=1200, color='red', linestyle='--', alpha=0.6, linewidth=2,
                    label='Target (1200 appm)')
    ax_h_abs.set_xlabel('Timestep Size (days)', fontsize=14, fontweight='bold')
    ax_h_abs.set_ylabel('H Production (appm)', fontsize=14, fontweight='bold')
    ax_h_abs.set_title('H Production @ 2y: With vs Without Impurities\n(* indicates 30-day choice)',
                      fontsize=15, fontweight='bold')
    ax_h_abs.tick_params(axis='both', which='major', labelsize=12)
    ax_h_abs.grid(True, alpha=0.3, linestyle='--')
    ax_h_abs.legend(fontsize=9, loc='best', ncol=2)
    
    # Plot 4: H relative difference due to impurities
    ax_h_rel = axes[1, 1]
    for mat_idx, material in enumerate(material_names):
        color = colors[mat_idx % len(colors)]
        
        h_with = np.array([gas_with.get(t, {}).get(material, {}).get('H_appm', np.nan) 
                           for t in all_timesteps])
        h_without = np.array([gas_without.get(t, {}).get(material, {}).get('H_appm', np.nan) 
                              for t in all_timesteps])
        
        # Relative difference
        with np.errstate(divide='ignore', invalid='ignore'):
            rel_diff = (h_with - h_without) / h_without
        
        ax_h_rel.plot(all_timesteps, rel_diff, marker='o', linewidth=2.5,
                     label=material, color=color, markersize=8)
        
        # Highlight chosen
        if CHOSEN_TIMESTEP in all_timesteps:
            idx = all_timesteps.index(CHOSEN_TIMESTEP)
            if idx < len(rel_diff) and np.isfinite(rel_diff[idx]):
                ax_h_rel.plot(CHOSEN_TIMESTEP, rel_diff[idx], marker='*',
                            markersize=20, color=color, markeredgecolor='black',
                            markeredgewidth=2, zorder=10)
    
    ax_h_rel.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=1)
    ax_h_rel.set_xlabel('Timestep Size (days)', fontsize=14, fontweight='bold')
    ax_h_rel.set_ylabel('Relative Change', fontsize=14, fontweight='bold')
    ax_h_rel.set_title('H Production: Impurity Effect\n(with - without) / without',
                      fontsize=15, fontweight='bold')
    ax_h_rel.tick_params(axis='both', which='major', labelsize=12)
    ax_h_rel.grid(True, alpha=0.3, linestyle='--')
    ax_h_rel.legend(fontsize=11, loc='best')
    
    plt.tight_layout()
    out_png = os.path.join(outdir, 'impurity_effect_comparison.png')
    out_pdf = os.path.join(outdir, 'impurity_effect_comparison.pdf')
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.savefig(out_pdf, bbox_inches='tight')
    plt.close()
    
    return out_png


if __name__ == '__main__':
    """Compare impurity effects on gas production."""
    
    print("=" * 60)
    print("IMPURITY EFFECT COMPARISON")
    print("=" * 60)
    print(f"With impurities: {WITH_IMPURITIES_DIR}")
    print(f"Without impurities: {WITHOUT_IMPURITIES_DIR}")
    print()
    
    # Timesteps to compare
    timesteps = [5.0, 15.0, 30.0, 60.0, 90.0, 182.5, 365.0]
    
    # Load gas production data
    print("Loading gas production data...")
    print(f"\nLoading WITH impurities from: {WITH_IMPURITIES_DIR}")
    gas_with = load_gas_production_from_results(WITH_IMPURITIES_DIR, timesteps)
    print(f"  Loaded {len(gas_with)} timesteps with data")
    
    print(f"\nLoading WITHOUT impurities from: {WITHOUT_IMPURITIES_DIR}")
    gas_without = load_gas_production_from_results(WITHOUT_IMPURITIES_DIR, timesteps)
    print(f"  Loaded {len(gas_without)} timesteps with data")
    
    if not gas_with:
        print(f"ERROR: No data found in {WITH_IMPURITIES_DIR}")
        print("Please run the timestep sweep with impurities first.")
        exit(1)
    
    if not gas_without:
        print(f"ERROR: No data found in {WITHOUT_IMPURITIES_DIR}")
        print("Please run the timestep sweep without impurities first.")
        print()
        print("To run without impurities:")
        print("1. Edit compute_data/timestep_sweep_depletion.py")
        print("2. Set IMPURITY_WPPM = {'C': 0.0, 'N': 0.0, 'O': 0.0} (line ~65-69)")
        print("3. Set RESULTS_BASE_DIR = 'analysis_results/timestep_sweep_depletion_no_impurities' (line ~55)")
        print("4. Run: python examples/scripts_for_paper/appendix_plots/compute_data/timestep_sweep_depletion.py")
        exit(1)
    
    # Create comparison plot
    outdir = "analysis_results/impurity_comparison"
    plot_path = plot_impurity_comparison(gas_with, gas_without, outdir)
    
    print()
    print("=" * 60)
    print("COMPARISON COMPLETE")
    print("=" * 60)
    print(f"Plot saved to: {plot_path}")
    
    # Print summary statistics
    print("\nSummary at 30-day timestep:")
    print("-" * 60)
    
    for material in sorted(set(gas_with.get(30.0, {}).keys()) | set(gas_without.get(30.0, {}).keys())):
        he_with = gas_with.get(30.0, {}).get(material, {}).get('He_appm', 0.0)
        he_without = gas_without.get(30.0, {}).get(material, {}).get('He_appm', 0.0)
        h_with = gas_with.get(30.0, {}).get(material, {}).get('H_appm', 0.0)
        h_without = gas_without.get(30.0, {}).get(material, {}).get('H_appm', 0.0)
        
        he_change = ((he_with - he_without) / he_without * 100) if he_without > 0 else 0
        h_change = ((h_with - h_without) / h_without * 100) if h_without > 0 else 0
        
        print(f"\n{material}:")
        print("  He production:")
        print(f"    With C,N,O: {he_with:.1f} appm")
        print(f"    Pure alloy: {he_without:.1f} appm")
        print(f"    Change: {he_change:+.1f}%")
        print("  H production:")
        print(f"    With C,N,O: {h_with:.1f} appm")
        print(f"    Pure alloy: {h_without:.1f} appm")
        print(f"    Change: {h_change:+.1f}%")
