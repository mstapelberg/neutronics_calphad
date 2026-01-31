#!/usr/bin/env python3
"""Timestep sensitivity sweep for depletion irradiation step.

This script evaluates how depletion results (activity, dose, gas production)
change when varying the timestep size during the 2-year irradiation phase.

Timesteps tested: 5, 15, 30, 45, 60 days
Reference: 30 days (matching production workflow)
Material: V-4Cr-4Ti alloy

Uses pre-computed flux and microxs from the impulse library.
"""

import os
import json
import csv
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import openmc
import openmc.deplete

from neutronics_calphad.neutronics.depletion import run_independent_depletion, extract_gas_production
from neutronics_calphad.neutronics.time_scheduler import TimeScheduler
from neutronics_calphad.utils.io import create_material
from neutronics_calphad.neutronics.config import SPHERICAL
from neutronics_calphad.neutronics.geometry_maker import create_model
from neutronics_calphad.neutronics.dose import contact_dose

# OpenMC nuclear data configuration
openmc.config['chain_file'] = '/home/myless/nuclear_data/tendl21-fispact20-gefy61-chain.xml'
openmc.config['cross_sections'] = '/home/myless/nuclear_data/tendl-2021-hdf5/cross_sections.xml'

# Environment
os.environ['OMP_NUM_THREADS'] = '32'

# Timestep sweep parameters (in days)
# Note: These are timesteps for the 2-year (730 day) irradiation phase
TIMESTEP_DAYS: List[float] = [
    5.0,      # 146 steps (finest resolution - reference)
    15.0,     # 49 steps
    30.0,     # 24 steps (production default - CHOSEN)
    60.0,     # 12 steps
    90.0,     # 8 steps (~3 months)
    182.5,    # 4 steps (~6 months)
    365.0,    # 2 steps (1 year)
]

REFERENCE_TIMESTEP: float = 5.0   # Reference for comparison (finest grid)
CHOSEN_TIMESTEP: float = 30.0     # Production choice to highlight

# Fixed simulation parameters
IRRADIATION_TIME_DAYS: float = 2.0 * 365.0  # 2 years
COOLING_TIMES: List[str] = ['30 days', '1 year', '5 years', '100 years']
RESULTS_BASE_DIR: str = 'analysis_results/timestep_sweep_depletion_t3'
#FLUX_MICROXS_DIR: str = 'analysis_results/impulse_library/library/flux_microxs'
FLUX_MICROXS_DIR: str = 'analysis_results/pure_element_neutronics_run_t2/microxs_and_flux'

# Power settings (same as production)
POWER_MW: float = 500.0
TORUS_TO_SPHERE_VOLUME_RATIO: float = 1.0 / 4.03
FUSION_POWER_MEV: float = 17.6
MEV_TO_J = 1.602176634e-13

# Impurity levels (weight ppm)
IMPURITY_WPPM: Dict[str, float] = {
    'C': 60.0,   # 60 wppm
    'N': 120.0,  # 120 wppm
    'O': 150.0,  # 150 wppm
}

"""
IMPURITY_WPPM: Dict[str, float] = {
    'C': 0.0,   # 60 wppm
    'N': 0.0,  # 120 wppm
    'O': 0.0,  # 150 wppm
}
"""


def load_flux_and_microxs(flux_microxs_dir: str) -> Tuple[np.ndarray, openmc.deplete.MicroXS]:
    """Load pre-computed flux spectrum and micro cross sections.
    
    Args:
        flux_microxs_dir: Directory containing flux_spectrum_1102.txt and microxs_1102.csv.
        
    Returns:
        Tuple of (flux array, MicroXS object).
    """
    flux_file = os.path.join(flux_microxs_dir, 'flux_spectrum_1102.txt')
    microxs_file = os.path.join(flux_microxs_dir, 'microxs_1102.csv')
    
    if not os.path.exists(flux_file):
        raise FileNotFoundError(f"Flux file not found: {flux_file}")
    if not os.path.exists(microxs_file):
        raise FileNotFoundError(f"MicroXS file not found: {microxs_file}")
    
    # Load flux spectrum
    flux_data = np.loadtxt(flux_file, comments='#')
    if flux_data.ndim == 1:
        flux = flux_data
    else:
        flux = flux_data[:, 1] if flux_data.shape[1] > 1 else flux_data[:, 0]
    
    # Load microxs
    microxs = openmc.deplete.MicroXS.from_csv(microxs_file)
    
    return flux, microxs


def setup_depletion_for_timestep(
    timestep_days: float,
    material_comp: Dict[str, float],
    material_name: str,
    flux: np.ndarray,
    microxs: openmc.deplete.MicroXS,
    run_dir: str,
) -> Dict[str, Any]:
    """Run depletion with specified timestep and return results.
    
    Args:
        timestep_days: Timestep size in days.
        material_comp: Material composition dict (e.g., {'V': 0.92, 'Cr': 0.04, 'Ti': 0.04}).
        material_name: Name for the material.
        flux: Flux spectrum array.
        microxs: Micro cross sections.
        run_dir: Output directory for this run.
        
    Returns:
        Dictionary containing depletion results and metadata.
    """
    os.makedirs(run_dir, exist_ok=True)
    
    # Calculate number of steps
    n_steps = int(np.ceil(IRRADIATION_TIME_DAYS / timestep_days))
    actual_timestep_days = IRRADIATION_TIME_DAYS / n_steps
    print(f"\n  [TIMESTEP DEBUG] Requested timestep: {timestep_days} days")
    print(f"  [TIMESTEP DEBUG] Irradiation time: {IRRADIATION_TIME_DAYS} days")
    print(f"  [TIMESTEP DEBUG] Number of steps: {n_steps}")
    print(f"  [TIMESTEP DEBUG] Actual timestep: {actual_timestep_days:.2f} days")
    
    # Source rate
    source_rate = POWER_MW * 1e6 / (FUSION_POWER_MEV * MEV_TO_J) * TORUS_TO_SPHERE_VOLUME_RATIO
    print(f"  [SOURCE DEBUG] Source rate: {source_rate:.2e} n/s")
    print(f"  [SOURCE DEBUG] Power: {POWER_MW} MW")
    print(f"  [SOURCE DEBUG] TORUS_TO_SPHERE_VOLUME_RATIO: {TORUS_TO_SPHERE_VOLUME_RATIO:.4f}")
    
    # Create time scheduler with specified timestep
    scheduler = TimeScheduler(
        irradiation_time=f'{IRRADIATION_TIME_DAYS} days',
        cooling_times=COOLING_TIMES,
        source_rate=source_rate,
        irradiation_steps=n_steps,
    )
    
    timesteps, sources = scheduler.get_timesteps_and_source_rates()
    
    # Create a minimal model (we only need it for depletion, not transport)
    # Use pre-computed flux/microxs
    config = SPHERICAL.copy()
    model = create_model(config=config)
    model.settings.particles = 100  # Minimal since we're not running transport
    model.settings.batches = 2
    
    # Create material with impurities (matching production approach)
    # Atomic weights for conversion
    atomic_weights = {
        'V': 50.9415, 'Cr': 51.9961, 'Ti': 47.867, 'W': 183.84, 'Zr': 91.224,
        'C': 12.011, 'N': 14.007, 'O': 15.999
    }
    
    # Convert impurities from wppm to weight fraction
    impurity_wtfrac = {elem: (wppm / 1e6) for elem, wppm in IMPURITY_WPPM.items()}
    
    # Total impurity weight fraction
    total_impurity_wt = sum(impurity_wtfrac.values())
    
    # Base alloy gets remaining weight (1 - impurities)
    base_alloy_wt = 1.0 - total_impurity_wt
    
    # Convert base alloy atomic fractions to weight fractions
    base_wt_unnorm = {}
    for elem, at_frac in material_comp.items():
        base_wt_unnorm[elem] = at_frac * atomic_weights[elem]
    
    base_wt_sum = sum(base_wt_unnorm.values())
    base_comp_wt = {elem: (wt / base_wt_sum) * base_alloy_wt for elem, wt in base_wt_unnorm.items()}
    
    # Combine base alloy + impurities (all in weight fractions)
    full_comp_wt = base_comp_wt.copy()
    full_comp_wt.update(impurity_wtfrac)
    
    # Convert to atomic fractions for OpenMC
    moles = {elem: (wt / atomic_weights[elem]) for elem, wt in full_comp_wt.items()}
    total_moles = sum(moles.values())
    comp_atomic = {elem: (mol / total_moles) for elem, mol in moles.items()}
    
    # DEBUG: Print composition details
    print(f"\n  [DEBUG] Material: {material_name}")
    print(f"  [DEBUG] Impurity wppm: {IMPURITY_WPPM}")
    print(f"  [DEBUG] Impurity wt frac total: {total_impurity_wt:.6e}")
    print(f"  [DEBUG] Base alloy weight fraction: {base_alloy_wt:.6f}")
    print("  [DEBUG] Final weight composition:")
    for elem in sorted(full_comp_wt.keys()):
        print(f"    {elem}: {full_comp_wt[elem]:.6e} wt")
    print("  [DEBUG] Final atomic composition:")
    for elem in sorted(comp_atomic.keys()):
        print(f"    {elem}: {comp_atomic[elem]:.6f} at")
    
    test_material = create_material(comp_atomic, material_name)
    test_material.depletable = True
    
    # Get existing vessel material to copy volume
    vessel_cell = model.geometry.get_cells_by_name('vessel')[0]
    print(f"\n  [VOLUME DEBUG] Timestep: {timestep_days} days, Material: {material_name}")
    print(f"  [VOLUME DEBUG] Vessel cell: {vessel_cell.name}")
    print(f"  [VOLUME DEBUG] Vessel cell fill: {vessel_cell.fill.name if vessel_cell.fill else 'None'}")
    
    if vessel_cell.fill and hasattr(vessel_cell.fill, 'volume') and vessel_cell.fill.volume:
        existing_volume = vessel_cell.fill.volume
        print(f"  [VOLUME DEBUG] Using existing vessel volume: {existing_volume:.2e} cm^3")
    else:
        existing_volume = 1000.0
        print(f"  [VOLUME DEBUG] WARNING: No existing volume found, using default: {existing_volume:.2e} cm^3")
    
    test_material.volume = existing_volume
    print(f"  [VOLUME DEBUG] Test material volume set to: {test_material.volume:.2e} cm^3")
    
    # DEBUG: Print OpenMC material
    print("  [DEBUG] OpenMC Material created:")
    print(f"    Name: {test_material.name}")
    print(f"    Volume: {test_material.volume} cm^3")
    print(f"    Depletable: {test_material.depletable}")
    print(f"    Nuclides: {[str(n) for n in test_material.nuclides]}")
    
    # Add to model
    model.materials.append(test_material)
    vessel_cell.fill = test_material
    
    # DEBUG: Print vessel cell info
    print("  [DEBUG] Vessel cell:")
    print(f"    Name: {vessel_cell.name}")
    print(f"    Fill: {vessel_cell.fill.name if vessel_cell.fill else 'None'}")
    
    # Run depletion
    material_outdir = os.path.join(run_dir, 'depletion_results', material_name)
    _ = run_independent_depletion(
        model=model,
        depletable_cell='vessel',
        microxs=microxs,
        flux=[flux],
        chain_file=openmc.config['chain_file'],
        timesteps=timesteps,
        source_rates=sources,
        outdir=material_outdir,
    )
    
    return {
        'timestep_days': float(timestep_days),
        'actual_timestep_days': float(actual_timestep_days),
        'n_steps': int(n_steps),
        'results_path': os.path.join(material_outdir, 'depletion_results.h5'),
        'run_dir': run_dir,
        'material_name': material_name,
    }


def read_depletion_results(results_path: str, material_name: str) -> Dict[str, Any]:
    """Extract activity, dose, and gas production from depletion results.
    
    Args:
        results_path: Path to depletion_results.h5.
        material_name: Material name for reference.
        
    Returns:
        Dictionary with cooling times, activities, dose rates, gas production,
        and specific dose values at target cooling times.
    """
    if not os.path.exists(results_path):
        print(f"Warning: Results file not found: {results_path}")
        return {}
    
    results = openmc.deplete.Results(results_path)
    
    # Get source rates to identify shutdown time
    source_rates = np.array([step.source_rate if step.source_rate is not None else 0 for step in results])
    
    # Target cooling times in seconds
    TARGET_COOLING_TIMES = {
        '30d': 30.0 * 24.0 * 3600.0,
        '1y': 365.0 * 24.0 * 3600.0,
        '5y': 5.0 * 365.0 * 24.0 * 3600.0,
        '100y': 100.0 * 365.0 * 24.0 * 3600.0,
    }
    
    try:
        material_id = list(results[0].index_mat.keys())[0]
        times_activity, activity_by_nuclide = results.get_activity(material_id, units='Bq/kg', by_nuclide=True)
        
        # Compute dose rates
        try:
            times_dose, dose_dicts = contact_dose(
                results,
                chain_file=openmc.config['chain_file'],
                abs_file='/home/myless/Packages/fispact/nuclear_data/decay/abs_2012',
            )
        except Exception as dose_error:
            print(f"Warning: Dose calculation failed: {dose_error}")
            times_dose = times_activity
            dose_dicts = [{} for _ in times_activity]
        
        # Extract gas production
        gas_production = extract_gas_production(results)
        
        # Identify shutdown and cooling phases
        # source_rates[i] is the rate DURING step i, so irradiation ENDS at times[irr_indices[-1] + 1]
        irr_indices = np.nonzero(source_rates)[0]
        if len(irr_indices) > 0:
            last_irr_step_idx = int(irr_indices[-1])
        else:
            last_irr_step_idx = 0
        
        # End of irradiation (shutdown) is at times[last_irr_step_idx + 1]
        # plot_start_idx is the first cooling data point (at shutdown or after)
        plot_start_idx = last_irr_step_idx + 1
        if plot_start_idx >= len(times_activity):
            print(f"Warning: Not enough cooling steps for {material_name}")
            return {}
        
        # shutdown_time is the END of irradiation (= times at plot_start_idx)
        shutdown_time = float(times_activity[plot_start_idx])
        cooling_times = times_activity[plot_start_idx:] - shutdown_time
        
        # Collect cooling phase data
        cooling_activities: Dict[float, Dict[str, float]] = {}
        for idx in range(plot_start_idx, len(times_activity)):
            cool_time = float(times_activity[idx] - shutdown_time)
            cooling_activities[cool_time] = activity_by_nuclide[idx]
        
        cooling_total_dose_rates: Dict[float, float] = {}
        for idx in range(plot_start_idx, len(times_dose)):
            cool_time = float(times_dose[idx] - shutdown_time)
            if idx < len(dose_dicts):
                cooling_total_dose_rates[cool_time] = float(sum(dose_dicts[idx].values()))
        
        # Extract dose at specific target cooling times (interpolate if needed)
        dose_at_targets: Dict[str, float] = {}
        cooling_times_dose = np.array([times_dose[i] - shutdown_time 
                                       for i in range(plot_start_idx, len(times_dose))])
        dose_values = np.array([cooling_total_dose_rates.get(ct, 0.0) 
                               for ct in cooling_times_dose])
        
        # Tolerance for matching cooling times: must be within 20% of target
        COOLING_TIME_TOLERANCE = 0.20
        
        for label, target_time in TARGET_COOLING_TIMES.items():
            if len(cooling_times_dose) > 0 and len(dose_values) > 0:
                # Find closest cooling time
                idx = np.argmin(np.abs(cooling_times_dose - target_time))
                closest_time = cooling_times_dose[idx]
                # Check if closest time is actually close to the target
                relative_error = abs(closest_time - target_time) / target_time if target_time > 0 else float('inf')
                if relative_error <= COOLING_TIME_TOLERANCE:
                    dose_at_targets[label] = float(dose_values[idx])
                else:
                    # No valid cooling time match - simulation doesn't have this point
                    print(f"    Warning: No valid {label} cooling time for {material_name} "
                          f"(closest={closest_time/86400:.1f}d vs target={target_time/86400:.1f}d)")
                    dose_at_targets[label] = float('nan')
            else:
                dose_at_targets[label] = float('nan')
        
        # Get nuclides list
        nuclides = list(activity_by_nuclide[0].keys()) if activity_by_nuclide else []
        
        return {
            'cooling_times': np.array(cooling_times, dtype=float),
            'nuclides': nuclides,
            'cooling_activities': cooling_activities,
            'cooling_total_dose_rates': cooling_total_dose_rates,
            'dose_at_targets': dose_at_targets,
            'gas_production': gas_production,
            'material_name': material_name,
        }
    except Exception as exc:
        print(f"Error reading depletion results for {material_name}: {exc}")
        return {}


def compute_total_activity_time_series(depletion_data: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    """Compute total activity over cooling time.
    
    Args:
        depletion_data: Dictionary from read_depletion_results.
        
    Returns:
        Tuple of (cooling_times_seconds, total_activity_Bq).
    """
    times_seconds = np.asarray(depletion_data['cooling_times'], dtype=float)
    activities = depletion_data['cooling_activities']
    
    total_activity: List[float] = []
    for time in times_seconds:
        activity_values = activities.get(float(time), {})
        if isinstance(activity_values, dict):
            total_activity.append(float(np.sum(list(activity_values.values()))))
        else:
            total_activity.append(0.0)
    
    return times_seconds, np.asarray(total_activity, dtype=float)


def run_timestep_sweep(
    timestep_days_list: List[float],
    test_materials: List[Dict[str, float]],
    base_results_dir: str,
    flux_microxs_dir: str,
) -> Tuple[Dict[float, Dict[str, Dict[str, np.ndarray]]], Dict[float, Dict[str, Dict[str, float]]], Dict[float, Dict[str, Dict[str, float]]]]:
    """Execute depletion runs across timestep settings.
    
    Args:
        timestep_days_list: List of timestep sizes in days.
        test_materials: List of material compositions.
        base_results_dir: Root directory for outputs.
        flux_microxs_dir: Directory with pre-computed flux/microxs.
        
    Returns:
        Tuple of (activity_by_timestep, gas_by_timestep, dose_targets_by_timestep).
    """
    os.makedirs(base_results_dir, exist_ok=True)
    
    # Load flux and microxs once
    print(f"Loading flux and microxs from {flux_microxs_dir}...")
    flux, microxs = load_flux_and_microxs(flux_microxs_dir)
    
    activity_by_timestep: Dict[float, Dict[str, Dict[str, np.ndarray]]] = {}
    gas_by_timestep: Dict[float, Dict[str, Dict[str, float]]] = {}
    dose_targets_by_timestep: Dict[float, Dict[str, Dict[str, float]]] = {}
    
    for timestep_days in timestep_days_list:
        print(f"\n=== Timestep: {timestep_days} days ===")
        run_dir = os.path.join(base_results_dir, f'timestep_{timestep_days:.0f}days')
        os.makedirs(run_dir, exist_ok=True)
        
        for test_material in test_materials:
            material_name = f"V-{test_material.get('Cr', 0)*100:.0f}Cr-{test_material.get('Ti', 0)*100:.0f}Ti"
            results_file = os.path.join(run_dir, 'depletion_results', material_name, 'depletion_results.h5')
            
            if os.path.exists(results_file):
                print(f"  - Found existing results for {material_name} at timestep {timestep_days}d")
            else:
                print(f"  - Running depletion for {material_name} with timestep {timestep_days}d")
                metadata = setup_depletion_for_timestep(
                    timestep_days=timestep_days,
                    material_comp=test_material,
                    material_name=material_name,
                    flux=flux,
                    microxs=microxs,
                    run_dir=run_dir,
                )
                results_file = metadata['results_path']
            
            # Read results
            dep_data = read_depletion_results(results_file, material_name)
            if not dep_data:
                continue
            
            times_s, total_activity = compute_total_activity_time_series(dep_data)
            activity_by_timestep.setdefault(timestep_days, {})[material_name] = {
                'times': times_s,
                'activity': total_activity,
            }
            gas_by_timestep.setdefault(timestep_days, {})[material_name] = dep_data.get('gas_production', {})
            dose_targets_by_timestep.setdefault(timestep_days, {})[material_name] = dep_data.get('dose_at_targets', {})
    
    return activity_by_timestep, gas_by_timestep, dose_targets_by_timestep


def plot_activity_sweep(
    activity_by_timestep: Dict[float, Dict[str, Dict[str, np.ndarray]]],
    outdir: str,
) -> List[str]:
    """Plot total activity vs cooling time for each timestep setting.
    
    Args:
        activity_by_timestep: Mapping of timestep -> material -> activity data.
        outdir: Output directory for plots.
        
    Returns:
        List of saved plot paths.
    """
    os.makedirs(outdir, exist_ok=True)
    
    # Collect material names
    material_names: List[str] = sorted({
        material
        for per_timestep in activity_by_timestep.values()
        for material in per_timestep.keys()
    })
    
    # Custom colors
    custom_colors = ['#2A33C3', '#A35D00', '#0B7285', '#8F2D56', '#6E8B00', '#D97706']
    
    saved_paths: List[str] = []
    for material_name in material_names:
        plt.figure(figsize=(12, 8))
        timesteps_sorted = sorted(activity_by_timestep.keys())
        colors = [custom_colors[i % len(custom_colors)] for i in range(len(timesteps_sorted))]
        
        for color, timestep in zip(colors, timesteps_sorted):
            per_mat = activity_by_timestep.get(timestep, {}).get(material_name)
            if not per_mat:
                continue
            times = per_mat['times']
            activity = per_mat['activity']
            times_plot = np.where(times <= 0, np.nan, times)
            activity_plot = np.where(activity <= 0, np.nan, activity)
            plt.plot(
                times_plot,
                activity_plot,
                label=f'Timestep={timestep:.0f}d',
                color=color,
                marker='o',
                linewidth=2,
            )
        
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('Cooling Time (s)', fontsize=14, fontweight='bold')
        plt.ylabel('Total Activity (Bq/kg)', fontsize=14, fontweight='bold')
        plt.title(f'{material_name}: Total Activity vs Cooling Time (Timestep Sweep)',
                 fontsize=16, fontweight='bold')
        plt.tick_params(axis='both', which='major', labelsize=12)
        plt.grid(True, which='both', alpha=0.3)
        plt.legend(fontsize=11)
        plt.tight_layout()
        
        out_png = os.path.join(outdir, f'activity_timestep_sweep_{material_name}.png')
        out_pdf = os.path.join(outdir, f'activity_timestep_sweep_{material_name}.pdf')
        plt.savefig(out_png, dpi=300, bbox_inches='tight')
        plt.savefig(out_pdf, bbox_inches='tight')
        plt.close()
        saved_paths.append(out_png)
    
    return saved_paths


def plot_gas_sweep(
    gas_by_timestep: Dict[float, Dict[str, Dict[str, float]]],
    outdir: str,
) -> str:
    """Plot He and H production across timestep settings.
    
    Args:
        gas_by_timestep: Mapping of timestep -> material -> gas production.
        outdir: Output directory.
        
    Returns:
        Path to saved plot.
    """
    os.makedirs(outdir, exist_ok=True)
    
    material_names: List[str] = sorted({
        material
        for per_timestep in gas_by_timestep.values()
        for material in per_timestep.keys()
    })
    
    if not material_names:
        return ""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    custom_colors = ['#2A33C3', '#A35D00', '#0B7285', '#8F2D56', '#6E8B00', '#D97706']
    
    timesteps_sorted = sorted(gas_by_timestep.keys())
    x_positions = np.arange(len(timesteps_sorted))
    width = 0.35
    
    for mat_idx, material_name in enumerate(material_names):
        he_values = []
        h_values = []
        labels = []
        color = custom_colors[mat_idx % len(custom_colors)]
        
        for timestep in timesteps_sorted:
            gas_data = gas_by_timestep.get(timestep, {}).get(material_name, {})
            he_values.append(gas_data.get('He_appm', 0.0))
            h_values.append(gas_data.get('H_appm', 0.0))
            labels.append(f'{timestep:.0f}d')
        
        # He production
        ax1.bar(x_positions + mat_idx * width, he_values, width,
                label=material_name, alpha=0.8, color=color)
        ax1.axhline(y=396, color='r', linestyle='--', linewidth=2,
                   label='Limit (396 appm)' if mat_idx == 0 else '')
        
        # H production
        ax2.bar(x_positions + mat_idx * width, h_values, width,
                label=material_name, alpha=0.8, color=color)
        ax2.axhline(y=1200, color='r', linestyle='--', linewidth=2,
                   label='Limit (1200 appm)' if mat_idx == 0 else '')
    
    # He plot formatting
    ax1.set_xlabel('Timestep Size (days)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('He Production (appm)', fontsize=14, fontweight='bold')
    ax1.set_title('He Production vs. Timestep', fontsize=16, fontweight='bold')
    ax1.set_xticks(x_positions + width * (len(material_names) - 1) / 2)
    ax1.set_xticklabels(labels, fontsize=11)
    ax1.tick_params(axis='y', labelsize=12)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # H plot formatting
    ax2.set_xlabel('Timestep Size (days)', fontsize=14, fontweight='bold')
    ax2.set_ylabel('H Production (appm)', fontsize=14, fontweight='bold')
    ax2.set_title('H Production vs. Timestep', fontsize=16, fontweight='bold')
    ax2.set_xticks(x_positions + width * (len(material_names) - 1) / 2)
    ax2.set_xticklabels(labels, fontsize=11)
    ax2.tick_params(axis='y', labelsize=12)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    out_png = os.path.join(outdir, 'gas_production_timestep_sweep.png')
    out_pdf = os.path.join(outdir, 'gas_production_timestep_sweep.pdf')
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.savefig(out_pdf, bbox_inches='tight')
    plt.close()
    return out_png


def _relative_l2(a: np.ndarray, b: np.ndarray) -> float:
    """Compute relative L2 norm ||a - b|| / ||b||.
    
    Args:
        a: Test array.
        b: Reference array.
        
    Returns:
        Relative L2 norm.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    denom = np.linalg.norm(b)
    if denom == 0.0:
        return float('nan')
    return float(np.linalg.norm(a - b) / denom)


def compute_relative_deviation_dose_targets(
    dose_targets_by_timestep: Dict[float, Dict[str, Dict[str, float]]],
    reference_timestep: float,
) -> Dict[str, Dict[float, Dict[str, float]]]:
    """Compute relative deviation for each dose target vs reference timestep.
    
    Args:
        dose_targets_by_timestep: Dose data by timestep (timestep -> material -> target_label -> dose).
        reference_timestep: Reference timestep in days.
        
    Returns:
        Mapping of target_label -> timestep -> material -> relative deviation.
        Relative deviation is computed as |dose - ref_dose| / ref_dose.
    """
    if reference_timestep not in dose_targets_by_timestep:
        return {}
    
    ref_materials = dose_targets_by_timestep[reference_timestep]
    
    # Get all target labels (30d, 1y, 5y, 100y)
    target_labels = set()
    for per_mat in dose_targets_by_timestep.values():
        for targets in per_mat.values():
            target_labels.update(targets.keys())
    
    # out[target_label][timestep][material] = deviation
    out: Dict[str, Dict[float, Dict[str, float]]] = {label: {} for label in target_labels}
    
    for timestep, per_mat in dose_targets_by_timestep.items():
        for target_label in target_labels:
            out[target_label][timestep] = {}
            for material_name, targets in per_mat.items():
                if material_name not in ref_materials:
                    out[target_label][timestep][material_name] = float('nan')
                    continue
                
                ref_dose = ref_materials[material_name].get(target_label, 0.0)
                cur_dose = targets.get(target_label, 0.0)
                
                if ref_dose == 0.0:
                    out[target_label][timestep][material_name] = float('nan')
                else:
                    out[target_label][timestep][material_name] = abs(cur_dose - ref_dose) / ref_dose
    
    return out


def compute_relative_deviation_gas(
    gas_by_timestep: Dict[float, Dict[str, Dict[str, float]]],
    reference_timestep: float,
) -> Dict[float, Dict[str, Dict[str, float]]]:
    """Compute relative deviation of gas production vs reference timestep.
    
    Args:
        gas_by_timestep: Gas production by timestep.
        reference_timestep: Reference timestep in days.
        
    Returns:
        Mapping of timestep -> material -> gas deviation dict.
    """
    if reference_timestep not in gas_by_timestep:
        return {}
    
    ref_materials = gas_by_timestep[reference_timestep]
    out: Dict[float, Dict[str, Dict[str, float]]] = {}
    
    for timestep, per_mat in gas_by_timestep.items():
        out[timestep] = {}
        for material_name, gas_data in per_mat.items():
            if material_name not in ref_materials:
                out[timestep][material_name] = {'He_rel_dev': float('nan'), 'H_rel_dev': float('nan')}
                continue
            
            ref_gas = ref_materials[material_name]
            he_ref = ref_gas.get('He_appm', 0.0)
            he_cur = gas_data.get('He_appm', 0.0)
            h_ref = ref_gas.get('H_appm', 0.0)
            h_cur = gas_data.get('H_appm', 0.0)
            
            he_dev = abs(he_cur - he_ref) / he_ref if he_ref > 0 else float('nan')
            h_dev = abs(h_cur - h_ref) / h_ref if h_ref > 0 else float('nan')
            
            out[timestep][material_name] = {
                'He_rel_dev': he_dev,
                'H_rel_dev': h_dev,
            }
    
    return out


def save_metrics(
    outdir: str,
    dose_deviation: Dict[str, Dict[float, Dict[str, float]]],
    gas_deviation: Dict[float, Dict[str, Dict[str, float]]],
) -> Tuple[str, str]:
    """Save deviation metrics to JSON and CSV.
    
    Args:
        outdir: Output directory.
        dose_deviation: Dose deviations by target (target -> timestep -> material -> deviation).
        gas_deviation: Gas production deviations.
        
    Returns:
        Tuple of (json_path, csv_path).
    """
    os.makedirs(outdir, exist_ok=True)
    json_path = os.path.join(outdir, 'timestep_deviation_metrics.json')
    csv_path = os.path.join(outdir, 'timestep_deviation_metrics.csv')
    
    # JSON
    jsonable = {
        'dose_targets': {
            target: {f'timestep_{t}d': per_mat for t, per_mat in by_timestep.items()}
            for target, by_timestep in dose_deviation.items()
        },
        'gas': {f'timestep_{t}d': per_mat for t, per_mat in gas_deviation.items()},
    }
    with open(json_path, 'w') as jf:
        json.dump(jsonable, jf, indent=2)
    
    # CSV
    with open(csv_path, 'w', newline='') as cf:
        writer = csv.writer(cf)
        writer.writerow(['timestep_days', 'metric', 'target', 'material', 'relative_deviation'])
        # Dose targets
        for target_label, by_timestep in dose_deviation.items():
            for timestep, per_mat in by_timestep.items():
                for mat, val in per_mat.items():
                    writer.writerow([timestep, 'dose_rate', target_label, mat, val])
        # Gas production
        for timestep, per_mat in gas_deviation.items():
            for mat, gas_devs in per_mat.items():
                writer.writerow([timestep, 'gas_production', 'He', mat, gas_devs.get('He_rel_dev', float('nan'))])
                writer.writerow([timestep, 'gas_production', 'H', mat, gas_devs.get('H_rel_dev', float('nan'))])
    
    return json_path, csv_path


def plot_convergence_chart(
    dose_deviation: Dict[str, Dict[float, Dict[str, float]]],
    gas_deviation: Dict[float, Dict[str, Dict[str, float]]],
    activity_by_timestep: Dict[float, Dict[str, Dict[str, np.ndarray]]],
    outdir: str,
) -> str:
    """Create simplified convergence chart for activity and gas production vs timestep.
    
    All deviations are relative to the 5-day reference timestep.
    The 30-day production choice is highlighted.
    
    Args:
        dose_deviation: Dose deviations by target (unused for now).
        gas_deviation: Gas deviations by timestep.
        activity_by_timestep: Activity time series by timestep.
        outdir: Output directory.
        
    Returns:
        Path to saved plot.
    """
    os.makedirs(outdir, exist_ok=True)
    
    # Create 2x2 grid: Activity time series + He + H + summary
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    ax_activity = fig.add_subplot(gs[0, :])  # Top row spans both columns
    ax_he = fig.add_subplot(gs[1, 0])
    ax_h = fig.add_subplot(gs[1, 1])
    
    # Expanded color palette (no reuse)
    colors = [
        '#2A33C3',  # Blue
        '#A35D00',  # Orange
        '#0B7285',  # Teal
        '#8F2D56',  # Magenta
        '#6E8B00',  # Olive
        '#D97706',  # Amber
        '#7C3AED',  # Violet
        '#DC2626',  # Red
        '#059669',  # Emerald
        '#7C2D12',  # Brown
    ]
    
    # Get timesteps and materials
    all_timesteps = sorted(gas_deviation.keys())
    material_names = sorted({mat for per_t in gas_deviation.values() for mat in per_t.keys()})
    
    # Compute activity time series deviation (L2 norm across cooling times)
    ref_timestep = REFERENCE_TIMESTEP
    chosen_timestep = CHOSEN_TIMESTEP
    
    activity_deviations: Dict[float, Dict[str, float]] = {}
    if ref_timestep in activity_by_timestep:
        ref_materials = activity_by_timestep[ref_timestep]
        for timestep, per_mat in activity_by_timestep.items():
            activity_deviations[timestep] = {}
            for material_name, series in per_mat.items():
                if material_name not in ref_materials:
                    activity_deviations[timestep][material_name] = float('nan')
                    continue
                a_ref = ref_materials[material_name]['activity']
                a_cur = series['activity']
                n = min(len(a_ref), len(a_cur))
                if n == 0:
                    activity_deviations[timestep][material_name] = float('nan')
                    continue
                # Relative L2 norm
                denom = np.linalg.norm(a_ref[:n])
                if denom == 0.0:
                    activity_deviations[timestep][material_name] = float('nan')
                else:
                    activity_deviations[timestep][material_name] = float(np.linalg.norm(a_cur[:n] - a_ref[:n]) / denom)
    
    # Plot 1: Activity time series deviation
    for mat_idx, material in enumerate(material_names):
        color = colors[mat_idx % len(colors)]
        deviations = [activity_deviations.get(t, {}).get(material, np.nan) for t in all_timesteps]
        
        # Plot all points
        ax_activity.plot(all_timesteps, deviations, marker='o', linewidth=2.5,
                        label=material, color=color, markersize=8)
        
        # Highlight chosen timestep (30 days)
        if chosen_timestep in all_timesteps:
            idx = all_timesteps.index(chosen_timestep)
            if idx < len(deviations) and np.isfinite(deviations[idx]):
                ax_activity.plot(chosen_timestep, deviations[idx], marker='*', 
                               markersize=20, color=color, markeredgecolor='black',
                               markeredgewidth=2, zorder=10)
    
    ax_activity.set_xlabel('Timestep Size (days)', fontsize=14, fontweight='bold')
    ax_activity.set_ylabel('Relative L2 Deviation', fontsize=14, fontweight='bold')
    ax_activity.set_title('Total Activity Time Series Convergence',
                         fontsize=15, fontweight='bold')
    ax_activity.tick_params(axis='both', which='major', labelsize=12)
    ax_activity.grid(True, alpha=0.3, linestyle='--')
    ax_activity.axhline(y=0.02, color='red', linestyle='--', alpha=0.6, linewidth=2, label='2% threshold')
    ax_activity.axhline(y=0.05, color='orange', linestyle='--', alpha=0.6, linewidth=2, label='5% threshold')
    ax_activity.legend(fontsize=11, loc='best', ncol=2)
    
    # Set tight y-axis limits to zoom into small deviations
    all_devs = [d for devs in [activity_deviations.get(t, {}).values() for t in all_timesteps] for d in devs if np.isfinite(d)]
    if all_devs:
        max_dev = max(all_devs)
        ax_activity.set_ylim(0, max(0.06, max_dev * 1.15))  # At least show up to 6%, or 15% above max
    else:
        ax_activity.set_ylim(0, 0.06)
    
    # Plot 2: He production deviation
    for mat_idx, material in enumerate(material_names):
        color = colors[mat_idx % len(colors)]
        he_devs = [gas_deviation.get(t, {}).get(material, {}).get('He_rel_dev', np.nan) 
                   for t in all_timesteps]
        
        ax_he.plot(all_timesteps, he_devs, marker='o', linewidth=2.5,
                  label=material, color=color, markersize=8)
        
        # Highlight chosen timestep
        if chosen_timestep in all_timesteps:
            idx = all_timesteps.index(chosen_timestep)
            if idx < len(he_devs) and np.isfinite(he_devs[idx]):
                ax_he.plot(chosen_timestep, he_devs[idx], marker='*',
                          markersize=20, color=color, markeredgecolor='black',
                          markeredgewidth=2, zorder=10)
    
    ax_he.set_xlabel('Timestep Size (days)', fontsize=14, fontweight='bold')
    ax_he.set_ylabel('Relative Deviation', fontsize=14, fontweight='bold')
    ax_he.set_title('He Production @ 2y',
                   fontsize=15, fontweight='bold')
    ax_he.tick_params(axis='both', which='major', labelsize=12)
    ax_he.grid(True, alpha=0.3, linestyle='--')
    ax_he.axhline(y=0.02, color='red', linestyle='--', alpha=0.6, linewidth=2, label='2% threshold')
    ax_he.axhline(y=0.05, color='orange', linestyle='--', alpha=0.6, linewidth=2, label='5% threshold')
    ax_he.legend(fontsize=11, loc='best')
    
    # Set tight y-axis limits
    all_he_devs = [gas_deviation.get(t, {}).get(m, {}).get('He_rel_dev', np.nan) 
                   for t in all_timesteps for m in material_names]
    all_he_devs = [d for d in all_he_devs if np.isfinite(d)]
    if all_he_devs:
        max_he = max(all_he_devs)
        ax_he.set_ylim(0, max(0.06, max_he * 1.15))
    else:
        ax_he.set_ylim(0, 0.06)
    
    # Plot 3: H production deviation
    for mat_idx, material in enumerate(material_names):
        color = colors[mat_idx % len(colors)]
        h_devs = [gas_deviation.get(t, {}).get(material, {}).get('H_rel_dev', np.nan) 
                  for t in all_timesteps]
        
        ax_h.plot(all_timesteps, h_devs, marker='o', linewidth=2.5,
                 label=material, color=color, markersize=8)
        
        # Highlight chosen timestep
        if chosen_timestep in all_timesteps:
            idx = all_timesteps.index(chosen_timestep)
            if idx < len(h_devs) and np.isfinite(h_devs[idx]):
                ax_h.plot(chosen_timestep, h_devs[idx], marker='*',
                         markersize=20, color=color, markeredgecolor='black',
                         markeredgewidth=2, zorder=10)
    
    ax_h.set_xlabel('Timestep Size (days)', fontsize=14, fontweight='bold')
    ax_h.set_ylabel('Relative Deviation', fontsize=14, fontweight='bold')
    ax_h.set_title('H Production @ 2y',
                  fontsize=15, fontweight='bold')
    ax_h.tick_params(axis='both', which='major', labelsize=12)
    ax_h.grid(True, alpha=0.3, linestyle='--')
    ax_h.axhline(y=0.02, color='red', linestyle='--', alpha=0.6, linewidth=2, label='2% threshold')
    ax_h.axhline(y=0.05, color='orange', linestyle='--', alpha=0.6, linewidth=2, label='5% threshold')
    ax_h.legend(fontsize=11, loc='best')
    
    # Set tight y-axis limits
    all_h_devs = [gas_deviation.get(t, {}).get(m, {}).get('H_rel_dev', np.nan) 
                  for t in all_timesteps for m in material_names]
    all_h_devs = [d for d in all_h_devs if np.isfinite(d)]
    if all_h_devs:
        max_h = max(all_h_devs)
        ax_h.set_ylim(0, max(0.06, max_h * 1.15))
    else:
        ax_h.set_ylim(0, 0.06)
    
    plt.tight_layout()
    out_png = os.path.join(outdir, 'timestep_convergence.png')
    out_pdf = os.path.join(outdir, 'timestep_convergence.pdf')
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.savefig(out_pdf, bbox_inches='tight')
    plt.close()
    return out_png


if __name__ == '__main__':
    """Run timestep sensitivity sweep for depletion."""
    
    # Test material: V-4Cr-4Ti
    test_materials: List[Dict[str, float]] = [
        {'V': 0.92, 'Cr': 0.04, 'Ti': 0.04},
    ]
    
    print("=" * 60)
    print("TIMESTEP SENSITIVITY SWEEP FOR DEPLETION")
    print("=" * 60)
    print(f"Timesteps to test: {TIMESTEP_DAYS} days")
    print(f"Reference timestep: {REFERENCE_TIMESTEP} days")
    print(f"Irradiation time: {IRRADIATION_TIME_DAYS} days ({IRRADIATION_TIME_DAYS/365:.1f} years)")
    print(f"Flux/MicroXS from: {FLUX_MICROXS_DIR}")
    print(f"Impurities: C={IMPURITY_WPPM['C']} wppm, N={IMPURITY_WPPM['N']} wppm, O={IMPURITY_WPPM['O']} wppm")
    print(f"Materials: {test_materials}")
    print()
    
    # Run sweep
    activity_by_timestep, gas_by_timestep, dose_targets_by_timestep = run_timestep_sweep(
        timestep_days_list=TIMESTEP_DAYS,
        test_materials=test_materials,
        base_results_dir=RESULTS_BASE_DIR,
        flux_microxs_dir=FLUX_MICROXS_DIR,
    )
    
    # Generate plots
    plots_dir = os.path.join(RESULTS_BASE_DIR, 'analysis_plots')
    
    activity_plot_paths = plot_activity_sweep(activity_by_timestep, plots_dir)
    gas_plot_path = plot_gas_sweep(gas_by_timestep, plots_dir)
    
    # Compute deviations vs reference (5-day timestep)
    metrics_dir = os.path.join(RESULTS_BASE_DIR, 'metrics')
    dose_dev = compute_relative_deviation_dose_targets(dose_targets_by_timestep, REFERENCE_TIMESTEP)
    gas_dev = compute_relative_deviation_gas(gas_by_timestep, REFERENCE_TIMESTEP)
    
    metrics_json, metrics_csv = save_metrics(metrics_dir, dose_dev, gas_dev)
    convergence_plot = plot_convergence_chart(dose_dev, gas_dev, activity_by_timestep, plots_dir)
    
    # Summary
    print('\n' + '=' * 60)
    print('TIMESTEP SWEEP COMPLETE')
    print('=' * 60)
    for ap in activity_plot_paths:
        print(f'- Activity plots saved to: {ap}')
    if gas_plot_path:
        print(f'- Gas production plot saved to: {gas_plot_path}')
    print(f'- Convergence plot saved to: {convergence_plot}')
    print(f'- Metrics saved to: {metrics_json} and {metrics_csv}')
    
    print(f"\nReference timestep: {REFERENCE_TIMESTEP} days (finest resolution)")
    print(f"Production choice: {CHOSEN_TIMESTEP} days (highlighted in plots)")
    print("\nTimestep Sensitivity Summary:")
    print("-" * 60)
    print("\nRelative deviation = |value - ref_value| / ref_value")
    print("  where ref_value is from the 5-day (finest) timestep")
    print("A deviation of 0.02 = 2% change, 0.10 = 10% change, etc.")
    print()
    
    # Find max deviations for each dose target
    if dose_dev:
        print("Dose Rate Targets:")
        dose_target_info = {
            '30d': ('30 days', 1e3),
            '1y': ('1 year', 1.0),
            '5y': ('5 years', 1e-2),
            '100y': ('100 years', 1e-4),
        }
        
        for target_label, (target_name, target_limit) in dose_target_info.items():
            if target_label in dose_dev:
                max_dev = 0.0
                max_timestep = None
                max_mat = None
                by_timestep = dose_dev[target_label]
                for timestep, mat_dict in by_timestep.items():
                    for mat, val in mat_dict.items():
                        if np.isfinite(val) and val > max_dev:
                            max_dev = val
                            max_timestep = timestep
                            max_mat = mat
                if max_timestep:
                    print(f"  {target_name} (target: {target_limit:.0e} Sv/h):")
                    print(f"    Max deviation: {max_dev:.1%} at timestep {max_timestep} days ({max_mat})")
    
    if gas_dev:
        print("\nGas Production Targets:")
        max_he_dev = 0.0
        max_he_timestep = None
        max_h_dev = 0.0
        max_h_timestep = None
        for timestep, mat_dict in gas_dev.items():
            for mat, gas_devs in mat_dict.items():
                he_val = gas_devs.get('He_rel_dev', 0.0)
                h_val = gas_devs.get('H_rel_dev', 0.0)
                if np.isfinite(he_val) and he_val > max_he_dev:
                    max_he_dev = he_val
                    max_he_timestep = timestep
                if np.isfinite(h_val) and h_val > max_h_dev:
                    max_h_dev = h_val
                    max_h_timestep = timestep
        if max_he_timestep:
            print("  He production (target: 396 appm @ 2y):")
            print(f"    Max deviation: {max_he_dev:.1%} at timestep {max_he_timestep} days")
        if max_h_timestep:
            print("  H production (target: 1200 appm @ 2y):")
            print(f"    Max deviation: {max_h_dev:.1%} at timestep {max_h_timestep} days")
    
    print("\n" + "-" * 60)
    print("Recommendation: Use timesteps with deviations < 2% (0.02) for accurate results.")
    print("Timesteps with deviations > 5% (0.05) should be considered too coarse.")

