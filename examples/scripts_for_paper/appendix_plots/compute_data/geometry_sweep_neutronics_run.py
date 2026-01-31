import os
import json
import csv
from typing import Any, Dict, List, Tuple
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import openmc
import openmc.deplete

from neutronics_calphad.neutronics.config import SPHERICAL, ELEMENT_DENSITIES
from neutronics_calphad.neutronics.depletion import run_independent_depletion, extract_gas_production
from neutronics_calphad.neutronics.flux import get_flux_and_microxs
from neutronics_calphad.neutronics.time_scheduler import TimeScheduler
from neutronics_calphad.utils.io import create_material
from neutronics_calphad.neutronics.geometry_maker import create_model
from neutronics_calphad.neutronics.dose import contact_dose


# OpenMC nuclear data configuration
openmc.config['chain_file'] = '/home/myless/nuclear_data/tendl21-fispact20-gefy61-chain.xml'
openmc.config['cross_sections'] = '/home/myless/nuclear_data/tendl-2021-hdf5/cross_sections.xml'


# Environment and defaults
os.environ['OMP_NUM_THREADS'] = '32'

# Geometry sweep parameters
FIRST_WALL_THICKNESSES_CM: List[float] = [0.1, 0.2, 0.5, 1.0]  # 1, 2, 5, 10 mm
VESSEL_THICKNESSES_CM: List[float] = [0.5, 1.0, 2.0, 5.0]  # 0.5, 1, 2, 5 cm
REFERENCE_GEOMETRY: Tuple[float, float] = (0.2, 1.0)  # (first_wall_cm, vessel_cm)

# Fixed simulation parameters
FIXED_BATCHES: int = 10
FIXED_PARTICLES: int = 10000
RESULTS_BASE_DIR: str = os.path.join('analysis_results', 'geometry_sweep_neutronics_run')
ANALYZE_ONLY: bool = False  # If True, skip simulations and only analyze existing results

# Tolerance gates for relative L2 norms
TOLERANCE_FLUX_REL_L2: float = 0.02  # 2% default
TOLERANCE_ACTIVITY_REL_L2: float = 0.02  # 2% default

# Dose times (in hours) matching the production code
DOSE_TIMES_D: List[float] = [30.0, 365.0, 5 * 365.0, 100 * 365.0]  # 30d, 1y, 5y, 100y
DOSE_TIMES_H: List[float] = [24.0 * d for d in DOSE_TIMES_D]


def create_geometry_config(first_wall_cm: float, vessel_cm: float) -> Dict[str, Any]:
    """Create a modified SPHERICAL config with specified layer thicknesses.
    
    Args:
        first_wall_cm: First wall thickness in cm.
        vessel_cm: Vessel thickness in cm.
        
    Returns:
        Modified configuration dictionary.
    """
    config = deepcopy(SPHERICAL)
    
    # Update layer thicknesses
    for layer in config['geometry']['layers']:
        if layer['name'] == 'first_wall':
            layer['thickness'] = first_wall_cm
        elif layer['name'] == 'vessel':
            layer['thickness'] = vessel_cm
            layer['material'] = 'v4cr4ti'  # Change to V-4Cr-4Ti
    
    # Add V-4Cr-4Ti material definition if not present
    if 'v4cr4ti' not in config['materials']:
        config['materials']['v4cr4ti'] = {
            'elements': {'V': 0.92, 'Cr': 0.04, 'Ti': 0.04},
            'density': ELEMENT_DENSITIES['V'],
            'depletable': True
        }
    
    return config


def setup_openmc_model_for_geometry(
    first_wall_cm: float, 
    vessel_cm: float, 
    run_dir: str
) -> Tuple[openmc.Model, List[np.ndarray], openmc.deplete.MicroXS, np.ndarray, np.ndarray, str]:
    """Create an OpenMC model configured for a given geometry.
    
    This function builds a spherical model with specified first wall and vessel
    thicknesses, configures the simulation settings, and prepares the group-collapsed
    micro cross sections and flux spectrum for later depletion.
    
    It caches microXS and flux files inside ``run_dir/microxs_and_flux`` so that
    repeated runs with the same geometry re-use previously computed data.
    
    Args:
        first_wall_cm: First wall thickness in cm.
        vessel_cm: Vessel thickness in cm.
        run_dir: Output directory specific to this geometry setting.
        
    Returns:
        Tuple containing:
            - model: The configured OpenMC model.
            - flux: List with a single numpy array of the multigroup flux.
            - microxs: The multigroup micro cross sections for depletion.
            - timesteps: Numpy array of simulation time steps (seconds).
            - source_rates: Numpy array of source rates per time step (n/s).
            - flux_file_path: Absolute path to the saved flux spectrum file.
    """
    config = create_geometry_config(first_wall_cm, vessel_cm)
    model = create_model(config=config)
    model.settings.particles = FIXED_PARTICLES
    model.settings.batches = FIXED_BATCHES
    
    # Prepare per-run microXS/flux directory
    microxs_and_flux_dir = os.path.join(run_dir, 'microxs_and_flux')
    os.makedirs(microxs_and_flux_dir, exist_ok=True)
    
    microxs_file = os.path.join(microxs_and_flux_dir, 'microxs_1102.csv')
    flux_file = os.path.join(microxs_and_flux_dir, 'flux_spectrum_1102.txt')
    
    if os.path.exists(microxs_file) and os.path.exists(flux_file):
        flux = [np.loadtxt(flux_file, comments='#', usecols=1)]
        microxs = openmc.deplete.MicroXS.from_csv(microxs_file)
    else:
        flux_path, microxs_path = get_flux_and_microxs(
            model,
            chain_file=openmc.config['chain_file'],
            group_structure='UKAEA-1102',
            outdir=microxs_and_flux_dir,
        )
        flux_file = flux_path
        microxs_file = microxs_path
        flux = [np.loadtxt(flux_path, comments='#', usecols=1)]
        microxs = openmc.deplete.MicroXS.from_csv(microxs_path)
    
    # Time scheduler (same as baseline example)
    POWER_MW = 500
    TORUS_TO_SPHERE_VOLUME_RATIO = 1 / 4.03
    FUSION_POWER_MEV = 17.6
    MEV_TO_J = 1.602176634e-13
    source_rate = POWER_MW * 1e6 / (FUSION_POWER_MEV * MEV_TO_J) * TORUS_TO_SPHERE_VOLUME_RATIO
    
    # Production cooling times only (30d, 1y, 5y, 100y)
    cooling_times = [
        '30 days', '1 year', '5 years', '100 years'
    ]
    
    scheduler = TimeScheduler(
        irradiation_time='2 years',
        cooling_times=cooling_times,
        source_rate=source_rate,
        irradiation_steps=24,
    )
    
    timesteps, sources = scheduler.get_timesteps_and_source_rates()
    
    return model, flux, microxs, timesteps, sources, flux_file


def collect_flux_spectrum(flux_file_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load the multigroup flux spectrum from file.
    
    The expected file format is two columns: energy (eV) and flux, with ``#`` comments allowed.
    
    Args:
        flux_file_path: Absolute path to the flux spectrum file.
        
    Returns:
        Tuple of (energies_eV, flux_values) as numpy arrays.
    """
    data = np.loadtxt(flux_file_path, comments='#')
    # If only a single column was saved, fall back to using group index as energy proxy
    if data.ndim == 1 or data.shape[1] == 1:
        values = data if data.ndim == 1 else data[:, 0]
        energies = np.arange(values.size)
        return energies.astype(float), values.astype(float)
    return data[:, 0].astype(float), data[:, 1].astype(float)


def load_existing_flux(
    base_results_dir: str, 
    first_wall_cm: float, 
    vessel_cm: float
) -> Tuple[bool, str, Tuple[np.ndarray, np.ndarray]]:
    """Attempt to load an existing flux spectrum from disk for a geometry setting.
    
    Args:
        base_results_dir: Root results directory.
        first_wall_cm: First wall thickness in cm.
        vessel_cm: Vessel thickness in cm.
        
    Returns:
        Tuple of (found, flux_file_path, (energies_eV, flux_values)). If not found, ``found`` is False
        and the remaining values are placeholders.
    """
    run_dir = os.path.join(
        base_results_dir, 
        f'fw_{first_wall_cm:.1f}cm_vessel_{vessel_cm:.1f}cm'
    )
    flux_file = os.path.join(run_dir, 'microxs_and_flux', 'flux_spectrum_1102.txt')
    if os.path.exists(flux_file):
        try:
            energies, flux_vals = collect_flux_spectrum(flux_file)
            return True, flux_file, (energies, flux_vals)
        except Exception as exc:
            print(f"Warning: Failed to read existing flux at {flux_file}: {exc}")
    return False, flux_file, (np.array([]), np.array([]))


def read_depletion_results(results_dir: str, material_name: str) -> Dict[str, Any]:
    """Read depletion results and extract cooling-time activity and dose.
    
    This mirrors the single-run reader from the base example, returning activity by nuclide
    and total dose rate time series referenced to the shutdown time.
    
    Args:
        results_dir: Base directory for a specific geometry run.
        material_name: Material name (e.g., 'V-4Cr-4Ti').
        
    Returns:
        Dictionary containing keys: ``cooling_times``, ``nuclides``, ``cooling_activities``,
        ``cooling_total_dose_rates``, ``material_name``. Returns an empty dict on failure.
    """
    material_dir = os.path.join(results_dir, 'depletion_results', material_name)
    results_file = os.path.join(material_dir, 'depletion_results.h5')
    if not os.path.exists(results_file):
        print(f"Warning: Results file not found: {results_file}")
        return {}
    
    results = openmc.deplete.Results(results_file)
    
    source_rates = np.array([step.source_rate if step.source_rate is not None else 0 for step in results])
    _ = results.get_times()
    
    try:
        material_id = list(results[0].index_mat.keys())[0]
        times_activity, activity_by_nuclide = results.get_activity(material_id, units='Bq/kg', by_nuclide=True)
        
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
        
        if activity_by_nuclide and len(activity_by_nuclide) > 0:
            nuclides = list(activity_by_nuclide[0].keys())
        else:
            print('Warning: No activity data found')
            return {}
        
        activities: Dict[float, Dict[str, float]] = {}
        for i, time in enumerate(times_activity):
            if i < len(activity_by_nuclide):
                activities[float(time)] = activity_by_nuclide[i]
        
        total_dose_rates: Dict[float, float] = {}
        for time, dose_dict in zip(times_dose, dose_dicts):
            total_dose_rates[float(time)] = float(sum(dose_dict.values()))
        
        irr_indices = np.nonzero(source_rates)[0]
        if len(irr_indices) > 0:
            shutdown_idx = int(irr_indices[-1])
        else:
            shutdown_idx = 0
        
        plot_start_idx = shutdown_idx + 1
        if plot_start_idx >= len(times_activity):
            print(f"Warning: Not enough cooling steps to plot for {material_name}")
            return {}
        
        shutdown_time = float(times_activity[shutdown_idx])
        
        cooling_times = times_activity[plot_start_idx:] - shutdown_time
        
        cooling_activities: Dict[float, Dict[str, float]] = {}
        for idx in range(plot_start_idx, len(times_activity)):
            cool_time = float(times_activity[idx] - shutdown_time)
            cooling_activities[cool_time] = activity_by_nuclide[idx]
        
        cooling_total_dose_rates: Dict[float, float] = {}
        for idx in range(plot_start_idx, len(times_dose)):
            cool_time = float(times_dose[idx] - shutdown_time)
            if float(times_dose[idx]) in total_dose_rates:
                cooling_total_dose_rates[cool_time] = total_dose_rates[float(times_dose[idx])]
        
        # Extract gas production (He and H) using the proper extraction function
        gas_production = extract_gas_production(results)
        
        return {
            'cooling_times': np.array(cooling_times, dtype=float),
            'nuclides': nuclides,
            'cooling_activities': cooling_activities,
            'cooling_total_dose_rates': cooling_total_dose_rates,
            'gas_production': gas_production,
            'material_name': material_name,
        }
    except Exception as exc:
        print(f"Error reading depletion results for {material_name}: {exc}")
        return {}


def compute_total_activity_time_series(depletion_data: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    """Compute total activity over cooling time from per-nuclide activities.
    
    Args:
        depletion_data: Dictionary returned by ``read_depletion_results``.
        
    Returns:
        Tuple of (cooling_times_seconds, total_activity_Bq) as numpy arrays.
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
    
    return times_seconds.astype(float), np.asarray(total_activity, dtype=float)


def run_grid_sweep(
    first_wall_thicknesses: List[float],
    vessel_thicknesses: List[float],
    test_materials: List[Dict[str, float]],
    base_results_dir: str,
) -> Tuple[Dict[Tuple[float, float], Tuple[np.ndarray, np.ndarray]], Dict[Tuple[float, float], Dict[str, Dict[str, np.ndarray]]], Dict[Tuple[float, float], Dict[str, Dict[str, float]]]]:
    """Execute depletion runs across a grid of geometry settings and collect outputs.
    
    For each (first_wall_cm, vessel_cm) point, this will:
      - Build the OpenMC model with the specified geometry and compute/load microXS and flux spectrum
      - Run depletion for each material, caching results to disk
      - Read depletion results and derive total activity vs cooling time and gas production
      
    Args:
        first_wall_thicknesses: List of first wall thicknesses in cm.
        vessel_thicknesses: List of vessel thicknesses in cm.
        test_materials: List of material composition dicts to run (e.g., [{"V": 0.92, "Cr": 0.04, "Ti": 0.04}]).
        base_results_dir: Root directory where all run outputs are stored.
        
    Returns:
        A tuple of:
          - flux_by_setting: maps (fw_cm, vessel_cm) -> (energies_eV, flux_values)
          - activity_by_setting: maps (fw_cm, vessel_cm) -> material_name -> {'times', 'activity'} arrays
          - gas_by_setting: maps (fw_cm, vessel_cm) -> material_name -> {'He_appm', 'H_appm'} dict
    """
    os.makedirs(base_results_dir, exist_ok=True)
    
    flux_by_setting: Dict[Tuple[float, float], Tuple[np.ndarray, np.ndarray]] = {}
    activity_by_setting: Dict[Tuple[float, float], Dict[str, Dict[str, np.ndarray]]] = {}
    gas_by_setting: Dict[Tuple[float, float], Dict[str, Dict[str, float]]] = {}
    
    for fw_cm in first_wall_thicknesses:
        for vessel_cm in vessel_thicknesses:
            print(f"\n=== Geometry case: {fw_cm} cm first wall, {vessel_cm} cm vessel ===")
            run_dir = os.path.join(base_results_dir, f'fw_{fw_cm:.1f}cm_vessel_{vessel_cm:.1f}cm')
            os.makedirs(run_dir, exist_ok=True)
            
            # Try to load existing flux without building model
            found_flux, flux_file_path, flux_tuple = load_existing_flux(base_results_dir, fw_cm, vessel_cm)
            if found_flux:
                flux_by_setting[(fw_cm, vessel_cm)] = flux_tuple
                model = None  # defer model creation unless we need to run a simulation
                flux = None
                microxs = None
                timesteps = None
                sources = None
            else:
                if ANALYZE_ONLY:
                    print("  - Flux not found and ANALYZE_ONLY is True; skipping flux for this setting.")
                    model = None
                    flux = None
                    microxs = None
                    timesteps = None
                    sources = None
                else:
                    model, flux, microxs, timesteps, sources, flux_file_path = setup_openmc_model_for_geometry(
                        fw_cm, vessel_cm, run_dir
                    )
                    energies_eV, flux_vals = collect_flux_spectrum(flux_file_path)
                    flux_by_setting[(fw_cm, vessel_cm)] = (energies_eV, flux_vals)
            
            # Depletion per material
            for test_material in test_materials:
                material_name = f"V-{test_material.get('Cr', 0)*100:.0f}Cr-{test_material.get('Ti', 0)*100:.0f}Ti"
                material_outdir = os.path.join(run_dir, 'depletion_results', material_name)
                results_file = os.path.join(material_outdir, 'depletion_results.h5')
                
                if os.path.exists(results_file):
                    print(f"  - Found existing results for {material_name}; will read for analysis.")
                else:
                    if ANALYZE_ONLY:
                        print(f"  - Results missing for {material_name} and ANALYZE_ONLY is True; skipping run.")
                        continue
                    if model is None or microxs is None or flux is None or timesteps is None or sources is None:
                        # Need a model context to run the simulation
                        model, flux, microxs, timesteps, sources, _ = setup_openmc_model_for_geometry(
                            fw_cm, vessel_cm, run_dir
                        )
                    # Create and insert the depletable material into the vessel cell
                    new_material = create_material(test_material, material_name)
                    new_material.depletable = True
                    
                    try:
                        model.materials.append(new_material)
                        vessel_cell = model.geometry.get_cells_by_name('vessel')[0]
                        # Match the volume of the original placeholder alloy for consistency
                        new_material.volume = next(m.volume for m in model.materials if m.name == 'v4cr4ti')
                        vessel_cell.fill = new_material
                    except Exception as geom_err:
                        print(f"  - Warning: Failed to configure vessel material for {material_name}: {geom_err}")
                        continue
                    
                    _ = run_independent_depletion(
                        model=model,
                        depletable_cell='vessel',
                        microxs=microxs,
                        flux=flux,
                        chain_file=openmc.config['chain_file'],
                        timesteps=timesteps,
                        source_rates=sources,
                        outdir=material_outdir,
                    )
                
                # Read results and compute total activity time series and gas production
                dep_data = read_depletion_results(run_dir, material_name)
                if not dep_data:
                    continue
                times_s, total_activity = compute_total_activity_time_series(dep_data)
                activity_by_setting.setdefault((fw_cm, vessel_cm), {})[material_name] = {
                    'times': times_s,
                    'activity': total_activity,
                }
                gas_by_setting.setdefault((fw_cm, vessel_cm), {})[material_name] = dep_data.get('gas_production', {})
    
    return flux_by_setting, activity_by_setting, gas_by_setting


def plot_flux_sweep(
    flux_by_setting: Dict[Tuple[float, float], Tuple[np.ndarray, np.ndarray]],
    outdir: str
) -> str:
    """Plot multigroup flux spectra for all geometry runs on one figure.
    
    Args:
        flux_by_setting: Mapping of (fw_cm, vessel_cm) -> (energies_eV, flux_values).
        outdir: Directory to save plots into.
        
    Returns:
        Absolute path to the saved PNG plot.
    """
    os.makedirs(outdir, exist_ok=True)
    plt.figure(figsize=(12, 8))
    
    # Custom color scheme
    custom_colors = ['#2A33C3', '#A35D00', '#0B7285', '#8F2D56', '#6E8B00', '#D97706', '#7C3AED']
    keys_sorted = sorted(flux_by_setting.keys())
    colors = [custom_colors[i % len(custom_colors)] for i in range(len(keys_sorted))]
    
    for color, key in zip(colors, keys_sorted):
        fw_cm, vessel_cm = key
        energies, flux_vals = flux_by_setting[key]
        # Ensure strictly positive values for log scale plotting
        energies_plot = np.where(energies <= 0, np.nan, energies)
        flux_plot = np.where(flux_vals <= 0, np.nan, flux_vals)
        plt.plot(
            energies_plot, 
            flux_plot, 
            label=f'FW={fw_cm:.1f}cm, V={vessel_cm:.1f}cm', 
            color=color, 
            marker='o', 
            linewidth=2
        )
    
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Energy (eV)', fontsize=14, fontweight='bold')
    plt.ylabel('Flux (a.u.)', fontsize=14, fontweight='bold')
    plt.title('Flux Spectrum vs. First Wall & Vessel Thickness', fontsize=16, fontweight='bold')
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.grid(True, which='both', alpha=0.3)
    plt.legend(fontsize=11)
    plt.tight_layout()
    
    out_png = os.path.join(outdir, 'flux_spectrum_geometry_sweep.png')
    out_pdf = os.path.join(outdir, 'flux_spectrum_geometry_sweep.pdf')
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.savefig(out_pdf, bbox_inches='tight')
    plt.close()
    return out_png


def plot_activity_sweep(
    activity_by_setting: Dict[Tuple[float, float], Dict[str, Dict[str, np.ndarray]]], 
    outdir: str
) -> List[str]:
    """Plot total activity vs cooling time across geometry settings for each material.
    
    Args:
        activity_by_setting: Mapping of (fw_cm, vessel_cm) -> material -> {'times', 'activity'} arrays.
        outdir: Directory to save plots into.
        
    Returns:
        List of absolute paths to the saved PNG plots (one per material).
    """
    os.makedirs(outdir, exist_ok=True)
    
    # Collect material names from any geometry bucket
    material_names: List[str] = sorted({
        material
        for per_setting in activity_by_setting.values()
        for material in per_setting.keys()
    })
    
    # Custom color scheme
    custom_colors = ['#2A33C3', '#A35D00', '#0B7285', '#8F2D56', '#6E8B00', '#D97706', '#7C3AED']
    
    saved_paths: List[str] = []
    for material_name in material_names:
        plt.figure(figsize=(12, 8))
        keys_sorted = sorted(activity_by_setting.keys())
        colors = [custom_colors[i % len(custom_colors)] for i in range(len(keys_sorted))]
        
        for color, key in zip(colors, keys_sorted):
            fw_cm, vessel_cm = key
            per_mat = activity_by_setting.get(key, {}).get(material_name)
            if not per_mat:
                continue
            times = per_mat['times']
            activity = per_mat['activity']
            times_plot = np.where(times <= 0, np.nan, times)
            activity_plot = np.where(activity <= 0, np.nan, activity)
            plt.plot(
                times_plot, 
                activity_plot, 
                label=f'FW={fw_cm:.1f}cm, V={vessel_cm:.1f}cm', 
                color=color, 
                marker='o', 
                linewidth=2
            )
        
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('Cooling Time (s)', fontsize=14, fontweight='bold')
        plt.ylabel('Total Activity (Bq/kg)', fontsize=14, fontweight='bold')
        plt.title(f'{material_name}: Total Activity vs Cooling Time across Geometries', 
                 fontsize=16, fontweight='bold')
        plt.tick_params(axis='both', which='major', labelsize=12)
        plt.grid(True, which='both', alpha=0.3)
        plt.legend(fontsize=11)
        plt.tight_layout()
        
        out_png = os.path.join(outdir, f'activity_sweep_{material_name}.png')
        out_pdf = os.path.join(outdir, f'activity_sweep_{material_name}.pdf')
        plt.savefig(out_png, dpi=300, bbox_inches='tight')
        plt.savefig(out_pdf, bbox_inches='tight')
        plt.close()
        saved_paths.append(out_png)
    
    return saved_paths


def plot_gas_sweep(
    gas_by_setting: Dict[Tuple[float, float], Dict[str, Dict[str, float]]],
    outdir: str
) -> str:
    """Plot He and H production across geometry settings for each material.
    
    Args:
        gas_by_setting: Mapping of (fw_cm, vessel_cm) -> material -> {'He_appm', 'H_appm'} dict.
        outdir: Directory to save plots into.
        
    Returns:
        Absolute path to the saved PNG plot.
    """
    os.makedirs(outdir, exist_ok=True)
    
    # Collect material names
    material_names: List[str] = sorted({
        material
        for per_setting in gas_by_setting.values()
        for material in per_setting.keys()
    })
    
    if not material_names:
        return ""
    
    # Create figure with subplots for He and H
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Custom color scheme
    custom_colors = ['#2A33C3', '#A35D00', '#0B7285', '#8F2D56', '#6E8B00', '#D97706', '#7C3AED']
    
    keys_sorted = sorted(gas_by_setting.keys())
    x_positions = np.arange(len(keys_sorted))
    width = 0.35
    
    for mat_idx, material_name in enumerate(material_names):
        he_values = []
        h_values = []
        labels = []
        color = custom_colors[mat_idx % len(custom_colors)]
        
        for key in keys_sorted:
            fw_cm, vessel_cm = key
            gas_data = gas_by_setting.get(key, {}).get(material_name, {})
            he_values.append(gas_data.get('He_appm', 0.0))
            h_values.append(gas_data.get('H_appm', 0.0))
            labels.append(f'{fw_cm:.1f},{vessel_cm:.1f}')
        
        # He production plot
        ax1.bar(x_positions + mat_idx * width, he_values, width, 
                label=material_name, alpha=0.8, color=color)
        ax1.axhline(y=396, color='r', linestyle='--', linewidth=2, label='Limit (396 appm)' if mat_idx == 0 else '')
        
        # H production plot
        ax2.bar(x_positions + mat_idx * width, h_values, width,
                label=material_name, alpha=0.8, color=color)
        ax2.axhline(y=1200, color='r', linestyle='--', linewidth=2, label='Limit (1200 appm)' if mat_idx == 0 else '')
    
    # He plot formatting
    ax1.set_xlabel('Geometry (FW cm, Vessel cm)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('He Production (appm)', fontsize=14, fontweight='bold')
    ax1.set_title('He Production vs. Geometry', fontsize=16, fontweight='bold')
    ax1.set_xticks(x_positions + width * (len(material_names) - 1) / 2)
    ax1.set_xticklabels(labels, rotation=45, ha='right', fontsize=11)
    ax1.tick_params(axis='y', labelsize=12)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # H plot formatting
    ax2.set_xlabel('Geometry (FW cm, Vessel cm)', fontsize=14, fontweight='bold')
    ax2.set_ylabel('H Production (appm)', fontsize=14, fontweight='bold')
    ax2.set_title('H Production vs. Geometry', fontsize=16, fontweight='bold')
    ax2.set_xticks(x_positions + width * (len(material_names) - 1) / 2)
    ax2.set_xticklabels(labels, rotation=45, ha='right', fontsize=11)
    ax2.tick_params(axis='y', labelsize=12)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    out_png = os.path.join(outdir, 'gas_production_geometry_sweep.png')
    out_pdf = os.path.join(outdir, 'gas_production_geometry_sweep.pdf')
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.savefig(out_pdf, bbox_inches='tight')
    plt.close()
    return out_png


def _relative_l2(a: np.ndarray, b: np.ndarray) -> float:
    """Compute relative L2 norm ||a - b|| / ||b|| with safe handling.
    
    Args:
        a: First array.
        b: Reference array.
        
    Returns:
        Relative L2 norm value; returns ``np.nan`` if the reference norm is zero.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    denom = np.linalg.norm(b)
    if denom == 0.0:
        return float('nan')
    return float(np.linalg.norm(a - b) / denom)


def compute_relative_deviation_flux(
    flux_by_setting: Dict[Tuple[float, float], Tuple[np.ndarray, np.ndarray]],
    reference_key: Tuple[float, float],
) -> Dict[Tuple[float, float], float]:
    """Compute relative deviation of flux vs reference geometry.
    
    Uses relative L2 norm across the multigroup flux values. If energy grids differ,
    the comparison is truncated to the minimum common length.
    
    Args:
        flux_by_setting: Mapping of (fw_cm, vessel_cm) -> (energies_eV, flux_values).
        reference_key: Key selecting the reference run.
        
    Returns:
        Mapping of (fw_cm, vessel_cm) -> relative deviation (float).
    """
    if reference_key not in flux_by_setting:
        return {}
    e_ref, f_ref = flux_by_setting[reference_key]
    dev: Dict[Tuple[float, float], float] = {}
    for key, (e, f) in flux_by_setting.items():
        n = min(len(f_ref), len(f))
        if n == 0:
            dev[key] = float('nan')
            continue
        dev[key] = _relative_l2(f[:n], f_ref[:n])
    return dev


def compute_relative_deviation_activity(
    activity_by_setting: Dict[Tuple[float, float], Dict[str, Dict[str, np.ndarray]]],
    reference_key: Tuple[float, float],
) -> Dict[Tuple[float, float], Dict[str, float]]:
    """Compute relative deviation of total activity vs reference for each material.
    
    Uses relative L2 norm across the activity time series. If time grids differ,
    the comparison is truncated to the minimum common length.
    
    Args:
        activity_by_setting: Mapping of (fw_cm, vessel_cm) -> material -> {'times', 'activity'} arrays.
        reference_key: Key selecting the reference run.
        
    Returns:
        Mapping of (fw_cm, vessel_cm) -> material_name -> relative deviation (float).
    """
    if reference_key not in activity_by_setting:
        return {}
    ref_materials = activity_by_setting[reference_key]
    out: Dict[Tuple[float, float], Dict[str, float]] = {}
    for key, per_mat in activity_by_setting.items():
        out[key] = {}
        for material_name, series in per_mat.items():
            if material_name not in ref_materials:
                out[key][material_name] = float('nan')
                continue
            a_ref = ref_materials[material_name]['activity']
            a_cur = series['activity']
            n = min(len(a_ref), len(a_cur))
            if n == 0:
                out[key][material_name] = float('nan')
                continue
            out[key][material_name] = _relative_l2(a_cur[:n], a_ref[:n])
    return out


def compute_relative_deviation_gas(
    gas_by_setting: Dict[Tuple[float, float], Dict[str, Dict[str, float]]],
    reference_key: Tuple[float, float],
) -> Dict[Tuple[float, float], Dict[str, Dict[str, float]]]:
    """Compute relative deviation of gas production vs reference for each material.
    
    Args:
        gas_by_setting: Mapping of (fw_cm, vessel_cm) -> material -> {'He_appm', 'H_appm'} dict.
        reference_key: Key selecting the reference run.
        
    Returns:
        Mapping of (fw_cm, vessel_cm) -> material_name -> {'He_rel_dev', 'H_rel_dev'} dict.
    """
    if reference_key not in gas_by_setting:
        return {}
    ref_materials = gas_by_setting[reference_key]
    out: Dict[Tuple[float, float], Dict[str, Dict[str, float]]] = {}
    for key, per_mat in gas_by_setting.items():
        out[key] = {}
        for material_name, gas_data in per_mat.items():
            if material_name not in ref_materials:
                out[key][material_name] = {'He_rel_dev': float('nan'), 'H_rel_dev': float('nan')}
                continue
            ref_gas = ref_materials[material_name]
            
            # Compute relative deviations for He and H
            he_ref = ref_gas.get('He_appm', 0.0)
            he_cur = gas_data.get('He_appm', 0.0)
            h_ref = ref_gas.get('H_appm', 0.0)
            h_cur = gas_data.get('H_appm', 0.0)
            
            he_dev = abs(he_cur - he_ref) / he_ref if he_ref > 0 else float('nan')
            h_dev = abs(h_cur - h_ref) / h_ref if h_ref > 0 else float('nan')
            
            out[key][material_name] = {
                'He_rel_dev': he_dev,
                'H_rel_dev': h_dev,
            }
    return out


def save_metrics(
    outdir: str,
    flux_deviation: Dict[Tuple[float, float], float],
    activity_deviation: Dict[Tuple[float, float], Dict[str, float]],
    gas_deviation: Dict[Tuple[float, float], Dict[str, Dict[str, float]]],
) -> Tuple[str, str]:
    """Save deviation metrics to JSON and CSV files.
    
    Args:
        outdir: Directory to save metrics into.
        flux_deviation: Relative flux deviation per setting.
        activity_deviation: Relative activity deviation per setting and material.
        gas_deviation: Relative gas production deviation per setting and material.
        
    Returns:
        Tuple of (json_path, csv_path).
    """
    os.makedirs(outdir, exist_ok=True)
    json_path = os.path.join(outdir, 'relative_deviation_metrics.json')
    csv_path = os.path.join(outdir, 'relative_deviation_metrics.csv')
    
    # JSON
    jsonable = {
        'flux': {f'fw_{fw}_vessel_{v}': val for (fw, v), val in flux_deviation.items()},
        'activity': {f'fw_{fw}_vessel_{v}': per_mat for (fw, v), per_mat in activity_deviation.items()},
        'gas': {f'fw_{fw}_vessel_{v}': per_mat for (fw, v), per_mat in gas_deviation.items()},
    }
    with open(json_path, 'w') as jf:
        json.dump(jsonable, jf, indent=2)
    
    # CSV: rows per setting-material-metric
    with open(csv_path, 'w', newline='') as cf:
        writer = csv.writer(cf)
        writer.writerow(['first_wall_cm', 'vessel_cm', 'metric', 'material', 'species', 'relative_deviation'])
        for (fw, v), val in flux_deviation.items():
            writer.writerow([fw, v, 'flux', '', '', val])
        for (fw, v), per_mat in activity_deviation.items():
            for mat, val in per_mat.items():
                writer.writerow([fw, v, 'activity_total', mat, '', val])
        for (fw, v), per_mat in gas_deviation.items():
            for mat, gas_devs in per_mat.items():
                writer.writerow([fw, v, 'gas_production', mat, 'He', gas_devs.get('He_rel_dev', float('nan'))])
                writer.writerow([fw, v, 'gas_production', mat, 'H', gas_devs.get('H_rel_dev', float('nan'))])
    
    return json_path, csv_path


def plot_sensitivity_heatmap(
    flux_deviation: Dict[Tuple[float, float], float],
    activity_deviation: Dict[Tuple[float, float], Dict[str, float]],
    gas_deviation: Dict[Tuple[float, float], Dict[str, Dict[str, float]]],
    first_wall_thicknesses: List[float],
    vessel_thicknesses: List[float],
    outdir: str,
) -> str:
    """Create heatmap showing sensitivity to geometry parameters.
    
    Args:
        flux_deviation: Relative flux deviation per setting.
        activity_deviation: Relative activity deviation per setting and material.
        gas_deviation: Relative gas production deviation per setting and material.
        first_wall_thicknesses: List of first wall thicknesses tested.
        vessel_thicknesses: List of vessel thicknesses tested.
        outdir: Directory to save plots into.
        
    Returns:
        Path to saved heatmap figure.
    """
    os.makedirs(outdir, exist_ok=True)
    
    # Create 2D grids for flux, activity, He, and H (averaged over materials)
    flux_grid = np.full((len(vessel_thicknesses), len(first_wall_thicknesses)), np.nan)
    activity_grid = np.full((len(vessel_thicknesses), len(first_wall_thicknesses)), np.nan)
    he_grid = np.full((len(vessel_thicknesses), len(first_wall_thicknesses)), np.nan)
    h_grid = np.full((len(vessel_thicknesses), len(first_wall_thicknesses)), np.nan)
    
    for i, vessel_cm in enumerate(vessel_thicknesses):
        for j, fw_cm in enumerate(first_wall_thicknesses):
            key = (fw_cm, vessel_cm)
            if key in flux_deviation:
                flux_grid[i, j] = flux_deviation[key]
            if key in activity_deviation:
                act_vals = list(activity_deviation[key].values())
                if act_vals:
                    activity_grid[i, j] = np.mean([v for v in act_vals if np.isfinite(v)])
            if key in gas_deviation:
                he_vals = [gas_devs.get('He_rel_dev', np.nan) 
                          for gas_devs in gas_deviation[key].values()]
                if he_vals:
                    he_grid[i, j] = np.mean([v for v in he_vals if np.isfinite(v)])
                h_vals = [gas_devs.get('H_rel_dev', np.nan) 
                         for gas_devs in gas_deviation[key].values()]
                if h_vals:
                    h_grid[i, j] = np.mean([v for v in h_vals if np.isfinite(v)])
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    ax1, ax2, ax3, ax4 = axes.flatten()
    
    # Flux deviation heatmap
    im1 = ax1.imshow(flux_grid, aspect='auto', cmap='RdYlGn_r', origin='lower')
    ax1.set_xticks(range(len(first_wall_thicknesses)))
    ax1.set_yticks(range(len(vessel_thicknesses)))
    ax1.set_xticklabels([f'{t:.1f}' for t in first_wall_thicknesses], fontsize=11)
    ax1.set_yticklabels([f'{t:.1f}' for t in vessel_thicknesses], fontsize=11)
    ax1.set_xlabel('First Wall Thickness (cm)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Vessel Thickness (cm)', fontsize=14, fontweight='bold')
    ax1.set_title('Flux Spectrum Relative L2 Deviation', fontsize=15, fontweight='bold')
    
    # Add text annotations
    for i in range(len(vessel_thicknesses)):
        for j in range(len(first_wall_thicknesses)):
            if np.isfinite(flux_grid[i, j]):
                _ = ax1.text(j, i, f'{flux_grid[i, j]:.3f}',
                               ha="center", va="center", color="black", fontsize=9)
    
    plt.colorbar(im1, ax=ax1, label='Relative Deviation')
    
    # Activity deviation heatmap
    im2 = ax2.imshow(activity_grid, aspect='auto', cmap='RdYlGn_r', origin='lower')
    ax2.set_xticks(range(len(first_wall_thicknesses)))
    ax2.set_yticks(range(len(vessel_thicknesses)))
    ax2.set_xticklabels([f'{t:.1f}' for t in first_wall_thicknesses], fontsize=11)
    ax2.set_yticklabels([f'{t:.1f}' for t in vessel_thicknesses], fontsize=11)
    ax2.set_xlabel('First Wall Thickness (cm)', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Vessel Thickness (cm)', fontsize=14, fontweight='bold')
    ax2.set_title('Activity (mean) Relative L2 Deviation', fontsize=15, fontweight='bold')
    
    # Add text annotations
    for i in range(len(vessel_thicknesses)):
        for j in range(len(first_wall_thicknesses)):
            if np.isfinite(activity_grid[i, j]):
                _ = ax2.text(j, i, f'{activity_grid[i, j]:.3f}',
                               ha="center", va="center", color="black", fontsize=9)
    
    plt.colorbar(im2, ax=ax2, label='Relative Deviation')
    
    # He production deviation heatmap
    im3 = ax3.imshow(he_grid, aspect='auto', cmap='RdYlGn_r', origin='lower')
    ax3.set_xticks(range(len(first_wall_thicknesses)))
    ax3.set_yticks(range(len(vessel_thicknesses)))
    ax3.set_xticklabels([f'{t:.1f}' for t in first_wall_thicknesses], fontsize=11)
    ax3.set_yticklabels([f'{t:.1f}' for t in vessel_thicknesses], fontsize=11)
    ax3.set_xlabel('First Wall Thickness (cm)', fontsize=14, fontweight='bold')
    ax3.set_ylabel('Vessel Thickness (cm)', fontsize=14, fontweight='bold')
    ax3.set_title('He Production Relative Deviation', fontsize=15, fontweight='bold')
    
    # Add text annotations
    for i in range(len(vessel_thicknesses)):
        for j in range(len(first_wall_thicknesses)):
            if np.isfinite(he_grid[i, j]):
                _ = ax3.text(j, i, f'{he_grid[i, j]:.3f}',
                               ha="center", va="center", color="black", fontsize=9)
    
    plt.colorbar(im3, ax=ax3, label='Relative Deviation')
    
    # H production deviation heatmap
    im4 = ax4.imshow(h_grid, aspect='auto', cmap='RdYlGn_r', origin='lower')
    ax4.set_xticks(range(len(first_wall_thicknesses)))
    ax4.set_yticks(range(len(vessel_thicknesses)))
    ax4.set_xticklabels([f'{t:.1f}' for t in first_wall_thicknesses], fontsize=11)
    ax4.set_yticklabels([f'{t:.1f}' for t in vessel_thicknesses], fontsize=11)
    ax4.set_xlabel('First Wall Thickness (cm)', fontsize=14, fontweight='bold')
    ax4.set_ylabel('Vessel Thickness (cm)', fontsize=14, fontweight='bold')
    ax4.set_title('H Production Relative Deviation', fontsize=15, fontweight='bold')
    
    # Add text annotations
    for i in range(len(vessel_thicknesses)):
        for j in range(len(first_wall_thicknesses)):
            if np.isfinite(h_grid[i, j]):
                _ = ax4.text(j, i, f'{h_grid[i, j]:.3f}',
                               ha="center", va="center", color="black", fontsize=9)
    
    plt.colorbar(im4, ax=ax4, label='Relative Deviation')
    
    plt.tight_layout()
    out_png = os.path.join(outdir, 'geometry_sensitivity_heatmap.png')
    out_pdf = os.path.join(outdir, 'geometry_sensitivity_heatmap.pdf')
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.savefig(out_pdf, bbox_inches='tight')
    plt.close()
    return out_png


if __name__ == '__main__':
    """Run geometry sweep and generate comparison plots.
    
    This script varies the first wall (pure W) and vessel (V-4Cr-4Ti) layer thicknesses
    to assess sensitivity of flux spectrum and activation results to geometry parameters.
    
    First wall thicknesses: 1, 2, 5, 10 mm (0.1, 0.2, 0.5, 1.0 cm)
    Vessel thicknesses: 0.5, 1, 2, 5 cm
    Reference geometry: 2 mm first wall, 1 cm vessel
    
    Results and plots are written under ``analysis_results/geometry_sweep_neutronics_run``.
    """
    # Test material: V-4Cr-4Ti for vessel
    test_materials: List[Dict[str, float]] = [
        {'V': 0.92, 'Cr': 0.04, 'Ti': 0.04},
    ]
    
    flux_by_setting, activity_by_setting, gas_by_setting = run_grid_sweep(
        first_wall_thicknesses=FIRST_WALL_THICKNESSES_CM,
        vessel_thicknesses=VESSEL_THICKNESSES_CM,
        test_materials=test_materials,
        base_results_dir=RESULTS_BASE_DIR,
    )
    
    plots_dir = os.path.join(RESULTS_BASE_DIR, 'analysis_plots')
    # Only plot when we have something to plot
    flux_plot_path = ''
    activity_plot_paths: List[str] = []
    gas_plot_path = ''
    if flux_by_setting:
        flux_plot_path = plot_flux_sweep(flux_by_setting, plots_dir)
    if activity_by_setting:
        activity_plot_paths = plot_activity_sweep(activity_by_setting, plots_dir)
    if gas_by_setting:
        gas_plot_path = plot_gas_sweep(gas_by_setting, plots_dir)
    
    # Relative deviation vs reference geometry
    ref_key = REFERENCE_GEOMETRY
    metrics_dir = os.path.join(RESULTS_BASE_DIR, 'metrics')
    flux_dev: Dict[Tuple[float, float], float] = {}
    activity_dev: Dict[Tuple[float, float], Dict[str, float]] = {}
    gas_dev: Dict[Tuple[float, float], Dict[str, Dict[str, float]]] = {}
    if ref_key in flux_by_setting:
        flux_dev = compute_relative_deviation_flux(flux_by_setting, ref_key)
    else:
        print('Warning: Reference geometry flux not available; skipping flux deviation computation.')
    if ref_key in activity_by_setting:
        activity_dev = compute_relative_deviation_activity(activity_by_setting, ref_key)
    else:
        print('Warning: Reference geometry activity not available; skipping activity deviation computation.')
    if ref_key in gas_by_setting:
        gas_dev = compute_relative_deviation_gas(gas_by_setting, ref_key)
    else:
        print('Warning: Reference geometry gas production not available; skipping gas deviation computation.')
    
    metrics_json, metrics_csv = save_metrics(metrics_dir, flux_dev, activity_dev, gas_dev)
    
    # Sensitivity heatmap
    heatmap_path = plot_sensitivity_heatmap(
        flux_dev, 
        activity_dev,
        gas_dev,
        FIRST_WALL_THICKNESSES_CM, 
        VESSEL_THICKNESSES_CM, 
        plots_dir
    )
    
    print('\n' + '=' * 60)
    print('GEOMETRY SWEEP COMPLETE')
    print('=' * 60)
    print(f'- Flux spectrum comparison saved to: {flux_plot_path}')
    for ap in activity_plot_paths:
        print(f'- Activity comparison saved to: {ap}')
    if gas_plot_path:
        print(f'- Gas production comparison saved to: {gas_plot_path}')
    print(f'- Sensitivity heatmap saved to: {heatmap_path}')
    print(f'- Relative deviation metrics saved to: {metrics_json} and {metrics_csv}')
    
    print(f"\nReference geometry: {ref_key[0]:.1f} cm first wall, {ref_key[1]:.1f} cm vessel")
    print("\nGeometry Sensitivity Summary:")
    print("-" * 60)
    
    # Find max deviations
    if flux_dev:
        max_flux_dev_key = max(flux_dev.items(), key=lambda x: x[1] if np.isfinite(x[1]) else -1)
        print(f"Maximum flux deviation: {max_flux_dev_key[1]:.4e}")
        print(f"  at geometry: {max_flux_dev_key[0][0]:.1f} cm FW, {max_flux_dev_key[0][1]:.1f} cm vessel")
    
    if activity_dev:
        max_activity_dev = 0.0
        max_activity_key = None
        max_activity_mat = None
        for key, mat_dict in activity_dev.items():
            for mat, val in mat_dict.items():
                if np.isfinite(val) and val > max_activity_dev:
                    max_activity_dev = val
                    max_activity_key = key
                    max_activity_mat = mat
        if max_activity_key:
            print(f"Maximum activity deviation: {max_activity_dev:.4e}")
            print(f"  at geometry: {max_activity_key[0]:.1f} cm FW, {max_activity_key[1]:.1f} cm vessel")
            print(f"  for material: {max_activity_mat}")
    
    if gas_dev:
        max_he_dev = 0.0
        max_he_key = None
        max_he_mat = None
        max_h_dev = 0.0
        max_h_key = None
        max_h_mat = None
        for key, mat_dict in gas_dev.items():
            for mat, gas_devs in mat_dict.items():
                he_val = gas_devs.get('He_rel_dev', 0.0)
                h_val = gas_devs.get('H_rel_dev', 0.0)
                if np.isfinite(he_val) and he_val > max_he_dev:
                    max_he_dev = he_val
                    max_he_key = key
                    max_he_mat = mat
                if np.isfinite(h_val) and h_val > max_h_dev:
                    max_h_dev = h_val
                    max_h_key = key
                    max_h_mat = mat
        if max_he_key:
            print(f"Maximum He production deviation: {max_he_dev:.4e}")
            print(f"  at geometry: {max_he_key[0]:.1f} cm FW, {max_he_key[1]:.1f} cm vessel")
            print(f"  for material: {max_he_mat}")
        if max_h_key:
            print(f"Maximum H production deviation: {max_h_dev:.4e}")
            print(f"  at geometry: {max_h_key[0]:.1f} cm FW, {max_h_key[1]:.1f} cm vessel")
            print(f"  for material: {max_h_mat}")
    
    print("\nUse these deviations to assess geometry sensitivity for your depletion model.")
    print("Consider geometries with deviations > 2% (0.02) as requiring finer modeling.")
    print("\nProduction limits for reference:")
    print("  He: 396 appm (2-year irradiation)")
    print("  H:  1200 appm (2-year irradiation)")
    print("  Dose: 30d=1e3, 1y=1.0, 5y=0.01, 100y=0.0001 Sv/h")

