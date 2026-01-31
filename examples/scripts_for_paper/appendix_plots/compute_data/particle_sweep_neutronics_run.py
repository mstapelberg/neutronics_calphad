import os
import json
import csv
from typing import Any, Dict, List, Tuple, Optional

import matplotlib.pyplot as plt
import numpy as np
import openmc
import openmc.deplete

from neutronics_calphad.neutronics.config import SPHERICAL
from neutronics_calphad.neutronics.depletion import run_independent_depletion
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

# Default simulation controls
DEFAULT_NUM_BATCHES: List[int] = [10, 50, 100]
PARTICLE_SWEEP: List[int] = [1000, 10000, 100000]
RESULTS_BASE_DIR: str = os.path.join('analysis_results', 'particle_sweep_neutronics_run_batches')
ANALYZE_ONLY: bool = False  # If True, do not run simulations; only read existing outputs and re-generate analysis

# Tolerance gates (relative L2 norms)
TOLERANCE_FLUX_REL_L2: float = 0.02  # 2% default
TOLERANCE_ACTIVITY_REL_L2: float = 0.02  # 2% default


def setup_openmc_model_for_particles(particles: int, batches: int, run_dir: str) -> Tuple[openmc.Model, List[np.ndarray], openmc.deplete.MicroXS, np.ndarray, np.ndarray, str]:
    """Create an OpenMC model configured for a given particle count.

    This function builds a spherical model, configures the simulation settings
    with the requested ``particles`` and ``batches``, and prepares the
    group-collapsed micro cross sections and flux spectrum for later depletion.

    It caches microXS and flux files inside ``run_dir/microxs_and_flux`` so that
    repeated runs with the same settings re-use previously computed data.

    Args:
        particles: Number of source particles to simulate per batch.
        batches: Number of batches to run (kept constant across the sweep).
        run_dir: Output directory specific to this particle setting.

    Returns:
        Tuple containing:
            - model: The configured OpenMC model.
            - flux: List with a single numpy array of the multigroup flux.
            - microxs: The multigroup micro cross sections for depletion.
            - timesteps: Numpy array of simulation time steps (seconds).
            - source_rates: Numpy array of source rates per time step (n/s).
            - flux_file_path: Absolute path to the saved flux spectrum file.
    """
    model = create_model(config=SPHERICAL)
    model.settings.particles = int(particles)
    model.settings.batches = int(batches)

    # Prepare per-run microXS/flux directory so flux differences are preserved per particle count
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

    cooling_times = [
        '1 second', '1 minute', '1 hour', '10 hours', '1 day', '1 week',
        '2 weeks', '30 days', '1 year', '2 years', '5 years', '10 years',
        '25 years', '100 years'
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


def load_existing_flux(base_results_dir: str, particles: int, batches: int) -> Tuple[bool, str, Tuple[np.ndarray, np.ndarray]]:
    """Attempt to load an existing flux spectrum from disk for a setting.

    Args:
        base_results_dir: Root results directory.
        particles: Particles per batch.
        batches: Number of batches.

    Returns:
        Tuple of (found, flux_file_path, (energies_eV, flux_values)). If not found, ``found`` is False
        and the remaining values are placeholders.
    """
    run_dir = os.path.join(base_results_dir, f'nparticles_{int(particles)}_nbatches_{int(batches)}')
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
        results_dir: Base directory for a specific particle-count run.
        material_name: Material name (e.g., 'V').

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

        return {
            'cooling_times': np.array(cooling_times, dtype=float),
            'nuclides': nuclides,
            'cooling_activities': cooling_activities,
            'cooling_total_dose_rates': cooling_total_dose_rates,
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
    particle_counts: List[int],
    batch_counts: List[int],
    test_materials: List[Dict[str, float]],
    base_results_dir: str,
) -> Tuple[Dict[Tuple[int, int], Tuple[np.ndarray, np.ndarray]], Dict[Tuple[int, int], Dict[str, Dict[str, np.ndarray]]]]:
    """Execute depletion runs across a grid of particle and batch counts and collect outputs.

    For each (particles, batches) point, this will:
      - Build the OpenMC model and compute/load microXS and flux spectrum
      - Run depletion for each material, caching results to disk
      - Read depletion results and derive total activity vs cooling time

    Args:
        particle_counts: List of particle counts per batch to simulate.
        batch_counts: List of batch counts to simulate.
        test_materials: List of material composition dicts to run (e.g., [{"V": 1.0}]).
        base_results_dir: Root directory where all run outputs are stored.

    Returns:
        A tuple of:
          - flux_by_setting: maps (particles, batches) -> (energies_eV, flux_values)
          - activity_by_setting: maps (particles, batches) -> material_name -> {'times', 'activity'} arrays
    """
    os.makedirs(base_results_dir, exist_ok=True)

    flux_by_setting: Dict[Tuple[int, int], Tuple[np.ndarray, np.ndarray]] = {}
    activity_by_setting: Dict[Tuple[int, int], Dict[str, Dict[str, np.ndarray]]] = {}

    for particles in particle_counts:
        for batches in batch_counts:
            print(f"\n=== Grid case: {particles} particles, {batches} batches ===")
            run_dir = os.path.join(base_results_dir, f'nparticles_{int(particles)}_nbatches_{int(batches)}')
            os.makedirs(run_dir, exist_ok=True)

            # Try to load existing flux without building model
            found_flux, flux_file_path, flux_tuple = load_existing_flux(base_results_dir, particles, batches)
            if found_flux:
                flux_by_setting[(int(particles), int(batches))] = flux_tuple
                model = None  # defer model creation unless we need to run a simulation for missing materials
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
                    model, flux, microxs, timesteps, sources, flux_file_path = setup_openmc_model_for_particles(particles, batches, run_dir)
                    energies_eV, flux_vals = collect_flux_spectrum(flux_file_path)
                    flux_by_setting[(int(particles), int(batches))] = (energies_eV, flux_vals)

            # Depletion per material
            for test_material in test_materials:
                material_name = list(test_material.keys())[0]
                material_outdir = os.path.join(run_dir, 'depletion_results', material_name)
                results_file = os.path.join(material_outdir, 'depletion_results.h5')

                if os.path.exists(results_file):
                    print(f"  - Found existing results for {material_name}; will read for analysis.")
                else:
                    if ANALYZE_ONLY:
                        print(f"  - Results missing for {material_name} and ANALYZE_ONLY is True; skipping run.")
                        # Skip reading since file does not exist
                        continue
                    if model is None or microxs is None or flux is None or timesteps is None or sources is None:
                        # Need a model context to run the simulation
                        model, flux, microxs, timesteps, sources, _ = setup_openmc_model_for_particles(particles, batches, run_dir)
                    # Create and insert the depletable material into the vessel cell
                    new_material = create_material(test_material, material_name)
                    new_material.depletable = True

                    try:
                        model.materials.append(new_material)
                        vessel_cell = model.geometry.get_cells_by_name('vessel')[0]
                        # Match the volume of the original placeholder alloy for consistency
                        new_material.volume = next(m.volume for m in model.materials if m.name == 'vcrtiwzr')
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

                # Read results and compute total activity time series
                dep_data = read_depletion_results(run_dir, material_name)
                if not dep_data:
                    continue
                times_s, total_activity = compute_total_activity_time_series(dep_data)
                activity_by_setting.setdefault((int(particles), int(batches)), {})[material_name] = {
                    'times': times_s,
                    'activity': total_activity,
                }

    return flux_by_setting, activity_by_setting


def plot_flux_sweep(
    flux_by_setting: Dict[Tuple[int, int], Tuple[np.ndarray, np.ndarray]],
    outdir: str
) -> str:
    """Plot multigroup flux spectra for all grid runs on one figure.

    Args:
        flux_by_setting: Mapping of (particles, batches) -> (energies_eV, flux_values).
        outdir: Directory to save plots into.

    Returns:
        Absolute path to the saved plot.
    """
    os.makedirs(outdir, exist_ok=True)
    plt.figure(figsize=(12, 8))

    # Stable color order across different settings
    keys_sorted = sorted(flux_by_setting.keys())
    colors = plt.cm.viridis(np.linspace(0, 1, len(keys_sorted)))

    for color, key in zip(colors, keys_sorted):
        particles, batches = key
        energies, flux_vals = flux_by_setting[key]
        # Ensure strictly positive values for log scale plotting
        energies_plot = np.where(energies <= 0, np.nan, energies)
        flux_plot = np.where(flux_vals <= 0, np.nan, flux_vals)
        plt.plot(energies_plot, flux_plot, label=f'{particles} p, {batches} b', color=color, marker='o', linewidth=2)

    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Energy (eV)')
    plt.ylabel('Flux (a.u.)')
    plt.title('Flux Spectrum vs. Particles and Batches')
    plt.grid(True, which='both', alpha=0.3)
    plt.legend()
    plt.tight_layout()

    out_path = os.path.join(outdir, 'flux_spectrum_particle_sweep.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    return out_path


def plot_activity_sweep(activity_by_setting: Dict[Tuple[int, int], Dict[str, Dict[str, np.ndarray]]], outdir: str) -> List[str]:
    """Plot total activity vs cooling time across grid settings for each material.

    Args:
        activity_by_setting: Mapping of (particles, batches) -> material -> {'times', 'activity'} arrays.
        outdir: Directory to save plots into.

    Returns:
        List of absolute paths to the saved plots (one per material).
    """
    os.makedirs(outdir, exist_ok=True)

    # Collect material names from any particles bucket
    material_names: List[str] = sorted({
        material
        for per_setting in activity_by_setting.values()
        for material in per_setting.keys()
    })

    saved_paths: List[str] = []
    for material_name in material_names:
        plt.figure(figsize=(12, 8))
        keys_sorted = sorted(activity_by_setting.keys())
        colors = plt.cm.plasma(np.linspace(0, 1, len(keys_sorted)))

        for color, key in zip(colors, keys_sorted):
            particles, batches = key
            per_mat = activity_by_setting.get(key, {}).get(material_name)
            if not per_mat:
                continue
            times = per_mat['times']
            activity = per_mat['activity']
            times_plot = np.where(times <= 0, np.nan, times)
            activity_plot = np.where(activity <= 0, np.nan, activity)
            plt.plot(times_plot, activity_plot, label=f'{particles} p, {batches} b', color=color, marker='o', linewidth=2)

        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('Cooling Time (s)')
        plt.ylabel('Total Activity (Bq/kg)')
        plt.title(f'{material_name}: Total Activity vs Cooling Time across Settings')
        plt.grid(True, which='both', alpha=0.3)
        plt.legend()
        plt.tight_layout()

        out_path = os.path.join(outdir, f'activity_sweep_{material_name}.png')
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close()
        saved_paths.append(out_path)

    return saved_paths


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
    flux_by_setting: Dict[Tuple[int, int], Tuple[np.ndarray, np.ndarray]],
    reference_key: Tuple[int, int],
) -> Dict[Tuple[int, int], float]:
    """Compute relative deviation of flux vs reference setting.

    Uses relative L2 norm across the multigroup flux values. If energy grids differ,
    the comparison is truncated to the minimum common length.

    Args:
        flux_by_setting: Mapping of (particles, batches) -> (energies_eV, flux_values).
        reference_key: Key selecting the reference run.

    Returns:
        Mapping of (particles, batches) -> relative deviation (float).
    """
    if reference_key not in flux_by_setting:
        return {}
    e_ref, f_ref = flux_by_setting[reference_key]
    dev: Dict[Tuple[int, int], float] = {}
    for key, (e, f) in flux_by_setting.items():
        n = min(len(f_ref), len(f))
        if n == 0:
            dev[key] = float('nan')
            continue
        dev[key] = _relative_l2(f[:n], f_ref[:n])
    return dev


def compute_relative_deviation_activity(
    activity_by_setting: Dict[Tuple[int, int], Dict[str, Dict[str, np.ndarray]]],
    reference_key: Tuple[int, int],
) -> Dict[Tuple[int, int], Dict[str, float]]:
    """Compute relative deviation of total activity vs reference for each material.

    Uses relative L2 norm across the activity time series. If time grids differ,
    the comparison is truncated to the minimum common length.

    Args:
        activity_by_setting: Mapping of (particles, batches) -> material -> {'times', 'activity'} arrays.
        reference_key: Key selecting the reference run.

    Returns:
        Mapping of (particles, batches) -> material_name -> relative deviation (float).
    """
    if reference_key not in activity_by_setting:
        return {}
    ref_materials = activity_by_setting[reference_key]
    out: Dict[Tuple[int, int], Dict[str, float]] = {}
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


def save_metrics(
    outdir: str,
    flux_deviation: Dict[Tuple[int, int], float],
    activity_deviation: Dict[Tuple[int, int], Dict[str, float]],
) -> Tuple[str, str]:
    """Save deviation metrics to JSON and CSV files.

    Args:
        outdir: Directory to save metrics into.
        flux_deviation: Relative flux deviation per setting.
        activity_deviation: Relative activity deviation per setting and material.

    Returns:
        Tuple of (json_path, csv_path).
    """
    os.makedirs(outdir, exist_ok=True)
    json_path = os.path.join(outdir, 'relative_deviation_metrics.json')
    csv_path = os.path.join(outdir, 'relative_deviation_metrics.csv')

    # JSON
    jsonable = {
        'flux': {f'{p}_{b}': v for (p, b), v in flux_deviation.items()},
        'activity': {f'{p}_{b}': per_mat for (p, b), per_mat in activity_deviation.items()},
    }
    with open(json_path, 'w') as jf:
        json.dump(jsonable, jf, indent=2)

    # CSV: rows per setting-material, plus one for flux-only
    with open(csv_path, 'w', newline='') as cf:
        writer = csv.writer(cf)
        writer.writerow(['particles', 'batches', 'metric', 'material', 'relative_deviation'])
        for (p, b), val in flux_deviation.items():
            writer.writerow([p, b, 'flux', '', val])
        for (p, b), per_mat in activity_deviation.items():
            for mat, val in per_mat.items():
                writer.writerow([p, b, 'activity_total', mat, val])

    return json_path, csv_path


def choose_minimal_setting_meeting_tolerance(
    particle_counts: List[int],
    batch_counts: List[int],
    flux_deviation: Dict[Tuple[int, int], float],
    activity_deviation: Dict[Tuple[int, int], Dict[str, float]],
    tol_flux: float,
    tol_activity: float,
    materials_priority: Optional[List[str]] = None,
) -> Optional[Tuple[int, int]]:
    """Pick the smallest-cost (particles*batches) setting meeting tolerance gates.

    A setting passes if:
      - flux deviation <= tol_flux, and
      - activity deviation <= tol_activity for all materials present at that setting.

    If ``materials_priority`` is provided, the activity criterion is applied
    in that order, but all must still pass.

    Among passing settings, the one with the lowest cost proxy (particles*batches)
    is selected. Ties are broken by smaller particles, then smaller batches.

    Args:
        particle_counts: Candidate particle values.
        batch_counts: Candidate batch values.
        flux_deviation: Relative flux deviation per setting.
        activity_deviation: Relative activity deviation per setting and material.
        tol_flux: Flux deviation tolerance (relative L2).
        tol_activity: Activity deviation tolerance (relative L2).
        materials_priority: Optional list of material names to evaluate first.

    Returns:
        The chosen (particles, batches) pair or None if no setting passes.
    """
    keys = [(p, b) for p in particle_counts for b in batch_counts]

    def passes(key: Tuple[int, int]) -> bool:
        fdev = flux_deviation.get(key)
        if fdev is None or not np.isfinite(fdev) or fdev > tol_flux:
            return False
        per_mat = activity_deviation.get(key, {})
        if not per_mat:
            return False
        # Evaluate in priority order if provided
        mats = list(per_mat.keys())
        if materials_priority:
            mats = [m for m in materials_priority if m in per_mat] + [m for m in mats if m not in materials_priority]
        for m in mats:
            v = per_mat.get(m)
            if v is None or not np.isfinite(v) or v > tol_activity:
                return False
        return True

    passing = [k for k in keys if passes(k)]
    if not passing:
        return None

    # Lowest cost proxy: particles * batches
    passing.sort(key=lambda k: (k[0] * k[1], k[0], k[1]))
    return passing[0]


if __name__ == '__main__':
    """Run particle-count sweep and generate comparison plots.

    This script mirrors the baseline example but varies the number of particles
    per batch (default of 10 batches) and compares:
      - Multigroup flux spectra vs particle count
      - Total activity vs cooling time vs particle count

    Adjust ``PARTICLE_SWEEP`` and ``DEFAULT_NUM_BATCHES`` as desired.
    Results and plots are written under ``analysis_results/particle_sweep_neutronics_run``.
    """
    # Materials identical to the baseline script subset
    test_materials: List[Dict[str, float]] = [
        {'V': 1.0},
        {'Zr': 1.0},
        {'W': 1.0},
    ]

    flux_by_setting, activity_by_setting = run_grid_sweep(
        particle_counts=PARTICLE_SWEEP,
        batch_counts=DEFAULT_NUM_BATCHES,
        test_materials=test_materials,
        base_results_dir=RESULTS_BASE_DIR,
    )

    plots_dir = os.path.join(RESULTS_BASE_DIR, 'analysis_plots')
    # Only plot when we have something to plot
    flux_plot_path = ''
    activity_plot_paths: List[str] = []
    if flux_by_setting:
        flux_plot_path = plot_flux_sweep(flux_by_setting, plots_dir)
    if activity_by_setting:
        activity_plot_paths = plot_activity_sweep(activity_by_setting, plots_dir)

    # Relative deviation vs reference (max particles and batches)
    ref_particles = max(PARTICLE_SWEEP)
    ref_batches = max(DEFAULT_NUM_BATCHES)
    ref_key = (ref_particles, ref_batches)
    metrics_dir = os.path.join(RESULTS_BASE_DIR, 'metrics')
    flux_dev: Dict[Tuple[int, int], float] = {}
    activity_dev: Dict[Tuple[int, int], Dict[str, float]] = {}
    if ref_key in flux_by_setting:
        flux_dev = compute_relative_deviation_flux(flux_by_setting, ref_key)
    else:
        print('Warning: Reference setting flux not available; skipping flux deviation computation.')
    if ref_key in activity_by_setting:
        activity_dev = compute_relative_deviation_activity(activity_by_setting, ref_key)
    else:
        print('Warning: Reference setting activity not available; skipping activity deviation computation.')
    metrics_json, metrics_csv = save_metrics(metrics_dir, flux_dev, activity_dev)

    # Specific check: 10 batches & 10000 particles, if present
    target_key = (10000, 10)
    flux_dev_target = flux_dev.get(target_key)
    activity_dev_target = activity_dev.get(target_key, {})

    # Choose minimal-cost setting meeting tolerance gates
    recommended = choose_minimal_setting_meeting_tolerance(
        particle_counts=PARTICLE_SWEEP,
        batch_counts=DEFAULT_NUM_BATCHES,
        flux_deviation=flux_dev,
        activity_deviation=activity_dev,
        tol_flux=TOLERANCE_FLUX_REL_L2,
        tol_activity=TOLERANCE_ACTIVITY_REL_L2,
        materials_priority=['V', 'Zr', 'W'],
    )

    # Persist recommendation
    recommendation_path = os.path.join(metrics_dir, 'tolerance_recommendation.json')
    with open(recommendation_path, 'w') as rf:
        json.dump({
            'tolerance': {
                'flux_rel_l2': TOLERANCE_FLUX_REL_L2,
                'activity_rel_l2': TOLERANCE_ACTIVITY_REL_L2,
            },
            'reference': {'particles': ref_particles, 'batches': ref_batches},
            'target_checked': {'particles': 10000, 'batches': 10},
            'target_flux_dev': flux_dev_target,
            'target_activity_dev': activity_dev_target,
            'recommended_setting': None if recommended is None else {'particles': recommended[0], 'batches': recommended[1]},
        }, rf, indent=2)

    print('\n' + '=' * 60)
    print('PARTICLE & BATCH SWEEP COMPLETE')
    print('=' * 60)
    print(f'- Flux spectrum comparison saved to: {flux_plot_path}')
    for ap in activity_plot_paths:
        print(f'- Activity comparison saved to: {ap}')
    print(f'- Relative deviation metrics saved to: {metrics_json} and {metrics_csv}')

    if flux_dev_target is not None:
        print(f"\nReference setting: {ref_particles} particles, {ref_batches} batches")
        print("Target setting: 10000 particles, 10 batches")
        print(f"Flux relative deviation (L2): {flux_dev_target:.4e}")
        if activity_dev_target:
            for mat, val in activity_dev_target.items():
                print(f"Activity relative deviation (L2) for {mat}: {val:.4e}")
        else:
            print('No activity data found for the target setting to compare.')
    else:
        print('\nTarget setting (10000 particles, 10 batches) not present among computed cases.')

    print('\nUse these deviations to assess if 10 batches and 10000 particles are sufficient for publication-grade stability.')


