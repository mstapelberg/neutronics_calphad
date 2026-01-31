"""
V-15Cr-5Ti ARC Neutronics Comparison

Compares neutronics results for V-15Cr-5Ti alloy between:
- Current spherical model  
- ARC publication reference values
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import openmc
import openmc.deplete
from neutronics_calphad.neutronics.config import SPHERICAL
from neutronics_calphad.neutronics.geometry_maker import create_model
from neutronics_calphad.neutronics.depletion import run_independent_depletion
from neutronics_calphad.neutronics.time_scheduler import TimeScheduler
from neutronics_calphad.neutronics.flux import get_flux_and_microxs
from neutronics_calphad.neutronics.dose import contact_dose
from neutronics_calphad.utils.io import material_string, create_material
from neutronics_calphad.utils.utils import filter_openmc_warnings, suppress_openmc_warnings

# Apply OpenMC warning filters
filter_openmc_warnings()

# TODO : Need to update the paths to use the stuff in data 
# Configuration 
#CHAIN_FILE = '/Users/myless/nuclear_data/endfb71-gefy61-decay2012.xml'
CHAIN_FILE = '/Users/myless/nuclear_data/tendl2021_fispact2020_chain.xml'
#CROSS_SECTIONS_FILE = '/Users/myless/nuclear_data/endfb-71-hdf5/cross_sections.xml'
CROSS_SECTIONS_FILE = '/Users/myless/nuclear_data/tendl-2021-hdf5/cross_sections.xml'
#CROSS_SECTIONS_FILE = '/Users/myless/nuclear_data/endfb-vii.1-hdf5/cross_sections.xml'
ABS_FILE = '/Users/myless/Packages/fispact/nuclear_data/decay/abs_2012'

openmc.config['chain_file'] = CHAIN_FILE
openmc.config['cross_sections'] = CROSS_SECTIONS_FILE

RESULTS_DIR = "analysis_results/vcrti_arc_comparison"
os.makedirs(RESULTS_DIR, exist_ok=True)

# V-15Cr-5Ti composition (wt%)
V_15CR_5TI_COMPOSITION = {'V': 0.80, 'Cr': 0.15, 'Ti': 0.05}
REFERENCE_FLUX = 7.54e14  # neutrons/cm²s

# Time points: 10E-4 to 10E4 years (32 points)
TIME_POINTS_YEARS = np.logspace(-4, 4, 32)

def main() -> None:
    """Main execution function."""
    print("=== V-15Cr-5Ti ARC Neutronics Comparison ===")
    
    # Set up model
    model = create_model(config=SPHERICAL)
    model.settings.particles = 10000
    model.export_to_xml(os.path.join(RESULTS_DIR, 'model.xml'))

    # Get vessel cell and ensure it has a volume
    vessel_cell = model.geometry.get_cells_by_name('vessel')[0]
    if vessel_cell.volume is None:
        # Calculate volume from geometry if not set
        # For spherical geometry, vessel is a shell between two spheres
        # Get the vessel material to check if it has volume
        vessel_material = vessel_cell.fill
        if hasattr(vessel_material, 'volume') and vessel_material.volume is not None:
            vessel_cell.volume = vessel_material.volume
            print(f"Set vessel cell volume from material: {vessel_cell.volume:.2e} cm³")
        else:
            # Calculate volume for spherical shell (vessel layer)
            # From config: vessel thickness = 2 cm, radius starts at 113 cm
            inner_radius = 113.0  # cm
            outer_radius = 113.0 + 2.0  # cm (vessel thickness)
            vessel_volume = 4/3 * np.pi * (outer_radius**3 - inner_radius**3)
            vessel_cell.volume = vessel_volume
            print(f"Calculated vessel cell volume: {vessel_cell.volume:.2e} cm³")
    else:
        print(f"Vessel cell volume: {vessel_cell.volume:.2e} cm³")
    
    # Calculate average flux first
    microxs_and_flux_dir = os.path.join(RESULTS_DIR, 'microxs_and_flux')
    os.makedirs(microxs_and_flux_dir, exist_ok=True)
    num_groups = 709  # From 'CCFE-709'
    flux_path = os.path.join(microxs_and_flux_dir, f"flux_spectrum_{num_groups}.txt")
    microxs_path = os.path.join(microxs_and_flux_dir, f"microxs_{num_groups}.csv")

    if os.path.exists(flux_path) and os.path.exists(microxs_path):
        print("Loading existing flux and microxs...")
        # Parse flux file
        flux_data = []
        material_volume = None
        with open(flux_path, 'r') as f:
            for line in f:
                if line.startswith('# Volume:'):
                    material_volume = float(line.split()[2])
                elif not line.startswith('#'):
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        flux_data.append(float(parts[1]))
        if material_volume is None:
            raise ValueError("Could not find volume in flux file")
        flux = [np.array(flux_data)]
        microxs = openmc.deplete.MicroXS.from_csv(microxs_path)
    else:
        print("Calculating flux and microxs...")
        with suppress_openmc_warnings():
            flux_path, microxs_path = get_flux_and_microxs(
                model, chain_file=CHAIN_FILE, group_structure='CCFE-709', outdir=microxs_and_flux_dir
            )
        flux = [np.loadtxt(flux_path, comments='#', usecols=1)]
        microxs = openmc.deplete.MicroXS.from_csv(microxs_path)
        # Parse volume after calculation
        with open(flux_path, 'r') as f:
            for line in f:
                if line.startswith('# Volume:'):
                    material_volume = float(line.split()[2])
                    break
        if material_volume is None:
            raise ValueError("Could not find volume in newly created flux file")

    # Set up time scheduler
    POWER_MW = 500
    TORUS_TO_SPHERE_VOLUME_RATIO = 1/4.03
    FUSION_POWER_MEV = 17.6
    MEV_TO_J = 1.602176634e-13
    SOURCE_RATE = POWER_MW * 1e6 / (FUSION_POWER_MEV * MEV_TO_J) * TORUS_TO_SPHERE_VOLUME_RATIO

    # Calculate average flux (integrated over energy)
    # Flux values from file are per-source tracklength [n-cm/src]; convert to [n/cm²/src] by dividing by volume
    average_flux_tracklength_per_source = np.sum(flux[0])
    average_flux_per_source_per_cm2 = average_flux_tracklength_per_source / material_volume
    average_flux = average_flux_per_source_per_cm2 * SOURCE_RATE
    print(f"Average flux (per-source tracklength): {average_flux_tracklength_per_source:.2e} n·cm/src")
    print(f"Material volume used: {material_volume:.2e} cm³")
    print(f"Average flux in vessel: {average_flux:.2e} neutrons/cm²s")
    print(f"Reference flux: {REFERENCE_FLUX:.2e} neutrons/cm²s")
    print(f"Ratio (model/reference): {average_flux/REFERENCE_FLUX:.3f}")
    
    # Create V-15Cr-5Ti material
    material_name = material_string(V_15CR_5TI_COMPOSITION, 'V', precision=1)
    material = create_material(V_15CR_5TI_COMPOSITION, material_name, percent_type='wo')
    material.depletable = True

    # Assign volume from original vessel material (like pipeline does)
    material.volume = next(m.volume for m in model.materials if m.name == 'vcrtiwzr')

    # Replace vessel material
    vessel_cell = model.geometry.get_cells_by_name('vessel')[0]
    vessel_cell.fill = material
    
        
    cooling_times = [f"{t:.2e} years" for t in TIME_POINTS_YEARS]
    scheduler = TimeScheduler(
        irradiation_time='2 years',
        cooling_times=cooling_times,
        source_rate=SOURCE_RATE,
        irradiation_steps=12,
    )
    
    timesteps, sources = scheduler.get_timesteps_and_source_rates()
    
    # Run depletion
    depletion_dir = os.path.join(RESULTS_DIR, 'depletion_results')
    os.makedirs(depletion_dir, exist_ok=True)
    depletion_results_path = os.path.join(depletion_dir, 'depletion_results.h5')

    if os.path.exists(depletion_results_path):
        print("Loading existing depletion results...")
        results = openmc.deplete.Results(depletion_results_path)
    else:
        print("Running depletion...")
        with suppress_openmc_warnings():
            results = run_independent_depletion(
                model=model,
                depletable_cell='vessel',
                microxs=microxs,
                flux=flux,
                chain_file=CHAIN_FILE,
                timesteps=timesteps,
                source_rates=sources,
                outdir=depletion_dir,
            )
    
    # Calculate activity and dose (align to cooling time after irradiation)
    material_id = list(results[0].index_mat.keys())[0]
    times_activity, activity_by_nuclide = results.get_activity(material_id, units="Bq/kg", by_nuclide=True)

    # Convert absolute times to cooling times (seconds since end of irradiation)
    irradiation_time_s = scheduler.irradiation_time

    # Calculate total activity mapped by cooling time
    total_activity = {}
    for i, time_abs in enumerate(times_activity):
        if i < len(activity_by_nuclide):
            cooling_t = time_abs - irradiation_time_s
            if cooling_t < 0:
                continue  # skip pre-shutdown times
            activity_values = activity_by_nuclide[i]
            if isinstance(activity_values, dict):
                total_activity[cooling_t] = sum(activity_values.values())

    # Calculate dose mapped by cooling time (contact_dose already returns Sv/h per kg per nuclide)
    times_dose_abs, dose_dicts = contact_dose(results=results, chain_file=CHAIN_FILE, abs_file=ABS_FILE)
    dose_per_hr = {}
    for time_abs, dose_dict in zip(times_dose_abs, dose_dicts):
        cooling_t = time_abs - irradiation_time_s
        if cooling_t < 0:
            continue
        total_dose = sum(dose_dict.values())  # Sv/h per kg
        dose_per_hr[cooling_t] = total_dose

    # Load reference activity data and create comparison/plot
    median_ratio = None
    p10_ratio = None
    p90_ratio = None
    plot_path = None

    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        ref_csv = os.path.join(
            os.path.dirname(os.path.dirname(script_dir)),
            'data',
            'neutronics-data',
            'v15cr5ti_reference_data.csv',
        )
        if os.path.exists(ref_csv):
            ref_data = np.loadtxt(ref_csv, delimiter=',')
            if ref_data.ndim == 1 and ref_data.size == 2:
                ref_data = ref_data.reshape(1, 2)
            ref_times_years = ref_data[:, 0]
            ref_activity_bqkg = ref_data[:, 1]
            ref_times_seconds = ref_times_years * 365.25 * 24 * 3600

            # Our activity as arrays (cooling time seconds → Bq/kg)
            if total_activity:
                my_times_s = np.array(sorted(total_activity.keys()))
                my_activity_bqkg = np.array([total_activity[t] for t in my_times_s])

                # Keep strictly positive for log interpolation
                pos = my_activity_bqkg > 0
                if np.any(pos):
                    my_times_pos = my_times_s[pos]
                    my_act_pos = my_activity_bqkg[pos]

                    # Interpolate our activity at reference times within overlap, in log-log space
                    def log_interp(x, xp, fp):
                        return np.exp(np.interp(np.log(x), np.log(xp), np.log(fp)))

                    overlap = (ref_times_seconds >= my_times_pos.min()) & (ref_times_seconds <= my_times_pos.max())
                    if np.any(overlap):
                        ref_t_common = ref_times_seconds[overlap]
                        ref_a_common = ref_activity_bqkg[overlap]
                        # Guard against non-positive reference values
                        ref_pos = ref_a_common > 0
                        ref_t_common = ref_t_common[ref_pos]
                        ref_a_common = ref_a_common[ref_pos]

                        if ref_t_common.size > 0:
                            my_interp = log_interp(ref_t_common, my_times_pos, my_act_pos)
                            ratios = my_interp / ref_a_common

                            # Plot
                            import matplotlib.pyplot as plt  # Local import to avoid side effects
                            plt.figure(figsize=(7, 5))
                            plt.loglog(my_times_pos / (365.25 * 24 * 3600), my_act_pos, 'b-', label='Model (Bq/kg)')
                            plt.loglog(ref_times_years, ref_activity_bqkg, 'r--', label='Reference (Bq/kg)')
                            plt.xlabel('Cooling time (years)')
                            plt.ylabel('Activity (Bq/kg)')
                            plt.title('V-15Cr-5Ti Activity Comparison')
                            plt.grid(True, which='both', alpha=0.3)
                            plt.legend()
                            plot_path = os.path.join(RESULTS_DIR, 'activity_comparison.png')
                            plt.savefig(plot_path, dpi=220)
                            plt.close()

                            # Ratio stats (geometric)
                            log_r = np.log(ratios)
                            median_ratio = float(np.exp(np.median(log_r)))
                            p10_ratio = float(np.exp(np.percentile(log_r, 10)))
                            p90_ratio = float(np.exp(np.percentile(log_r, 90)))
    except Exception as e:
        print(f"Warning: Could not create reference activity comparison: {e}")
    
    # Save results
    results_file = os.path.join(RESULTS_DIR, 'comparison_results.txt')
    with open(results_file, 'w') as f:
        f.write("V-15Cr-5Ti ARC Comparison Results\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"Average Flux Comparison:\n")
        f.write(f"  Model flux: {average_flux:.2e} neutrons/cm²s\n")
        f.write(f"  Reference flux: {REFERENCE_FLUX:.2e} neutrons/cm²s\n")
        f.write(f"  Ratio (model/reference): {average_flux/REFERENCE_FLUX:.3f}\n\n")
        
        f.write("Activity and Dose Data:\n")
        f.write("Time(years) | Activity(Bq/kg) | Dose(Sv/hr)\n")
        f.write("-" * 50 + "\n")
        
        for t in TIME_POINTS_YEARS:
            # desired cooling time in seconds
            t_seconds = t * 365.25 * 24 * 3600
            
            # Find closest values
            activity_times = np.array(list(total_activity.keys())) if total_activity else np.array([])
            dose_times = np.array(list(dose_per_hr.keys())) if dose_per_hr else np.array([])
            
            if activity_times.size:
                activity_idx = np.argmin(np.abs(activity_times - t_seconds))
                activity_val = list(total_activity.values())[activity_idx]
            else:
                activity_val = 0.0
            
            if dose_times.size:
                dose_idx = np.argmin(np.abs(dose_times - t_seconds))
                dose_val = list(dose_per_hr.values())[dose_idx]
            else:
                dose_val = 0.0
            
            f.write(f"{t:.2e} | {activity_val:.2e} | {dose_val:.2e}\n")

        # Append comparison summary if available
        if median_ratio is not None:
            f.write("\nActivity comparison vs reference (geometric ratios)\n")
            f.write(f"  Median ratio (model/ref): {median_ratio:.2f}\n")
            f.write(f"  P10–P90 ratio range: {p10_ratio:.2f} – {p90_ratio:.2f}\n")
        if plot_path:
            f.write(f"\nSaved activity comparison plot: {plot_path}\n")
    
    print(f"Results saved to: {results_file}")

if __name__ == "__main__":
    main()
