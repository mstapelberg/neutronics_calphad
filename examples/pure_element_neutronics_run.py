import openmc
import openmc.deplete
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

from typing import Dict, List, Any, Optional, Tuple
from neutronics_calphad.neutronics.config import SPHERICAL
from neutronics_calphad.neutronics.flux import get_flux_and_microxs
from neutronics_calphad.neutronics.geometry_maker import create_model
from neutronics_calphad.neutronics.depletion import run_independent_depletion
from neutronics_calphad.neutronics.time_scheduler import TimeScheduler
from neutronics_calphad.optimizer.parsers import parse_openmc_results
from neutronics_calphad.utils.io import material_string, create_material
from neutronics_calphad.neutronics.dose import contact_dose


openmc.config['chain_file'] = '/home/myless/nuclear_data/tendl21-fispact20-gefy61-chain.xml'
openmc.config['cross_sections'] = '/home/myless/nuclear_data/tendl-2021-hdf5/cross_sections.xml'

chain = openmc.deplete.Chain.from_xml(openmc.config['chain_file'])

print(dir(chain))
print(chain.reactions)
OPENMC_NUM_PARTICLES = 10000
RESULTS_DIR = os.path.join("analysis_results", "pure_element_neutronics_run")

# Neutronics limits
CRIT_LIMITS = {"He_appm": 1172.2/2, "H_appm": 1200}
DOSE_LIMITS = {14: 1e4, 365: 1, 3650: 1e-2, 36500: 1e-4}

def calculate_max_compositions(results_dict: Dict[str, Dict[str, Any]], 
                             crit_limits: Dict[str, float], 
                             dose_limits: Dict[int, float]) -> Dict[str, float]:
    """
    Calculate maximum compositions for V-X binary alloys based on neutronics limits.
    
    This function determines the maximum allowable fraction of each alloying element
    (Cr, Ti, W, Zr) in a V-X binary system using linear weighted averages.
    For each V-X binary, it finds the maximum X fraction that keeps the weighted
    average of V and X properties within the specified limits.
    
    Args:
        results_dict: Dictionary containing neutronics results for pure elements
        crit_limits: Dictionary with critical gas production limits (He_appm, H_appm)
        dose_limits: Dictionary with dose rate limits at different cooling times (days)
    
    Returns:
        Dictionary with maximum compositions for each element in V-X binary
    """
    if 'V' not in results_dict:
        raise ValueError("Vanadium (V) results are required as the base element")
    
    v_results = results_dict['V']
    max_compositions = {'V': 0.95}  # Base V composition
    
    # Elements to calculate max compositions for (excluding V)
    alloying_elements = ['Cr', 'Ti', 'W', 'Zr']
    
    for element in alloying_elements:
        if element not in results_dict:
            print(f"Warning: No results found for {element}")
            max_compositions[element] = 0.0
            continue
            
        element_results = results_dict[element]
        
        # For a V-X binary with composition f_X (fraction of X), the weighted average is:
        # weighted_property = (1 - f_X) * V_property + f_X * X_property
        # We want: weighted_property <= limit
        # Solving for f_X: f_X <= (limit - V_property) / (X_property - V_property)
        
        max_gas_fraction = 1.0
        max_dose_fraction = 1.0
        
        # Check gas production limits
        for gas_type, limit in crit_limits.items():
            if (gas_type in v_results.get('gas_production', {}) and 
                gas_type in element_results.get('gas_production', {})):
                
                v_gas = v_results['gas_production'][gas_type]
                x_gas = element_results['gas_production'][gas_type]
                
                # Calculate maximum X fraction for this gas type
                if x_gas != v_gas:  # Avoid division by zero
                    gas_fraction = (limit - v_gas) / (x_gas - v_gas)
                    # Ensure fraction is between 0 and 1
                    gas_fraction = max(0.0, min(1.0, gas_fraction))
                    max_gas_fraction = min(max_gas_fraction, gas_fraction)
                else:
                    # Same gas production as V, no additional constraint
                    pass
            else:
                print(f"Warning: Missing {gas_type} data for V or {element}")
        
        # Check dose rate limits
        for cooling_time, limit in dose_limits.items():
            if (cooling_time in v_results.get('dose_at_cooling_times', {}) and 
                cooling_time in element_results.get('dose_at_cooling_times', {})):
                
                v_dose = v_results['dose_at_cooling_times'][cooling_time]
                x_dose = element_results['dose_at_cooling_times'][cooling_time]
                
                # Calculate maximum X fraction for this dose rate
                if x_dose != v_dose:  # Avoid division by zero
                    dose_fraction = (limit - v_dose) / (x_dose - v_dose)
                    # Ensure fraction is between 0 and 1
                    dose_fraction = max(0.0, min(1.0, dose_fraction))
                    max_dose_fraction = min(max_dose_fraction, dose_fraction)
                else:
                    # Same dose rate as V, no additional constraint
                    pass
            else:
                print(f"Warning: Missing dose data for {cooling_time} days for V or {element}")
        
        # Take the more restrictive limit
        max_fraction = min(max_gas_fraction, max_dose_fraction)
        
        # Ensure we don't exceed 0.95 (leaving at least 5% for V)
        max_fraction = min(max_fraction, 0.95)
        
        max_compositions[element] = max_fraction
        
        print(f"\n{element}: Max fraction = {max_fraction:.4f}")
        print(f"  - Gas production limit: {max_gas_fraction:.4f}")
        print(f"  - Dose rate limit: {max_dose_fraction:.4f}")
        
        # Show example calculation for verification
        if max_fraction > 0:
            print(f"  - Example V-{element} alloy with {max_fraction:.3f} {element}:")
            for gas_type in crit_limits:
                if gas_type in v_results.get('gas_production', {}) and gas_type in element_results.get('gas_production', {}):
                    v_gas = v_results['gas_production'][gas_type]
                    x_gas = element_results['gas_production'][gas_type]
                    weighted_gas = (1 - max_fraction) * v_gas + max_fraction * x_gas
                    print(f"    {gas_type}: {weighted_gas:.2f} appm (limit: {crit_limits[gas_type]:.1f})")
    
    return max_compositions


def read_depletion_results(results_dir: str, material_name: str) -> Dict[str, Any]:
    """
    Read depletion results from OpenMC output files.
    
    Args:
        results_dir: Directory containing depletion results
        material_name: Name of the material
    
    Returns:
        Dictionary containing depletion results data
    """
    material_dir = os.path.join(results_dir, 'depletion_results', material_name)
    
    # Read the results.h5 file
    results_file = os.path.join(material_dir, 'depletion_results.h5')
    if not os.path.exists(results_file):
        print(f"Warning: Results file not found: {results_file}")
        return {}
    
    # Use OpenMC to read the results
    results = openmc.deplete.Results(results_file)

    # Extract source rates
    import numpy as np
    source_rates = np.array([step.source_rate if step.source_rate is not None else 0 for step in results])
    
    # Extract time points
    times = results.get_times()
    print(f"Times shape: {times.shape}, first few: {times[:5]}")
    
    try:
        # Get activity data by nuclide
        material_id = list(results[0].index_mat.keys())[0]
        times_activity, activity_by_nuclide = results.get_activity(material_id, units="Bq", by_nuclide=True)

        # Get total dose rates (not by nuclide)
        try:
            times_dose, dose_dicts = contact_dose(results,
                                               chain_file=openmc.config['chain_file'],
                                               abs_file='/home/myless/Packages/fispact/nuclear_data/decay/abs_2012')
        except Exception as dose_error:
            print(f"Warning: Dose calculation failed: {dose_error}")
            print("Continuing without dose rate data...")
            times_dose = times_activity
            dose_dicts = [{} for _ in times_activity]

        # Extract nuclide names from the first time step
        if activity_by_nuclide and len(activity_by_nuclide) > 0:
            nuclides = list(activity_by_nuclide[0].keys())
        else:
            print("Warning: No activity data found")
            return {}
        
        print(f"Found {len(nuclides)} nuclides: {nuclides[:10]}...")
        
        # Build time series data for activities
        activities = {}
        for i, time in enumerate(times_activity):
            if i < len(activity_by_nuclide):
                activities[time] = activity_by_nuclide[i]
        
        # Build total dose rate data
        total_dose_rates = {}
        for time, dose_dict in zip(times_dose, dose_dicts):
            total_dose_rates[time] = sum(dose_dict.values())
        
        # Filter for cooling period, excluding the t=0 point
        irr_indices = np.nonzero(source_rates)[0]
        if len(irr_indices) > 0:
            shutdown_idx = irr_indices[-1]
        else:
            shutdown_idx = 0
        
        # We start plotting from the first cooling step, skipping the exact time of shutdown
        plot_start_idx = shutdown_idx + 1

        if plot_start_idx >= len(times_activity):
            print(f"Warning: Not enough cooling steps to plot for {material_name}")
            return {}

        shutdown_time = times_activity[shutdown_idx]
        
        cooling_times = times_activity[plot_start_idx:] - shutdown_time
        
        cooling_activities = {}
        for idx in range(plot_start_idx, len(times_activity)):
            cool_time = times_activity[idx] - shutdown_time
            cooling_activities[cool_time] = activity_by_nuclide[idx]
        
        cooling_total_dose_rates = {}
        for idx in range(plot_start_idx, len(times_dose)):
            cool_time = times_dose[idx] - shutdown_time
            if times_dose[idx] in total_dose_rates:
                cooling_total_dose_rates[cool_time] = total_dose_rates[times_dose[idx]]
        
        return {
            'cooling_times': cooling_times,
            'nuclides': nuclides,
            'cooling_activities': cooling_activities,
            'cooling_total_dose_rates': cooling_total_dose_rates,
            'material_name': material_name
        }
        
    except Exception as e:
        print(f"Error reading depletion results: {e}")
        return {}


def plot_activity_and_dose_analysis(results_dict: Dict[str, Dict[str, Any]], 
                                   results_dir: str) -> None:
    """
    Create comprehensive plots showing activity by nuclide and total dose rates for all elements.
    
    Args:
        results_dict: Dictionary containing neutronics results for all materials
        results_dir: Directory containing depletion results
    """
    # Create output directory for plots
    plots_dir = os.path.join(results_dir, 'analysis_plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    # Dictionary to store dose rate data for the combined plot
    all_dose_data = {}
    all_activity_data = {}
    
    # Create individual activity plots for each material
    for material_name in results_dict.keys():
        print(f"\nAnalyzing {material_name}...")
        
        # Read depletion results
        depletion_data = read_depletion_results(results_dir, material_name)
        if not depletion_data:
            continue
        
        times_seconds = depletion_data['cooling_times']
        nuclides = depletion_data['nuclides']
        activities = depletion_data['cooling_activities']
        total_dose_rates = depletion_data['cooling_total_dose_rates']
        
        # Store dose data for combined plot
        all_dose_data[material_name] = total_dose_rates
        
        # Create activity plot (single panel)
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        # Plot total activity
        total_activity = []
        for time in times_seconds:
            if time in activities:
                activity_values = activities[time]
                if isinstance(activity_values, dict):
                    total_activity.append(np.sum(list(activity_values.values())))
                else:
                    print(f"DEBUG: activities[{time}] is not a dict: {type(activity_values)}")
                    total_activity.append(0.0)
            else:
                total_activity.append(0.0)
        
        # Debug: Check total_activity before plotting
        print(f"DEBUG: total_activity type: {type(total_activity)}")
        print(f"DEBUG: total_activity length: {len(total_activity)}")
        print(f"DEBUG: total_activity first few values: {total_activity[:5]}")
        print(f"DEBUG: times_seconds type: {type(times_seconds)}")
        print(f"DEBUG: times_seconds length: {len(times_seconds)}")
        
        # Ensure both arrays are numpy arrays and have the same length
        total_activity = np.array(total_activity, dtype=float)
        times_seconds = np.array(times_seconds, dtype=float)
        
        if len(total_activity) != len(times_seconds):
            print(f"ERROR: Length mismatch - total_activity: {len(total_activity)}, times_seconds: {len(times_seconds)}")
            # Truncate to shorter length
            min_len = min(len(total_activity), len(times_seconds))
            total_activity = total_activity[:min_len]
            times_seconds = times_seconds[:min_len]
        
        all_activity_data[material_name] = {
            'times': times_seconds,
            'activity': total_activity
        }
        
        ax.plot(times_seconds, total_activity, 'k-o', linewidth=2, label='Total Activity')
        
        # Plot top contributing nuclides (by maximum activity)
        nuclide_max_activities = {}
        for nuclide in nuclides:
            nuclide_activities = [activities.get(time, {}).get(nuclide, 0.0) for time in times_seconds]
            nuclide_max_activities[nuclide] = max(nuclide_activities)
        
        # Sort by maximum activity and take top 10
        top_nuclides = sorted(nuclide_max_activities.items(), 
                             key=lambda x: x[1], reverse=True)[:10]
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(top_nuclides)))
        for i, (nuclide, max_activity) in enumerate(top_nuclides):
            nuclide_activities = [activities.get(time, {}).get(nuclide, 0.0) for time in times_seconds]
            ax.plot(times_seconds, nuclide_activities, 
                    color=colors[i], alpha=0.7, label=f'{nuclide}', marker='o')
        
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('Time (seconds)')
        ax.set_ylabel('Activity (Bq)')
        ax.set_ylim(1e0, 1e20)
        ax.set_title(f'{material_name} - Activity vs Cooling Time')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f'{material_name}_activity_analysis.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  - Activity plot saved to: {plots_dir}")
    
    # Create combined dose rate plot for all elements
    if all_dose_data:
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(all_dose_data)))
        for i, (material_name, dose_rates) in enumerate(all_dose_data.items()):
            times = np.array(list(dose_rates.keys()))
            dose_values = np.array(list(dose_rates.values()))
            
            # Sort by time
            sort_idx = np.argsort(times)
            times = times[sort_idx]
            dose_values = dose_values[sort_idx]
            
            ax.plot(times, dose_values, 
                   color=colors[i], linewidth=2, label=material_name, marker='o')
        
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('Time (seconds)')
        ax.set_ylabel('Dose Rate (Sv/h/kg)')
        ax.set_ylim(1e-10, 1e10)
        ax.set_title('Total Dose Rate Comparison during Cooling - All Elements')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'all_elements_dose_rate_comparison.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  - Combined dose rate plot saved to: {plots_dir}")

    # Create combined activity plot for all elements
    if all_activity_data:
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(all_activity_data)))
        for i, (material_name, activity_data) in enumerate(all_activity_data.items()):
            times = activity_data['times']
            activity_values = activity_data['activity']
            
            ax.plot(times, activity_values, 
                   color=colors[i], linewidth=2, label=material_name, marker='o')
        
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('Time (seconds)')
        ax.set_ylabel('Total Activity (Bq)')
        ax.set_ylim(1e0, 1e20)
        ax.set_title('Total Activity Comparison during Cooling - All Elements')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'all_elements_activity_comparison.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  - Combined activity plot saved to: {plots_dir}")


def analyze_gas_production_reactions(results_dict: Dict[str, Dict[str, Any]], 
                                   results_dir: str) -> None:
    """
    Analyze which reactions are most responsible for H1 and He4 production.
    
    Args:
        results_dict: Dictionary containing neutronics results for all materials
        results_dir: Directory containing depletion results
    """
    print("\n" + "="*60)
    print("GAS PRODUCTION REACTION ANALYSIS")
    print("="*60)
    
    # Load the depletion chain to understand reactions
    chain = openmc.deplete.Chain.from_xml(openmc.config['chain_file'])
    
    for material_name in results_dict.keys():
        print(f"\n{material_name}:")
        
        # Read depletion results
        depletion_data = read_depletion_results(results_dir, material_name)
        if not depletion_data:
            continue

        #print(f"Depletion data: {depletion_data}")
        
        nuclides = depletion_data['nuclides']
        
        # Analyze the depletion chain for gas-producing reactions
        try:
            # Get the material's gas production from results_dict
            material_results = results_dict[material_name]
            gas_production = material_results.get('gas_production', {})
            
            print(f"  Gas production: {gas_production}")
            
            # Analyze reactions that produce He4 and H1
            he4_producers = []
            h1_producers = []
            
            # Look through the chain for reactions that produce He4 and H1
            for nuclide_name, nuclide_data in chain.nuclides.items():
                for reaction_name, reaction_data in nuclide_data.reactions.items():
                    # Check if this reaction produces He4
                    if 'He4' in reaction_data.products:
                        he4_producers.append({
                            'parent': nuclide_name,
                            'reaction': reaction_name,
                            'products': reaction_data.products
                        })
                    
                    # Check if this reaction produces H1
                    if 'H1' in reaction_data.products:
                        h1_producers.append({
                            'parent': nuclide_name,
                            'reaction': reaction_name,
                            'products': reaction_data.products
                        })
            
            print(f"  Found {len(he4_producers)} He4-producing reactions in chain")
            print(f"  Found {len(h1_producers)} H1-producing reactions in chain")
            
            # Show top reactions (this is qualitative since we don't have reaction rates)
            if he4_producers:
                print("  Example He4-producing reactions:")
                for i, reaction in enumerate(he4_producers[:5]):
                    print(f"    {reaction['parent']}({reaction['reaction']}) -> He4 + ...")
            
            if h1_producers:
                print("  Example H1-producing reactions:")
                for i, reaction in enumerate(h1_producers[:5]):
                    print(f"    {reaction['parent']}({reaction['reaction']}) -> H1 + ...")
                    
        except Exception as e:
            print(f"  Error analyzing reactions: {e}")
            print("  Note: Detailed reaction analysis requires additional OpenMC data processing")


def create_comprehensive_analysis(results_dict: Dict[str, Dict[str, Any]], 
                                results_dir: str) -> None:
    """
    Create comprehensive analysis including plots and reaction analysis.
    
    Args:
        results_dict: Dictionary containing neutronics results for all materials
        results_dir: Directory containing depletion results
    """
    print("\n" + "="*60)
    print("CREATING COMPREHENSIVE ANALYSIS")
    print("="*60)
    
    # Create activity and dose rate plots
    plot_activity_and_dose_analysis(results_dict, results_dir)
    
    # Analyze gas production reactions
    analyze_gas_production_reactions(results_dict, results_dir)
    
    print(f"\nAnalysis complete! Check the 'analysis_plots' directory for visualizations.")

def setup_openmc_model():
    model = create_model(config=SPHERICAL)
    model.settings.particles = OPENMC_NUM_PARTICLES
    cells = list(model.geometry.get_all_cells().values())
    
    # Create necessary directories
    microxs_and_flux_dir = os.path.join(RESULTS_DIR, 'microxs_and_flux')
    os.makedirs(microxs_and_flux_dir, exist_ok=True)
    
    microxs_file = os.path.join(microxs_and_flux_dir, 'microxs_1102.csv')
    flux_file = os.path.join(microxs_and_flux_dir, 'flux_spectrum_1102.txt')
    
    if os.path.exists(microxs_file) and os.path.exists(flux_file):
        print("Using existing microxs and flux")
        flux = [np.loadtxt(flux_file, comments='#', usecols=1)]
        microxs = openmc.deplete.MicroXS.from_csv(microxs_file)
    else:
        print("Calculating microxs and flux")
        flux_path, microxs_path = get_flux_and_microxs(
            model,
            chain_file=openmc.config['chain_file'],
            group_structure='UKAEA-1102',
            outdir=microxs_and_flux_dir,
        )
        flux = [np.loadtxt(flux_path, comments='#', usecols=1)]
        microxs = openmc.deplete.MicroXS.from_csv(microxs_path)
    
    # Time scheduler
    POWER_MW = 500
    TORUS_TO_SPHERE_VOLUME_RATIO = 1/4.03 # from notebooks/compare_volume_spherical_toroidal.ipynb
    FUSION_POWER_MEV = 17.6
    MEV_TO_J = 1.602176634e-13
    SOURCE_RATE = POWER_MW * 1e6 / (FUSION_POWER_MEV * MEV_TO_J)  * TORUS_TO_SPHERE_VOLUME_RATIO
    cooling_times = ['1 second', '1 minute', '1 hour', '10 hours', '1 day', '1 week', '2 weeks', '1 year', '2 years', '5 years', '10 years', '25 years', '100 years']
    
    scheduler = TimeScheduler(
        irradiation_time='1 year',
        cooling_times=cooling_times,
        source_rate=SOURCE_RATE,
        irradiation_steps=12,
    )
    
    timesteps, sources = scheduler.get_timesteps_and_source_rates()
    
    return model, flux, microxs, timesteps, sources



# 1) Create a spherical geometry using the config file 
model, flux, microxs, timesteps, sources = setup_openmc_model()

test_materials = [{'V' : 1.0}, {'Cr' : 1.0}, {'Ti' : 1.0}, {'W' : 1.0}, {'Zr' : 1.0}]
results_dict = {}
for test_material in test_materials:
    #material_name = material_string(test_material, 'V')
    material_name = list(test_material.keys())[0]
    
    # Check if results file already exists
    results_file = os.path.join(RESULTS_DIR, 'depletion_results', material_name, 'depletion_results.h5')
    if os.path.exists(results_file):
        print(f"Results file already exists for {material_name}, skipping...")
        # Try to load existing results
        try:
            results = openmc.deplete.Results(results_file)
            material_score = parse_openmc_results(results=results,
                                                  chain_file=openmc.config['chain_file'],
                                                  abs_file='/home/myless/Packages/fispact/nuclear_data/decay/abs_2012')
            results_dict[material_name] = material_score
            print(f"Loaded existing results for {material_name}: {material_score}")
            continue
        except Exception as e:
            print(f"Failed to load existing results for {material_name}: {e}")
            print("Will recalculate...")
    
    new_material = create_material(test_material, material_name)
    # make new material depletable 
    new_material.depletable = True
    
    # Replace vessel material
    model.materials.append(new_material)
    vessel_cell = model.geometry.get_cells_by_name('vessel')[0]
    new_material.volume = next(m.volume for m in model.materials if m.name == 'vcrtiwzr')
    vessel_cell.fill = new_material


    results = run_independent_depletion(
                                        model=model,
                                        depletable_cell='vessel',
                                        microxs=microxs,
                                        flux=flux,
                                        chain_file=openmc.config['chain_file'],
                                        timesteps=timesteps,
                                        source_rates=sources,
                                        outdir=os.path.join(RESULTS_DIR, 'depletion_results', material_name)
    )

    # 7) evaluate the batch of material's dose rates and gas production rates 
    material_score = parse_openmc_results(results=results,
                                          chain_file=openmc.config['chain_file'],
                                          abs_file='/home/myless/Packages/fispact/nuclear_data/decay/abs_2012')
    
    results_dict[material_name] = material_score

    print(f" Score for {material_name} is {material_score}")

# Calculate maximum compositions based on neutronics limits
print("\n" + "="*50)
print("CALCULATING MAXIMUM COMPOSITIONS")
print("="*50)

MAX_COMPOSITIONS = calculate_max_compositions(
    results_dict=results_dict,
    crit_limits=CRIT_LIMITS,
    dose_limits=DOSE_LIMITS
)

print("\n" + "="*50)
print("MAXIMUM COMPOSITIONS FOR V-X BINARY ALLOYS")
print("="*50)
for element, max_comp in MAX_COMPOSITIONS.items():
    print(f"{element}: {max_comp:.4f}")

print("\n" + "="*50)
print("SUMMARY")
print("="*50)
print("These compositions ensure:")
print(f"- He production ≤ {CRIT_LIMITS['He_appm']:.1f} appm")
print(f"- H production ≤ {CRIT_LIMITS['H_appm']:.1f} appm")
for days, limit in DOSE_LIMITS.items():
    print(f"- Dose rate at {days} days ≤ {limit:.2e} Sv/h")

print("\nUse MAX_COMPOSITIONS as bounds for your optimization problem.")

# Create comprehensive analysis plots and reaction analysis
create_comprehensive_analysis(results_dict, RESULTS_DIR)