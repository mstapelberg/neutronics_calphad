"""Utility script to migrate depletion results from old runs into current pipeline.

This script helps you:
1. Copy depletion results from old pipeline runs
2. Merge them into the current pipeline structure
3. Use existing results to fit surrogate models and suggest new compositions
"""

import os
import json
import shutil
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import numpy as np

from neutronics_calphad.optimizer.composition_sampler import CompositionSampler
from neutronics_calphad.neutronics.dose import contact_dose
from neutronics_calphad.neutronics.depletion import extract_gas_production


def migrate_depletion_results(
    old_pipeline_dir: str,
    new_pipeline_dir: str,
    elements: List[str],
    critical_limits: Dict[str, float],
    dose_limits: Dict[int, float]
) -> Dict[str, Any]:
    """Migrate depletion results from old pipeline to new pipeline.
    
    Parameters
    ----------
    old_pipeline_dir : str
        Path to the old pipeline results directory
    new_pipeline_dir : str
        Path to the new pipeline results directory
    elements : List[str]
        List of element symbols
    critical_limits : Dict[str, float]
        Gas production limits
    dose_limits : Dict[int, float]
        Dose rate limits
        
    Returns
    -------
    Dict[str, Any]
        Merged neutronics results with all evaluated compositions
    """
    
    # Create new pipeline directory structure
    new_neutronics_dir = os.path.join(new_pipeline_dir, 'neutronics_depletion')
    os.makedirs(new_neutronics_dir, exist_ok=True)
    
    # Find old depletion results
    old_neutronics_dir = os.path.join(old_pipeline_dir, 'neutronics_depletion')
    if not os.path.exists(old_neutronics_dir):
        raise ValueError(f"Old neutronics directory not found: {old_neutronics_dir}")
    
    # Copy all depletion results
    print(f"Copying depletion results from {old_neutronics_dir} to {new_neutronics_dir}")
    for item in os.listdir(old_neutronics_dir):
        old_path = os.path.join(old_neutronics_dir, item)
        new_path = os.path.join(new_neutronics_dir, item)
        
        if os.path.isdir(old_path):
            shutil.copytree(old_path, new_path, dirs_exist_ok=True)
            print(f"  Copied directory: {item}")
        else:
            shutil.copy2(old_path, new_path)
            print(f"  Copied file: {item}")
    
    # Load old neutronics results if they exist
    old_results_file = os.path.join(old_pipeline_dir, "neutronics_optimization_results.json")
    if os.path.exists(old_results_file):
        with open(old_results_file, 'r') as f:
            old_results = json.load(f)
        print(f"Loaded existing results from {old_results_file}")
    else:
        print("No existing neutronics results found, will reconstruct from depletion data")
        old_results = None
    
    # Reconstruct results from depletion data
    reconstructed_results = reconstruct_results_from_depletion(
        new_neutronics_dir, elements, critical_limits, dose_limits, old_results
    )
    
    # Save merged results
    new_results_file = os.path.join(new_pipeline_dir, "neutronics_optimization_results.json")
    with open(new_results_file, 'w') as f:
        json.dump(reconstructed_results, f, indent=2, default=str)
    
    print(f"Saved merged results to {new_results_file}")
    return reconstructed_results


def reconstruct_results_from_depletion(
    neutronics_dir: str,
    elements: List[str],
    critical_limits: Dict[str, float],
    dose_limits: Dict[int, float],
    existing_results: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Reconstruct neutronics results from depletion calculation files.
    
    Parameters
    ----------
    neutronics_dir : str
        Directory containing depletion results
    elements : List[str]
        List of element symbols
    critical_limits : Dict[str, float]
        Gas production limits
    dose_limits : Dict[int, float]
        Dose rate limits
    existing_results : Optional[Dict[str, Any]]
        Existing results to merge with
        
    Returns
    -------
    Dict[str, Any]
        Reconstructed neutronics results
    """
    
    # Initialize results structure
    if existing_results:
        results = existing_results
        print(f"Using existing results with {len(results.get('materials', []))} materials")
    else:
        results = {
            'metadata': {
                'elements': elements,
                'critical_limits': critical_limits,
                'dose_limits': dose_limits,
                'reconstructed_from_depletion': True
            },
            'materials': []  # Flat list instead of iterations
        }
    
    # Find all depletion result directories
    depletion_dirs = []
    for item in os.listdir(neutronics_dir):
        item_path = os.path.join(neutronics_dir, item)
        if os.path.isdir(item_path) and os.path.exists(os.path.join(item_path, 'depletion_results.h5')):
            depletion_dirs.append(item)
    
    print(f"Found {len(depletion_dirs)} depletion result directories")
    
    # Process each depletion result
    all_materials = []
    for material_name in depletion_dirs:
        try:
            # Parse composition from material name
            # Expected format: "V0.85Cr0.10Ti0.03W0.01Zr0.01"
            comp_dict = parse_composition_from_name(material_name, elements)
            comp_array = np.array([comp_dict[el] for el in elements])
            
            # Load depletion results
            depletion_path = os.path.join(neutronics_dir, material_name, 'depletion_results.h5')
            if not os.path.exists(depletion_path):
                print(f"Warning: No depletion results for {material_name}")
                continue
            
            # Evaluate material (this will load the results and compute dose/gas)
            detailed_results = evaluate_material_from_depletion(
                depletion_path, critical_limits, dose_limits
            )
            
            # Store material result
            material_result = {
                'material_name': material_name,
                'composition': comp_dict,
                'composition_array': comp_array.tolist(),
                'satisfy_dose': detailed_results['satisfy_dose'],
                'satisfy_gas': detailed_results['satisfy_gas'],
                'dose_rates': detailed_results['dose_at_limit'],
                'gas_production': detailed_results['gas_production_rates'],
                'neutronics_outputs': detailed_results['neutronics_outputs'].tolist(),
                'raw_neutronics_data': {
                    'times_s': detailed_results['times_s'],
                    'total_dose': detailed_results['total_dose'],
                    'final_irr_time': detailed_results['final_irr_time'],
                    'cool_start_time': detailed_results['cool_start_time'],
                    'source_rates': detailed_results['source_rates']
                }
            }
            
            all_materials.append(material_result)
            print(f"  Processed {material_name}: dose_ok={detailed_results['satisfy_dose']}, "
                  f"gas_ok={detailed_results['satisfy_gas']}")
            
        except Exception as e:
            print(f"Error processing {material_name}: {e}")
            continue
    
    # Update results with flat materials list
    results['materials'] = all_materials
    results['metadata']['total_materials_processed'] = len(all_materials)
    
    print(f"Reconstructed {len(all_materials)} materials")
    return results


def parse_composition_from_name(material_name: str, elements: List[str]) -> Dict[str, float]:
    """Parse composition from material name like 'V-2.1Cr-3.0Ti-4.2W-6.0Zr'.
    
    Parameters
    ----------
    material_name : str
        Material name containing composition
    elements : List[str]
        List of element symbols
        
    Returns
    -------
    Dict[str, float]
        Composition dictionary
    """
    import re
    
    comp_dict = {}
    
    # Parse each element (handle both old format V0.85Cr0.10 and new format V-2.1Cr-3.0Ti)
    for element in elements:
        # Try new format first: V-2.1Cr-3.0Ti-4.2W-6.0Zr
        pattern = f"{element}(\\d+\\.\\d+)"
        match = re.search(pattern, material_name)
        if match:
            percentage = float(match.group(1))
            comp_dict[element] = percentage / 100.0  # Convert percentage to fraction
        else:
            # Try old format: V0.85Cr0.10Ti0.03W0.01Zr0.01
            pattern_old = f"{element}(\\d+\\.\\d+)"
            match_old = re.search(pattern_old, material_name)
            if match_old:
                fraction = float(match_old.group(1))
                comp_dict[element] = fraction
            else:
                comp_dict[element] = 0.0
    
    # Normalize to sum to 1.0
    total = sum(comp_dict.values())
    if total > 0:
        for element in comp_dict:
            comp_dict[element] /= total
    
    return comp_dict


def evaluate_material_from_depletion(
    depletion_path: str,
    critical_limits: Dict[str, float],
    dose_limits: Dict[int, float]
) -> Dict[str, Any]:
    """Evaluate material from depletion results file.
    
    Parameters
    ----------
    depletion_path : str
        Path to depletion_results.h5 file
    critical_limits : Dict[str, float]
        Gas production limits
    dose_limits : Dict[int, float]
        Dose rate limits
        
    Returns
    -------
    Dict[str, Any]
        Detailed evaluation results
    """
    import openmc.deplete
    
    # Load depletion results
    results = openmc.deplete.Results(depletion_path)
    
    # Compute dose rates
    chain_file = '/home/myless/nuclear_data/tendl21-fispact20-gefy61-chain.xml'
    abs_file = '/home/myless/nuclear_data/decay/abs_2012'
    
    times_s, dose_dicts = contact_dose(results=results, chain_file=chain_file, abs_file=abs_file)
    
    # Compute gas production
    gas_production_rates = extract_gas_production(results)
    
    # Identify end of irradiation
    source_rates = results.get_source_rates()
    final_idx = np.nonzero(source_rates)[0][-1]
    final_irr_time = times_s[final_idx]
    
    # Build total-dose lookup
    total_dose = {t: sum(d.values()) for t, d in zip(times_s, dose_dicts)}
    
    # Locate cooling step
    cool_start_idx = final_idx + 1
    if cool_start_idx >= len(times_s):
        raise RuntimeError("No post-irradiation time steps available")
    
    cool_start_time = times_s[cool_start_idx]
    cool_times = times_s[cool_start_idx:] - cool_start_time
    
    # Check dose limits
    satisfy_dose = True
    dose_at_limit = {}
    for days_after, limit in dose_limits.items():
        target_s = days_after * 24 * 3600
        rel_idx = np.searchsorted(cool_times, target_s, side='left')
        if rel_idx >= len(cool_times):
            rel_idx = len(cool_times) - 1
        
        abs_idx = cool_start_idx + rel_idx
        t_actual = times_s[abs_idx]
        rate_actual = total_dose[t_actual]
        
        dose_at_limit[days_after] = rate_actual
        if rate_actual > limit:
            satisfy_dose = False
    
    # Check gas limits
    satisfy_gas = True
    for gas, produced in gas_production_rates.items():
        limit = critical_limits.get(gas, np.inf)
        if produced > limit:
            satisfy_gas = False
    
    # Raw outputs
    neutronics_outputs = np.array([
        dose_at_limit.get(14, 0),
        dose_at_limit.get(365, 0),
        dose_at_limit.get(3650, 0),
        dose_at_limit.get(36500, 0),
        gas_production_rates.get('He_appm', 0),
        gas_production_rates.get('H_appm', 0)
    ])
    
    return {
        'satisfy_dose': satisfy_dose,
        'satisfy_gas': satisfy_gas,
        'dose_at_limit': dose_at_limit,
        'gas_production_rates': gas_production_rates,
        'neutronics_outputs': neutronics_outputs,
        'times_s': times_s.tolist(),
        'total_dose': {str(t): v for t, v in total_dose.items()},
        'final_irr_time': final_irr_time,
        'cool_start_time': cool_start_time,
        'source_rates': source_rates.tolist()
    }


def use_existing_results_for_surrogate(
    results_file: str,
    elements: List[str],
    min_compositions: Dict[str, float],
    max_compositions: Dict[str, float],
    num_suggestions: int = 1000
) -> Tuple[CompositionSampler, np.ndarray]:
    """Use existing neutronics results to fit surrogate models and suggest new compositions.
    
    Parameters
    ----------
    results_file : str
        Path to neutronics results JSON file
    elements : List[str]
        List of element symbols
    min_compositions : Dict[str, float]
        Minimum composition constraints
    max_compositions : Dict[str, float]
        Maximum composition constraints
    num_suggestions : int
        Number of new compositions to suggest
        
    Returns
    -------
    Tuple[CompositionSampler, np.ndarray]
        Fitted sampler and suggested compositions
    """
    
    # Load existing results
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    # Create sampler
    sampler = CompositionSampler(
        elements=elements,
        batch_size=10,
        min_compositions=min_compositions,
        max_compositions=max_compositions
    )
    
    # Extract training data
    all_compositions = []
    all_outputs = []
    
    for iteration in results['iterations']:
        for material in iteration['materials']:
            comp_array = material['composition_array']
            outputs = material['neutronics_outputs']
            
            all_compositions.append(comp_array)
            all_outputs.append(outputs)
    
    if not all_compositions:
        raise ValueError("No training data found in results file")
    
    # Update sampler with existing data
    compositions_array = np.array(all_compositions)
    outputs_array = np.array(all_outputs)
    sampler.update(compositions_array, outputs_array)
    
    print(f"Loaded {len(all_compositions)} training samples")
    
    # Fit surrogate models
    sampler.fit_surrogates()
    print("Fitted surrogate models")
    
    # Suggest new compositions
    suggested_comps = sampler.suggest(num_suggestions)
    feasible = sampler.predict_feasibility(suggested_comps)
    good_comps = suggested_comps[feasible]
    
    print(f"Suggested {num_suggestions} compositions")
    print(f"Found {len(good_comps)} feasible compositions")
    
    return sampler, good_comps


if __name__ == "__main__":
    # Example usage
    
    # Configuration
    ELEMENTS = ['V', 'Cr', 'Ti', 'W', 'Zr']
    MIN_COMPOSITIONS = {'V': 0.70}
    MAX_COMPOSITIONS = {'V': 0.95, 'Cr': 0.20, 'Ti': 0.15, 'W': 0.20, 'Zr': 0.1}
    CRIT_LIMITS = {"He_appm": 1172.2/2, "H_appm": 1500}
    DOSE_LIMITS = {14: 1e5, 365: 1, 3650: 1e-2, 36500: 1e-4}
    
    # Example 1: Migrate results from old run
    # old_dir = "sequential_materials_pipeline_old_run"
    # new_dir = "sequential_materials_pipeline_run_1"
    # migrated_results = migrate_depletion_results(old_dir, new_dir, ELEMENTS, CRIT_LIMITS, DOSE_LIMITS)
    
    # Example 2: Use existing results for surrogate modeling
    # results_file = "sequential_materials_pipeline_run_1/neutronics_optimization_results.json"
    # sampler, good_comps = use_existing_results_for_surrogate(
    #     results_file, ELEMENTS, MIN_COMPOSITIONS, MAX_COMPOSITIONS, num_suggestions=1000
    # )
    
    print("Migration utility ready. Uncomment examples above to use.") 