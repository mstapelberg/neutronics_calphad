"""Comprehensive script to consolidate all past pipeline results.

This script:
1. Finds all neutronics_optimization_results.json files
2. Checks for duplicate material names across all runs
3. Migrates all depletion results to a single consolidated folder
4. Creates a unified neutronics_optimization_results.json
5. Verifies the migration process
6. Prepares the data for surrogate modeling
"""

import os
import json
import shutil
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
from collections import defaultdict

from neutronics_calphad.utils.io import material_string
from neutronics_calphad.optimizer.composition_sampler import CompositionSampler


def find_all_pipeline_results(base_dir: str = ".") -> List[str]:
    """Find all neutronics_optimization_results.json files.
    
    Parameters
    ----------
    base_dir : str
        Base directory to search in
        
    Returns
    -------
    List[str]
        List of paths to results files
    """
    
    results_files = []
    for root, dirs, files in os.walk(base_dir):
        if "neutronics_optimization_results.json" in files:
            results_path = os.path.join(root, "neutronics_optimization_results.json")
            results_files.append(results_path)
    
    return sorted(results_files)


def load_all_results(results_files: List[str]) -> Dict[str, Dict[str, Any]]:
    """Load all results files and organize by pipeline name.
    
    Parameters
    ----------
    results_files : List[str]
        List of paths to results files
        
    Returns
    -------
    Dict[str, Dict[str, Any]]
        Dictionary mapping pipeline names to their results
    """
    
    all_results = {}
    
    for results_file in results_files:
        try:
            # Extract pipeline name from path
            pipeline_dir = os.path.dirname(results_file)
            pipeline_name = os.path.basename(pipeline_dir)
            
            print(f"Loading {pipeline_name}: {results_file}")
            
            with open(results_file, 'r') as f:
                results = json.load(f)
            
            all_results[pipeline_name] = {
                'file_path': results_file,
                'dir_path': pipeline_dir,
                'results': results
            }
            
        except Exception as e:
            print(f"Error loading {results_file}: {e}")
            continue
    
    return all_results


def check_duplicates_across_all_runs(all_results: Dict[str, Dict[str, Any]], elements: List[str]) -> Dict[str, Any]:
    """Check for duplicate material names across all pipeline runs.
    
    Parameters
    ----------
    all_results : Dict[str, Dict[str, Any]]
        All loaded results organized by pipeline name
    elements : List[str]
        List of element symbols
        
    Returns
    -------
    Dict[str, Any]
        Comprehensive duplicate analysis
    """
    
    print("\n=== CHECKING DUPLICATES ACROSS ALL RUNS ===")
    
    # Collect all material names and their sources
    all_material_names = []
    compositions_by_name = defaultdict(list)
    
    for pipeline_name, pipeline_data in all_results.items():
        results = pipeline_data['results']
        
        for iteration in results.get('iterations', []):
            for material in iteration.get('materials', []):
                material_name = material['material_name']
                composition = material['composition']
                
                all_material_names.append(material_name)
                compositions_by_name[material_name].append({
                    'composition': composition,
                    'pipeline': pipeline_name,
                    'iteration': iteration.get('iteration', 0),
                    'material_index': len(compositions_by_name[material_name])
                })
    
    # Find duplicates
    duplicates = {name: comps for name, comps in compositions_by_name.items() if len(comps) > 1}
    
    # Analysis results
    analysis = {
        'total_materials': len(all_material_names),
        'unique_names': len(set(all_material_names)),
        'duplicate_names': len(duplicates),
        'duplicates': duplicates,
        'duplicate_details': {},
        'pipeline_summary': {}
    }
    
    # Pipeline summary
    for pipeline_name, pipeline_data in all_results.items():
        results = pipeline_data['results']
        pipeline_materials = 0
        for iteration in results.get('iterations', []):
            pipeline_materials += len(iteration.get('materials', []))
        
        analysis['pipeline_summary'][pipeline_name] = {
            'materials': pipeline_materials,
            'iterations': len(results.get('iterations', []))
        }
    
    # Analyze each duplicate
    for name, comps in duplicates.items():
        print(f"\nDuplicate material name: {name}")
        print(f"  Found {len(comps)} materials with this name:")
        
        duplicate_detail = {
            'name': name,
            'count': len(comps),
            'compositions': []
        }
        
        for i, comp_info in enumerate(comps):
            comp = comp_info['composition']
            print(f"    {i+1}. {comp_info['pipeline']} - Iteration {comp_info['iteration']}")
            print(f"       Composition: {comp}")
            
            # Generate new name with higher precision
            new_name_prec1 = material_string(comp, 'V', precision=1)
            new_name_prec2 = material_string(comp, 'V', precision=2)
            
            print(f"       New name (precision=1): {new_name_prec1}")
            print(f"       New name (precision=2): {new_name_prec2}")
            
            duplicate_detail['compositions'].append({
                'original_composition': comp,
                'pipeline': comp_info['pipeline'],
                'iteration': comp_info['iteration'],
                'material_index': comp_info['material_index'],
                'new_name_prec1': new_name_prec1,
                'new_name_prec2': new_name_prec2
            })
        
        analysis['duplicate_details'][name] = duplicate_detail
    
    return analysis


def consolidate_all_results(
    all_results: Dict[str, Dict[str, Any]],
    output_dir: str,
    elements: List[str],
    critical_limits: Dict[str, float],
    dose_limits: Dict[int, float],
    precision: int = 2
) -> Dict[str, Any]:
    """Consolidate all results into a single unified dataset.
    
    Parameters
    ----------
    all_results : Dict[str, Dict[str, Any]]
        All loaded results organized by pipeline name
    output_dir : str
        Output directory for consolidated results
    elements : List[str]
        List of element symbols
    critical_limits : Dict[str, float]
        Gas production limits
    dose_limits : Dict[int, float]
        Dose rate limits
    precision : int
        Precision for material naming
        
    Returns
    -------
    Dict[str, Any]
        Consolidated results
    """
    
    print(f"\n=== CONSOLIDATING ALL RESULTS TO {output_dir} ===")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    neutronics_dir = os.path.join(output_dir, 'neutronics_depletion')
    os.makedirs(neutronics_dir, exist_ok=True)
    
    # Track all materials and handle duplicates
    all_materials = []
    material_counter = defaultdict(int)
    
    for pipeline_name, pipeline_data in all_results.items():
        results = pipeline_data['results']
        pipeline_dir = pipeline_data['dir_path']
        
        print(f"\nProcessing {pipeline_name}...")
        
        for iteration in results.get('iterations', []):
            for material in iteration.get('materials', []):
                composition = material['composition']
                old_material_name = material['material_name']
                
                # Generate new material name with precision
                new_material_name = material_string(composition, 'V', precision=precision)
                
                # Handle duplicates by adding counter
                if new_material_name in material_counter:
                    material_counter[new_material_name] += 1
                    new_material_name = f"{new_material_name}_v{material_counter[new_material_name]}"
                else:
                    material_counter[new_material_name] = 0
                
                # Copy depletion results if they exist
                old_depletion_dir = os.path.join(pipeline_dir, 'neutronics_depletion', old_material_name)
                new_depletion_dir = os.path.join(neutronics_dir, new_material_name)
                
                if os.path.exists(old_depletion_dir):
                    if os.path.exists(new_depletion_dir):
                        print(f"  Warning: {new_material_name} already exists, skipping copy")
                    else:
                        shutil.copytree(old_depletion_dir, new_depletion_dir)
                        print(f"  Copied: {old_material_name} -> {new_material_name}")
                else:
                    print(f"  Warning: No depletion results for {old_material_name}")
                
                # Create material result
                material_result = {
                    'material_name': new_material_name,
                    'original_name': old_material_name,
                    'source_pipeline': pipeline_name,
                    'source_iteration': iteration.get('iteration', 0),
                    'composition': composition,
                    'composition_array': material.get('composition_array', []),
                    'satisfy_dose': material.get('satisfy_dose', False),
                    'satisfy_gas': material.get('satisfy_gas', False),
                    'dose_rates': material.get('dose_rates', {}),
                    'gas_production': material.get('gas_production', {}),
                    'neutronics_outputs': material.get('neutronics_outputs', []),
                    'raw_neutronics_data': material.get('raw_neutronics_data', {})
                }
                
                all_materials.append(material_result)
    
    # Create consolidated results structure
    consolidated_results = {
        'metadata': {
            'consolidated_from': list(all_results.keys()),
            'elements': elements,
            'critical_limits': critical_limits,
            'dose_limits': dose_limits,
            'naming_precision': precision,
            'total_materials': len(all_materials),
            'consolidation_timestamp': str(np.datetime64('now'))
        },
        'materials': all_materials  # Flat list instead of iterations
    }
    
    # Save consolidated results
    results_file = os.path.join(output_dir, "neutronics_optimization_results.json")
    with open(results_file, 'w') as f:
        json.dump(consolidated_results, f, indent=2, default=str)
    
    print(f"\nConsolidated {len(all_materials)} materials from {len(all_results)} pipelines")
    print(f"Results saved to: {results_file}")
    
    return consolidated_results


def verify_consolidation(consolidated_results: Dict[str, Any], output_dir: str) -> Dict[str, Any]:
    """Verify the consolidation process.
    
    Parameters
    ----------
    consolidated_results : Dict[str, Any]
        Consolidated results
    output_dir : str
        Output directory
        
    Returns
    -------
    Dict[str, Any]
        Verification results
    """
    
    print(f"\n=== VERIFYING CONSOLIDATION ===")
    
    verification = {
        'total_materials': len(consolidated_results['materials']),
        'depletion_dirs_exist': 0,
        'depletion_dirs_missing': 0,
        'unique_compositions': 0,
        'duplicate_compositions': 0
    }
    
    # Check depletion directories
    neutronics_dir = os.path.join(output_dir, 'neutronics_depletion')
    compositions_seen = set()
    
    for material in consolidated_results['materials']:
        material_name = material['material_name']
        depletion_dir = os.path.join(neutronics_dir, material_name)
        
        if os.path.exists(depletion_dir):
            verification['depletion_dirs_exist'] += 1
        else:
            verification['depletion_dirs_missing'] += 1
            print(f"  Warning: Missing depletion dir for {material_name}")
        
        # Check for duplicate compositions
        comp_tuple = tuple(material['composition'].items())
        if comp_tuple in compositions_seen:
            verification['duplicate_compositions'] += 1
        else:
            compositions_seen.add(comp_tuple)
            verification['unique_compositions'] += 1
    
    print(f"Verification Results:")
    print(f"  Total materials: {verification['total_materials']}")
    print(f"  Depletion dirs exist: {verification['depletion_dirs_exist']}")
    print(f"  Depletion dirs missing: {verification['depletion_dirs_missing']}")
    print(f"  Unique compositions: {verification['unique_compositions']}")
    print(f"  Duplicate compositions: {verification['duplicate_compositions']}")
    
    return verification


def prepare_for_surrogate_modeling(consolidated_results: Dict[str, Any], elements: List[str]) -> CompositionSampler:
    """Prepare consolidated results for surrogate modeling.
    
    Parameters
    ----------
    consolidated_results : Dict[str, Any]
        Consolidated results
    elements : List[str]
        List of element symbols
        
    Returns
    -------
    CompositionSampler
        Fitted sampler ready for surrogate modeling
    """
    
    print(f"\n=== PREPARING FOR SURROGATE MODELING ===")
    
    # Extract training data
    all_compositions = []
    all_outputs = []
    
    for material in consolidated_results['materials']:
        if 'composition_array' in material and material['composition_array']:
            comp_array = material['composition_array']
        else:
            # Convert composition dict to array
            comp_array = [material['composition'][el] for el in elements]
        
        if 'neutronics_outputs' in material and material['neutronics_outputs']:
            outputs = material['neutronics_outputs']
        else:
            # Reconstruct outputs from individual values
            dose_rates = material.get('dose_rates', {})
            gas_production = material.get('gas_production', {})
            outputs = [
                dose_rates.get(14, 0),
                dose_rates.get(365, 0),
                dose_rates.get(3650, 0),
                dose_rates.get(36500, 0),
                gas_production.get('He_appm', 0),
                gas_production.get('H_appm', 0)
            ]
        
        all_compositions.append(comp_array)
        all_outputs.append(outputs)
    
    if not all_compositions:
        raise ValueError("No training data found in consolidated results")
    
    # Create and fit sampler
    sampler = CompositionSampler(
        elements=elements,
        batch_size=10,
        min_compositions={'V': 0.70},
        max_compositions={'V': 0.95, 'Cr': 0.20, 'Ti': 0.15, 'W': 0.20, 'Zr': 0.1}
    )
    
    compositions_array = np.array(all_compositions)
    outputs_array = np.array(all_outputs)
    sampler.update(compositions_array, outputs_array)
    
    print(f"Loaded {len(all_compositions)} training samples")
    
    # Fit surrogate models
    sampler.fit_surrogates()
    print("Surrogate models fitted successfully!")
    
    return sampler


def main():
    """Main consolidation workflow."""
    
    # Configuration
    ELEMENTS = ['V', 'Cr', 'Ti', 'W', 'Zr']
    CRIT_LIMITS = {"He_appm": 1172.2/2, "H_appm": 1500}
    DOSE_LIMITS = {14: 1e5, 365: 1, 3650: 1e-2, 36500: 1e-4}
    OUTPUT_DIR = "consolidated_pipeline_results"
    
    print("=== PIPELINE RESULTS CONSOLIDATION ===")
    
    # Step 1: Find all results files
    print("\nStep 1: Finding all pipeline results...")
    results_files = find_all_pipeline_results()
    print(f"Found {len(results_files)} results files:")
    for f in results_files:
        print(f"  {f}")
    
    if not results_files:
        print("No results files found!")
        return
    
    # Step 2: Load all results
    print("\nStep 2: Loading all results...")
    all_results = load_all_results(results_files)
    print(f"Loaded {len(all_results)} pipeline results")
    
    # Step 3: Check for duplicates
    print("\nStep 3: Checking for duplicates...")
    duplicate_analysis = check_duplicates_across_all_runs(all_results, ELEMENTS)
    
    print(f"\nDuplicate Summary:")
    print(f"  Total materials: {duplicate_analysis['total_materials']}")
    print(f"  Unique names: {duplicate_analysis['unique_names']}")
    print(f"  Duplicate names: {duplicate_analysis['duplicate_names']}")
    
    if duplicate_analysis['duplicate_names'] > 0:
        print(f"  Duplicate rate: {100 * duplicate_analysis['duplicate_names'] / duplicate_analysis['unique_names']:.1f}%")
    
    # Step 4: Consolidate results
    print("\nStep 4: Consolidating results...")
    consolidated_results = consolidate_all_results(
        all_results, OUTPUT_DIR, ELEMENTS, CRIT_LIMITS, DOSE_LIMITS, precision=2
    )
    
    # Step 5: Verify consolidation
    print("\nStep 5: Verifying consolidation...")
    verification = verify_consolidation(consolidated_results, OUTPUT_DIR)
    
    # Step 6: Prepare for surrogate modeling
    print("\nStep 6: Preparing for surrogate modeling...")
    try:
        sampler = prepare_for_surrogate_modeling(consolidated_results, ELEMENTS)
        
        # Test surrogate modeling
        print("\nTesting surrogate modeling...")
        test_comps = sampler.suggest(1000)
        feasible = sampler.predict_feasibility(test_comps)
        good_comps = test_comps[feasible]
        
        print(f"Generated {len(test_comps)} test compositions")
        print(f"Found {len(good_comps)} feasible compositions ({100*len(good_comps)/len(test_comps):.1f}%)")
        
    except Exception as e:
        print(f"Error in surrogate modeling: {e}")
    
    print(f"\n=== CONSOLIDATION COMPLETE ===")
    print(f"All results consolidated in: {OUTPUT_DIR}")
    print(f"You can now use the consolidated data for surrogate modeling!")


if __name__ == "__main__":
    main() 