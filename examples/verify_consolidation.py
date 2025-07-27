"""Simple script to verify the consolidation process and show results structure."""

import os
import json
from typing import Dict, List, Any


def verify_consolidated_results(consolidated_dir: str = "consolidated_pipeline_results"):
    """Verify the consolidated results structure.
    
    Parameters
    ----------
    consolidated_dir : str
        Directory containing consolidated results
    """
    
    print(f"=== VERIFYING CONSOLIDATED RESULTS IN {consolidated_dir} ===")
    
    # Check if directory exists
    if not os.path.exists(consolidated_dir):
        print(f"Error: Directory {consolidated_dir} does not exist!")
        return
    
    # Load consolidated results
    results_file = os.path.join(consolidated_dir, "neutronics_optimization_results.json")
    if not os.path.exists(results_file):
        print(f"Error: Results file {results_file} does not exist!")
        return
    
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    # Check structure
    print(f"\nResults Structure:")
    print(f"  Metadata keys: {list(results.get('metadata', {}).keys())}")
    print(f"  Materials count: {len(results.get('materials', []))}")
    
    # Show metadata
    metadata = results.get('metadata', {})
    print(f"\nMetadata:")
    print(f"  Consolidated from: {metadata.get('consolidated_from', [])}")
    print(f"  Elements: {metadata.get('elements', [])}")
    print(f"  Total materials: {metadata.get('total_materials', 0)}")
    print(f"  Naming precision: {metadata.get('naming_precision', 'unknown')}")
    print(f"  Consolidation timestamp: {metadata.get('consolidation_timestamp', 'unknown')}")
    
    # Check depletion directories
    neutronics_dir = os.path.join(consolidated_dir, 'neutronics_depletion')
    if os.path.exists(neutronics_dir):
        depletion_dirs = [d for d in os.listdir(neutronics_dir) 
                         if os.path.isdir(os.path.join(neutronics_dir, d))]
        print(f"\nDepletion directories: {len(depletion_dirs)}")
        
        # Check for missing depletion results
        missing_depletion = 0
        for material in results.get('materials', []):
            material_name = material['material_name']
            depletion_path = os.path.join(neutronics_dir, material_name)
            if not os.path.exists(depletion_path):
                missing_depletion += 1
        
        print(f"  Materials with missing depletion: {missing_depletion}")
    else:
        print(f"\nWarning: No neutronics_depletion directory found!")
    
    # Show sample materials
    materials = results.get('materials', [])
    if materials:
        print(f"\nSample Materials (first 3):")
        for i, material in enumerate(materials[:3]):
            print(f"  {i+1}. {material['material_name']}")
            print(f"     Source: {material.get('source_pipeline', 'unknown')}")
            print(f"     Composition: {material['composition']}")
            print(f"     Satisfy dose: {material.get('satisfy_dose', False)}")
            print(f"     Satisfy gas: {material.get('satisfy_gas', False)}")
    
    # Check for duplicates
    material_names = [m['material_name'] for m in materials]
    unique_names = set(material_names)
    duplicates = len(material_names) - len(unique_names)
    
    print(f"\nDuplicate Analysis:")
    print(f"  Total material names: {len(material_names)}")
    print(f"  Unique material names: {len(unique_names)}")
    print(f"  Duplicate names: {duplicates}")
    
    if duplicates > 0:
        print(f"  Duplicate rate: {100 * duplicates / len(material_names):.1f}%")
    
    # Check composition uniqueness
    compositions = [tuple(sorted(m['composition'].items())) for m in materials]
    unique_compositions = set(compositions)
    duplicate_compositions = len(compositions) - len(unique_compositions)
    
    print(f"\nComposition Analysis:")
    print(f"  Total compositions: {len(compositions)}")
    print(f"  Unique compositions: {len(unique_compositions)}")
    print(f"  Duplicate compositions: {duplicate_compositions}")
    
    if duplicate_compositions > 0:
        print(f"  Duplicate composition rate: {100 * duplicate_compositions / len(compositions):.1f}%")


def show_pipeline_summary(consolidated_dir: str = "consolidated_pipeline_results"):
    """Show summary by source pipeline.
    
    Parameters
    ----------
    consolidated_dir : str
        Directory containing consolidated results
    """
    
    results_file = os.path.join(consolidated_dir, "neutronics_optimization_results.json")
    if not os.path.exists(results_file):
        print(f"Error: Results file {results_file} does not exist!")
        return
    
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    materials = results.get('materials', [])
    
    # Group by source pipeline
    pipeline_summary = {}
    for material in materials:
        pipeline = material.get('source_pipeline', 'unknown')
        if pipeline not in pipeline_summary:
            pipeline_summary[pipeline] = {
                'total': 0,
                'satisfy_dose': 0,
                'satisfy_gas': 0,
                'satisfy_both': 0
            }
        
        pipeline_summary[pipeline]['total'] += 1
        if material.get('satisfy_dose', False):
            pipeline_summary[pipeline]['satisfy_dose'] += 1
        if material.get('satisfy_gas', False):
            pipeline_summary[pipeline]['satisfy_gas'] += 1
        if material.get('satisfy_dose', False) and material.get('satisfy_gas', False):
            pipeline_summary[pipeline]['satisfy_both'] += 1
    
    print(f"\n=== PIPELINE SUMMARY ===")
    for pipeline, stats in pipeline_summary.items():
        print(f"\n{pipeline}:")
        print(f"  Total materials: {stats['total']}")
        print(f"  Satisfy dose: {stats['satisfy_dose']} ({100*stats['satisfy_dose']/stats['total']:.1f}%)")
        print(f"  Satisfy gas: {stats['satisfy_gas']} ({100*stats['satisfy_gas']/stats['total']:.1f}%)")
        print(f"  Satisfy both: {stats['satisfy_both']} ({100*stats['satisfy_both']/stats['total']:.1f}%)")


def main():
    """Main verification function."""
    
    consolidated_dir = "consolidated_pipeline_results"
    
    if not os.path.exists(consolidated_dir):
        print(f"Consolidated directory {consolidated_dir} not found!")
        print("Run consolidate_all_results.py first to create the consolidated dataset.")
        return
    
    # Verify structure
    verify_consolidated_results(consolidated_dir)
    
    # Show pipeline summary
    show_pipeline_summary(consolidated_dir)
    
    print(f"\n=== VERIFICATION COMPLETE ===")
    print(f"You can now use the consolidated data for surrogate modeling!")


if __name__ == "__main__":
    main() 