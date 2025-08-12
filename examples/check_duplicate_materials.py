"""Utility script to check for duplicate material names in neutronics results.

This script helps identify and fix issues where similar compositions
generate the same material name, potentially causing data overwrites.
"""

import os
import json
import numpy as np
from typing import Dict, List, Any, Tuple
from collections import defaultdict

from neutronics_calphad.utils.io import material_string


def check_duplicate_material_names(results_file: str, elements: List[str]) -> Dict[str, Any]:
    """Check for duplicate material names in neutronics results.
    
    Parameters
    ----------
    results_file : str
        Path to neutronics results JSON file
    elements : List[str]
        List of element symbols
        
    Returns
    -------
    Dict[str, Any]
        Analysis results with duplicate information
    """
    
    print(f"Checking for duplicate material names in {results_file}")
    
    # Load results
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    # Collect all material names and their compositions
    material_names = []
    compositions_by_name = defaultdict(list)
    
    for iteration in results['iterations']:
        for material in iteration['materials']:
            material_name = material['material_name']
            composition = material['composition']
            material_names.append(material_name)
            compositions_by_name[material_name].append({
                'composition': composition,
                'iteration': iteration['iteration'],
                'material_index': len(compositions_by_name[material_name])
            })
    
    # Find duplicates
    duplicates = {name: comps for name, comps in compositions_by_name.items() if len(comps) > 1}
    
    # Analysis results
    analysis = {
        'total_materials': len(material_names),
        'unique_names': len(set(material_names)),
        'duplicate_names': len(duplicates),
        'duplicates': duplicates,
        'duplicate_details': {}
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
            print(f"    {i+1}. Iteration {comp_info['iteration']}, Material {comp_info['material_index']}")
            print(f"       Composition: {comp}")
            
            # Generate new name with higher precision
            new_name_prec1 = material_string(comp, 'V', precision=1)
            new_name_prec2 = material_string(comp, 'V', precision=2)
            
            print(f"       New name (precision=1): {new_name_prec1}")
            print(f"       New name (precision=2): {new_name_prec2}")
            
            duplicate_detail['compositions'].append({
                'original_composition': comp,
                'iteration': comp_info['iteration'],
                'material_index': comp_info['material_index'],
                'new_name_prec1': new_name_prec1,
                'new_name_prec2': new_name_prec2
            })
        
        analysis['duplicate_details'][name] = duplicate_detail
    
    return analysis


def suggest_fixes(analysis: Dict[str, Any]) -> List[str]:
    """Suggest fixes for duplicate material names.
    
    Parameters
    ----------
    analysis : Dict[str, Any]
        Analysis results from check_duplicate_material_names
        
    Returns
    -------
    List[str]
        List of suggested fixes
    """
    
    suggestions = []
    
    if analysis['duplicate_names'] == 0:
        suggestions.append("No duplicate material names found. No fixes needed.")
        return suggestions
    
    suggestions.append(f"Found {analysis['duplicate_names']} duplicate material names.")
    suggestions.append("Suggested fixes:")
    
    for name, detail in analysis['duplicate_details'].items():
        suggestions.append(f"\nFor duplicate '{name}':")
        
        # Check if precision=1 fixes the issue
        new_names_prec1 = [comp['new_name_prec1'] for comp in detail['compositions']]
        unique_prec1 = len(set(new_names_prec1))
        
        if unique_prec1 == len(new_names_prec1):
            suggestions.append(f"  ✓ Precision=1 fixes the issue (all names unique)")
        else:
            suggestions.append(f"  ✗ Precision=1 does not fix the issue ({unique_prec1} unique names)")
            
            # Check precision=2
            new_names_prec2 = [comp['new_name_prec2'] for comp in detail['compositions']]
            unique_prec2 = len(set(new_names_prec2))
            
            if unique_prec2 == len(new_names_prec2):
                suggestions.append(f"  ✓ Precision=2 fixes the issue (all names unique)")
            else:
                suggestions.append(f"  ✗ Even precision=2 does not fix the issue ({unique_prec2} unique names)")
                suggestions.append(f"    Consider using a hash-based naming scheme")
    
    return suggestions


def test_material_naming_precision():
    """Test the material naming function with different precision levels."""
    
    print("Testing material naming with different precision levels:")
    print("=" * 60)
    
    # Test cases that would cause collisions with precision=0
    test_compositions = [
        {'V': 0.85, 'Cr': 0.021, 'Ti': 0.03, 'W': 0.042, 'Zr': 0.06},  # V-2.1Cr-3.0Ti-4.2W-6.0Zr
        {'V': 0.85, 'Cr': 0.026, 'Ti': 0.031, 'W': 0.042, 'Zr': 0.062}, # V-2.6Cr-3.1Ti-4.2W-6.2Zr
        {'V': 0.85, 'Cr': 0.025, 'Ti': 0.03, 'W': 0.041, 'Zr': 0.061},  # V-2.5Cr-3.0Ti-4.1W-6.1Zr
    ]
    
    for i, comp in enumerate(test_compositions):
        print(f"\nTest composition {i+1}: {comp}")
        
        name_prec0 = material_string(comp, 'V', precision=0)
        name_prec1 = material_string(comp, 'V', precision=1)
        name_prec2 = material_string(comp, 'V', precision=2)
        
        print(f"  Precision 0: {name_prec0}")
        print(f"  Precision 1: {name_prec1}")
        print(f"  Precision 2: {name_prec2}")


def main():
    """Main function to check for duplicates in existing results."""
    
    # Configuration
    ELEMENTS = ['V', 'Cr', 'Ti', 'W', 'Zr']
    
    # Test material naming
    test_material_naming_precision()
    
    # Check existing results
    results_file = "sequential_materials_pipeline_run_1/neutronics_optimization_results.json"
    
    if not os.path.exists(results_file):
        print(f"\nResults file not found: {results_file}")
        print("Run the pipeline first or update the path to check for duplicates.")
        return
    
    # Check for duplicates
    analysis = check_duplicate_material_names(results_file, ELEMENTS)
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"SUMMARY:")
    print(f"  Total materials: {analysis['total_materials']}")
    print(f"  Unique names: {analysis['unique_names']}")
    print(f"  Duplicate names: {analysis['duplicate_names']}")
    
    if analysis['duplicate_names'] > 0:
        print(f"  Duplicate rate: {100 * analysis['duplicate_names'] / analysis['unique_names']:.1f}%")
    
    # Suggest fixes
    suggestions = suggest_fixes(analysis)
    for suggestion in suggestions:
        print(suggestion)


if __name__ == "__main__":
    main() 