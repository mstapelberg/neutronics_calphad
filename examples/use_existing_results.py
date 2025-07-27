"""Example script showing how to use existing neutronics results for surrogate modeling.

This script demonstrates how to:
1. Load existing neutronics results
2. Fit surrogate models to the existing data
3. Suggest new compositions based on the fitted models
4. Save the suggested compositions for further analysis
"""

import os
import json
import numpy as np
from typing import Dict, List, Any

from neutronics_calphad.optimizer.composition_sampler import CompositionSampler


def load_and_fit_surrogate(
    results_file: str,
    elements: List[str],
    min_compositions: Dict[str, float],
    max_compositions: Dict[str, float]
) -> CompositionSampler:
    """Load existing results and fit surrogate models.
    
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
        
    Returns
    -------
    CompositionSampler
        Fitted sampler with surrogate models
    """
    
    print(f"Loading results from {results_file}")
    
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
    
    print("Extracting training data...")
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
    print("Fitting surrogate models...")
    sampler.fit_surrogates()
    print("Surrogate models fitted successfully!")
    
    return sampler


def suggest_new_compositions(
    sampler: CompositionSampler,
    num_suggestions: int = 1000,
    save_to_file: str = None
) -> np.ndarray:
    """Suggest new compositions using fitted surrogate models.
    
    Parameters
    ----------
    sampler : CompositionSampler
        Fitted sampler with surrogate models
    num_suggestions : int
        Number of compositions to suggest
    save_to_file : str, optional
        File path to save suggested compositions
        
    Returns
    -------
    np.ndarray
        Array of suggested compositions
    """
    
    print(f"Suggesting {num_suggestions} new compositions...")
    
    # Generate suggestions
    suggested_comps = sampler.suggest(num_suggestions)
    
    # Predict feasibility
    feasible = sampler.predict_feasibility(suggested_comps)
    good_comps = suggested_comps[feasible]
    
    print(f"Generated {num_suggestions} compositions")
    print(f"Found {len(good_comps)} feasible compositions ({100*len(good_comps)/num_suggestions:.1f}%)")
    
    # Save if requested
    if save_to_file:
        results = {
            'metadata': {
                'num_suggested': num_suggestions,
                'num_feasible': len(good_comps),
                'feasibility_rate': len(good_comps) / num_suggestions,
                'elements': sampler.elements
            },
            'all_suggested_compositions': suggested_comps.tolist(),
            'feasible_compositions': good_comps.tolist(),
            'feasibility_mask': feasible.tolist()
        }
        
        with open(save_to_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Saved results to {save_to_file}")
    
    return good_comps


def analyze_suggested_compositions(
    compositions: np.ndarray,
    elements: List[str]
) -> Dict[str, Any]:
    """Analyze the suggested compositions.
    
    Parameters
    ----------
    compositions : np.ndarray
        Array of suggested compositions
    elements : List[str]
        List of element symbols
        
    Returns
    -------
    Dict[str, Any]
        Analysis results
    """
    
    if len(compositions) == 0:
        return {'error': 'No compositions to analyze'}
    
    analysis = {
        'total_compositions': len(compositions),
        'element_statistics': {},
        'composition_ranges': {}
    }
    
    # Analyze each element
    for i, element in enumerate(elements):
        element_comps = compositions[:, i]
        
        analysis['element_statistics'][element] = {
            'mean': float(np.mean(element_comps)),
            'std': float(np.std(element_comps)),
            'min': float(np.min(element_comps)),
            'max': float(np.max(element_comps)),
            'median': float(np.median(element_comps))
        }
        
        analysis['composition_ranges'][element] = {
            'min': float(np.min(element_comps)),
            'max': float(np.max(element_comps))
        }
    
    return analysis


def main():
    """Main function demonstrating the workflow."""
    
    # Configuration
    ELEMENTS = ['V', 'Cr', 'Ti', 'W', 'Zr']
    MIN_COMPOSITIONS = {'V': 0.70}
    MAX_COMPOSITIONS = {'V': 0.95, 'Cr': 0.20, 'Ti': 0.15, 'W': 0.20, 'Zr': 0.1}
    
    # Path to existing results
    results_file = "sequential_materials_pipeline_run_1/neutronics_optimization_results.json"
    
    if not os.path.exists(results_file):
        print(f"Results file not found: {results_file}")
        print("Please run the pipeline first or update the path to your results file.")
        return
    
    try:
        # Step 1: Load and fit surrogate models
        print("=== STEP 1: Loading and Fitting Surrogate Models ===")
        sampler = load_and_fit_surrogate(
            results_file, ELEMENTS, MIN_COMPOSITIONS, MAX_COMPOSITIONS
        )
        
        # Step 2: Suggest new compositions
        print("\n=== STEP 2: Suggesting New Compositions ===")
        good_comps = suggest_new_compositions(
            sampler,
            num_suggestions=1000,
            save_to_file="suggested_compositions.json"
        )
        
        # Step 3: Analyze suggested compositions
        print("\n=== STEP 3: Analyzing Suggested Compositions ===")
        analysis = analyze_suggested_compositions(good_comps, ELEMENTS)
        
        print(f"Analysis Results:")
        print(f"  Total feasible compositions: {analysis['total_compositions']}")
        print(f"  Element ranges:")
        for element, ranges in analysis['composition_ranges'].items():
            print(f"    {element}: {ranges['min']:.3f} - {ranges['max']:.3f}")
        
        # Step 4: Show some example compositions
        print(f"\n=== STEP 4: Example Compositions ===")
        if len(good_comps) > 0:
            print("First 5 suggested compositions:")
            for i, comp in enumerate(good_comps[:5]):
                comp_dict = dict(zip(ELEMENTS, comp))
                print(f"  {i+1}. {comp_dict}")
        
        print(f"\n=== SUMMARY ===")
        print(f"Successfully loaded existing results and fitted surrogate models")
        print(f"Generated {len(good_comps)} feasible compositions")
        print(f"Results saved to 'suggested_compositions.json'")
        print(f"You can now use these compositions for CALPHAD analysis or further evaluation")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 