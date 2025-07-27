#!/usr/bin/env python3
"""
Simple test of the existing CALPHAD calculator with the problematic composition.
"""

import numpy as np
import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from neutronics_calphad.calphad.phase_calculator import CALPHADBatchCalculator

def main():
    """Test the existing CALPHAD calculator."""
    print("=== Testing Existing CALPHAD Calculator ===")
    
    # Test parameters
    elements = ['V', 'Cr', 'Ti', 'W', 'Zr']
    
    # The problematic composition
    problematic_comp = np.array([0.66920999, 0.12934373, 0.03863475, 0.11088052, 0.05193101])
    
    # Simple test composition
    simple_comp = np.array([0.8, 0.1, 0.05, 0.03, 0.02])
    
    test_compositions = [problematic_comp, simple_comp]
    
    print(f"Testing {len(test_compositions)} compositions")
    print(f"Elements: {elements}")
    print(f"Problematic composition: {problematic_comp}")
    print(f"Simple composition: {simple_comp}")
    
    # Initialize calculator
    print("\nInitializing CALPHAD calculator...")
    calc = CALPHADBatchCalculator(
        database="TCHEA7",
        temperature=823.5
    )
    
    # Convert to batch format
    comp_arrays = np.array(test_compositions)
    
    # Run calculation
    print("\nRunning CALPHAD calculations...")
    try:
        results = calc.calculate_batch(comp_arrays, elements)
        print("✓ Calculations completed successfully!")
        
        # Print results
        print("\nResults:")
        for i, (comp, row) in enumerate(zip(test_compositions, results.itertuples())):
            print(f"\nComposition {i+1}: {comp}")
            print(f"  Phase count: {row.phase_count}")
            print(f"  Dominant phase: {row.dominant_phase}")
            print(f"  Single phase: {row.single_phase}")
            print(f"  Phases: {row.phases}")
            
    except Exception as e:
        print(f"✗ Calculation failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 