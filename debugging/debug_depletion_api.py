#!/usr/bin/env python3
"""
Debug script to test the OpenMC depletion results API.
"""

import openmc.deplete
import os

def debug_depletion_api():
    """Debug the depletion results API to see available methods."""
    
    # Check if any depletion results exist
    results_dir = "elem_lib/V/depletion"
    results_file = os.path.join(results_dir, "depletion_results.h5")
    
    if not os.path.exists(results_file):
        print(f"No depletion results found at {results_file}")
        print("Run 'neutronics-calphad build-library' to generate results first.")
        return
    
    print(f"Loading depletion results from {results_file}")
    results = openmc.deplete.Results(results_file)
    
    print(f"Number of time steps: {len(results)}")
    
    # Get the first result (after irradiation)
    if len(results) > 0:
        result = results[0]
        print(f"First result time: {result.time}")
        
        # Try to get a material (assuming material ID 1 is the vacuum vessel)
        material_ids = result.get_material_ids()
        print(f"Available material IDs: {material_ids}")
        
        if material_ids:
            mat_id = material_ids[0]
            mat = result.get_material(str(mat_id))
            
            print(f"\nMaterial {mat_id} methods:")
            methods = [method for method in dir(mat) if not method.startswith('_')]
            for method in sorted(methods):
                print(f"  {method}")
            
            print(f"\nMaterial {mat_id} attributes:")
            if hasattr(mat, 'nuclides'):
                print(f"  Available nuclides: {len(mat.nuclides)} total")
                # Show first few nuclides
                nuclides = list(mat.nuclides)[:10]
                print(f"  First 10 nuclides: {nuclides}")
            
            # Try different methods to get nuclide information
            print(f"\nTesting different API methods:")
            
            test_nuclides = ['He4', 'He3', 'H1', 'H2', 'H3']
            for nuclide in test_nuclides:
                print(f"\n  Testing {nuclide}:")
                
                # Test get_atom_density
                if hasattr(mat, 'get_atom_density'):
                    try:
                        density = mat.get_atom_density(nuclide)
                        print(f"    get_atom_density({nuclide}): {density}")
                    except Exception as e:
                        print(f"    get_atom_density({nuclide}): Error - {e}")
                
                # Test accessing densities directly
                if hasattr(mat, 'get_nuclide_densities'):
                    try:
                        densities = mat.get_nuclide_densities()
                        if nuclide in densities:
                            print(f"    get_nuclide_densities()[{nuclide}]: {densities[nuclide]}")
                        else:
                            print(f"    get_nuclide_densities()[{nuclide}]: Not found")
                    except Exception as e:
                        print(f"    get_nuclide_densities(): Error - {e}")

if __name__ == "__main__":
    debug_depletion_api() 