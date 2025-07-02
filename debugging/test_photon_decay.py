#!/usr/bin/env python3
"""
Test script to verify photon decay energy calculation.
"""

import openmc
import openmc.deplete
import os
from pathlib import Path

def test_photon_decay():
    """Test that photon decay energy calculation works."""
    
    # Check if depletion results exist
    results_dir = "elem_lib/V/depletion"
    results_file = os.path.join(results_dir, "depletion_results.h5")
    chain_file = os.path.join("elem_lib/V", "reduced_chain.xml")
    
    if not os.path.exists(results_file):
        print(f"No depletion results found at {results_file}")
        return False
        
    if not os.path.exists(chain_file):
        print(f"No chain file found at {chain_file}")
        return False
    
    print(f"Testing photon decay energy calculation...")
    
    # Set the chain file
    openmc.config['chain_file'] = chain_file
    print(f"Set chain file to: {openmc.config['chain_file']}")
    
    # Load results
    results = openmc.deplete.Results(results_file)
    print(f"Loaded results with {len(results)} time steps")
    
    # Test decay photon energy for first cooling step
    if len(results) > 1:
        try:
            # Get material from first cooling step (index 1)
            activated_mat = results[1].get_material('1')  # Assuming material ID 1
            
            # Try to get decay photon energy
            photon_energy = activated_mat.get_decay_photon_energy()
            print(f"Successfully got decay photon energy: {type(photon_energy)}")
            
            if hasattr(photon_energy, 'x'):
                print(f"  Energy grid points: {len(photon_energy.x)}")
            
            return True
            
        except Exception as e:
            print(f"Error getting decay photon energy: {e}")
            return False
    else:
        print("Not enough time steps in results")
        return False

if __name__ == "__main__":
    success = test_photon_decay()
    if success:
        print("✅ Photon decay energy test passed!")
    else:
        print("❌ Photon decay energy test failed!") 