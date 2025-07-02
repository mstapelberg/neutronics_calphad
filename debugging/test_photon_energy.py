#!/usr/bin/env python3
"""
Direct test of photon energy calculation at 10 years.
"""

import numpy as np
import openmc
import openmc.deplete
import os
from pathlib import Path

def test_photon_energy_direct():
    """Directly test the photon energy calculation at different time steps."""
    
    element = 'V'
    outdir = 'test_single_element'
    
    # Set up paths
    depletion_file = os.path.join(outdir, element, 'depletion', 'depletion_results.h5')
    reduced_chain_path = os.path.join(outdir, element, 'reduced_chain.xml')
    
    if not os.path.exists(depletion_file):
        print(f"❌ Depletion results not found")
        return
    
    # Set the chain file
    if os.path.exists(reduced_chain_path):
        openmc.config['chain_file'] = os.path.abspath(reduced_chain_path)
    
    # Load results
    results = openmc.deplete.Results(depletion_file)
    
    # Find the VV material ID (should be 2)
    material_id = '2'
    
    # Check specific time steps
    # Index 9: 5 years, Index 10: 10 years, Index 11: 25 years
    for idx in [9, 10, 11]:
        result = results[idx]
        time_seconds = result.time[0]
        time_years = (time_seconds - 365*24*3600) / (365.25*24*3600)  # Subtract irradiation time
        
        print(f"\n{'='*60}")
        print(f"Time step {idx}: {time_years:.1f} years cooling")
        print(f"{'='*60}")
        
        try:
            mat = result.get_material(material_id)
            
            # Get decay photon energy
            photon_energy = mat.get_decay_photon_energy()
            
            if photon_energy is None:
                print("❌ photon_energy is None")
                continue
            
            # Calculate integral
            integral = photon_energy.integral()
            print(f"Photon integral: {integral:.6e} photons/s")
            
            # Check the distribution details
            if hasattr(photon_energy, 'x') and hasattr(photon_energy, 'p'):
                energies = photon_energy.x
                probs = photon_energy.p
                
                print(f"Distribution info:")
                print(f"  - Number of energy bins: {len(energies)}")
                print(f"  - Energy range: {np.min(energies):.2e} - {np.max(energies):.2e} eV")
                print(f"  - Probability sum: {np.sum(probs):.6e}")
                print(f"  - Non-zero bins: {np.sum(probs > 0)}")
                
                # Check for very small values
                if integral == 0 and np.sum(probs) > 0:
                    print(f"  ⚠️  WARNING: Integral is 0 but probabilities sum to {np.sum(probs):.6e}")
                    print(f"  - Min non-zero probability: {np.min(probs[probs > 0]):.6e}")
                    print(f"  - Max probability: {np.max(probs):.6e}")
                    
                    # Check if it's a normalization issue
                    if hasattr(photon_energy, '_intensity'):
                        print(f"  - Internal intensity: {photon_energy._intensity:.6e}")
            
            # Also check the material's total activity
            activity = mat.get_activity()
            print(f"Total activity: {activity:.6e} Bq")
            
            # Get top nuclides
            nuclides = list(mat.get_nuclides())
            print(f"Number of nuclides: {len(nuclides)}")
            
            # Check specific isotopes known to be important
            important_isotopes = ['V50', 'V51', 'Cr51', 'Ti51', 'Sc46', 'Sc47', 'Sc48']
            print("\nChecking important isotopes:")
            for iso in important_isotopes:
                if iso in nuclides:
                    act = mat.get_activity(nuclides=[iso])
                    print(f"  {iso}: {act:.6e} Bq")
            
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()

    # Additional test: Check if it's a time-stepping issue
    print(f"\n{'='*60}")
    print("Checking exact time values:")
    print(f"{'='*60}")
    
    TIMES = [
        1, 3600, 10*3600, 24*3600, 7*24*3600, 14*24*3600,
        30*24*3600, 60*24*3600, 365*24*3600, 5*365*24*3600,
        10*365*24*3600, 25*365*24*3600, 100*365*24*3600
    ]
    
    for i in range(len(results)):
        time = results[i].time[0]
        if i == 0:
            print(f"Step {i}: {time:.0f} s (initial)")
        elif i == 1:
            print(f"Step {i}: {time:.0f} s (after irradiation)")
        else:
            cooling_time = time - 365*24*3600
            expected_cooling = TIMES[i-2] if i-2 < len(TIMES) else None
            match = "✅" if expected_cooling and abs(cooling_time - expected_cooling) < 1 else "❌"
            print(f"Step {i}: {time:.0f} s (cooling: {cooling_time:.0f} s, expected: {expected_cooling}, {match})")

if __name__ == "__main__":
    test_photon_energy_direct() 