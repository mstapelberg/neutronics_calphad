#!/usr/bin/env python3
"""
Diagnostic script to investigate the 10-year dose rate issue.
"""

import h5py
import numpy as np
import os
import openmc
import openmc.deplete
import sys
from pathlib import Path
sys.path.append('.')

from neutronics_calphad.geometry_maker import ELEMENT_DENSITIES

def check_depletion_results(element='V', outdir='test_single_element'):
    """Check the depletion results at each time step."""
    
    depletion_file = os.path.join(outdir, element, 'depletion', 'depletion_results.h5')
    reduced_chain_path = os.path.join(outdir, element, 'reduced_chain.xml')
    
    if not os.path.exists(depletion_file):
        print(f"❌ Depletion results not found at: {depletion_file}")
        return
    
    # Set the chain file for decay data
    if os.path.exists(reduced_chain_path):
        openmc.config['chain_file'] = os.path.abspath(reduced_chain_path)
        print(f"✅ Using chain file: {reduced_chain_path}")
    else:
        print(f"⚠️  Warning: Chain file not found at {reduced_chain_path}")
        print("   Decay heat and photon energy calculations may fail")
    
    print(f"\n📊 Analyzing depletion results: {depletion_file}")
    print("="*70)
    
    # Load results
    results = openmc.deplete.Results(depletion_file)
    
    # We need to know the material ID - let's check the first result
    # In the simulation, the VV material should have ID 2
    # Let's try different IDs to find the right one
    material_id = None
    material_name = f"vv_{element}"
    
    # Try to find the material ID by checking the first result
    print(f"Looking for material: {material_name}")
    
    # Focus on the time steps around 10 years
    # From the data: index 10 is 10 years (315360000 s)
    target_indices = [9, 10, 11, 12]  # 5 years, 10 years, 25 years, 100 years
    
    # Check each time step
    for i in range(len(results)):
        result = results[i]
        time = result.time[0]  # Time in seconds
        
        # Skip time steps we're not interested in for now
        if i not in target_indices and i > 1:
            continue
        
        # Convert to readable format
        time_label = f"{time:.0f} s"
        if time == 0:
            time_label = "Initial"
        elif time == 365*24*3600:
            time_label = "After irradiation (1 year)"
        elif abs(time - (365*24*3600 + 5*365*24*3600)) < 1:
            time_label = "5 years cooling"
        elif abs(time - (365*24*3600 + 10*365*24*3600)) < 1:
            time_label = "🔴 10 years cooling (PROBLEM TIME)"
        elif abs(time - (365*24*3600 + 25*365*24*3600)) < 1:
            time_label = "25 years cooling"
        elif abs(time - (365*24*3600 + 100*365*24*3600)) < 1:
            time_label = "100 years cooling"
            
        print(f"\nTime step {i}: {time_label}")
        print("-"*50)
        
        # Try to find the VV material
        mat = None
        if material_id is None:
            # Try common material IDs
            for test_id in ['1', '2', '3', '4', '5']:
                try:
                    test_mat = result.get_material(test_id)
                    # Check if this might be our VV material
                    nuclides = list(test_mat.get_nuclides())
                    # VV material should have the element we're looking for
                    if element in nuclides or f"{element}51" in nuclides or f"{element}50" in nuclides:
                        material_id = test_id
                        mat = test_mat
                        print(f"  Found VV material with ID: {material_id}")
                        break
                except:
                    continue
            
            if material_id is None:
                print(f"  ❌ Could not find VV material")
                continue
        else:
            try:
                mat = result.get_material(material_id)
            except Exception as e:
                print(f"  Error getting material {material_id}: {e}")
                continue
        
        # Check basic properties
        try:
            activity = mat.get_activity()
            decay_heat = mat.get_decay_heat()
            
            print(f"  Total activity: {activity:.2e} Bq")
            print(f"  Decay heat: {decay_heat:.2e} W")
            
            # Try to get decay photon energy
            try:
                photon_energy = mat.get_decay_photon_energy()
                if photon_energy is None:
                    print(f"  Photon energy: None ❌")
                else:
                    photon_rate = photon_energy.integral()
                    print(f"  Photon rate: {photon_rate:.2e} photons/s")
                    
                    # Check if it's exactly zero
                    if photon_rate == 0:
                        print(f"    ⚠️  WARNING: Photon rate is exactly zero!")
                        # Check the distribution
                        if hasattr(photon_energy, 'x') and hasattr(photon_energy, 'p'):
                            print(f"    Energy bins: {len(photon_energy.x)}")
                            print(f"    Probability array shape: {photon_energy.p.shape}")
                            print(f"    Sum of probabilities: {np.sum(photon_energy.p):.2e}")
                            # Check for any non-zero probabilities
                            non_zero = np.sum(photon_energy.p > 0)
                            print(f"    Non-zero probability bins: {non_zero}")
                            if non_zero > 0:
                                print(f"    Max probability: {np.max(photon_energy.p):.2e}")
                            
                            # For debugging: check the actual values
                            if i == 10:  # 10-year time step
                                print(f"    🔍 Detailed analysis at 10 years:")
                                print(f"    First 10 energy bins (eV): {photon_energy.x[:10]}")
                                print(f"    First 10 probabilities: {photon_energy.p[:10]}")
                                # Check if it's a numerical precision issue
                                very_small = np.sum((photon_energy.p > 0) & (photon_energy.p < 1e-300))
                                print(f"    Extremely small (>0 but <1e-300) probability bins: {very_small}")
                            
            except Exception as e:
                print(f"  Error getting photon energy: {e}")
                import traceback
                traceback.print_exc()
            
            # Check nuclide inventory
            nuclides = list(mat.get_nuclides())
            print(f"  Number of nuclides: {len(nuclides)}")
            
            # Get top 5 by activity
            activities = []
            for nuc in nuclides:
                try:
                    act = mat.get_activity(nuclides=[nuc])
                    activities.append((nuc, act))
                except:
                    pass
            
            activities.sort(key=lambda x: x[1], reverse=True)
            print(f"  Top nuclides by activity:")
            for nuc, act in activities[:5]:
                fraction = act / activity * 100 if activity > 0 else 0
                print(f"    {nuc}: {act:.2e} Bq ({fraction:.1f}%)")
                
                # Special check for nuclides at 10 years
                if i == 10:
                    # Check if this nuclide has photon data
                    try:
                        print(f"      Checking {nuc} for photon emission...")
                        # This is a bit of a hack to check individual nuclide photon emission
                        # We'd need access to the decay data directly
                    except:
                        pass
                
        except Exception as e:
            print(f"  Error analyzing material: {e}")
            import traceback
            traceback.print_exc()
    
    # Also check the saved dose rates
    h5_file = os.path.join(outdir, element, f"{element}.h5")
    if os.path.exists(h5_file):
        print(f"\n📈 Saved dose rates from: {h5_file}")
        print("="*70)
        with h5py.File(h5_file, 'r') as f:
            times = f['dose_times'][:]
            doses = f['dose'][:]
            
            print("\nFocusing on the problem area (5-25 years):")
            print("-"*50)
            for i, (t, d) in enumerate(zip(times, doses)):
                time_years = t / (365.25 * 24 * 3600)
                if 4 < time_years < 26:  # Focus on 5-25 year range
                    status = "❌ ZERO!" if d == 0 else "✅"
                    print(f"  Index {i}: Time: {t:.0f} s ({time_years:.2f} years) -> Dose: {d:.2e} µSv/h {status}")

if __name__ == "__main__":
    check_depletion_results() 