#!/usr/bin/env python3
import openmc.deplete
import os
import numpy as np
from pathlib import Path

def quick_debug():
    element = 'V'
    outdir = 'test_fix_verification'
    
    # Check if depletion results exist
    depletion_file = os.path.join(outdir, element, 'depletion', 'depletion_results.h5')
    chain_file = os.path.join(outdir, element, 'reduced_chain.xml')
    
    print(f"🔍 Quick Debug:")
    print(f"  Depletion file exists: {os.path.exists(depletion_file)}")
    print(f"  Chain file exists: {os.path.exists(chain_file)}")
    
    if not os.path.exists(depletion_file):
        print("❌ No depletion results found")
        return
    
    # Set chain file
    openmc.config['chain_file'] = os.path.abspath(chain_file)
    
    # Load results
    results = openmc.deplete.Results(depletion_file)
    print(f"  Number of results: {len(results)}")
    
    # Check times and activities at each step
    irradiation_time = 365 * 24 * 3600  # 1 year
    
    for i in range(len(results)):
        try:
            result = results[i]
            time = result.time[0]
            cooling_time = time - irradiation_time if i > 0 else 0
            
            # Get material (try ID 1 first, then 2)
            mat = None
            for mat_id in ['1', '2']:
                try:
                    mat = result.get_material(mat_id)
                    break
                except:
                    continue
            
            if mat:
                activity = mat.get_activity()
                try:
                    photon_energy = mat.get_decay_photon_energy()
                    photon_rate = photon_energy.integral() if photon_energy else 0
                except:
                    photon_rate = 0
                
                print(f"  Step {i}: time={time:.0f}s, cooling={cooling_time:.0f}s, activity={activity:.2e}Bq, photons={photon_rate:.2e}/s")
            else:
                print(f"  Step {i}: time={time:.0f}s, cooling={cooling_time:.0f}s, NO MATERIAL FOUND")
                
        except Exception as e:
            print(f"  Step {i}: ERROR - {e}")

if __name__ == "__main__":
    quick_debug() 