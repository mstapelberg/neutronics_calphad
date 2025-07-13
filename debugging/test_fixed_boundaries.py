#!/usr/bin/env python3
"""
Test the fixed boundary conditions with the full tokamak source.
"""

import openmc
import numpy as np
from pathlib import Path
from neutronics_calphad.geometry_maker import create_model
from neutronics_calphad.config import ARC_D_SHAPE

def test_fixed_boundaries():
    """Test if the fixed boundaries allow proper neutron transport."""
    
    print("🔧 Testing Fixed Boundary Conditions")
    print("="*40)
    
    # Set cross sections
    cross_sections = Path.home() / 'nuclear_data' / 'cross_sections.xml'
    openmc.config['cross_sections'] = str(cross_sections)
    
    # Create model with fixed boundaries
    model = create_model(ARC_D_SHAPE)
    
    # Use the full tokamak source (not simplified)
    print(f"Number of sources: {len(model.settings.source) if isinstance(model.settings.source, list) else 1}")
    print(f"Particles per batch: {model.settings.particles}")
    
    # Keep original settings but add a simple flux tally
    vv_cell = model.vv_cell
    vv_tally = openmc.Tally(name='vv_flux')
    vv_cell_filter = openmc.CellFilter(vv_cell)
    energy_filter = openmc.EnergyFilter([0.0, 1e-6, 1.0, 1e6, 20e6])
    vv_tally.filters = [vv_cell_filter, energy_filter]
    vv_tally.scores = ['flux']
    
    # Also add a total flux tally
    total_tally = openmc.Tally(name='total_flux')
    total_tally.filters = [energy_filter]
    total_tally.scores = ['flux']
    
    model.tallies = openmc.Tallies([vv_tally, total_tally])
    
    # Run with slightly more particles to get better statistics
    model.settings.batches = 10
    model.settings.particles = 10000
    model.settings.photon_transport = False
    
    print(f"\nRunning simulation with:")
    print(f"  - Fixed vacuum boundaries (outer surfaces)")
    print(f"  - Reflective boundaries (x=0, y=0 planes only)")
    print(f"  - {len(model.settings.source) if isinstance(model.settings.source, list) else 1} tokamak sources")
    print(f"  - {model.settings.particles} particles per batch")
    
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            statepoint_file = model.run(cwd=tmpdir)
            
            with openmc.StatePoint(statepoint_file) as sp:
                # Get VV flux
                vv_tally_result = sp.get_tally(name='vv_flux')
                vv_flux = vv_tally_result.mean.flatten()
                vv_flux_std = vv_tally_result.std_dev.flatten()
                
                # Get total flux  
                total_tally_result = sp.get_tally(name='total_flux')
                total_flux = total_tally_result.mean.flatten()
                
                print(f"\n📊 Results with Fixed Boundaries:")
                
                energy_labels = ['Thermal', 'Epithermal', 'Fast', 'Fusion']
                print(f"\nVacuum Vessel Flux:")
                for i, label in enumerate(energy_labels):
                    if vv_flux[i] > 0:
                        rel_err = vv_flux_std[i]/vv_flux[i]*100
                        print(f"  {label}: {vv_flux[i]:.2e} ± {rel_err:.1f}%")
                    else:
                        print(f"  {label}: {vv_flux[i]:.2e}")
                
                print(f"\nTotal System Flux:")
                for i, label in enumerate(energy_labels):
                    print(f"  {label}: {total_flux[i]:.2e}")
                
                # Check flux utilization
                print(f"\nFlux Utilization (VV/Total):")
                for i, label in enumerate(energy_labels):
                    if total_flux[i] > 0:
                        utilization = vv_flux[i] / total_flux[i] * 100
                        print(f"  {label}: {utilization:.1f}%")
                
                # Overall assessment
                total_vv = np.sum(vv_flux)
                total_system = np.sum(total_flux)
                
                print(f"\n🎯 Summary:")
                print(f"  Total VV flux: {total_vv:.2e}")
                print(f"  Total system flux: {total_system:.2e}")
                
                if total_vv > 0:
                    print(f"\n✅ SUCCESS! Neutrons are now reaching the VV!")
                    print(f"   The boundary fix worked!")
                else:
                    print(f"\n❌ Still no VV flux - additional debugging needed")
                    
        except Exception as e:
            print(f"\n❌ Error running simulation: {e}")
            if "maximum number of events" in str(e).lower():
                print("   → Particles still hitting event limit")
            elif "lost particles" in str(e).lower():
                print("   → Particles being lost from geometry")

if __name__ == "__main__":
    test_fixed_boundaries() 