#!/usr/bin/env python3
"""
Test with a simple isotropic source to verify neutron transport works.
"""

import openmc
import numpy as np
from pathlib import Path
from neutronics_calphad.geometry_maker import create_model

def test_simple_source():
    """Test if a simple source in plasma center reaches the VV."""
    
    print("🧪 Testing Simple Isotropic Source")
    print("="*40)
    
    # Set cross sections
    cross_sections = Path.home() / 'nuclear_data' / 'cross_sections.xml'
    openmc.config['cross_sections'] = str(cross_sections)
    
    # Create model
    model = create_model('V')
    
    # Replace tokamak source with simple isotropic source
    simple_source = openmc.IndependentSource()
    simple_source.space = openmc.stats.Point((330, 0, 0))  # Plasma center
    simple_source.angle = openmc.stats.Isotropic()
    simple_source.energy = openmc.stats.Discrete([14.1e6], [1.0])  # 14.1 MeV
    simple_source.particle = 'neutron'
    
    model.settings.source = [simple_source]
    model.settings.batches = 10
    model.settings.particles = 10000
    model.settings.photon_transport = False
    
    # Add simple flux tally for VV
    vv_cell = model.vv_cell
    vv_tally = openmc.Tally(name='vv_flux')
    vv_cell_filter = openmc.CellFilter(vv_cell)
    energy_filter = openmc.EnergyFilter([0.0, 1e6, 20e6])  # thermal+epithermal, fast
    vv_tally.filters = [vv_cell_filter, energy_filter]
    vv_tally.scores = ['flux']
    
    model.tallies = openmc.Tallies([vv_tally])
    
    print(f"Source position: (330, 0, 0) cm")
    print(f"Source energy: 14.1 MeV")
    print(f"Running with {model.settings.particles} particles...")
    
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            statepoint_file = model.run(cwd=tmpdir)
            
            with openmc.StatePoint(statepoint_file) as sp:
                vv_tally_result = sp.get_tally(name='vv_flux')
                vv_flux = vv_tally_result.mean.flatten()
                vv_flux_std = vv_tally_result.std_dev.flatten()
                
                print(f"\n✅ VV Flux Results:")
                print(f"  Low energy (< 1 MeV): {vv_flux[0]:.2e} ± {vv_flux_std[0]:.2e}")
                print(f"  High energy (> 1 MeV): {vv_flux[1]:.2e} ± {vv_flux_std[1]:.2e}")
                print(f"  Total flux: {np.sum(vv_flux):.2e}")
                
                if np.sum(vv_flux) > 0:
                    print(f"\n✅ SUCCESS: Neutrons ARE reaching the VV!")
                    print(f"   → The geometry is OK")
                    print(f"   → Problem is with the tokamak source positioning")
                else:
                    print(f"\n❌ FAIL: Even simple source doesn't reach VV")
                    print(f"   → Fundamental geometry problem")
                    
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_simple_source() 