#!/usr/bin/env python3
"""
Debug the tokamak source positioning to see where neutrons are actually being generated.
"""

import openmc
import numpy as np
from pathlib import Path
from neutronics_calphad.geometry_maker import create_model
import matplotlib.pyplot as plt

def debug_source_positioning():
    """Check where the tokamak source is actually generating neutrons."""
    
    print(f"🎯 Tokamak Source Position Debug")
    print("="*50)
    
    # Set cross sections
    cross_sections = Path.home() / 'nuclear_data' / 'cross_sections.xml'
    openmc.config['cross_sections'] = str(cross_sections)
    
    # Create model
    model = create_model('V')
    
    # Get the source
    source = model.settings.source[0]
    print(f"Source type: {type(source)}")
    print(f"Source space type: {type(source.space)}")
    
    # Try to sample from the source directly
    print(f"\n📍 Sampling source particles...")
    
    # Method 1: Try to sample particles directly
    try:
        particles = []
        for i in range(100):
            particle = source[i].sample()
            particles.append({
                'x': particle.r[0],
                'y': particle.r[1], 
                'z': particle.r[2],
                'energy': particle.E
            })
        
        if particles:
            x_coords = [p['x'] for p in particles]
            y_coords = [p['y'] for p in particles]
            z_coords = [p['z'] for p in particles]
            energies = [p['energy'] for p in particles]
            
            print(f"  ✅ Sampled {len(particles)} particles successfully")
            print(f"  X range: {min(x_coords):.1f} - {max(x_coords):.1f} cm")
            print(f"  Y range: {min(y_coords):.1f} - {max(y_coords):.1f} cm") 
            print(f"  Z range: {min(z_coords):.1f} - {max(z_coords):.1f} cm")
            print(f"  Energy range: {min(energies):.2e} - {max(energies):.2e} eV")
            
            # Convert to cylindrical coordinates
            r_coords = np.sqrt(np.array(x_coords)**2 + np.array(y_coords)**2)
            print(f"  R range: {min(r_coords):.1f} - {max(r_coords):.1f} cm")
            
            # Check if source is within expected plasma boundaries
            plasma_r_min = 330 - 113  # major_radius - minor_radius
            plasma_r_max = 330 + 113  # major_radius + minor_radius
            plasma_z_max = 1.8 * 113  # elongation * minor_radius
            
            print(f"\n🏗️ Expected plasma boundaries:")
            print(f"  R: {plasma_r_min:.1f} - {plasma_r_max:.1f} cm")
            print(f"  Z: {-plasma_z_max:.1f} - {+plasma_z_max:.1f} cm")
            
            # Check overlap
            r_in_range = np.logical_and(r_coords >= plasma_r_min, r_coords <= plasma_r_max)
            z_in_range = np.logical_and(np.array(z_coords) >= -plasma_z_max, np.array(z_coords) <= plasma_z_max)
            in_plasma = np.logical_and(r_in_range, z_in_range)
            
            print(f"\n✅ Source-plasma overlap check:")
            print(f"  Particles in plasma R range: {np.sum(r_in_range)}/{len(particles)} ({np.sum(r_in_range)/len(particles)*100:.1f}%)")
            print(f"  Particles in plasma Z range: {np.sum(z_in_range)}/{len(particles)} ({np.sum(z_in_range)/len(particles)*100:.1f}%)")
            print(f"  Particles in plasma region: {np.sum(in_plasma)}/{len(particles)} ({np.sum(in_plasma)/len(particles)*100:.1f}%)")
            
            if np.sum(in_plasma) < len(particles) * 0.8:
                print(f"  ⚠️  <80% of source particles are in plasma region!")
                print(f"     This could explain why neutrons aren't reaching VV")
            
    except Exception as e:
        print(f"  ❌ Error sampling particles: {e}")
        import traceback
        traceback.print_exc()

def test_simple_point_source():
    """Test with a simple point source to isolate geometry issues."""
    
    print(f"\n🔧 Simple Point Source Test")
    print("="*40)
    
    # Set cross sections
    cross_sections = Path.home() / 'nuclear_data' / 'cross_sections.xml'
    openmc.config['cross_sections'] = str(cross_sections)
    
    # Create model but replace source
    model = create_model('V')
    
    # Create simple point source at plasma center
    point_source = openmc.IndependentSource()
    point_source.space = openmc.stats.Point((330, 0, 0))  # Major radius center
    point_source.angle = openmc.stats.Isotropic()
    point_source.energy = openmc.stats.Discrete([14.1e6], [1.0])  # 14.1 MeV
    point_source.particle = 'neutron'
    
    model.settings.source = [point_source]
    model.settings.batches = 5
    model.settings.particles = 5000
    
    # Add simple flux tally
    vv_cell = model.vv_cell
    tally = openmc.Tally(name='vv_flux')
    cell_filter = openmc.CellFilter(vv_cell)
    tally.filters = [cell_filter]
    tally.scores = ['flux']
    
    model.tallies = openmc.Tallies([tally])
    
    print(f"  Point source at: (330, 0, 0) cm")
    print(f"  Energy: 14.1 MeV")
    print(f"  Running test...")
    
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            statepoint_file = model.run(cwd=tmpdir)
            
            with openmc.StatePoint(statepoint_file) as sp:
                flux_tally = sp.get_tally(name='vv_flux')
                total_flux = flux_tally.mean.sum()
                rel_error = flux_tally.std_dev.sum() / flux_tally.mean.sum() * 100 if flux_tally.mean.sum() > 0 else 0
                
                print(f"  VV neutron flux: {total_flux:.2e} ± {rel_error:.1f}%")
                
                if total_flux > 1e-10:
                    print(f"  ✅ Point source WORKS - neutrons reach VV!")
                    print(f"     → Issue is with tokamak source positioning")
                else:
                    print(f"  ❌ Point source FAILS - geometry problem")
                    print(f"     → Issue is with geometry, not source")
                    
        except Exception as e:
            print(f"  Error in point source test: {e}")

def recommendations():
    """Provide specific recommendations."""
    print(f"\n💡 Next Steps:")
    print(f"="*20)
    print(f"1. If point source works but tokamak source fails:")
    print(f"   → Fix tokamak source positioning")
    print(f"   → Check quarter-torus geometry setup")
    print(f"")
    print(f"2. If both sources fail:")
    print(f"   → Geometry has gaps/overlaps")
    print(f"   → Check cell definitions and regions")
    print(f"")
    print(f"3. Check for common issues:")
    print(f"   → Vacuum cells with no material")
    print(f"   → Undefined regions")
    print(f"   → Reflective boundary problems")

if __name__ == "__main__":
    debug_source_positioning()
    test_simple_point_source()
    recommendations() 