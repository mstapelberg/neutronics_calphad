#!/usr/bin/env python3
"""
Check geometry and source positioning to diagnose why neutrons aren't reaching the VV.
"""

import openmc
import numpy as np
from pathlib import Path
from neutronics_calphad.geometry_maker import create_model
from neutronics_calphad.config import ARC_D_SHAPE

def check_source_geometry_overlap(config=ARC_D_SHAPE):
    """Check if the plasma source overlaps with the geometry correctly."""
    
    print(f"🔬 Source-Geometry Overlap Analysis")
    print("="*50)
    
    # Set cross sections
    cross_sections = Path.home() / 'nuclear_data' / 'cross_sections.xml'
    openmc.config['cross_sections'] = str(cross_sections)
    
    # Create model
    model = create_model(config)
    
    # Get source information
    source = model.settings.source
    sources = source if isinstance(source, list) else [source]

    if not sources:
        print("❌ No source found in model")
        return
    
    print(f"\n🚀 Source Configuration:")
    print(f"  Source type: {type(sources[0].space).__name__}")
    
    # Sample source positions
    print(f"\n📍 Sampling source positions...")
    positions = []
    for i in range(1000):
        try:
            # sample from a random source in the list
            s = np.random.choice(sources)
            if hasattr(s.space, 'sample'):
                pos = s.space.sample()
                positions.append(pos)
            else:
                # For tokamak source, need to sample differently
                pos = s.sample()  # This should give us a full particle
                positions.append([pos.r, pos.phi, pos.z])
        except Exception as e:
            if i == 0:
                print(f"  Error sampling positions: {e}")
            break
    
    if positions:
        positions = np.array(positions)
        print(f"  Sampled {len(positions)} source positions")
        
        if positions.shape[1] == 3:  # x, y, z or r, phi, z
            print(f"  X/R range: {np.min(positions[:,0]):.1f} - {np.max(positions[:,0]):.1f} cm")
            print(f"  Y/φ range: {np.min(positions[:,1]):.1f} - {np.max(positions[:,1]):.1f}")
            print(f"  Z range: {np.min(positions[:,2]):.1f} - {np.max(positions[:,2]):.1f} cm")
            
            # Convert to cartesian if needed
            if np.max(positions[:,1]) < 10:  # Probably phi in radians
                x_pos = positions[:,0] * np.cos(positions[:,1])
                y_pos = positions[:,0] * np.sin(positions[:,1])
                z_pos = positions[:,2]
                print(f"  Converted to Cartesian:")
                print(f"    X range: {np.min(x_pos):.1f} - {np.max(x_pos):.1f} cm")
                print(f"    Y range: {np.min(y_pos):.1f} - {np.max(y_pos):.1f} cm")
                print(f"    Z range: {np.min(z_pos):.1f} - {np.max(z_pos):.1f} cm")
    
    # Get geometry bounds
    print(f"\n🏗️ Geometry Analysis:")
    
    # Get all cells
    cells = model.geometry.get_all_cells()
    for cell in cells.values():
        if cell.name in ['plasma', 'first_wall', 'vacuum_vessel']:
            print(f"  {cell.name}: ID={cell.id}")
            try:
                bbox = cell.bounding_box
                print(f"    Bounding box: x=[{bbox[0][0]:.1f}, {bbox[1][0]:.1f}], "
                      f"y=[{bbox[0][1]:.1f}, {bbox[1][1]:.1f}], z=[{bbox[0][2]:.1f}, {bbox[1][2]:.1f}]")
            except Exception as e:
                print(f"    Could not get bounding box: {e}")
    
    # Check material path from plasma to VV
    print(f"\n🛤️ Material Path Analysis:")
    print(f"Expected neutron path: Plasma → First Wall → Vacuum Vessel")
    
    # Get materials
    materials = {mat.id: mat for mat in model.materials}
    
    for cell in cells.values():
        if cell.fill and hasattr(cell.fill, 'name'):
            print(f"  {cell.name}: {cell.fill.name} (ρ={cell.fill.density:.2f} g/cm³)")
            
            # Check neutron absorption
            nuclides = [nuc.name for nuc in cell.fill.nuclides]
            print(f"    Nuclides: {nuclides}")
            
            # Identify high absorbers
            if any(nuc in ['B10', 'B11', 'Cd113', 'Gd155', 'Gd157'] for nuc in nuclides):
                print(f"    ⚠️  Contains strong neutron absorbers!")

def check_neutron_leakage():
    """Check neutron leakage using a simpler test case."""
    
    print(f"\n🔧 Neutron Leakage Test")
    print("="*30)
    
    # Create simple test model
    cross_sections = Path.home() / 'nuclear_data' / 'cross_sections.xml'
    openmc.config['cross_sections'] = str(cross_sections)
    
    model = create_model(ARC_D_SHAPE)
    
    # Simple leakage tally
    leakage_tally = openmc.Tally(name='leakage')
    surface_filter = openmc.SurfaceFilter([])  # All surfaces
    leakage_tally.filters = [surface_filter]
    leakage_tally.scores = ['current']
    
    # Add to model
    model.tallies = openmc.Tallies([leakage_tally])
    
    # Short run
    model.settings.batches = 5
    model.settings.particles = 5000
    model.settings.photon_transport = False
    
    print("Running leakage test...")
    
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            statepoint_file = model.run(cwd=tmpdir)
            
            with openmc.StatePoint(statepoint_file) as sp:
                leakage = sp.get_tally(name='leakage')
                total_leakage = np.sum(np.abs(leakage.mean))
                
                print(f"  Total neutron leakage: {total_leakage:.2e}")
                
                if total_leakage > 0.5:  # More than 50% leakage
                    print(f"  ⚠️  High neutron leakage detected!")
                    print(f"     → Neutrons escaping geometry without reaching VV")
                else:
                    print(f"  ✅ Leakage within reasonable bounds")
                    
        except Exception as e:
            print(f"  Error in leakage test: {e}")

def recommendations():
    """Provide specific recommendations based on findings."""
    
    print(f"\n💡 Specific Recommendations:")
    print(f"="*30)
    
    print(f"Based on 0.0% neutron utilization in VV:")
    print(f"")
    print(f"1. **Check source positioning**:")
    print(f"   - Ensure plasma source is inside the plasma cell")
    print(f"   - Verify quarter-torus geometry doesn't have gaps")
    print(f"")
    print(f"2. **Check first wall thickness**:")
    print(f"   - FW thickness = 0.2 cm (very thin)")
    print(f"   - Tungsten density = 19.3 g/cm³ (high)")
    print(f"   - May be absorbing all neutrons despite thinness")
    print(f"")
    print(f"3. **Test without first wall**:")
    print(f"   - Temporarily remove FW to test direct plasma→VV transport")
    print(f"")
    print(f"4. **Check for geometry overlaps**:")
    print(f"   - Use OpenMC geometry debugging")
    print(f"   - Look for undefined regions")

if __name__ == "__main__":
    check_source_geometry_overlap()
    check_neutron_leakage()
    recommendations() 