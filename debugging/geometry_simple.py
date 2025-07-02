#!/usr/bin/env python3
"""
Simplified cylindrical tokamak geometry that should work reliably.
"""

import openmc
import math
from pathlib import Path

# Use the same parameters as the complex geometry
ELEMENT_DENSITIES = {
    'V': 6.11, 'Cr': 7.19, 'Ti': 4.54, 'W': 19.35, 'Zr': 6.51,
}

def create_simple_model(element_symbol: str):
    """Create a simplified cylindrical tokamak model."""
    
    # Simplified parameters (cylindrical approximation)
    major_radius = 330
    minor_radius = 113
    fw_thickness = 0.2
    vv_thickness = 2
    
    # Create simple cylindrical surfaces
    plasma_inner = openmc.ZCylinder(r=major_radius - minor_radius)
    plasma_outer = openmc.ZCylinder(r=major_radius + minor_radius)
    fw_outer = openmc.ZCylinder(r=major_radius + minor_radius + fw_thickness)
    vv_outer = openmc.ZCylinder(r=major_radius + minor_radius + fw_thickness + vv_thickness)
    
    # Vertical boundaries
    z_top = openmc.ZPlane(z0=150)
    z_bottom = openmc.ZPlane(z0=-150)
    
    # Outer boundary
    outer_boundary = openmc.ZCylinder(r=600, boundary_type='vacuum')
    z_top_boundary = openmc.ZPlane(z0=300, boundary_type='vacuum') 
    z_bottom_boundary = openmc.ZPlane(z0=-300, boundary_type='vacuum')
    
    # Define regions
    plasma_region = +plasma_inner & -plasma_outer & -z_top & +z_bottom
    fw_region = +plasma_outer & -fw_outer & -z_top & +z_bottom
    vv_region = +fw_outer & -vv_outer & -z_top & +z_bottom
    outer_region = (+vv_outer | +z_top | -z_bottom) & -outer_boundary & -z_top_boundary & +z_bottom_boundary
    
    # Materials
    vv_material = openmc.Material(name=f'vv_{element_symbol}')
    vv_material.add_element(element_symbol, 1.0, 'ao')
    vv_material.set_density('g/cm3', ELEMENT_DENSITIES[element_symbol])
    vv_material.depletable = True
    
    w_material = openmc.Material(name='tungsten')
    w_material.add_element('W', 1.0, 'ao')
    w_material.set_density('g/cm3', 19.3)
    w_material.depletable = True
    
    void_material = openmc.Material(name='void')
    void_material.add_element('H', 1.0, 'ao')
    void_material.set_density('g/cm3', 1e-10)  # Very low density
    
    # Calculate volumes
    plasma_volume = math.pi * (plasma_outer.r**2 - plasma_inner.r**2) * (z_top.z0 - z_bottom.z0)
    fw_volume = math.pi * (fw_outer.r**2 - plasma_outer.r**2) * (z_top.z0 - z_bottom.z0)
    vv_volume = math.pi * (vv_outer.r**2 - fw_outer.r**2) * (z_top.z0 - z_bottom.z0)
    
    w_material.volume = fw_volume
    vv_material.volume = vv_volume
    
    # Cells
    plasma_cell = openmc.Cell(region=plasma_region, name='plasma')  # Void
    fw_cell = openmc.Cell(region=fw_region, fill=w_material, name='first_wall')
    vv_cell = openmc.Cell(region=vv_region, fill=vv_material, name='vacuum_vessel')
    outer_cell = openmc.Cell(region=outer_region, fill=void_material, name='outer_void')
    
    # Geometry
    universe = openmc.Universe(cells=[plasma_cell, fw_cell, vv_cell, outer_cell])
    geometry = openmc.Geometry(universe)
    
    # Simple point source in plasma center
    source = openmc.IndependentSource()
    source.space = openmc.stats.Point((major_radius, 0, 0))
    source.angle = openmc.stats.Isotropic()
    source.energy = openmc.stats.Discrete([14.1e6], [1.0])  # 14.1 MeV D-T
    source.particle = 'neutron'
    
    # Settings
    settings = openmc.Settings()
    settings.run_mode = "fixed source"
    settings.source = [source]
    settings.batches = 5
    settings.particles = 5000
    
    # Materials
    materials = openmc.Materials([vv_material, w_material, void_material])
    
    model = openmc.model.Model(materials=materials, geometry=geometry, settings=settings)
    model.vv_cell = vv_cell
    
    return model

def test_simple_model():
    """Test the simplified model."""
    
    print("🧪 Testing Simplified Tokamak Model")
    print("="*40)
    
    # Set cross sections
    cross_sections = Path.home() / 'nuclear_data' / 'cross_sections.xml'
    openmc.config['cross_sections'] = str(cross_sections)
    
    # Create model
    model = create_simple_model('V')
    
    # Add flux tally
    vv_cell = model.vv_cell
    tally = openmc.Tally(name='vv_flux')
    cell_filter = openmc.CellFilter(vv_cell)
    energy_filter = openmc.EnergyFilter([0.0, 1e6, 20e6])
    tally.filters = [cell_filter, energy_filter]
    tally.scores = ['flux']
    
    model.tallies = openmc.Tallies([tally])
    
    print("Running simplified model test...")
    
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            statepoint_file = model.run(cwd=tmpdir)
            
            with openmc.StatePoint(statepoint_file) as sp:
                flux_tally = sp.get_tally(name='vv_flux')
                flux_values = flux_tally.mean
                
                print(f"VV Flux Results:")
                print(f"  Thermal flux:  {flux_values[0]:.2e}")
                print(f"  Fast flux:     {flux_values[1]:.2e}")
                print(f"  Fusion flux:   {flux_values[2]:.2e}")
                print(f"  Total flux:    {flux_values.sum():.2e}")
                
                if flux_values.sum() > 1e-5:
                    print(f"✅ SUCCESS: Simplified model works!")
                    improvement = flux_values.sum() / 3.76e-05
                    print(f"   Improvement over complex geometry: {improvement:.0f}x")
                    return True
                else:
                    print(f"❌ Still low flux")
                    return False
                    
        except Exception as e:
            print(f"Error: {e}")
            return False

if __name__ == "__main__":
    test_simple_model() 