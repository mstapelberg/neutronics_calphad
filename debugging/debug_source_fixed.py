#!/usr/bin/env python3
"""
Fixed debug script for tokamak source and geometry issues.
"""

import openmc
import numpy as np
from pathlib import Path
from neutronics_calphad.geometry_maker import create_model

def debug_source_positioning():
    """Check where the tokamak source is actually generating neutrons."""
    
    print(f"🎯 Tokamak Source Position Debug")
    print("="*50)
    
    # Set cross sections
    cross_sections = Path.home() / 'nuclear_data' / 'cross_sections.xml'
    openmc.config['cross_sections'] = str(cross_sections)
    
    # Create model
    model = create_model('V')
    
    # Get the source list - tokamak_source returns a list of IndependentSource objects
    sources = model.settings.source
    print(f"Number of source objects: {len(sources)}")
    
    # Instead of sampling, let's inspect the source definitions directly
    print(f"\n📍 Source Definition Analysis...")
    
    try:
        # Analyze the first few sources to understand the pattern
        sources_to_check = min(5, len(sources))
        
        r_ranges = []
        z_ranges = []
        phi_ranges = []
        energy_ranges = []
        
        for i in range(sources_to_check):
            source = sources[i]
            print(f"\n  Source {i+1}/{sources_to_check}:")
            print(f"    Type: {type(source)}")
            print(f"    Particle: {source.particle}")
            print(f"    Strength: {getattr(source, 'strength', 'default')}")
            
            # Inspect space distribution
            if hasattr(source, 'space'):
                space = source.space
                print(f"    Space distribution: {type(space)}")
                
                # For CylindricalIndependent, check individual components
                if hasattr(space, 'r') and hasattr(space, 'phi') and hasattr(space, 'z'):
                    r_dist = space.r
                    phi_dist = space.phi
                    z_dist = space.z
                    
                    print(f"      R distribution: {type(r_dist)}")
                    if hasattr(r_dist, 'a') and hasattr(r_dist, 'b'):
                        r_min, r_max = r_dist.a, r_dist.b
                        print(f"        R range: {r_min:.1f} - {r_max:.1f} cm")
                        r_ranges.extend([r_min, r_max])
                    elif hasattr(r_dist, 'x') and hasattr(r_dist, 'p'):
                        # Discrete distribution
                        r_values = r_dist.x
                        r_probs = r_dist.p
                        print(f"        R discrete values: {r_values[:5]}..." if len(r_values) > 5 else f"        R discrete values: {r_values}")
                        print(f"        R probabilities: {r_probs[:5]}..." if len(r_probs) > 5 else f"        R probabilities: {r_probs}")
                        r_ranges.extend([min(r_values), max(r_values)])
                    
                    print(f"      Phi distribution: {type(phi_dist)}")
                    if hasattr(phi_dist, 'a') and hasattr(phi_dist, 'b'):
                        phi_min, phi_max = phi_dist.a, phi_dist.b
                        print(f"        Phi range: {phi_min:.3f} - {phi_max:.3f} rad")
                        phi_ranges.extend([phi_min, phi_max])
                    
                    print(f"      Z distribution: {type(z_dist)}")
                    if hasattr(z_dist, 'a') and hasattr(z_dist, 'b'):
                        z_min, z_max = z_dist.a, z_dist.b
                        print(f"        Z range: {z_min:.1f} - {z_max:.1f} cm")
                        z_ranges.extend([z_min, z_max])
                    elif hasattr(z_dist, 'x') and hasattr(z_dist, 'p'):
                        # Discrete distribution
                        z_values = z_dist.x
                        z_probs = z_dist.p
                        print(f"        Z discrete values: {z_values[:5]}..." if len(z_values) > 5 else f"        Z discrete values: {z_values}")
                        print(f"        Z probabilities: {z_probs[:5]}..." if len(z_probs) > 5 else f"        Z probabilities: {z_probs}")
                        z_ranges.extend([min(z_values), max(z_values)])
            
            # Inspect energy distribution
            if hasattr(source, 'energy'):
                energy = source.energy
                print(f"    Energy distribution: {type(energy)}")
                
                if hasattr(energy, 'distribution') and hasattr(energy, 'probabilities'):
                    # Mixture distribution
                    distributions = energy.distribution
                    probabilities = energy.probabilities
                    print(f"      Mixture with {len(distributions)} components")
                    print(f"      Probabilities: {probabilities[:3]}..." if len(probabilities) > 3 else f"      Probabilities: {probabilities}")
                    
                    # Check first distribution component
                    if len(distributions) > 0:
                        first_dist = distributions[0]
                        print(f"      First component: {type(first_dist)}")
                        if hasattr(first_dist, 'x') and hasattr(first_dist, 'p'):
                            energies = first_dist.x
                            energy_ranges.extend([min(energies), max(energies)])
                            print(f"        Energy range: {min(energies):.2e} - {max(energies):.2e} eV")
        
        # Summarize overall source characteristics
        print(f"\n🏗️ Overall Source Characteristics:")
        if r_ranges:
            print(f"  Overall R range: {min(r_ranges):.1f} - {max(r_ranges):.1f} cm")
        if z_ranges:
            print(f"  Overall Z range: {min(z_ranges):.1f} - {max(z_ranges):.1f} cm")
        if phi_ranges:
            print(f"  Overall Phi range: {min(phi_ranges):.3f} - {max(phi_ranges):.3f} rad")
        if energy_ranges:
            print(f"  Overall Energy range: {min(energy_ranges):.2e} - {max(energy_ranges):.2e} eV")
        
        # Compare with expected plasma boundaries
        plasma_r_min = 330 - 113 * 0.9  # Expected source boundaries
        plasma_r_max = 330 + 113 * 0.9
        plasma_z_max = 1.8 * 113 * 0.9
        
        print(f"\n✅ Source Boundary Check:")
        print(f"  Expected plasma R: {plasma_r_min:.1f} - {plasma_r_max:.1f} cm")
        print(f"  Expected plasma Z: {-plasma_z_max:.1f} - {+plasma_z_max:.1f} cm")
        
        if r_ranges:
            r_min_actual, r_max_actual = min(r_ranges), max(r_ranges)
            if plasma_r_min <= r_min_actual and r_max_actual <= plasma_r_max:
                print(f"  ✅ R range looks good!")
            else:
                print(f"  ⚠️  R range may be outside plasma boundaries")
        
        if z_ranges:
            z_min_actual, z_max_actual = min(z_ranges), max(z_ranges)
            if -plasma_z_max <= z_min_actual and z_max_actual <= plasma_z_max:
                print(f"  ✅ Z range looks good!")
            else:
                print(f"  ⚠️  Z range may be outside plasma boundaries")
        
        # Check if energies are fusion-like
        if energy_ranges:
            fusion_energy = 14.1e6  # eV
            if min(energy_ranges) <= fusion_energy <= max(energy_ranges):
                print(f"  ✅ Energy range includes fusion energies (~14 MeV)")
            else:
                print(f"  ⚠️  Energy range doesn't include typical fusion energies")
        
        print(f"\n📊 Source Distribution Summary:")
        print(f"  Total sources: {len(sources)}")
        print(f"  All sources are IndependentSource objects: {all(isinstance(s, openmc.IndependentSource) for s in sources)}")
        print(f"  All sources have CylindricalIndependent space: {all(isinstance(s.space, openmc.stats.CylindricalIndependent) for s in sources if hasattr(s, 'space'))}")
        print(f"  All sources have Mixture energy: {all(isinstance(s.energy, openmc.stats.Mixture) for s in sources if hasattr(s, 'energy'))}")
        
    except Exception as e:
        print(f"❌ Error analyzing sources: {e}")
        import traceback
        traceback.print_exc()
        
        # Fallback: Just show basic source info
        print(f"\nFallback source inspection:")
        for i, source in enumerate(sources[:3]):
            print(f"  Source {i}: {type(source)}")
            if hasattr(source, 'space'):
                print(f"    Space: {type(source.space)}")
            if hasattr(source, 'energy'):
                print(f"    Energy: {type(source.energy)}")
            if hasattr(source, 'particle'):
                print(f"    Particle: {source.particle}")

def check_geometry_problems():
    """Check for geometry issues that cause infinite loops."""
    
    print(f"\n🔧 Geometry Problem Diagnosis")
    print("="*40)
    
    # Set cross sections
    cross_sections = Path.home() / 'nuclear_data' / 'cross_sections.xml'
    openmc.config['cross_sections'] = str(cross_sections)
    
    # Create model
    model = create_model('V')
    
    # Check for common geometry issues
    print("Checking geometry for common issues...")
    
    # 1. Check for overlapping cells
    print("\n1. Cell Overlap Check:")
    cells = model.geometry.get_all_cells()
    print(f"   Total cells: {len(cells)}")
    
    cell_names = []
    for cell in cells.values():
        if hasattr(cell, 'name') and cell.name:
            cell_names.append(cell.name)
        if cell.fill is None and cell.name != 'world_void':
            print(f"   ⚠️  Cell '{cell.name}' has no material (void)")
    
    print(f"   Cell names: {cell_names}")
    
    # 2. Check for undefined regions
    print(f"\n2. Boundary Conditions:")
    all_surfaces = model.geometry.get_all_surfaces()
    boundary_surfaces = [s for s in all_surfaces.values() if hasattr(s, 'boundary_type') and s.boundary_type]
    print(f"   Boundary surfaces: {len(boundary_surfaces)}")
    for surf in boundary_surfaces:
        print(f"     {type(surf).__name__}: {surf.boundary_type}")
        
    # 3. Test with very conservative settings
    print(f"\n3. Conservative Transport Test:")
    
    # Create very simple point source in plasma center
    simple_source = openmc.IndependentSource()
    simple_source.space = openmc.stats.Point((330, 0, 0))
    simple_source.angle = openmc.stats.Isotropic()
    simple_source.energy = openmc.stats.Discrete([14.1e6], [1.0])
    simple_source.particle = 'neutron'
    
    model.settings.source = [simple_source]
    model.settings.batches = 10
    model.settings.particles = 5_000  # Very few particles
    model.settings.max_events = 10_000  # Limit events per particle
    model.settings.max_lost_particles = 10_000  # Limit lost particles

    print(f"   Testing with {model.settings.particles} particles...")
    print(f"   Using simple point source at plasma center")
    
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            # Just run without tallies to test basic transport
            model.tallies = openmc.Tallies([])
            statepoint_file = model.run(geometry_debug=True,cwd=tmpdir)
            print(f"   ✅ Basic transport successful")
            
            # Check if any particles hit max events
            with openmc.StatePoint(statepoint_file) as sp:
                # This is basic - just seeing if we can read results
                print(f"   Simulation completed normally")
                
        except Exception as e:
            print(f"   ❌ Basic transport failed: {e}")
            if "maximum number of events" in str(e).lower():
                print(f"      → Particles hitting event limit = geometry problems")
            if "overlap" in str(e).lower():
                print(f"      → Geometry overlaps detected")
            if "undefined" in str(e).lower():
                print(f"      → Undefined regions detected")

def test_simplified_geometry():
    """Test with a very simple geometry to isolate issues."""
    
    print(f"\n🧪 Simplified Geometry Test")
    print("="*35)
    
    # Create ultra-simple model: just source in vacuum vessel
    print("Creating simplified model...")
    
    # Set cross sections
    cross_sections = Path.home() / 'nuclear_data' / 'cross_sections.xml'
    openmc.config['cross_sections'] = str(cross_sections)
    
    # Simple materials
    vv_material = openmc.Material(name='vv_V_simple')
    vv_material.add_element('V', 1.0, 'ao')
    vv_material.set_density('g/cm3', 6.11)
    
    # Simple geometry - just a sphere
    sphere_surf = openmc.Sphere(r=500, boundary_type='vacuum')
    sphere_region = -sphere_surf
    sphere_cell = openmc.Cell(region=sphere_region, fill=vv_material, name='simple_vv')
    
    universe = openmc.Universe(cells=[sphere_cell])
    geometry = openmc.Geometry(universe)
    
    # Simple source
    source = openmc.IndependentSource()
    source.space = openmc.stats.Point((0, 0, 0))
    source.angle = openmc.stats.Isotropic()
    source.energy = openmc.stats.Discrete([14.1e6], [1.0])
    source.particle = 'neutron'
    
    # Settings
    settings = openmc.Settings()
    settings.run_mode = "fixed source"
    settings.source = [source]
    settings.batches = 2
    settings.particles = 1000
    
    # Simple flux tally
    tally = openmc.Tally(name='simple_flux')
    cell_filter = openmc.CellFilter(sphere_cell)
    tally.filters = [cell_filter]
    tally.scores = ['flux']
    
    materials = openmc.Materials([vv_material])
    tallies = openmc.Tallies([tally])
    
    simple_model = openmc.model.Model(
        materials=materials,
        geometry=geometry,
        settings=settings,
        tallies=tallies
    )
    
    print("Running simplified test...")
    
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            statepoint_file = simple_model.run(cwd=tmpdir)
            
            with openmc.StatePoint(statepoint_file) as sp:
                flux_tally = sp.get_tally(name='simple_flux')
                total_flux = flux_tally.mean.sum()
                
                print(f"  Simple model flux: {total_flux:.2e}")
                
                if total_flux > 1e-10:
                    print(f"  ✅ Simple geometry WORKS")
                    print(f"     → Problem is with complex tokamak geometry")
                else:
                    print(f"  ❌ Even simple geometry fails")
                    print(f"     → Fundamental setup issue")
                    
        except Exception as e:
            print(f"  Error in simplified test: {e}")

def check_source_energy():
    """Check the neutron source energy spectrum."""
    print(f"\n🔋 Source Energy Analysis:")
    
    from openmc_plasma_source import tokamak_source
    import math
    
    # Create the same source as in geometry_maker
    # Revert source to a 90-degree wedge for the quarter torus
    sources = tokamak_source(
        elongation=1.8,
        ion_density_centre=1.09e20,
        ion_density_pedestal=1.09e20,
        ion_density_peaking_factor=1,
        ion_density_separatrix=3e19,
        ion_temperature_centre=45.9e3,
        ion_temperature_pedestal=6.09e3,
        ion_temperature_separatrix=0.1e3,
        ion_temperature_peaking_factor=8.06,
        ion_temperature_beta=6,
        major_radius=330,
        minor_radius=113,
        pedestal_radius=0.8 * 113,
        mode="H",
        shafranov_factor=0.0,
        angles=(0, math.pi/2),
        sample_seed=42,
        triangularity=0.5,
        fuel={"D": 0.5, "T": 0.5},
    )
    
    print(f"  Source type: Tokamak D-T fusion")
    print(f"  Expected energy: ~14.1 MeV (D-T fusion)")
    print(f"  Number of source objects: {len(sources)}")
    
    # Sample some energies to check
    print(f"  Sampling source energies...")
    try:
        energies = []
        positions = []
        
        # Sample energies from all sources
        total_samples = 1000
        samples_per_source = max(1, total_samples // len(sources))
        
        for i, source in enumerate(sources):
            #print(f"    Sampling from source {i+1}/{len(sources)}...")
            #print(f"      Energy distribution type: {type(source.energy)}")
            
            for j in range(samples_per_source):
                try:
                    energy = source.energy.sample()
                    energies.append(energy)
                    
                    # Also sample position to understand source spatial distribution
                    if hasattr(source.space, 'sample'):
                        pos = source.space.sample()
                        r = np.sqrt(pos[0]**2 + pos[1]**2)
                        positions.append(r)
                except Exception as e:
                    if j == 0:  # Only print error once per source
                        print(f"      Error sampling from source {i}: {e}")
                    continue
        
        if energies:
            energies = np.array(energies)
            positions = np.array(positions) if positions else np.array([])
            
            print(f"    ✅ Successfully sampled {len(energies)} energies")
            print(f"    Mean energy: {np.mean(energies):.2e} eV")
            print(f"    Energy range: {np.min(energies):.2e} - {np.max(energies):.2e} eV")
            print(f"    14 MeV equivalent: {14.1e6:.2e} eV")
            print(f"    Standard deviation: {np.std(energies):.2e} eV")
            
            # Check if we have fast neutrons
            fast_fraction = np.sum(energies > 1e6) / len(energies)
            fusion_fraction = np.sum(np.abs(energies - 14.1e6) < 1e6) / len(energies)
            
            print(f"    Fast neutrons (>1 MeV): {fast_fraction:.1%}")
            print(f"    Near fusion energy (13-15 MeV): {fusion_fraction:.1%}")
            
            # Energy distribution analysis
            if np.max(energies) > np.min(energies):
                print(f"    Energy spread: {(np.max(energies) - np.min(energies))/np.mean(energies)*100:.1f}%")
            
            # Spatial distribution if available
            if len(positions) > 0:
                print(f"    Radial position range: {np.min(positions):.1f} - {np.max(positions):.1f} cm")
                print(f"    Mean radial position: {np.mean(positions):.1f} cm")
                
        else:
            print(f"    ❌ Could not sample any source energies")
            
        # Try to inspect energy distributions directly
        print(f"\n  📊 Direct energy distribution inspection:")
        for i, source in enumerate(sources[:3]):  # Check first 3 sources
            try:
                print(f"    Source {i}:")
                print(f"      Energy type: {type(source.energy)}")
                
                if hasattr(source.energy, 'x') and hasattr(source.energy, 'p'):
                    energies_direct = source.energy.x
                    probabilities = source.energy.p
                    print(f"      Energy bins: {len(energies_direct)}")
                    print(f"      Energy range: {min(energies_direct):.2e} - {max(energies_direct):.2e} eV")
                    
                    # Calculate mean energy
                    if len(probabilities) == len(energies_direct):
                        mean_energy = np.sum(energies_direct * probabilities) / np.sum(probabilities)
                        print(f"      Weighted mean energy: {mean_energy:.2e} eV")
                elif hasattr(source.energy, 'a'):
                    # Watt spectrum or similar
                    print(f"      Distribution parameter 'a': {source.energy.a}")
                    if hasattr(source.energy, 'b'):
                        print(f"      Distribution parameter 'b': {source.energy.b}")
                else:
                    print(f"      Available attributes: {[attr for attr in dir(source.energy) if not attr.startswith('_')]}")
                    
            except Exception as e:
                print(f"      Error inspecting source {i}: {e}")
            
    except Exception as e:
        print(f"    ❌ Error sampling energies: {e}")
        
        # Fallback: Try to inspect the source structure
        try:
            print(f"    Fallback inspection of source structure:")
            for i, source in enumerate(sources[:2]):  # Just check first 2
                print(f"      Source {i}: {type(source)}")
                if hasattr(source, 'energy'):
                    print(f"        Energy: {type(source.energy)}")
                if hasattr(source, 'space'):
                    print(f"        Space: {type(source.space)}")
                if hasattr(source, 'particle'):
                    print(f"        Particle: {source.particle}")
        except Exception as e2:
            print(f"    Could not inspect source structure: {e2}")

if __name__ == "__main__":
    debug_source_positioning()
    check_geometry_problems()
    test_simplified_geometry()
    check_source_energy() 