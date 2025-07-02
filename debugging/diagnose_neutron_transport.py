#!/usr/bin/env python3
"""
Diagnostic script to check neutron transport effectiveness.
"""

import openmc
import numpy as np
from pathlib import Path
from neutronics_calphad.geometry_maker import create_model

def diagnose_neutron_transport(element='V', cross_sections=Path.home() / 'nuclear_data' / 'cross_sections.xml'):
    """Check if neutrons are actually reaching the vacuum vessel."""
    
    print(f"🔬 Neutron Transport Diagnostics for {element}")
    print("="*60)
    
    # Set cross sections
    print(f"Setting cross sections: {cross_sections}")
    openmc.config['cross_sections'] = str(cross_sections)
    
    # Create model
    model = create_model(element)
    
    # Add neutron flux tallies
    print("Setting up flux tallies...")
    
    # Get the vacuum vessel cell
    vv_cell = model.vv_cell
    
    # Cell tally for vacuum vessel
    vv_tally = openmc.Tally(name='vv_flux')
    vv_cell_filter = openmc.CellFilter(vv_cell)
    energy_filter = openmc.EnergyFilter([0.0, 1e-6, 1.0, 1e6, 20e6])  # thermal, epithermal, fast, fusion
    vv_tally.filters = [vv_cell_filter, energy_filter]
    vv_tally.scores = ['flux', 'fission']
    
    # Overall flux tally
    total_tally = openmc.Tally(name='total_flux')
    total_tally.filters = [energy_filter]
    total_tally.scores = ['flux']
    
    # Source leakage tally
    surface_filter = openmc.SurfaceFilter([])  # Will tally on all surfaces
    leakage_tally = openmc.Tally(name='leakage')
    leakage_tally.filters = [surface_filter]
    leakage_tally.scores = ['current']
    
    model.tallies = openmc.Tallies([vv_tally, total_tally, leakage_tally])
    
    # Run short simulation to check flux
    model.settings.batches = 10
    model.settings.particles = 10000
    model.settings.photon_transport = False
    
    print("Running neutron transport diagnostic...")
    
    # Run in temporary directory
    import tempfile, os
    with tempfile.TemporaryDirectory() as tmpdir:
        statepoint_file = model.run(cwd=tmpdir)
        
        # Read results
        with openmc.StatePoint(statepoint_file) as sp:
            print("\n📊 Neutron Transport Results:")
            
            # Total flux
            total_tally_result = sp.get_tally(name='total_flux')
            total_flux = total_tally_result.mean.flatten()
            total_flux_std = total_tally_result.std_dev.flatten()
            
            print(f"\n🌊 Total Neutron Flux by Energy Group:")
            energy_bins = ['Thermal (0-1 µeV)', 'Epithermal (1 µeV-1 eV)', 'Fast (1 eV-1 MeV)', 'Fusion (1-20 MeV)']
            for i, (flux, std, label) in enumerate(zip(total_flux, total_flux_std, energy_bins)):
                rel_err = std/flux*100 if flux > 0 else 0
                print(f"  {label}: {flux:.2e} ± {rel_err:.1f}%")
            
            # Vacuum vessel flux
            try:
                vv_tally_result = sp.get_tally(name='vv_flux')
                vv_flux = vv_tally_result.mean.flatten()
                vv_flux_std = vv_tally_result.std_dev.flatten()
                
                print(f"\n🎯 Vacuum Vessel Flux by Energy Group:")
                for i, (flux, std, label) in enumerate(zip(vv_flux, vv_flux_std, energy_bins)):
                    rel_err = std/flux*100 if flux > 0 else 0
                    print(f"  {label}: {flux:.2e} ± {rel_err:.1f}%")
                    
                # Calculate flux ratios
                print(f"\n📈 Flux Utilization:")
                for i, label in enumerate(energy_bins):
                    if total_flux[i] > 0:
                        ratio = vv_flux[i] / total_flux[i]
                        print(f"  {label}: {ratio:.1%} of total flux reaches VV")
                    else:
                        print(f"  {label}: No flux detected")
                        
            except Exception as e:
                print(f"❌ Error reading VV flux tally: {e}")
            
            # Check source information
            print(f"\n🚀 Source Analysis:")
            source = model.settings.source[0] if model.settings.source else None
            if source:
                print(f"  Source type: {type(source.space).__name__}")
                print(f"  Source energy: {type(source.energy).__name__}")
                if hasattr(source.space, 'r'):
                    print(f"  Source radius range: {source.space.r}")
                if hasattr(source.space, 'z'):
                    print(f"  Source z range: {source.space.z}")
                    
            # Check geometry
            print(f"\n🏗️ Geometry Analysis:")
            print(f"  VV cell ID: {vv_cell.id}")
            print(f"  VV cell name: {vv_cell.name}")
            print(f"  VV bounding box: {vv_cell.bounding_box}")
            
            # Material analysis
            vv_material = vv_cell.fill
            if vv_material:
                print(f"  VV material: {vv_material.name}")
                print(f"  VV density: {vv_material.density} g/cm³")
                print(f"  VV nuclides: {[nuc.name for nuc in vv_material.nuclides]}")
            
    print(f"\n💡 Diagnostic Summary:")
    
    # Check if flux is reaching VV
    if 'vv_flux' in locals():
        total_vv_flux = np.sum(vv_flux)
        if total_vv_flux < 1e-10:
            print(f"  ❌ Very low neutron flux in vacuum vessel")
            print(f"     → Check source position vs. geometry")
            print(f"     → Check for neutron absorption in surrounding materials")
        elif total_vv_flux < 1e-5:
            print(f"  ⚠️  Low neutron flux in vacuum vessel")
            print(f"     → May need higher source rate or longer irradiation")
        else:
            print(f"  ✅ Reasonable neutron flux detected")
    
    # Check total flux
    total_system_flux = np.sum(total_flux)
    if total_system_flux < 1e-10:
        print(f"  ❌ Very low total neutron flux in system")
        print(f"     → Check source definition")
        print(f"     → Check particle count and batch settings")
    
    print(f"\n🔧 Recommendations:")
    print(f"  1. If VV flux is low: Check source-geometry overlap")
    print(f"  2. If total flux is low: Increase particles/batches")
    print(f"  3. If fast flux is zero: Check source energy spectrum")
    print(f"  4. For activation: Need fast neutrons (>1 MeV) for gas production")

def check_source_energy():
    """Check the neutron source energy spectrum."""
    print(f"\n🔋 Source Energy Analysis:")
    
    from openmc_plasma_source import tokamak_source
    import math
    
    # Create the same source as in geometry_maker
    # Revert source to a 90-degree wedge for the quarter torus
    source = tokamak_source(
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
    #print(f"  Energy distribution: {type(source.energy).__name__}")
    
    # Sample some energies to check
    print(f"  Sampling source energies...")
    try:
        # This is a bit hacky, but let's see what energies we get
        energies = []
        for i in range(1000):
            try:
                energy = source[i].energy.sample()
                energies.append(energy)
            except:
                break
        
        if energies:
            energies = np.array(energies)
            print(f"    Mean energy: {np.mean(energies):.2e} eV")
            print(f"    Energy range: {np.min(energies):.2e} - {np.max(energies):.2e} eV")
            print(f"    14 MeV equivalent: {14.1e6:.2e} eV")
            
            # Check if we have fast neutrons
            fast_fraction = np.sum(energies > 1e6) / len(energies)
            print(f"    Fast neutrons (>1 MeV): {fast_fraction:.1%}")
        else:
            print(f"    Could not sample source energies")
            
    except Exception as e:
        print(f"    Error sampling energies: {e}")

if __name__ == "__main__":
    diagnose_neutron_transport('V')
    check_source_energy() 