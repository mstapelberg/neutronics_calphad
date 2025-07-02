#!/usr/bin/env python3
"""
Quick script to check if neutron flux and source rates are reasonable for fusion applications.
"""

import numpy as np

def check_fusion_neutron_parameters():
    """Check if the neutron parameters are realistic for fusion reactors."""
    
    print("🔬 Fusion Reactor Neutron Flux Analysis")
    print("="*50)
    
    # Current simulation parameters
    source_rate_sim = 1e20  # neutrons/s from the simulation
    power_days = 365        # days of irradiation
    
    print(f"Current simulation parameters:")
    print(f"  Source rate: {source_rate_sim:.1e} neutrons/s")
    print(f"  Irradiation time: {power_days} days")
    
    # ITER reference values
    print(f"\n📚 ITER Reference Values:")
    iter_fusion_power = 500e6  # 500 MW fusion power
    energy_per_fusion = 17.6e6 * 1.602e-19  # 17.6 MeV in Joules
    iter_fusion_rate = iter_fusion_power / energy_per_fusion  # fusions/s
    iter_neutron_rate = iter_fusion_rate  # 1 neutron per D-T fusion
    
    print(f"  ITER fusion power: {iter_fusion_power/1e6:.0f} MW")
    print(f"  ITER neutron rate: {iter_neutron_rate:.2e} neutrons/s")
    print(f"  Simulation vs ITER: {source_rate_sim/iter_neutron_rate:.1f}x")
    
    # Neutron wall loading
    print(f"\n🏗️ Neutron Wall Loading:")
    # ITER first wall area (rough estimate)
    iter_first_wall_area = 700  # m^2 (approximate)
    iter_neutron_wall_loading = iter_neutron_rate / iter_first_wall_area  # neutrons/m^2/s
    
    print(f"  ITER neutron wall loading: {iter_neutron_wall_loading:.2e} neutrons/m²/s")
    print(f"  ITER neutron wall loading: {iter_neutron_wall_loading * 1e-4:.2e} neutrons/cm²/s")
    
    # Typical neutron flux levels
    print(f"\n⚡ Typical Neutron Flux Levels:")
    print(f"  Fast flux in fusion first wall: ~1e15 neutrons/cm²/s")
    print(f"  Thermal flux in research reactors: ~1e14 neutrons/cm²/s")
    print(f"  Fast flux in fission reactors: ~1e14 neutrons/cm²/s")
    
    # Fluence calculation
    irradiation_seconds = power_days * 24 * 3600
    typical_flux = 1e15  # neutrons/cm²/s for fusion
    total_fluence = typical_flux * irradiation_seconds
    
    print(f"\n🎯 Expected Fluence (1 year at 1e15 n/cm²/s):")
    print(f"  Total fluence: {total_fluence:.2e} neutrons/cm²")
    print(f"  Displacement damage: ~{total_fluence * 1e-24:.1f} dpa (for steel)")
    
    # Gas production estimates
    print(f"\n💨 Expected Gas Production (per atom of material):")
    print(f"  He production in V: ~10-100 appm per 1e22 n/cm² (14 MeV)")
    print(f"  H production in V: ~50-200 appm per 1e22 n/cm²")
    print(f"  ")
    print(f"  For {total_fluence:.1e} n/cm² fluence:")
    fluence_factor = total_fluence / 1e22
    print(f"  Expected He in V: ~{10*fluence_factor:.1f}-{100*fluence_factor:.1f} appm")
    print(f"  Expected H in V: ~{50*fluence_factor:.1f}-{200*fluence_factor:.1f} appm")
    
    # Volume and atom calculations
    print(f"\n🧮 Material Volume Analysis:")
    # From geometry_maker.py - vacuum vessel volume
    major_radius = 330  # cm
    minor_radius = 113  # cm
    elongation = 1.8
    fw_thickness = 0.2  # cm
    vv_thickness = 2    # cm
    
    # Volume calculation (quarter torus)
    import math
    a_outer = minor_radius + fw_thickness + vv_thickness
    a_inner = minor_radius + fw_thickness
    vv_volume = math.pi * major_radius * (a_outer**2 - a_inner**2) * elongation / 4.0
    
    print(f"  VV volume (quarter torus): {vv_volume:.2e} cm³")
    
    # Atom density calculation
    element = 'V'
    density = 6.11  # g/cm³ for vanadium
    atomic_mass = 50.9415  # g/mol for V
    avogadro = 6.022e23  # atoms/mol
    
    total_mass = density * vv_volume
    total_moles = total_mass / atomic_mass
    total_atoms = total_moles * avogadro
    
    print(f"  VV mass ({element}): {total_mass:.2e} g")
    print(f"  VV atoms ({element}): {total_atoms:.2e} atoms")
    
    # Expected gas atoms
    he_appm = 50 * fluence_factor  # middle estimate
    h_appm = 125 * fluence_factor   # middle estimate
    
    he_atoms = total_atoms * he_appm * 1e-6
    h_atoms = total_atoms * h_appm * 1e-6
    
    print(f"\n✨ Expected Gas Atoms After 1 Year:")
    print(f"  He atoms: {he_atoms:.2e}")
    print(f"  H atoms: {h_atoms:.2e}")
    
    # Recommendations
    print(f"\n💡 Recommendations:")
    if source_rate_sim < iter_neutron_rate * 0.1:
        print(f"  ⚠️  Source rate may be too low - consider increasing to ~{iter_neutron_rate:.1e}")
    else:
        print(f"  ✅ Source rate seems reasonable")
    
    if power_days < 30:
        print(f"  ⚠️  Irradiation time may be too short for significant gas production")
    else:
        print(f"  ✅ Irradiation time seems sufficient")
    
    print(f"\n🔍 If you see zero gas production, check:")
    print(f"  1. Nuclear data includes gas-producing reactions (n,α), (n,p), etc.")
    print(f"  2. Neutron energy spectrum includes fast neutrons (>1 MeV)")
    print(f"  3. Cross sections are available for your materials")
    print(f"  4. Depletion chain reduction didn't remove gas products")

if __name__ == "__main__":
    check_fusion_neutron_parameters() 