"""
Flux calculation and neutron transport utilities.

This module contains functions for calculating neutron flux spectra,
handling multi-group cross sections, and managing flux normalization
for neutronics simulations.
"""
import numpy as np
import openmc
import openmc.deplete
import pandas as pd
from pathlib import Path
from typing import Tuple, List, Optional

# Constants
FUSION_ENERGY_MEV = 17.6 # eV per D-T fusion event
MEV_TO_J = 1.602176634e-13


def get_flux_and_microxs(model: openmc.Model, 
                        chain_file: str, 
                        group_structure: str,
                        outdir: Path) -> Tuple[Path, Path]:
    """Run transport calculation and extract spatially-resolved flux and microscopic cross sections.
    
    Args:
        model: The OpenMC model to run.
        chain_file: Path to the depletion chain file.
        group_structure: The group structure to use for the flux and microxs calculations. CCFE-709 or UKAEA-1102
        outdir: Directory to save output files.
        
    Returns:
        Tuple of paths to the flux file and the multi-group cross-section CSV file.
        
    Raises:
        ValueError: If no depletable materials are found in the model.
    """
    # get the number of energy groups from the group structure string there will be a number after the -
    num_groups = int(group_structure.split('-')[1])

    flux_file = outdir / f"flux_spectrum_{num_groups}.txt"
    microxs_csv = outdir / f"microxs_{num_groups}.csv"
    fispact_flux_file = outdir / f"fispact_flux_{num_groups}.txt"

    depletable_mats = [m for m in model.materials if m.depletable]
    if not depletable_mats:
        raise ValueError("No depletable materials found in the model.")

    # Get multi-group flux and cross sections using UKAEA-1102 structure
    flux_list, microxs_list = openmc.deplete.get_microxs_and_flux(
        model,
        depletable_mats,
        energies=group_structure,  
        chain_file=chain_file,
        run_kwargs={'cwd': str(outdir)}
    )

    # Focus on the vacuum vessel material ('vcrti')
    try:
        vcrti_index = [m.name for m in depletable_mats].index('vcrti')
        flux = flux_list[vcrti_index]
        microxs = microxs_list[vcrti_index]
        vv_material = depletable_mats[vcrti_index]
    except ValueError:
        print("Warning: 'vcrti' material not found. Using the first depletable material.")
        flux = flux_list[0]
        microxs = microxs_list[0]
        vv_material = depletable_mats[0]

    # Get material volume for proper flux normalization
    material_volume = getattr(vv_material, 'volume', None)
    if material_volume is None:
        print("Warning: Material volume not set. Using estimated VV volume.")
        material_volume = 2.13e5  # cm³, estimated quarter-torus VV volume

    print(f"Material: {vv_material.name}")
    print(f"Material volume: {material_volume:.2e} cm³")
    print(f"Flux spectrum: {len(flux)} energy groups")
    
    # Debug: Check what openmc.deplete.get_microxs_and_flux returns
    print(f"DEBUG: OpenMC flux analysis:")
    print(f"  - Flux array shape: {flux.shape}")
    print(f"  - Flux units from OpenMC: assumed to be neutrons/cm²/s per source neutron")
    print(f"  - Total flux per source neutron: {np.sum(flux):.2e}")
    if hasattr(flux, 'dtype'):
        print(f"  - Flux dtype: {flux.dtype}")
    
    # Save flux spectrum for internal use
    flux_data = {
        'energy_groups': len(flux),
        'material_volume_cm3': material_volume,
        'flux_per_source_neutron': flux.tolist()
    }
    
    with open(flux_file, 'w') as f:
        f.write("# OpenMC Flux Spectrum\n")
        f.write(f"# Material: {vv_material.name}\n")
        f.write(f"# Volume: {material_volume:.6e} cm³\n")
        f.write(f"# Energy groups: {len(flux)}\n")
        f.write("# Flux values are per source neutron\n")
        for i, phi in enumerate(flux):
            f.write(f"{i+1:4d} {phi:.6e}\n")
    
    # Save cross sections to CSV (simple format for easy manipulation)
    microxs.to_csv(microxs_csv)
    print(f"Saved {len(microxs.nuclides)} nuclides with cross sections to {microxs_csv}")
    
    # Create FISPACT-compatible flux file (normalized per unit volume)
    # FISPACT expects flux in neutrons/cm²/s, so we normalize by volume
    flux_per_cm3 = flux / material_volume
    
    # Debug: Check if flux normalization seems reasonable
    print(f"DEBUG: Flux normalization check:")
    print(f"  - Flux per source neutron (total): {np.sum(flux):.2e}")
    print(f"  - Material volume: {material_volume:.2e} cm³")
    print(f"  - Flux per cm³ per source neutron: {np.sum(flux_per_cm3):.2e}")
    
    # Sanity check: flux per source neutron should be << 1
    if np.sum(flux) > 1:
        print(f"WARNING: Flux per source neutron ({np.sum(flux):.1e}) seems high!")
        print(f"         Expected: < 1 (flux should be less than 1 per source neutron)")
        print(f"         This suggests the flux may not be per-source-neutron normalized")
        
        # Check if this is a cross-section weighted flux
        print(f"DEBUG: Flux analysis:")
        print(f"  - Max flux in any group: {np.max(flux):.2e}")
        print(f"  - Min flux in any group: {np.min(flux[flux > 0]):.2e}")
        print(f"  - Number of non-zero groups: {np.sum(flux > 0)}")
        print(f"  - This may be a volume-averaged or cross-section weighted flux")
    
    with open(fispact_flux_file, 'w') as f:
        f.write(f"{len(flux)}\n")  # Number of energy groups
        for phi in flux_per_cm3:
            f.write(f"{phi:.6e}\n")  # Flux per cm³ per source neutron
    
    print(f"Saved FISPACT-compatible flux (normalized per cm³) to {fispact_flux_file}")
    print(f"  - Total flux per source neutron: {np.sum(flux):.2e}")
    print(f"  - Total flux per cm³ per source neutron: {np.sum(flux_per_cm3):.2e}")
    
    return flux_file, microxs_csv


def collapse_cross_sections(flux_file: Path, 
                           microxs_csv: Path, 
                           outdir: Path) -> Tuple[openmc.deplete.MicroXS, np.ndarray]:
    """Collapse multi-group cross sections using OpenMC's built-in flux weighting.
    
    Args:
        flux_file: Path to the flux spectrum file.
        microxs_csv: Path to the multi-group cross-section CSV file (not used, kept for compatibility).
        outdir: Directory to save the collapsed data (for reference files).
        
    Returns:
        Tuple of (collapsed_microxs, flux_array) - The collapsed MicroXS object and flux array.
        
    Raises:
        ValueError: If no flux data is found in the flux file.
    """
    # Read flux spectrum
    flux_data = []
    material_volume = None
    
    with open(flux_file, 'r') as f:
        for line in f:
            if line.startswith('# Volume:'):
                material_volume = float(line.split()[2])
            elif not line.startswith('#'):
                parts = line.strip().split()
                if len(parts) >= 2:
                    flux_data.append(float(parts[1]))  # Second column is flux
    
    flux = np.array(flux_data)
    
    if len(flux) == 0:
        raise ValueError("No flux data found in flux file")
    
    print(f"DEBUG: Flux spectrum loaded - {len(flux)} groups, total: {np.sum(flux):.2e}")
    
    # Use OpenMC's built-in method to collapse cross sections with flux weighting
    # This automatically handles all the nuclides and reactions properly
    try:
        # Get the chain file from openmc config for consistency
        chain_file = openmc.config.get('chain_file')
        if not chain_file:
            raise ValueError("Chain file not found in openmc.config")
        
        print(f"DEBUG: Using chain file: {chain_file}")
        print(f"DEBUG: Calling MicroXS.from_multigroup_flux...")
        
        collapsed_microxs = openmc.deplete.MicroXS.from_multigroup_flux(
            energies="UKAEA-1102",  # Same energy structure we used for transport
            multigroup_flux=flux,   # Our calculated flux spectrum
            chain_file=chain_file   # Depletion chain for nuclides/reactions
        )
        
        print(f"✅ Successfully collapsed cross sections using flux weighting")
        print(f"  - {len(collapsed_microxs.nuclides)} nuclides")
        print(f"  - {len(collapsed_microxs.reactions)} reactions")
        
        # Save a summary CSV for reference (optional)
        summary_data = []
        for nuc in collapsed_microxs.nuclides:
            for rx in collapsed_microxs.reactions:
                try:
                    xs_val = collapsed_microxs[nuc, rx][0]  # Single group
                    if xs_val > 0:
                        summary_data.append({
                            'nuclide': nuc,
                            'reaction': rx,
                            'xs_barns': xs_val
                        })
                except (KeyError, IndexError):
                    continue
        
        if summary_data:
            summary_csv = outdir / "collapsed_xs_summary.csv"
            pd.DataFrame(summary_data).to_csv(summary_csv, index=False)
            print(f"Saved cross-section summary to {summary_csv}")
        
        return collapsed_microxs, flux
        
    except Exception as e:
        print(f"❌ Error creating collapsed cross sections: {e}")
        print(f"Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        
        # Try a fallback approach - use the original multi-group data directly
        print("\n🔄 Attempting fallback: using original multi-group cross sections...")
        
        try:
            # Load the original multi-group cross sections
            original_microxs = openmc.deplete.MicroXS.from_csv(microxs_csv)
            print(f"Loaded original MicroXS: {len(original_microxs.nuclides)} nuclides")
            
            # For now, just return the multi-group cross sections and flux
            # This allows the simulation to continue, though not collapsed
            print("⚠️  WARNING: Using multi-group cross sections (not collapsed)")
            return original_microxs, flux
            
        except Exception as fallback_error:
            print(f"❌ Fallback also failed: {fallback_error}")
            raise


def calculate_actual_flux(flux_per_source_neutron: np.ndarray, 
                         power: float) -> np.ndarray:
    """Calculate actual neutron flux from per-source-neutron flux and fusion power.
    
    Args:
        flux_per_source_neutron: Flux spectrum normalized per source neutron.
        power: Fusion power in Megawatts.
        
    Returns:
        Actual flux spectrum in neutrons/cm²/s.
    """
    source_rate = power * 1E6 / (FUSION_ENERGY_MEV * MEV_TO_J)  # neutrons/s
    actual_flux = flux_per_source_neutron * source_rate  # Scale flux by neutron production rate
    
    print(f"Fusion power: {power/1e6:.1f} MW")
    print(f"Neutron source rate: {source_rate:.2e} neutrons/s")
    print(f"Flux per source neutron: {np.sum(flux_per_source_neutron):.2e}")
    print(f"Total flux in material: {np.sum(actual_flux):.2e} neutrons/cm²/s")
    
    # Sanity check: typical fusion reactor flux should be ~1e15 neutrons/cm²/s
    typical_fusion_flux = 1e15  # neutrons/cm²/s
    flux_ratio = np.sum(actual_flux) / typical_fusion_flux
    if flux_ratio > 100:
        print(f"WARNING: Calculated flux is {flux_ratio:.1f}x higher than typical fusion reactor flux!")
        print(f"         Expected: ~{typical_fusion_flux:.1e} neutrons/cm²/s")
        print(f"         This may indicate a flux normalization issue.")
        
        # Geometry correction factor analysis
        print(f"DEBUG: Geometry analysis:")
        spherical_area = 4 * np.pi * (113/2)**2  # cm² for minor radius sphere
        torus_area = 4 * np.pi**2 * 330 * 113    # cm² for torus (R=330, r=113)
        geometry_factor = torus_area / spherical_area
        print(f"  - Spherical model area: {spherical_area:.1e} cm²")
        print(f"  - Actual torus area: {torus_area:.1e} cm²") 
        print(f"  - Geometry correction factor: {geometry_factor:.1f}x")
        corrected_flux = np.sum(actual_flux) / geometry_factor
        print(f"  - Geometry-corrected flux: {corrected_flux:.2e} neutrons/cm²/s")
        print(f"  - Still {corrected_flux/typical_fusion_flux:.1f}x higher than typical")
    
    return actual_flux


def validate_flux_normalization(flux: np.ndarray, 
                               material_volume: Optional[float] = None) -> bool:
    """Validate that flux normalization is physically reasonable.
    
    Args:
        flux: Flux spectrum array (should be per source neutron).
        material_volume: Material volume in cm³ (optional).
        
    Returns:
        True if flux normalization seems reasonable, False otherwise.
    """
    total_flux = np.sum(flux)
    
    print(f"Flux validation:")
    print(f"  - Total flux per source neutron: {total_flux:.2e}")
    
    # Key test: flux per source neutron should be much less than 1
    if total_flux > 1.0:
        print(f"  ❌ FAIL: Flux per source neutron ({total_flux:.2e}) > 1.0")
        print(f"       This is physically unreasonable and indicates a normalization error.")
        return False
    elif total_flux > 0.1:
        print(f"  ⚠️  WARNING: Flux per source neutron ({total_flux:.2e}) is high but < 1.0")
        print(f"       This may still indicate a normalization issue.")
        return False
    else:
        print(f"  ✅ PASS: Flux per source neutron ({total_flux:.2e}) seems reasonable")
        
    # Additional checks
    if len(flux) > 0:
        max_flux = np.max(flux)
        min_flux = np.min(flux[flux > 0]) if np.any(flux > 0) else 0
        non_zero_groups = np.sum(flux > 0)
        
        print(f"  - Energy groups: {len(flux)} total, {non_zero_groups} non-zero")
        print(f"  - Flux range: {min_flux:.2e} to {max_flux:.2e}")
        
        if non_zero_groups == 0:
            print(f"  ❌ FAIL: No non-zero flux groups found")
            return False
    
    return True 