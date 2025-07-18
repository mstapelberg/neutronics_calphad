import openmc
import openmc.deplete
import numpy as np
from pathlib import Path
import os
from neutronics_calphad.geometry_maker import create_model
from neutronics_calphad.config import SPHERICAL
from typing import Tuple

# --- Simulation Constants ---
POWER_MW = 500  # Megawatts
ENERGY_PER_FUSION_MEV = 14.1

def calculate_source_rate(power_mw: float, energy_per_fusion_mev: float) -> float:
    """
    Calculates the total neutron source rate from fusion power.

    Args:
        power_mw: Fusion power in megawatts.
        energy_per_fusion_mev: Energy per fusion reaction in MeV.

    Returns:
        The total neutron source rate in n/s.
    """
    power_watts = power_mw * 1e6
    energy_joules = energy_per_fusion_mev * 1.602e-13
    source_rate = power_watts / energy_joules
    print(f"INFO: Calculated neutron source rate: {source_rate:.3e} n/s")
    return source_rate

def get_flux_and_microxs_1102(model: openmc.Model,
                                 chain_file: str,
                                 outdir: Path) -> Tuple[Path, Path]:
    """
    Run transport calculation and extract flux and cross sections in UKAEA-1102 format.

    Args:
        model: The OpenMC model to run.
        chain_file: Path to the depletion chain file.
        outdir: Directory to save output files.

    Returns:
        Tuple of paths to the flux NumPy file and the multi-group cross-section CSV file.
    """
    flux_file = outdir / "vv_flux.npy"
    microxs_csv = outdir / "microxs_1102.csv"
    fispact_flux_file = outdir / "fispact_flux_1102.txt"

    depletable_mats = [m for m in model.materials if m.depletable]
    if not depletable_mats:
        raise ValueError("No depletable materials found in the model.")

    print("Running transport calculation to get flux and microscopic cross sections...")
    flux_list, microxs_list = openmc.deplete.get_microxs_and_flux(
        model,
        depletable_mats,
        energies="UKAEA-1102",
        chain_file=chain_file,
        run_kwargs={'cwd': str(outdir)}
    )

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

    np.save(flux_file, flux)
    microxs.to_csv(microxs_csv)
    print(f"✅ Flux data saved to {flux_file}")
    print(f"✅ Microscopic cross sections saved to {microxs_csv}")

    # Save the volume for use in the next step
    volume_file = outdir / "vv_volume.txt"
    material_volume = getattr(vv_material, 'volume', 2.13e5) # fallback volume
    with open(volume_file, 'w') as f:
        f.write(str(material_volume))
    print(f"✅ VV material volume saved to {volume_file}")

    # Create FISPACT-compatible flux file
    source_rate = calculate_source_rate(POWER_MW, ENERGY_PER_FUSION_MEV)
    
    with open(fispact_flux_file, 'w') as f:
        f.write(f"{len(flux)}\n")
        # Normalize by volume and multiply by source rate for FISPACT
        normalized_flux = (flux / material_volume) * source_rate
        for phi in normalized_flux:
            f.write(f"{phi:.6e}\n")
    print(f"✅ FISPACT-compatible flux saved to {fispact_flux_file}")
    
    return flux_file, microxs_csv

def main():
    """Main function to generate flux and cross sections."""
    script_dir = Path('step_1_get_flux_and_xs')
    
    # --- Setup OpenMC Data Path ---
    if 'OPENMC_CROSS_SECTIONS' not in os.environ:
         raise ValueError("❌ Cross section path not set. "
                     "Please set the OPENMC_CROSS_SECTIONS environment variable.")
    openmc.config['cross_sections'] = os.environ['OPENMC_CROSS_SECTIONS']
    
    chain_file = os.environ.get('OPENMC_CHAIN_FILE')
    if not chain_file:
        raise ValueError("❌ Chain file path not set. "
                         "Please set the OPENMC_CHAIN_FILE environment variable.")

    print(f"✅ Using cross sections from: {openmc.config['cross_sections']}")
    print(f"✅ Using chain file from: {chain_file}")

    # Create the OpenMC model
    model = create_model(SPHERICAL)
    model.settings.particles = 10000

    # Get flux and cross sections
    get_flux_and_microxs_1102(model, chain_file, script_dir)

if __name__ == "__main__":
    main()