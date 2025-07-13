import openmc
import numpy as np
from pathlib import Path
import os
from neutronics_calphad.geometry_maker import create_model
from neutronics_calphad.config import ARC_D_SHAPE

# --- User Configuration ---
# Option 1: Explicitly set the path to your cross_sections.xml file.
# This is the recommended approach for testing different data libraries.
# Example: CROSS_SECTIONS_PATH = '/path/to/your/libs/cross_sections.xml'
CROSS_SECTIONS_PATH = '/Users/myless/nuclear_data/tendl_hdf5/cross_sections.xml'

# --- Setup OpenMC Data Path ---
# This logic prefers the explicit path, falls back to the environment variable,
# and raises an error if neither is set.
if CROSS_SECTIONS_PATH:
    cross_sections_file = Path(CROSS_SECTIONS_PATH)
    if not cross_sections_file.is_file():
        raise FileNotFoundError(f"❌ Cross sections file not found at '{cross_sections_file}'")
    openmc.config['cross_sections'] = str(cross_sections_file)
    print(f"✅ Using cross sections from: {cross_sections_file}")
elif 'OPENMC_CROSS_SECTIONS' in os.environ:
    openmc.config['cross_sections'] = os.environ['OPENMC_CROSS_SECTIONS']
    print(f"✅ Using cross sections from OPENMC_CROSS_SECTIONS env var.")
else:
    raise ValueError("❌ Cross section path not set. "
                     "Please set the CROSS_SECTIONS_PATH variable in this script or "
                     "set the OPENMC_CROSS_SECTIONS environment variable.")

# 1. Read the 709-group energy boundaries
ebins_file = Path(__file__).parent / 'ebins_709'
with open(ebins_file, 'r') as f:
    # Read the text file
    text = f.read()

# Split the text by commas and convert to a list of floats
energies_list = [float(e) for e in text.split(',')]
energy_bins = np.array(energies_list)

# OpenMC expects energy bins to be monotonically increasing.
# The ebins file is in descending order, so we sort it.
energy_bins = np.sort(energy_bins)

# 2. Create the OpenMC model
model = create_model(ARC_D_SHAPE)

# Replace tokamak source with simple isotropic source
simple_source = openmc.IndependentSource()
simple_source.space = openmc.stats.Point((330, 0, 0))  # Plasma center
simple_source.angle = openmc.stats.Isotropic()
simple_source.energy = openmc.stats.Discrete([14.1e6], [1.0])
simple_source.particle = 'neutron'

model.settings.source = [simple_source]
model.settings.batches = 10
model.settings.particles = 10000
model.settings.photon_transport = False

# 3. Create the 709-group flux tally
vv_cell = model.vv_cell
flux_tally = openmc.Tally(name='vv_flux')
cell_filter = openmc.CellFilter(vv_cell)
energy_filter = openmc.EnergyFilter(energy_bins)
flux_tally.filters = [cell_filter, energy_filter]
flux_tally.scores = ['flux']

model.tallies = openmc.Tallies([flux_tally])

# 4. Run the simulation
output_dir = Path(__file__).parent
statepoint_filename = model.run(cwd=output_dir)

# 5. Extract and save the flux
with openmc.StatePoint(statepoint_filename) as sp:
    flux_tally_result = sp.get_tally(name='vv_flux')
    vv_flux = flux_tally_result.mean.flatten()

    # Save the flux to a file
    flux_output_file = output_dir / 'vv_flux.npy'
    np.save(flux_output_file, vv_flux)

    print(f"✅ Flux data saved to {flux_output_file}")

    # For verification, print the total flux
    print(f"Total flux in vv_cell: {np.sum(vv_flux):.3e} n/cm^2-s") 