import numpy as np
from pathlib import Path
import sys

# It's good practice to check for dependencies and guide the user.
try:
    import pypact as pp
except ImportError:
    print("❌ pypact is not installed. Please install it with 'pip install pypact'")
    sys.exit(1)

# --- Simulation Constants ---
POWER_MW = 2.25e6
ENERGY_PER_FUSION_MEV = 17.6

# New cumulative irradiation times in years
CUMULATIVE_TIMES_YEARS = np.array([
    1 / (365.25 * 24 * 3600),  # 1 second
    1 / (365.25 * 24),         # 1 hour
    10 / (365.25 * 24),        # 10 hours
    7 / 365.25,                # 1 week
    30 / 365.25,               # 1 month
    1.0,                       # 1 year
    5.0,
    7.0,
    10.0,
    50.0,
    100.0
])

# Convert cumulative times to durations for each step
IRRADIATION_DURATIONS_YEARS = np.diff(CUMULATIVE_TIMES_YEARS, prepend=0)

def calculate_source_rate(power_mw, energy_per_fusion_mev):
    """Calculates the total neutron source rate from fusion power."""
    power_watts = power_mw * 1e6
    energy_joules = energy_per_fusion_mev * 1.602e-13
    source_rate = power_watts / energy_joules
    print(f"INFO: Calculated neutron source rate: {source_rate:.3e} n/s")
    return source_rate

def main():
    """Main function to set up and write the FISPACT-II input files."""
    script_dir = Path(__file__).parent
    output_name = "vv_vanadium"

    # 1. Check for required input files from step 1
    flux_file = script_dir / 'vv_flux.npy'
    ebins_file = script_dir / 'ebins_709'

    if not all([flux_file.exists(), ebins_file.exists()]):
        print(f"❌ ERROR: Missing 'vv_flux.npy' or 'ebins_709'.")
        print("➡️ Please run '1_generate_flux.py' first.")
        return
        
    print("✅ Found required flux and energy bin files.")

    # 2. Load and scale flux from OpenMC
    flux_per_source_n = np.load(flux_file)
    with open(ebins_file, 'r') as f:
        energy_bins = np.array([float(e) for e in f.read().split(',')])
    
    flux_per_source_n = np.flip(flux_per_source_n)
    source_rate = calculate_source_rate(POWER_MW, ENERGY_PER_FUSION_MEV)
    absolute_flux = flux_per_source_n * source_rate
    total_flux = np.sum(absolute_flux)
    print(f"INFO: Total absolute flux in vv_cell: {total_flux:.3e} n/cm^2-s")

    # 3. Write the flux file for FISPACT
    fluxes_filename = f"{output_name}.fluxes"
    flux_out_path = script_dir / fluxes_filename
    flux_obj = pp.Flux(flux_values=absolute_flux / total_flux if total_flux > 0 else absolute_flux,
                       group_boundaries=energy_bins)
    flux_obj.write(str(flux_out_path))
    print(f"✅ Wrote flux spectrum to '{flux_out_path}'")

    # 4. Create FISPACT input data object with user-specified settings
    inp = pp.InputData(name=output_name)
    inp.overwriteExisting()
    inp.enableJSON()
    inp.approxGammaSpectrum()
    inp.readXSData(709)
    inp.readDecayData()
    inp.enableHalflifeInOutput()
    inp.enableHazardsInOutput()
    inp.setProjectile(pp.PROJECTILE_NEUTRON)
    inp.enableSystemMonitor()
    inp.readGammaGroup()
    inp.enableInitialInventoryInOutput()
    inp.setLogLevel(pp.LOG_SEVERITY_ERROR)
    inp.setFluxesFile(fluxes_filename)

    # Thresholds
    inp.setXSThreshold(1e-12)
    inp.setAtomsThreshold(1e5)

    # Target material (100% Vanadium to match OpenMC model)
    # Density for Vanadium is ~6.1 g/cm3, but we use 1.0 to match OpenMC model
    inp.setDensity(1.0) 
    inp.setMass(1.0)
    inp.addElement('V', percentage=100.0)

    # Irradiation schedule
    for duration_years in IRRADIATION_DURATIONS_YEARS:
        duration_seconds = duration_years * 365.25 * 24 * 3600
        inp.addIrradiation(duration_seconds, total_flux)

    # 5. Validate and write input file
    try:
        inp.validate()
        input_filename = script_dir / f"{output_name}.i"
        pp.to_file(inp, str(input_filename))
        print(f"✅ Wrote main input file to '{input_filename}'")
        print("\n➡️  You can now manually run FISPACT-II with these generated files.")
    except Exception as e:
        print(f"❌ An error occurred during input validation or writing: {e}")

if __name__ == "__main__":
    main() 