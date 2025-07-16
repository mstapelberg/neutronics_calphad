# fusion_opt/library.py
import openmc
import openmc.deplete
import openmc.lib
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import os
import h5py
from matplotlib.colors import LogNorm
from pathlib import Path

from .geometry_maker import create_model
from openmc_regular_mesh_plotter import plot_mesh_tally
from .config import ARC_D_SHAPE, ELEMENT_DENSITIES
import copy


ELMS = ['V', 'Cr', 'Ti', 'W', 'Zr']
# Times in seconds after shutdown
TIMES = [
    1,                    # 1 second
    3600,                 # 1 hour
    10*3600,              # 10 hours
    24*3600,              # 1 day
    7*24*3600,            # 1 week
    14*24*3600,           # 2 weeks
    30*24*3600,           # 1 month (30 days)
    60*24*3600,           # 2 months (60 days)
    365*24*3600,          # 1 year
    5*365*24*3600,        # 5 years
    10*365*24*3600,       # 10 years
    25*365*24*3600,       # 25 years
    100*365*24*3600       # 100 years
]

# Constants for FISPACT-like calculations
E_PER_FUSION_eV = 17.6e6  # eV per D-T fusion event
UNITS_EV_TO_J = 1.60218e-19

def get_material_by_name(materials, name):
    """Helper function to find a material by name."""
    for material in materials:
        if material.name == name:
            return material
    raise ValueError(f"Material with name '{name}' not found")

def get_reference_dose_rates(element, time_after_shutdown):
    """Provides reference dose rate ranges for comparison.
    
    Based on typical fusion reactor activation studies and ITER estimates.
    Returns (low_estimate, high_estimate) in µSv/h for contact dose rate.
    """
    # Reference data from fusion literature
    # These are contact dose rates for structural materials after ~1 year of operation
    
    reference_data = {
        'V': {
            1: (1e10, 1e12),         # 1 second: 100-10000 Sv/h
            3600: (1e10, 1e12),       # 1 hour: 10-1000 Sv/h
            24*3600: (1e8, 1e12),    # 1 day: 1-100 Sv/h
            14*24*3600: (1e8, 1e12), # 2 weeks: 0.01-1 Sv/h (hands-on = 0.01 Sv/h)
            365*24*3600: (1e6, 1e10), # 1 year: 0.001-0.1 Sv/h
            5*365*24*3600: (1e4, 1e8), # 5 years: 0.0001-0.01 Sv/h
            7*365*24*3600: (1e2, 1e5), # 7 years
            100*365*24*3600: (1e0, 1e4), # 100 years: 0.0001-0.01 Sv/h
        },
        'Cr': {
            1: (1e10, 1e12),         # 1 second: 100-10000 Sv/h
            3600: (1e10, 1e12),       # 1 hour: 10-1000 Sv/h
            24*3600: (1e8, 1e12),    # 1 day: 1-100 Sv/h
            14*24*3600: (1e8, 1e12), # 2 weeks: 0.01-1 Sv/h (hands-on = 0.01 Sv/h)
            365*24*3600: (1e6, 1e10), # 1 year: 0.001-0.1 Sv/h
            5*365*24*3600: (1e4, 1e8), # 5 years: 0.0001-0.01 Sv/h
            7*365*24*3600: (1e2, 1e5), # 7 years
            100*365*24*3600: (1e0, 1e4), # 100 years: 0.0001-0.01 Sv/h
        }
        # Similar patterns for other elements...
    }
    
    # Default to V estimates if element not found
    element_data = reference_data.get(element, reference_data['V'])
    
    # Find closest time point
    closest_time = min(element_data.keys(), key=lambda t: abs(t - time_after_shutdown))
    return element_data[closest_time]

def _get_flux_and_microxs(model, chain_file, outdir):
    """Runs transport, gets flux and microscopic cross sections.
    Args:
        model (openmc.Model): The OpenMC model to run.
        chain_file (str): Path to the depletion chain file.
        outdir (pathlib.Path): Directory to save output files.
    Returns:
        tuple: Paths to the flux HDF5 file and the multi-group cross-section CSV file.
    """
    flux_h5 = outdir / "flux.h5"
    microxs_csv = outdir / "microxs.csv"
    fispact_flux_file = outdir / "fispact_flux.txt"

    depletable_mats = [m for m in model.materials if m.depletable]
    if not depletable_mats:
        raise ValueError("No depletable materials found in the model.")

    flux_list, microxs_list = openmc.deplete.get_microxs_and_flux(
        model,
        depletable_mats,
        energies="UKAEA-1102", # CCFE-709 not available for TENDL21 in FISPACT
        chain_file=chain_file,
        run_kwargs={'cwd': str(outdir)}
    )

    # In the context of run_element, we are interested in the 'vcrti' material
    try:
        vcrti_index = [m.name for m in depletable_mats].index('vcrti')
        flux = flux_list[vcrti_index]
        microxs = microxs_list[vcrti_index]
    except ValueError:
        print("Warning: 'vcrti' material not found. Using the first depletable material.")
        flux = flux_list[0]
        microxs = microxs_list[0]

    # Save flux to an HDF5 file for internal use (e.g., collapsing XS)
    with h5py.File(flux_h5, 'w') as hf:
        hf.create_dataset('phi', data=flux)

    # Save microxs object to CSV
    microxs.to_csv(microxs_csv)
    print(f"Wrote multi-group cross sections to {microxs_csv}")
    
    # Save the flux in a FISPACT-readable format
    with open(fispact_flux_file, 'w') as f:
        f.write(f"{len(flux)}\n")
        for f_val in flux:
            f.write(f"{f_val:.6e}\n")
    print(f"Wrote FISPACT-readable flux file to {fispact_flux_file}")
    
    return flux_h5, microxs_csv

def _collapse_cross_sections(flux_h5, microxs_csv, outdir):
    """Collapses multi-group cross sections using a given flux spectrum.
    Args:
        flux_h5 (pathlib.Path): Path to the HDF5 file with CCFE-709 flux.
        microxs_csv (pathlib.Path): Path to the OpenMC micro-xs CSV file.
        outdir (pathlib.Path): Directory to save the output CSV.
    Returns:
        pathlib.Path: Path to the output CSV file.
    """
    out_csv = outdir / "collapsed_xs.csv"
    
    with h5py.File(flux_h5, 'r') as hf:
        # Assuming single material, flux has shape (1, n_groups)
        phi_E = hf['phi'][:][0]

    microxs = openmc.deplete.MicroXS.from_csv(microxs_csv)
    
    collapsed_data = []
    sum_phi_E = np.sum(phi_E)
    if sum_phi_E == 0:
        raise ValueError("Total flux is zero, cannot normalize.")

    for i, nuc in enumerate(microxs.nuclides):
        for j, rx in enumerate(microxs.reactions[nuc]):
            sigma_E = microxs[nuc, rx]
            sigma_eff = np.sum(sigma_E * phi_E) / sum_phi_E
            if sigma_eff > 0:
                collapsed_data.append({
                    'ZAID': nuc.zaid,
                    'MT': rx.mt,
                    'sigma_b': sigma_eff
                })

    pd.DataFrame(collapsed_data).to_csv(out_csv, index=False)
    return out_csv

def _run_independent_depletion(model, xs_csv, flux_h5, chain_file, timesteps, power, outdir):
    """Runs depletion with IndependentOperator.
    Args:
        model (openmc.Model): The OpenMC model with initial compositions.
        xs_csv (pathlib.Path): Path to the collapsed cross-sections CSV file.
        flux_h5 (pathlib.Path): Path to the HDF5 flux file.
        chain_file (str): Path to the depletion chain file.
        timesteps (list): List of irradiation/cooling durations in seconds.
        power (float): Fusion power in Watts.
        outdir (pathlib.Path): Directory to save depletion results.
    Returns:
        openmc.deplete.Results: The depletion results object.
    """
    with h5py.File(flux_h5, 'r') as hf:
        initial_flux = hf['phi'][:][0]

    micro_xs = openmc.deplete.MicroXS.from_csv(str(xs_csv))
    
    op = openmc.deplete.IndependentOperator(
        materials=model.materials,
        micro_xs=micro_xs,
        chain_file=str(chain_file),
        initial_flux=initial_flux,
        normalization_mode='source-rate'
    )
    
    source_rate = power / (E_PER_FUSION_eV * UNITS_EV_TO_J)
    source_rates = [source_rate] + [0.0] * (len(timesteps) - 1)

    integrator = openmc.deplete.CECMIntegrator(op, timesteps, source_rates=source_rates)
    integrator.integrate(cwd=str(outdir))
    
    return openmc.deplete.Results(outdir / "depletion_results.h5")

def _calculate_fispact_dose(results, data_dir):
    """Calculates dose rate using the FISPACT semi-empirical formula.
    Args:
        results (openmc.deplete.Results): Depletion results object.
        data_dir (pathlib.Path): Path to the data directory containing
            'mass_energy_abs_coeff_air.csv'.
    Returns:
        np.ndarray: Array of dose rates in µSv/h for each cooling step.
    """
    coeff_file = data_dir / "mass_energy_abs_coeff_air.csv"
    if not coeff_file.exists():
        raise FileNotFoundError(f"Required data file not found: {coeff_file}")
    
    coeffs = pd.read_csv(coeff_file)
    interp_func = interp1d(
        coeffs["Energy (MeV)"], 
        coeffs["mu_en/rho (cm^2/g)"],
        bounds_error=False, 
        fill_value="extrapolate"
    )

    dose_rates = []
    # Skip initial and irradiation steps (indices 0 and 1)
    for i in range(2, len(results)):
        mat = results[i].get_material(str(results[i].mat_to_ind.keys()[0]))
        activities = mat.get_activity(by_nuclide=True, units='Bq')
        gamma_spec = mat.get_decay_photon_energy()

        if not gamma_spec:
            dose_rates.append(0.0)
            continue
        
        # Gamma spectrum energies are in eV, convert to MeV
        energies_mev = gamma_spec.x * 1e-6
        mu_en_rho = interp_func(energies_mev)
        
        # Dose rate formula from FISPACT manual (Eq. 60)
        # Dose (Sv/h) = 1.602E-13 * 3600 * sum(Activity * Energy * Coeff)
        # Here we get µSv/h
        dose_sv_h = 1.602e-13 * 3600 * np.sum(gamma_spec.p * energies_mev * mu_en_rho)
        dose_usv_h = dose_sv_h * 1e6
        dose_rates.append(dose_usv_h)
        
    return np.array(dose_rates)


def _get_fispact_xs(printlib_path, outdir):
    """Parses a FISPACT printlib file to extract collapsed cross sections.

    This function reads a FISPACT-II `printlib` file, finds the block
    containing one-group cross-sections, parses the data, and saves it
    to a CSV file compatible with `openmc.deplete.MicroXS.from_csv`.

    Args:
        printlib_path (str or Path): Path to the FISPACT printlib file.
        outdir (pathlib.Path): Directory to save the output CSV.

    Returns:
        pathlib.Path: Path to the output CSV file with collapsed xs.
    """
    import re
    print(f"Parsing FISPACT printlib file: {printlib_path}")
    collapsed_data = []
    in_xs_block = False

    with open(printlib_path, 'r') as f:
        for line in f:
            # Block 3 with cross sections starts with this header
            if "CROSS-SECTIONS FOR ALL REACTIONS" in line:
                in_xs_block = True
                # Skip the next 3 header lines to get to the data
                next(f, None); next(f, None); next(f, None)
                continue

            if not in_xs_block:
                continue

            # A blank line or a line with "TOTAL" indicates the end of the block
            if not line.strip() or "TOTAL" in line:
                break
            
            # Attempt to parse the line. This is based on a common fixed-width format.
            try:
                # Example: V 51      102 (n,gamma)         CR 52           1.0945E-02 ...
                parts = line.split()
                if len(parts) < 5:
                    continue

                # The MT number is the first integer that is not the mass number
                mt = -1
                for part in parts:
                    if part.isdigit() and int(part) != int(parts[1]):
                        mt = int(part)
                        break
                if mt == -1:
                    continue

                # Reconstruct nuclide string, e.g., 'V51', 'Am242m' -> 'Am242_m1'
                element = parts[0]
                mass_number = parts[1]
                nuclide_str = f"{element.capitalize()}{mass_number}"
                if 'm' in nuclide_str:
                    nuclide_str = re.sub(r'm(\d*)', r'_m\1', nuclide_str)

                zaid = openmc.Nuclide(nuclide_str).zaid
                
                # The cross section is usually the first floating point number
                # after the reaction string like '(n,gamma)'
                xs_val = 0.0
                for part in parts:
                    try:
                        if 'E' in part or '.' in part:
                           val = float(part)
                           if val >= 0.0:
                               xs_val = val
                               break 
                    except ValueError:
                        continue

                if xs_val > 0.0:
                    collapsed_data.append({
                        'ZAID': zaid,
                        'MT': mt,
                        'sigma_b': xs_val
                    })
            except (ValueError, IndexError, openmc.exceptions.DataError) as e:
                if "No data available" not in str(e): # Suppress common benign errors
                    print(f" could not parse line: '{line.strip()}'. Error: {e}")
                continue

    if not collapsed_data:
        raise ValueError(
            f"Could not parse any cross-section data from {printlib_path}. "
            "Please check if the file is a valid printlib file containing "
            "the 'CROSS-SECTIONS FOR ALL REACTIONS' block."
        )

    out_csv = outdir / "fispact_collapsed_xs.csv"
    df = pd.DataFrame(collapsed_data)
    
    # Remove duplicates that might arise from parsing, keeping the first
    df = df.drop_duplicates(subset=['ZAID', 'MT'], keep='first')
    
    df.to_csv(out_csv, index=False)
    
    print(f" Successfully parsed {len(df)} unique cross sections from printlib.")
    print(f"Saved FISPACT collapsed cross sections to {out_csv}")
    
    return out_csv


def run_element(element,
                config,
                outdir="results/elem_lib",
                chain_file=None,
                cross_sections=None,
                use_reduced_chain=True,
                mpi_args=None,
                workflow='r2s',
                power=500e6,
                printlib_file=None,
                debug_particles=None,
                verbose=False):
    """Runs a full depletion simulation for a single element.
    This function can orchestrate multiple workflows:
    - 'r2s': A "Rigorous 2-Step" analysis with coupled neutron-photon transport.
    - 'fispact_path_a': A FISPACT-like analysis using pre-calculated fluxes
      and a semi-empirical dose formula.
    - 'fispact_path_b': A FISPACT-like depletion with cross sections read from
      a FISPACT printlib file.
    Args:
        element (str): The chemical symbol of the element to use.
        config (dict): The base configuration for the model.
        outdir (str, optional): The root directory for output files.
        chain_file (str or Path, optional): Path to the depletion chain file.
        cross_sections (str or Path, optional): Path to cross_sections.xml.
        use_reduced_chain (bool, optional): Whether to use a reduced chain.
        mpi_args (list, optional): MPI arguments for parallel execution.
        workflow (str, optional): The calculation workflow to use.
            One of ['r2s', 'fispact_path_a', 'fispact_path_b']. Defaults to 'r2s'.
        power (float, optional): The fusion power in Watts. Defaults to 500e6.
        printlib_file (str or Path, optional): Path to a FISPACT printlib file,
            required for the 'fispact_path_b' workflow.
        debug_particles (int, optional): If set, overrides the number of
            particles per batch for a faster, less precise run.
        verbose (bool, optional): Whether to enable verbose output.
    """
    print(f"Running '{workflow}' workflow for element: {element}")
    
    # --- Resolve cross section and chain paths ---
    if cross_sections:
        openmc.config['cross_sections'] = str(cross_sections)
    elif 'OPENMC_CROSS_SECTIONS' in os.environ:
        openmc.config['cross_sections'] = os.environ['OPENMC_CROSS_SECTIONS']
    else:
        raise ValueError("Cross sections must be specified via argument or OPENMC_CROSS_SECTIONS env var.")

    if not chain_file:
        chain_file = os.environ.get('OPENMC_CHAIN_FILE')
        if not chain_file:
            # Provide a default path if not set, but warn the user.
            default_path = Path.home() / 'nuclear_data' / 'chain_endfb80_sfr.xml'
            print(f" WARNING: `chain_file` not specified and OPENMC_CHAIN_FILE not set.")
            print(f"    -> Defaulting to '{default_path}'")
            chain_file = default_path

    print(f"Using Cross Sections: {openmc.config['cross_sections']}")
    print(f"Using Depletion Chain: {chain_file}")

    # Create a deep copy of the config to modify for the specific element
    elem_config = copy.deepcopy(config)
    
    # Modify the config for the specific element
    # This assumes the vv material is 'vcrti' and we replace it
    elem_config['materials']['vcrti'] = {
        'elements': {element: 1.0},
        'density': ELEMENT_DENSITIES[element],
        'depletable': True
    }

    # Create the OpenMC model for the given element
    model = create_model(elem_config)

    if debug_particles:
        model.settings.particles = int(debug_particles)
        print(f"DEBUG: Particle count overridden to {model.settings.particles}")

    if verbose:
        model.settings.verbosity = 10
        print(" verbose logging enabled.")
    else:
        model.settings.verbosity = 5


    element_outdir = Path(outdir) / element
    element_outdir.mkdir(parents=True, exist_ok=True)

    # --- Setup depletion chain (with reduction) ---
    if use_reduced_chain:
        print("Preparing depletion chain with reduction...")
        full_chain = openmc.deplete.Chain.from_xml(chain_file)
        
        # Get all nuclides from the model's depletable materials
        initial_nuclides = {n for mat in model.materials if mat.depletable for n in mat.get_nuclides()}

        # Filter out nuclides that are in the model but not in the chain.
        # This is a critical step to prevent errors during reduction.
        chain_nuclides = set(full_chain.nuclide_dict.keys())
        nuclides_for_reduction = list(initial_nuclides.intersection(chain_nuclides))

        # Warn about any nuclides that are being ignored
        missing_from_chain = initial_nuclides.difference(chain_nuclides)
        if missing_from_chain:
            print(f"WARNING: The following nuclides are in the model but not in the "
                  f"depletion chain and will be ignored: {missing_from_chain}")

        reduced_chain_path = element_outdir / f"reduced_chain_{Path(chain_file).stem}.xml"

        if not reduced_chain_path.exists():
            print("Generating reduced depletion chain...")
            try:
                reduced_chain = full_chain.reduce(nuclides_for_reduction)
                reduced_chain.export_to_xml(reduced_chain_path)
                print(f" Wrote reduced chain to {reduced_chain_path} ({len(reduced_chain.nuclides)} nuclides)")
            except KeyError as e:
                print(f"WARNING: Failed to create reduced chain due to a missing nuclide: {e}")
                print("         This can happen if a reaction product is not in the base chain.")
                print("         Falling back to using the full, un-reduced depletion chain.")
                use_reduced_chain = False
        else:
            print(f"️ Using existing reduced chain: {reduced_chain_path}")
    
    if use_reduced_chain:
        chain_file_to_use = element_outdir / f"reduced_chain_{Path(chain_file).stem}.xml"
    else:
        print("Using full depletion chain as specified.")
        chain_file_to_use = chain_file

    # --- Workflow-dependent execution ---
    if workflow == 'r2s':
        # --- Neutron transport and depletion (Coupled) ---
        model.settings.photon_transport = False
        model.settings.batches = 10
        model.settings.particles = 10000

        depletion_dir = element_outdir / "depletion_r2s"
        depletion_dir.mkdir(parents=True, exist_ok=True)
        model.settings.output_path = depletion_dir
        
        source_rate = power / (E_PER_FUSION_eV * UNITS_EV_TO_J)
        source_rates = [source_rate] + [0.0] * (len(TIMES) - 1)

        operator = openmc.deplete.CoupledOperator(
            model,
            chain_file=chain_file_to_use,
            normalization_mode='source-rate',
        )
        
        integrator = openmc.deplete.PredictorIntegrator(
            operator, TIMES, source_rates=source_rates)
        
        original_cwd = os.getcwd()
        os.chdir(depletion_dir)
        try:
            integrator.integrate()
        finally:
            os.chdir(original_cwd)
        
        results = openmc.deplete.Results(depletion_dir / "depletion_results.h5")
        # TODO: Refactor photon transport logic here for R2S dose calc
        dose_rates = [0.0] * len(TIMES) # Placeholder

    elif workflow in ['fispact_path_a', 'fispact_path_b']:
        depletion_dir = element_outdir / "depletion_independent"
        depletion_dir.mkdir(parents=True, exist_ok=True)
        
        flux_h5, microxs_csv = _get_flux_and_microxs(model, chain_file_to_use, depletion_dir)

        if workflow == 'fispact_path_a':
            xs_csv = _collapse_cross_sections(flux_h5, microxs_csv, depletion_dir)
        else:  # fispact_path_b
            if not printlib_file:
                raise ValueError("The 'fispact_path_b' workflow requires a `printlib_file`.")
            print(f"Using FISPACT cross sections from: {printlib_file}")
            xs_csv = _get_fispact_xs(printlib_file, depletion_dir)

        results = _run_independent_depletion(
            model, xs_csv, flux_h5, chain_file_to_use, TIMES, power, depletion_dir
        )
        
        if workflow == 'fispact_path_a':
            # Assumes 'data' dir is sibling to 'neutronics_calphad'
            data_dir = Path(__file__).parent.parent / 'data'
            dose_rates = _calculate_fispact_dose(results, data_dir)
        else: # fispact_path_b
            # TODO: Refactor photon transport logic here
            dose_rates = [0.0] * len(TIMES) # Placeholder
    else:
        raise ValueError(f"Unknown workflow: {workflow}")

    # --- Extract metrics ---
    results_file = depletion_dir / "depletion_results.h5"
    results = openmc.deplete.Results(results_file)
    
    # Gas production - Use the correct OpenMC Results API
    vv_material = get_material_by_name(model.materials, "vcrti")
    
    print(f"DEBUG: Results object methods: {[m for m in dir(results) if not m.startswith('_') and callable(getattr(results, m))]}")
    
    # DEBUG: Check neutron activation effectiveness
    print(f"\nDEBUG: Neutron activation analysis for {element}")
    try:
        # Get material after irradiation (step 0 is initial, step 1 is after irradiation)
        initial_mat = results[0].get_material(str(vv_material.id))
        irradiated_mat = results[1].get_material(str(vv_material.id))
        
        # Use the original material's volume since it's not preserved in results
        original_volume = vv_material.volume if hasattr(vv_material, 'volume') else None
        original_density = vv_material.density if hasattr(vv_material, 'density') else None
        
        # Calculate mass from density and volume
        initial_density = getattr(initial_mat, 'density', None)
        irradiated_density = getattr(irradiated_mat, 'density', None)
        
        if initial_density is not None and original_volume is not None:
            initial_mass = initial_density * original_volume if original_volume > 0 else 0
            irradiated_mass = irradiated_density * original_volume if irradiated_density is not None and original_volume > 0 else 0
            print(f"  - Initial material mass: {initial_mass:.2e} g (ρ={initial_density:.2f} g/cm³, V={original_volume:.2e} cm³)")
            if irradiated_density is not None:
                print(f"  - Irradiated material mass: {irradiated_mass:.2e} g")
            else:
                print(f"  - Irradiated material density not available")
        else:
            print(f"  - Material volume/density not available")
            print(f"  - Material attributes: {[attr for attr in dir(initial_mat) if not attr.startswith('_')]}")
        
        # Check total activity after irradiation
        total_activity = irradiated_mat.get_activity()  # Bq
        print(f"  - Total activity after irradiation: {total_activity:.2e} Bq")
        
        # Check decay heat
        decay_heat = irradiated_mat.get_decay_heat()  # W
        print(f"  - Decay heat after irradiation: {decay_heat:.2e} W")
        
        # List major activation products
        nuclide_list = list(irradiated_mat.get_nuclides())
        print(f"  - Number of nuclides after irradiation: {len(nuclide_list)}")
        print(f"  - Nuclides present: {nuclide_list[:10]}...")  # Show first 10
        
        # Just list the nuclides for now (individual activity calculation needs different API)
        print(f"  - Major activation products: {nuclide_list[:5]}")  # Show first 5
            
    except Exception as e:
        print(f"  - Error analyzing activation: {e}")
        import traceback
        traceback.print_exc()
    
    gases = {}
    print(f"\nDEBUG: Gas production analysis for {element}")
    print(f"  - VV material ID: {vv_material.id}")
    print(f"  - VV material name: {vv_material.name}")
    print(f"  - Number of time steps in results: {len(results)}")
    
    for gas in ['He3', 'He4', 'H1', 'H2', 'H3']:
        try:
            # Use Results.get_atoms() method - this is the correct API
            times, atom_counts = results.get_atoms(vv_material, gas, nuc_units="atoms")
            print(f"  - {gas}: Found {len(atom_counts)} time points")
            if len(atom_counts) > 0:
                print(f"    Time 0 (initial): {atom_counts[0]:.2e} atoms")
                if len(atom_counts) > 1:
                    print(f"    Time 1 (after irradiation): {atom_counts[1]:.2e} atoms")
                    gases[gas] = atom_counts[1]  # After irradiation, not initial
                else:
                    gases[gas] = atom_counts[0]
            else:
                gases[gas] = 0.0
                print(f"    No data found for {gas}")
            print(f"DEBUG: {gas} atoms after irradiation: {gases[gas]:.2e}")
        except Exception as e:
            print(f"Warning: Could not get {gas} atoms from results: {e}")
            # Try alternative approach - check if the nuclide exists at all
            try:
                # Check if this gas nuclide is present in any time step
                for i in range(len(results)):
                    mat = results[i].get_material(str(vv_material.id))
                    nuclides = list(mat.get_nuclides())
                    if gas in nuclides:
                        print(f"  - {gas} found in time step {i}: {nuclides}")
                        break
                else:
                    print(f"  - {gas} not found in any time step")
            except:
                pass
            gases[gas] = 0.0

    # --- Photon transport for dose rate ---
    # Set the chain file for decay photon energy calculations - use absolute path
    openmc.config['chain_file'] = os.path.abspath(chain_file_to_use)
    
    print(f"\nStarting photon transport analysis...")
    print(f"   - Chain file set to: {openmc.config['chain_file']}")
    print(f"   - Results file: {results_file}")
    print(f"   - Number of results: {len(results)}")
    
    dose_rates = []
    
    # Create a mesh for the dose tally - explicitly define bounds
    mesh = openmc.RegularMesh()
    mesh.dimension = [100, 100, 100]  # x, y, z
    # Define mesh bounds to cover the tokamak geometry
    mesh.lower_left = [0, 0, -350]     # Quarter torus starts at x=0, y=0
    mesh.upper_right = [600, 600, 350] # Extends to outer boundaries
    
    energies, pSv_cm2 = openmc.data.dose_coefficients(particle="photon", geometry="AP")
    dose_filter = openmc.EnergyFunctionFilter(energies, pSv_cm2)
    particle_filter = openmc.ParticleFilter(["photon"])
    mesh_filter = openmc.MeshFilter(mesh)
    dose_tally = openmc.Tally(name="photon_dose_on_mesh")
    dose_tally.filters = [mesh_filter, dose_filter, particle_filter]
    dose_tally.scores = ["flux"]
    
    model.tallies = openmc.Tallies([dose_tally])
    
    # Get the vacuum vessel cell for source definition
    vv_cell = model.vv_cell
    
    print(f"\nDEBUG: Starting photon transport calculations...")
    print(f"  Processing {len(TIMES)} cooling time steps")
    
    for i in range(1, len(TIMES)): # Skip the irradiation step (index 0)
        cooling_time = TIMES[i-1]  # TIMES[0] corresponds to results[2], etc.
        result_index = i + 1  # Correct index: results[0]=initial, results[1]=after irradiation, results[2]=first cooling, etc.
        time_label = f"{cooling_time} s"
        
        # Convert to readable format
        if cooling_time == 10*365*24*3600:
            time_label = "10 years"
        elif cooling_time == 5*365*24*3600:
            time_label = "5 years"
        elif cooling_time == 25*365*24*3600:
            time_label = "25 years"
        elif cooling_time == 100*365*24*3600:
            time_label = "100 years"
            
        print(f"\n{'='*60}")
        print(f"Calculating dose rate at cooling time {cooling_time} s ({time_label})")
        print(f"Using result index: {result_index} (corrected from {i})")
        print(f"{'='*60}")
        
        activated_mat = results[result_index].get_material(str(vv_material.id))
        
        # Enhanced debugging for material state
        print(f"\nDEBUG: Material state at {time_label}:")
        try:
            activity = activated_mat.get_activity()
            decay_heat = activated_mat.get_decay_heat()
            nuclides = list(activated_mat.get_nuclides())[:5]  # First 5 nuclides
            print(f"  - Total activity: {activity:.2e} Bq")
            print(f"  - Decay heat: {decay_heat:.2e} W")
            print(f"  - Sample nuclides: {nuclides}")
        except Exception as e:
            print(f"  - Error getting material properties: {e}")
        
        # DEBUG: Check the activated material properties
        # Use the original material's volume since it's not preserved in depletion results
        original_volume = vv_material.volume if hasattr(vv_material, 'volume') else None
        original_density = vv_material.density if hasattr(vv_material, 'density') else None
        activated_density = getattr(activated_mat, 'density', None)
        
        if original_volume is not None and activated_density is not None:
            material_mass = activated_density * original_volume if original_volume > 0 else 0
            print(f"DEBUG: Activated material volume (from original): {original_volume:.2e} cm³")
            print(f"DEBUG: Material density: {activated_density:.2f} g/cm³")
            print(f"DEBUG: Activated material mass: {material_mass:.2e} g")
        elif original_volume is not None and original_density is not None:
            # Use original density if activated density is None
            material_mass = original_density * original_volume if original_volume > 0 else 0
            print(f"DEBUG: Activated material volume (from original): {original_volume:.2e} cm³")
            print(f"DEBUG: Material density (from original): {original_density:.2f} g/cm³")
            print(f"DEBUG: Activated material mass (estimated): {material_mass:.2e} g")
        else:
            print(f"DEBUG: Material volume/density not available")
            print(f"DEBUG: Available attributes: {[attr for attr in dir(activated_mat) if not attr.startswith('_')]}")
            # Fallback to reasonable estimates if both are not available
            if original_volume is None:
                print(f"WARNING: Using estimated VV volume")
                original_volume = 2.13e+05  # cm³ for quarter torus VV
            if original_density is None:
                print(f"WARNING: Using estimated VV density")
                original_density = 6.11  # g/cm³ for Vanadium
            material_mass = original_density * original_volume
            print(f"DEBUG: Using fallback estimates: V={original_volume:.2e} cm³, ρ={original_density:.2f} g/cm³")
        
        # DEBUG: Check decay photon source properties
        decay_photon_energy = None
        total_photon_rate = 0
        
        try:
            print(f"\nDEBUG: Getting decay photon energy for {time_label}...")
            decay_photon_energy = activated_mat.get_decay_photon_energy()
            
            if decay_photon_energy is None:
                print(f"  - get_decay_photon_energy() returned None")
            else:
                print(f"  - Decay photon energy object created")
                
                # Check if it's a valid distribution
                if hasattr(decay_photon_energy, 'integral'):
                    total_photon_rate = decay_photon_energy.integral()  # Total photons/s
                    print(f"  - Total photon rate: {total_photon_rate:.2e} photons/s")
                    
                    if total_photon_rate == 0:
                        print(f"  - WARNING: Photon rate is exactly zero!")
                        # Try to understand why
                        if hasattr(decay_photon_energy, 'x') and hasattr(decay_photon_energy, 'p'):
                            print(f"    - Energy bins: {len(decay_photon_energy.x)}")
                            print(f"    - Probability sum: {np.sum(decay_photon_energy.p):.2e}")
                else:
                    print(f"  - Decay photon energy has no 'integral' method")
                    print(f"  - Available methods: {[m for m in dir(decay_photon_energy) if not m.startswith('_')]}")
            
            # Get energy spectrum info
            if decay_photon_energy and hasattr(decay_photon_energy, 'x') and hasattr(decay_photon_energy, 'p'):
                energies = decay_photon_energy.x
                probabilities = decay_photon_energy.p
                if len(energies) > 0 and np.sum(probabilities) > 0:
                    avg_energy = np.sum(energies * probabilities) / np.sum(probabilities)
                    print(f"DEBUG: Average photon energy: {avg_energy:.3f} eV")
                    print(f"DEBUG: Energy range: {min(energies):.2e} - {max(energies):.2e} eV")
                else:
                    print(f"DEBUG: No valid energy spectrum data")
            
        except Exception as e:
            print(f"DEBUG: Exception getting photon energy distribution: {e}")
            import traceback
            traceback.print_exc()
            decay_photon_energy = None
        
        if decay_photon_energy is None or total_photon_rate == 0:
            print(f"\nWARNING: No decay photons at time {cooling_time} s ({time_label})")
            print(f"    - Appending 0.0 to dose_rates")
            dose_rates.append(0.0)
            continue
        
        photon_source = openmc.Source()
        photon_source.space = openmc.stats.Box(*vv_cell.bounding_box)
        photon_source.particle = 'photon'
        photon_source.energy = decay_photon_energy
        
        # DEBUG: Check source box dimensions
        bbox = vv_cell.bounding_box
        print(f"DEBUG: Source box: x=[{bbox[0][0]:.1f}, {bbox[1][0]:.1f}], "
              f"y=[{bbox[0][1]:.1f}, {bbox[1][1]:.1f}], z=[{bbox[0][2]:.1f}, {bbox[1][2]:.1f}] cm")
        
        model.settings.source = photon_source
        model.settings.particles = 10000
        
        photon_outdir = element_outdir / f"photon_transport_{cooling_time}s"
        photon_outdir.mkdir(parents=True, exist_ok=True)
        
        # Run photon transport with MPI if specified
        if mpi_args:
            print(f"DEBUG: Using MPI for photon transport: {mpi_args}")
            # Ensure OpenMC is re-initialized for MPI in a clean state
            openmc.lib.reset()
            statepoint_file = model.run(cwd=photon_outdir, mpi_args=mpi_args)
        else:
            openmc.lib.reset()
            statepoint_file = model.run(cwd=photon_outdir)
        
        with openmc.StatePoint(statepoint_file) as sp:
            tally = sp.get_tally(name="photon_dose_on_mesh")
            
            # DEBUG: Check tally statistics
            raw_mean = tally.mean.flatten()
            raw_std = tally.std_dev.flatten()
            non_zero_cells = np.sum(raw_mean > 0)
            print(f"DEBUG: Tally statistics:")
            print(f"  - Non-zero mesh cells: {non_zero_cells}/{len(raw_mean)}")
            print(f"  - Raw tally range: {np.min(raw_mean):.2e} - {np.max(raw_mean):.2e}")
            print(f"  - Raw tally mean: {np.mean(raw_mean):.2e}")
            print(f"  - Relative error: {np.mean(raw_std[raw_mean > 0])/np.mean(raw_mean[raw_mean > 0])*100:.1f}%" if non_zero_cells > 0 else "  - Relative error: N/A")
            
            # Convert from pSv-cm^3/source-particle to uSv/h
            # The tally gives dose per source particle, but we need total dose rate
            volume = mesh.volumes[0][0][0]
            print(f"DEBUG: Mesh cell volume: {volume:.2e} cm³")
            
            # Scaling breakdown:
            # 1. tally.mean is in pSv⋅cm³/source-particle 
            # 2. multiply by total_photon_rate (photons/s) to get pSv⋅cm³/s
            # 3. divide by volume to get pSv/s  
            # 4. multiply by 3600 to get pSv/h
            # 5. multiply by 1e-6 to get µSv/h
            
            scaling_factor = total_photon_rate * 1e-6 * 3600 / volume
            print(f"DEBUG: Scaling factor breakdown:")
            print(f"  - Total photon rate: {total_photon_rate:.2e} photons/s")
            print(f"  - Volume normalization: 1/{volume:.2e} = {1/volume:.2e} /cm³")
            print(f"  - Unit conversion: 1e-6 * 3600 = {1e-6 * 3600:.2e} (pSv→µSv, s→h)")
            print(f"  - Final scaling factor: {scaling_factor:.2e}")
            
            dose_mean = raw_mean * scaling_factor
            average_dose = np.mean(dose_mean)
            max_dose = np.max(dose_mean)
            
            print(f"DEBUG: Final dose values:")
            print(f"  - Average dose rate: {average_dose:.2e} µSv/h")
            print(f"  - Maximum dose rate: {max_dose:.2e} µSv/h")
            
            # Compare with reference values
            ref_low, ref_high = get_reference_dose_rates(element, cooling_time)
            print(f"DEBUG: Reference comparison:")
            print(f"  - Expected range: {ref_low:.2e} - {ref_high:.2e} µSv/h")
            if average_dose < ref_low:
                ratio = ref_low / average_dose if average_dose > 0 else float('inf')
                print(f"  - RESULT TOO LOW by factor of {ratio:.1e}")
            elif average_dose > ref_high:
                ratio = average_dose / ref_high
                print(f"  - RESULT TOO HIGH by factor of {ratio:.1e}")
            else:
                print(f"  - Result within expected range")
            
            dose_rates.append(average_dose)
            
            # Finalize the OpenMC process to release file handles
            openmc.lib.finalize()
            
            # Plot the dose map - add validation for LogNorm
            try:
                # Check if tally data is valid for LogNorm (no zeros, negative values, or NaN)
                tally_data = tally.mean.flatten() * scaling_factor
                valid_data = tally_data[~np.isnan(tally_data) & (tally_data > 0)]
                
                if len(valid_data) > 0 and np.max(valid_data) > np.min(valid_data):
                    # Data is valid for LogNorm
                    plot_norm = LogNorm(vmin=max(np.min(valid_data), 1e-10), vmax=np.max(valid_data))
                else:
                    # Fall back to linear normalization
                    plot_norm = None
                    print(f"Warning: Using linear scale for dose plot due to invalid data for log scale")
                
                plot = plot_mesh_tally(
                    tally=tally,
                    basis="xz",
                    value="mean",
                    colorbar_kwargs={'label': "Decay photon dose [µSv/h]"},
                    norm=plot_norm,
                    volume_normalization=False,
                    scaling_factor=scaling_factor,
                )
                plot.figure.savefig(photon_outdir / f'dose_map_{element}_{workflow}_time_{cooling_time}s.png')
            except Exception as e:
                print(f"Warning: Could not create dose plot for {element} at time {cooling_time}s: {e}")
                # Continue without plotting

    # --- Store results ---
    output_filename = f"{element}_{workflow}.h5"
    with h5py.File(element_outdir / output_filename, "w") as h:
        h.create_dataset('dose_times', data=np.array(TIMES))
        h.create_dataset('dose', data=np.array(dose_rates))
        for k, v in gases.items():
            h.create_dataset(f"gas/{k}", data=v)
    
    # --- Create summary table and plot ---
    print(f"\nDose Rate Summary for {element}:")
    print("-"*70)
    print(f"{'Time':<20} {'Dose Rate (Sv/h)':<20} {'Reference (Sv/h)':<20} {'Status':<10}")
    print("-"*70)
    
    # Convert times to years for display
    time_labels = {
        1: "1 second",
        3600: "1 hour",
        10*3600: "10 hours",
        24*3600: "1 day",
        7*24*3600: "1 week",
        14*24*3600: "2 weeks",
        30*24*3600: "1 month",
        60*24*3600: "2 months",
        365*24*3600: "1 year",
        5*365*24*3600: "5 years",
        10*365*24*3600: "10 years",
        25*365*24*3600: "25 years",
        100*365*24*3600: "100 years"
    }
    
    # Reference values in Sv/h (not µSv/h)
    reference_values_Sv = {
        1: 1e5,                  # 100,000 Sv/h at 1 second
        3600: 5e4,               # 10,000 Sv/h at 1 hour
        10*3600: 1e4,            # 5,000 Sv/h at 10 hours
        24*3600: 1e4,            # 1,000 Sv/h at 1 day
        7*24*3600: 1e2,          # 100 Sv/h at 1 week
        14*24*3600: 5e1,         # 50 Sv/h at 2 weeks
        30*24*3600: 1e1,         # 10 Sv/h at 1 month
        60*24*3600: 5e0,         # 5 Sv/h at 2 months
        365*24*3600: 1e0,        # 1 Sv/h at 1 year
        5*365*24*3600: 1e-1,     # 0.1 Sv/h at 5 years
        10*365*24*3600: 1e-2,    # 0.05 Sv/h at 10 years
        25*365*24*3600: 1e-3,    # 0.01 Sv/h at 25 years
        100*365*24*3600: 1e-4    # 0.0001 Sv/h at 100 years
    }
    
    for i, (time_s, dose_uSv) in enumerate(zip(TIMES, dose_rates)):
        dose_Sv = dose_uSv * 1e-6  # Convert µSv/h to Sv/h
        ref_Sv = reference_values_Sv.get(time_s, 1e-4)
        
        # Status check
        if dose_Sv == 0:
            status = "Zero"
        elif dose_Sv > ref_Sv * 10:
            status = "High"
        elif dose_Sv < ref_Sv / 10:
            status = "Low"
        else:
            status = "OK"
            
        print(f"{time_labels[time_s]:<20} {dose_Sv:<20.2e} {ref_Sv:<20.2e} {status:<10}")
    
    print("-"*70)
    
    # Create dose rate plot
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Convert to years for x-axis
    times_years = np.array(TIMES) / (365.25 * 24 * 3600)
    doses_Sv = np.array(dose_rates) * 1e-6  # Convert to Sv/h
    
    # Plot calculated dose rates
    ax.loglog(times_years, doses_Sv, 'bo-', linewidth=2, markersize=8, 
              label=f'{element} (calculated)')
    
    # Plot reference values
    ref_times_years = np.array(list(reference_values_Sv.keys())) / (365.25 * 24 * 3600)
    ref_doses_Sv = np.array(list(reference_values_Sv.values()))
    ax.loglog(ref_times_years, ref_doses_Sv, 'r--', linewidth=2, 
              label='Reference (typical)')
    
    # Add hands-on maintenance limit
    ax.axhline(y=1e-2, color='green', linestyle=':', linewidth=2, 
               label='Hands-on limit (10 mSv/h)')
    
    ax.set_xlabel('Time after shutdown (years)')
    ax.set_ylabel('Contact dose rate (Sv/h)')
    ax.set_title(f'Contact Dose Rate vs. Cooling Time - {element}')
    ax.grid(True, which="both", ls="-", alpha=0.2)
    ax.legend()
    
    # Set reasonable y-axis limits
    ax.set_ylim(1e-8, 1e6)
    ax.set_xlim(1e-7, 1e3)
    
    # Add time markers
    for time_s, label in time_labels.items():
        time_y = time_s / (365.25 * 24 * 3600)
        if 1e-7 < time_y < 1e3:
            ax.axvline(x=time_y, color='gray', linestyle=':', alpha=0.3)
    
    plt.tight_layout()
    plot_path = element_outdir / f'dose_rate_vs_time_{element}.png'
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"\nDose rate plot saved to: {plot_path}")
    
    print(f"\nFinished simulation for element: {element}")

def build_library(elements=None,
                  config=None,
                  outdir="results/elem_lib",
                  chain_file=None,
                  cross_sections=None,
                  use_reduced_chain=True,
                  mpi_args=None,
                  workflow='r2s',
                  power=500e6,
                  printlib_file=None,
                  debug_particles=None,
                  verbose=False):
    """Builds a library of depletion results for a list of elements.

    This function iterates through the given list of elements and calls
    `run_element` for each one, effectively building a library of
    activation results for different materials.
    
    Args:
        elements (list of str, optional): A list of element symbols to run.
            If None, uses the default `ELMS` list. Defaults to None.
        config (dict, optional): The base model configuration.
            If None, defaults to ARC_D_SHAPE.
        outdir (str, optional): The root directory for output files.
            Defaults to "elem_lib".
        chain_file (str or Path, optional): Path to the depletion chain file.
            Passed to `run_element`.
        cross_sections (str or Path, optional): Path to the cross_sections.xml file.
            Passed to `run_element`.
        use_reduced_chain (bool, optional): Whether to use a reduced chain.
            Defaults to True.
        mpi_args (list, optional): MPI arguments for parallel execution.
            Example: ['mpiexec', '-n', '8'] for 8 processes. Defaults to None (serial).
        workflow (str, optional): The calculation workflow to use.
            Passed to `run_element`.
        power (float, optional): The fusion power in Watts.
            Passed to `run_element`.
        printlib_file (str or Path, optional): Path to a FISPACT printlib file.
            Passed to `run_element`.
        debug_particles (int, optional): If set, overrides the number of
            particles per batch for a faster, less precise run.
        verbose (bool, optional): Whether to enable verbose output.
    """
    if elements is None:
        elements = ELMS
    
    if config is None:
        from .config import ARC_D_SHAPE
        config = ARC_D_SHAPE
        config_name = 'arc_d_shape'
    else:
        # Attempt to find the name of the config for logging
        config_name = config.get('geometry', {}).get('type', 'custom')

    print(f"Building element library for: {', '.join(elements)}")
    print(f"Using base configuration: {config_name}")
    
    for element in elements:
        print(f"\n{'='*80}")
        print(f" Starting element: {element}")
        print(f"{'='*80}")
        try:
            run_element(element,
                        config=config,
                        outdir=outdir,
                        chain_file=chain_file,
                        cross_sections=cross_sections,
                        use_reduced_chain=use_reduced_chain,
                        mpi_args=mpi_args,
                        workflow=workflow,
                        power=power,
                        printlib_file=printlib_file,
                        debug_particles=debug_particles,
                        verbose=verbose)
            print(f"\n Successfully completed element: {element}")
        except Exception as e:
            print(f" FAILED to complete element: {element}")
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()
        print(f"\n{'='*80}")
        
    print(f"\n Library build complete for: {', '.join(elements)}")
    print(f"Results saved in '{outdir}' directory.")
