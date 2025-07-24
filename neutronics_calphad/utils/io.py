"""
I/O module for handling nuclear data and creating depletion chains.
"""

import argparse
import math
from typing import Dict
from collections import defaultdict
from io import StringIO
from pathlib import Path
import warnings
import h5py
import pandas as pd
import requests
import numpy as np

import openmc
import openmc.data
import openmc.deplete
from openmc.deplete import Nuclide, FissionYieldDistribution, REACTIONS


def material_string(comp_dict: Dict[str, float], bal_element: str) -> str:
    """Generate a material string of the form 'bal_element-2Cr-4Ti-3W-1Zr'.

    Args:
        comp_dict (Dict[str, float]): Dictionary of element fractions.
        bal_element (str): The balance element to use as prefix.

    Returns:
        str: Formatted material string.
    """
    parts = []
    for element, value in comp_dict.items():
        if element != bal_element:
            parts.append(f"{int(round(value * 100))}{element}")
    return f"{bal_element}-" + "-".join(parts)

def create_material(comp_dict, material_name, bal_element = 'V', density = 6.11, percent_type = 'ao'):
    new_material = openmc.Material()
    new_material.set_density('g/cm3', density)
    for element, percent in comp_dict.items():
        new_material.add_element(element, percent, percent_type)
    new_material.name = material_name
    
    return new_material

def create_ccfe709_h5(data_dir: Path):
    """Reads CCFE-709 energy boundaries and saves them to HDF5.
    The source text file is expected to have two header lines followed by
    comma-separated energy values in eV, typically in descending order. The
    output HDF5 file will contain datasets for lower and upper energy
    bounds suitable for OpenMC energy filters.
    Args:
        data_dir (Path): The directory containing 'ccfe_709.txt' and
            where 'ccfe_709.h5' will be saved.
    """
    txt_path = data_dir / "ccfe_709.txt"
    h5_path = data_dir / "ccfe_709.h5"

    if not txt_path.exists():
        raise FileNotFoundError(
            f"Source file '{txt_path}' not found. Please ensure it exists, "
            f"e.g., by running 'cp fispact_comparison/ebins_709 {txt_path}'."
        )

    with open(txt_path, 'r') as f:
        lines = f.readlines()

    # Skip header lines, join, and parse comma-separated values
    data_string = "".join(lines[2:])
    boundaries = np.array([float(e) for e in data_string.split(',') if e.strip()])
    boundaries = np.sort(boundaries)  # Ensure ascending order for OpenMC

    with h5py.File(h5_path, 'w') as hf:
        hf.create_dataset("E_lower", data=boundaries[:-1])
        hf.create_dataset("E_upper", data=boundaries[1:])
        hf.create_dataset("group_boundaries", data=boundaries)

    print(f"Successfully created '{h5_path}' with {len(boundaries) - 1} groups.")

def fetch_air_mass_energy_coeffs(data_dir: Path):
    """Fetches mass-energy absorption coefficients for Air from NIST.
    Downloads the data from the NIST XCOM database for dry air and saves it
    to a CSV file. If the download fails, a placeholder file is created.
    Args:
        data_dir (Path): The directory where the output
            'mass_energy_abs_coeff_air.csv' will be saved.
    """
    csv_path = data_dir / "mass_energy_abs_coeff_air.csv"
    url = "https://physics.nist.gov/cgi-bin/Xcom/xcom2"
    payload = {'Z': '7.218', 'CP': 'Air, Dry (near sea level)', 'NumEn': '200',
               'Energies': '1.00000-03..2.00000+01', 'Out': '2', 'Graph': '1'}

    try:
        response = requests.post(url, data=payload)
        response.raise_for_status()
        
        start_marker = "Energy (MeV),mu/rho (cm^2/g),mu_en/rho (cm^2/g)"
        end_marker = "</pre>"
        start_index = response.text.find(start_marker)
        if start_index == -1:
            raise ValueError("Data start marker not in NIST response.")
        
        csv_data = response.text[start_index:response.text.find(end_marker, start_index)].strip()
        df = pd.read_csv(StringIO(csv_data))
        df.rename(columns=lambda x: x.strip(), inplace=True)
        df.to_csv(csv_path, index=False)
        print(f"Successfully downloaded and saved '{csv_path}'.")
        
    except (requests.exceptions.RequestException, ValueError) as e:
        warnings.warn(f"Failed to fetch data from NIST: {e}. Creating placeholder.")
        placeholder = {"Energy (MeV)": [1e-3, 1e-2, 1e-1, 1e0, 1e1, 2e1],
                       "mu/rho (cm^2/g)": [4.7e3, 5.7, 0.16, 0.07, 0.027, 0.022],
                       "mu_en/rho (cm^2/g)": [4.7e3, 5.0, 0.028, 0.03, 0.022, 0.018]}
        pd.DataFrame(placeholder).to_csv(csv_path, index=False)
        print(f"Created placeholder file '{csv_path}'.")

# Helper functions adapted from openmc.deplete.chain

def _replace_missing(nuclide, data):
    """Replace a nuclide that is not present in the decay library.

    Parameters
    ----------
    nuclide : str
        Name of nuclide to be replaced, e.g., "Am242_m1"
    data : dict
        Dictionary of all available decay data

    Returns
    -------
    str or None
        Name of nuclide to replace with, or None if no suitable replacement can
        be found.

    """
    name, A, m = openmc.data.zam(nuclide)
    if m > 0:
        # Check for ground state
        if (name, A, 0) in data:
            return openmc.data.gnd_name(nuclide)
    else:
        # Check for metastable state
        if (name, A, 1) in data:
            return nuclide + "_m1"

    # Check for nuclide with one higher/lower mass number
    if (name, A + 1, 0) in data:
        return f"{name}{A + 1}"
    if (name, A - 1, 0) in data:
        return f"{name}{A - 1}"

    return None


def _replace_missing_fpy(parent, fpy_data, decay_data):
    """Find a replacement for a nuclide with missing fission product yields.

    Parameters
    ----------
    parent : str
        Name of nuclide with missing FPY data
    fpy_data : dict
        Dictionary of all available FPY data
    decay_data : dict
        Dictionary of all available decay data

    Returns
    -------
    str
        Name of nuclide whose FPY to use as a replacement

    """
    # If the nuclide is in a ground state and a metastable state exists with
    # fission yields, copy the yields from the metastable
    name, A, m = openmc.data.zam(parent)
    if m == 0:
        parent_m1 = parent + "_m1"
        if parent_m1 in fpy_data:
            return parent_m1

    # Find an isotone (same number of neutrons) and copy those yields
    n_neutrons = A - openmc.data.ATOMIC_NUMBER[name]
    for nuc, fpy in fpy_data.items():
        name_i, A_i, m_i = openmc.data.zam(nuc)
        if (A_i - openmc.data.ATOMIC_NUMBER[name_i]) == n_neutrons:
            return nuc

    # As a last resort, use U235 yields
    return "U235"


def create_chain(decay_files, fpy_files, neutron_files,
        reactions=('(n,2n)', '(n,3n)', '(n,4n)', '(n,gamma)', '(n,p)', '(n,a)', '(n,t)', '(n,d)'),
        progress=True
    ):
    """Create a depletion chain from ENDF files.

    Parameters
    ----------
    decay_files : list of str or openmc.data.endf.Evaluation
        List of ENDF decay sub-library files
    fpy_files : list of str or openmc.data.endf.Evaluation
        List of ENDF neutron-induced fission product yield sub-library files
    neutron_files : list of str or openmc.data.endf.Evaluation
        List of ENDF neutron reaction sub-library files
    reactions : iterable of str, optional
        Transmutation reactions to include in the depletion chain.
    progress : bool, optional
        Flag to print status messages during processing.

    Returns
    -------
    openmc.deplete.Chain
    """
    # Create dictionary mapping target to filename
    if progress:
        print('Processing neutron sub-library files...')
    
    rx_data = {}
    for f in neutron_files:
        try:
            evaluation = openmc.data.endf.Evaluation(f)
            name = evaluation.gnds_name
            rx_data[name] = {}
            for mf, mt, nc, mod in evaluation.reaction_list:
                if mf == 3:
                    file_obj = StringIO(evaluation.section[3, mt])
                    openmc.data.endf.get_head_record(file_obj)
                    q_value = openmc.data.endf.get_cont_record(file_obj)[1]
                    rx_data[name][mt] = q_value
        except Exception as e:
            warnings.warn(f"Could not process neutron file {f}: {e}")

    # Determine what decay and FPY nuclides are available
    if progress:
        print('Processing decay sub-library files...')
    decay_data = {}
    for f in decay_files:
        try:
            data = openmc.data.Decay(f)
            # Skip decay data for neutron itself
            if data.nuclide['atomic_number'] == 0:
                continue
            decay_data[data.nuclide['name']] = data
        except Exception as e:
            warnings.warn(f"Could not process decay file {f}: {e}")

    if progress:
        print('Processing fission product yield sub-library files...')
    fpy_data = {}
    for f in fpy_files:
        try:
            data = openmc.data.FissionProductYields(f)
            fpy_data[data.nuclide['name']] = data
        except Exception as e:
            warnings.warn(f"Could not process FPY file {f}: {e}")

    if progress:
        print('Creating depletion chain...')
    
    missing_daughter = []
    missing_rx_product = []
    missing_fpy = []
    missing_fp = []

    chain = openmc.deplete.Chain()
    for parent in sorted(decay_data, key=openmc.data.zam):
        data = decay_data[parent]
        nuclide = Nuclide(parent)

        if not data.nuclide['stable'] and data.half_life.nominal_value != 0.0:
            nuclide.half_life = data.half_life.nominal_value
            nuclide.decay_energy = data.decay_energy.nominal_value
            for mode in data.modes:
                type_ = ','.join(mode.modes)
                if mode.daughter in decay_data:
                    target = mode.daughter
                else:
                    target = _replace_missing(mode.daughter, decay_data)
                    if target is None:
                        missing_daughter.append(f"{parent} -> {type_} -> {mode.daughter}")
                        continue
                nuclide.add_decay_mode(type_, target, mode.branching_ratio.nominal_value)

        fissionable = False
        if parent in rx_data:
            reactions_available = set(rx_data[parent].keys())
            for name in reactions:
                mts = REACTIONS[name].mts
                if mts & reactions_available:
                    # Get dZ, dA for reaction and calculate daughter
                    delta_A, delta_Z = openmc.data.DADZ[name]
                    parent_A = data.nuclide['mass_number']
                    parent_Z = data.nuclide['atomic_number']
                    daughter_A = parent_A + delta_A
                    daughter_Z = parent_Z + delta_Z

                    # Check for physically possible products
                    if daughter_Z < 0 or daughter_A <= 0 or daughter_Z >= len(openmc.data.ATOMIC_SYMBOL):
                        continue
                    daughter_name = f"{openmc.data.ATOMIC_SYMBOL[daughter_Z]}{daughter_A}"

                    if daughter_name not in decay_data:
                        daughter_name = _replace_missing(daughter_name, decay_data)
                        if daughter_name is None:
                            missing_rx_product.append((parent, name, daughter_name))
                            continue
                    
                    for mt in sorted(mts):
                        if mt in rx_data[parent]:
                            q_value = rx_data[parent][mt]
                            break
                    else:
                        q_value = 0.0
                    nuclide.add_reaction(name, daughter_name, q_value, 1.0)
            
            if any(mt in reactions_available for mt in openmc.data.FISSION_MTS):
                q_value = rx_data[parent].get(18, 0.0) # MT=18 for total fission
                nuclide.add_reaction('fission', None, q_value, 1.0)
                fissionable = True

        if fissionable:
            if parent in fpy_data:
                fpy = fpy_data[parent]
                yield_energies = fpy.energies or [0.0]
                yield_data = {}
                for E, yield_table in zip(yield_energies, fpy.independent):
                    yields = defaultdict(float)
                    yield_replace_sum = 0.0
                    for product, y in yield_table.items():
                        if product not in decay_data:
                            daughter = _replace_missing(product, decay_data)
                            if daughter is not None:
                                yields[daughter] += y.nominal_value
                            yield_replace_sum += y.nominal_value
                        else:
                            yields[product] += y.nominal_value
                    if yield_replace_sum > 0:
                        missing_fp.append((parent, E, yield_replace_sum))
                    yield_data[E] = yields
                nuclide.yield_data = FissionYieldDistribution(yield_data)
            else:
                nuclide._fpy = _replace_missing_fpy(parent, fpy_data, decay_data)
                missing_fpy.append((parent, nuclide._fpy))

        #if nuclide.reactions or nuclide.decay_modes:
        chain.add_nuclide(nuclide)

    # Replace missing FPY data
    for nuclide in chain.nuclides:
        if hasattr(nuclide, '_fpy'):
            if nuclide._fpy in chain:
                nuclide.yield_data = chain[nuclide._fpy].yield_data
            del nuclide._fpy
    
    # Display warnings
    if missing_daughter:
        print('\nThe following decay modes have daughters with no decay data:')
        for mode in missing_daughter:
            print(f'  {mode}')

    if missing_rx_product:
        print('\nThe following reaction products have no decay data:')
        for vals in missing_rx_product:
            print('{} {} -> {}'.format(*vals))

    if missing_fpy:
        print('\nThe following fissionable nuclides have no fission product yields:')
        for parent, replacement in missing_fpy:
            print(f'  {parent}, replaced with {replacement}')

    if missing_fp:
        print('\nThe following nuclides have fission products with no decay data:')
        for vals in missing_fp:
            print('  {}, E={} eV (total yield={})'.format(*vals))

    return chain


def cmd_chain_builder(args):
    """Wrapper function for the command-line interface."""
    # Get file lists from directories
    neutron_dir = Path(args.neutron_dir)
    decay_dir = Path(args.decay_dir)
    fpy_dir = Path(args.fpy_dir)

    if not neutron_dir.is_dir() or not decay_dir.is_dir() or not fpy_dir.is_dir():
        print("Error: One or more provided paths is not a directory.")
        return

    neutron_files = list(neutron_dir.glob('*'))
    decay_files = list(decay_dir.glob('*'))
    fpy_files = list(fpy_dir.glob('*'))

    print(f"Found {len(neutron_files)} neutron files in {neutron_dir}")
    print(f"Found {len(decay_files)} decay files in {decay_dir}")
    print(f"Found {len(fpy_files)} FPY files in {fpy_dir}")

    # Create the chain
    chain = create_chain(decay_files, fpy_files, neutron_files)

    # Write to XML
    print(f"\nWriting chain to {args.output_file}...")
    chain.export_to_xml(args.output_file)
    print("Chain creation complete.")


def cmd_prepare_data(args):
    """Wrapper function to prepare all necessary data files."""
    data_path = Path("neutronics_calphad") / "data"
    data_path.mkdir(exist_ok=True)
    print("--- Preparing Data Files ---")
    create_ccfe709_h5(data_path)
    fetch_air_mass_energy_coeffs(data_path)
    print("--- Data Preparation Complete ---")


def main():
    """Command-line interface for I/O and data preparation tasks."""
    parser = argparse.ArgumentParser(description="I/O and data tasks for neutronics_calphad.")
    subparsers = parser.add_subparsers(dest='command', help='Sub-command help')

    # Sub-parser for the chain builder
    parser_chain = subparsers.add_parser('build-chain', help='Create an OpenMC depletion chain.')
    parser_chain.add_argument("--neutron-dir", required=True, help="Path to directory with TENDL neutron files.")
    parser_chain.add_argument("--decay-dir", required=True, help="Path to directory with FISPACT decay files.")
    parser_chain.add_argument("--fpy-dir", required=True, help="Path to directory with GEFY FPY files.")
    parser_chain.add_argument("--output-file", default="chain.xml", help="Path to write the output chain file.")
    
    # Sub-parser for data preparation
    parser_data = subparsers.add_parser('prepare-data', help='Download and process necessary data files.')

    args = parser.parse_args()
    
    if args.command == 'build-chain':
        cmd_chain_builder(args)
    elif args.command == 'prepare-data':
        cmd_prepare_data(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main() 