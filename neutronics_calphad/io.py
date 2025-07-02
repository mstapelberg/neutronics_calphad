"""
I/O module for handling nuclear data and creating depletion chains.
"""

import argparse
import math
from collections import defaultdict
from io import StringIO
from pathlib import Path
import warnings

import openmc
import openmc.data
import openmc.deplete
from openmc.deplete import Nuclide, FissionYieldDistribution, REACTIONS

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

        if nuclide.reactions or nuclide.decay_modes:
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


def main():
    """Command-line interface for creating a depletion chain."""
    parser = argparse.ArgumentParser(description="Create an OpenMC depletion chain from ENDF files.")
    parser.add_argument("--neutron-dir", required=True, help="Path to directory with TENDL 2021 neutron files.")
    parser.add_argument("--decay-dir", required=True, help="Path to directory with FISPACT 2020 decay files.")
    parser.add_argument("--fpy-dir", required=True, help="Path to directory with GEFY 6.1 FPY files.")
    parser.add_argument("--output-file", default="chain.xml", help="Path to write the output chain file.")

    args = parser.parse_args()
    cmd_chain_builder(args)


if __name__ == '__main__':
    main() 