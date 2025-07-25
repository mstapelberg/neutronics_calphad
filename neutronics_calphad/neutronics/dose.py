from __future__ import annotations
import warnings, re
from pathlib import Path
from typing import Tuple, List, Dict

import numpy as np
import openmc
from openmc.deplete import Results
import periodictable

# ===========================================================================
# NOTE: The functions _read_abs, _group_midpoints, _atomic_mass, and 
# _atomic_number_from_nuclide are assumed to be correct from previous steps
# and are included here for completeness. The only change is in contact_dose.
# ===========================================================================

def _read_abs(abs_file: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[int, np.ndarray]]:
    """
    Return (bounds[25], mu_air[24], muen_air[24], mu_of_element{Z}).

    This robust version handles data that may span multiple lines for all data blocks.
    """
    with open(abs_file) as fh:
        lines = [line.strip() for line in fh if line.strip()]

    bounds, mu_air, muen_air = None, None, None
    mu_elem: dict[int, np.ndarray] = {}
    i = 0
    while i < len(lines):
        # A helper function to parse a multi-line data block
        def parse_multiline_data(start_index: int) -> tuple[np.ndarray, int]:
            numbers = []
            j = start_index
            while j < len(lines):
                try:
                    line_tokens = lines[j].split()
                    float(line_tokens[0])  # Will raise ValueError if not a number
                    numbers.extend([float(x) for x in line_tokens])
                    j += 1
                except (ValueError, IndexError):
                    break # Stop when a non-numeric line is found
            return np.array(numbers), j

        # --- Main Parsing Loop ---
        current_line = lines[i]
        tokens = current_line.split()

        if current_line == "MeV":
            data, i = parse_multiline_data(i + 1)
            if data.size != 24:
                raise ValueError(f"Expected 24 energy bounds, but found {data.size}")
            bounds = np.array([0.0] + data.tolist())
        
        elif current_line.replace(" ", "").upper() == "AIRMU":
            mu_air, i = parse_multiline_data(i + 1)
            if mu_air.size != 24:
                 raise ValueError(f"Expected 24 AIRMU values, but found {mu_air.size}")

        elif current_line.replace(" ", "").upper() == "AIRMUEN":
            muen_air, i = parse_multiline_data(i + 1)
            if muen_air.size != 24:
                 raise ValueError(f"Expected 24 AIRMUEN values, but found {muen_air.size}")

        elif len(tokens) == 2 and tokens[0].isalpha() and tokens[1].isdigit():
            Z = int(tokens[1])
            mu_elem[Z], i = parse_multiline_data(i + 1)
            if mu_elem[Z].size != 24:
                raise ValueError(f"Element Z={Z} has {mu_elem[Z].size} values, but expected 24")
        
        else: # Unrecognized line, just advance the index
            i += 1
            
    # --- Final Validation ---
    if bounds is None: raise RuntimeError("Failed to locate 'MeV' bounds in file.")
    if mu_air is None: raise RuntimeError("Failed to locate 'AIRMU' in file.")
    if muen_air is None: raise RuntimeError("Failed to locate 'AIRMUEN' in file.")

    return bounds, mu_air, muen_air, mu_elem

_electron_charge_MeV_to_J = 1.602176634e-13
_C = 0.5 * 3600.0 * _electron_charge_MeV_to_J        # Eq.(60) constant

def _group_midpoints(bounds: np.ndarray) -> np.ndarray:
    """Geometric mid‑points of each energy group."""
    return np.sqrt(bounds[:-1] * bounds[1:])

def _atomic_mass(nuclide: str) -> float:
    """Return atomic mass in g/mol for any nuclide name."""
    return openmc.data.atomic_mass(nuclide)

def _atomic_number_from_nuclide(nuclide: str) -> int:
    """
    Extracts the atomic number from a nuclide string by identifying the
    element symbol at the start of the string.
    """
    match = re.match(r'^[A-Z][a-z]?', nuclide)
    if match:
        symbol = match.group(0)
        try:
            return periodictable.elements.symbol(symbol).number
        except ValueError as e:
            raise ValueError(f"Unknown element symbol '{symbol}' from nuclide '{nuclide}'") from e
    else:
        raise ValueError(f"Could not extract a valid element symbol from nuclide '{nuclide}'")


def contact_dose(
    results: Results,
    chain_file: str,
    abs_file: str = "/home/myless/Packages/fispact/nuclear_data/decay/abs_2012",
) -> Tuple[np.ndarray, List[Dict[str, float]]]:
    """
    Per‑timestep gamma contact dose (Sv h‑1 kg‑1) using Jaeger kernel.
    
    This function calculates dose rates using the depleted materials from the results file,
    which contain the proper decay photon energy spectra.
    """
    # --- static data --------------------------------------------------------
    bounds, mu_air, muen_air, mu_elem = _read_abs(abs_file)
    E_mid = _group_midpoints(bounds)

    openmc.config["chain_file"] = chain_file
    yield_cache: Dict[str, np.ndarray] = {}

    def photon_yield(nuc: str) -> np.ndarray:
        if nuc in yield_cache:
            return yield_cache[nuc]
        
        spec = openmc.data.decay_photon_energy(nuc)
        
        if spec is None:
            yield_cache[nuc] = None
            return None
        
        # Handle Mixture objects that don't have .x and .p attributes
        if not hasattr(spec, 'x') or not hasattr(spec, 'p'):
            
            # Check if it's a Mixture object with integral method
            if hasattr(spec, 'integral'):
                try:
                    _ = spec.integral()
                except Exception:
                    pass  # Error calling integral, skip this nuclide
                
                # For Mixture objects, we need to handle them differently
                # Since we can't easily extract the energy spectrum, we'll skip this nuclide
                warnings.warn(f"Nuclide {nuc} has decay photon data but it's a Mixture object without accessible spectrum")
                yield_cache[nuc] = None
                return None
            else:
                # Unknown object type
                warnings.warn(f"Nuclide {nuc} has decay photon data but it's not a recognized spectrum type")
                yield_cache[nuc] = None
                return None
        # If .x and .p attributes exist, proceed as normal
        
        y = np.zeros(24)
        for E_eV, I in zip(spec.x, spec.p):
            g = np.searchsorted(bounds, E_eV * 1e-6, "right") - 1
            # Handle gammas above the highest energy boundary
            if g >= 23: # Index 23 is the last bin
                g = 23
            y[g] += I
        yield_cache[nuc] = y
        return y

    # ----------------------------------------------------------------------
    # get the material id from the results
    material_id = list(results[0].index_mat.keys())[0]
    times_s, activities = results.get_activity(material_id, units="Bq", by_nuclide=True)
    dose_dicts: List[Dict[str, float]] = []

    # ----------------------------------------------------------------------
    for step_idx, act in enumerate(activities):
        #print(f"DEBUG: Processing time step {step_idx}")
        #print(f"DEBUG: Number of active nuclides: {len(act)}")
        #print(f"DEBUG: Nuclides with activity: {list(act.keys())}")
        
        m_total = 0.0
        mass_Z: Dict[int, float] = {}
        for nuc, A_Bq in act.items():
            N = results.get_atoms(material_id, nuc)[1][step_idx]
            m = N * _atomic_mass(nuc) / openmc.data.AVOGADRO
            Z = _atomic_number_from_nuclide(nuc)
            mass_Z[Z] = mass_Z.get(Z, 0.0) + m
            m_total += m
        if m_total == 0.0:
            warnings.warn(f"Material mass is zero at step {step_idx}")
            dose_dicts.append({})
            continue
        mass_frac = {Z: m / m_total for Z, m in mass_Z.items()}

        mu_mat = np.zeros(24)
        for Z, f in mass_frac.items():
            if Z not in mu_elem:
                warnings.warn(f"No μ/ρ data for Z={Z}; ignoring")
                continue
            mu_mat += f * mu_elem[Z]

        # --- DOSE CALCULATIONS ----------------------------------------------
        
        # Calculate dose per nuclide
        nuclide_dose = {}
        for nuc, A_Bq in act.items():
            y = photon_yield(nuc)
            if y is None:
                continue
            # Ensure all array math happens before the final .sum()
            nuclide_dose[nuc] = (_C * (muen_air / mu_mat) * (A_Bq * y) * E_mid).sum()

        # Normalize by mass and structure the output dictionary
        kg_mass = m_total / 1000.0
        dose_dicts.append(
            {k: 3600 * v / kg_mass for k, v in nuclide_dose.items()}
        )

    return times_s, dose_dicts
