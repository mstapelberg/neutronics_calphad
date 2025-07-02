# fusion_opt/evaluate.py
import numpy as np, h5py, json
import os
from pathlib import Path
import warnings

LIB = "elem_lib"
ELMS = ['V','Cr','Ti','W','Zr']
TIMES = np.array([14, 365*5, 365*7, 365*100])  # days

def get_material_by_name(materials, name):
    """Helper function to find a material by name."""
    for material in materials:
        if material.name == name:
            return material
    raise ValueError(f"Material with name '{name}' not found")

# Load critical limits from package directory, with fallback values
try:
    critical_json_path = Path(__file__).parent / "critical.json"
    with open(critical_json_path) as f:
        CRIT = json.load(f)
except FileNotFoundError:
    # Fallback values if critical.json is not found
    CRIT = {"He_appm": 1000, "H_appm": 500}
    print(f"Warning: critical.json not found, using fallback values: {CRIT}")

_library = None

def get_library():
    """Loads the element library on demand and caches it."""
    global _library
    if _library is None:
        _library = {}
        print("Loading element library...")
        for e in ELMS:
            lib_file = f"{LIB}/{e}/{e}.h5"
            if os.path.exists(lib_file):
                _library[e] = h5py.File(lib_file, 'r')
            else:
                # This warning will now only appear if evaluate is called without the library present
                warnings.warn(f"Library file {lib_file} not found. Run 'neutronics-calphad build-library' first.")
    return _library

def evaluate(x):
    """Evaluates the performance of a single alloy composition.

    This function takes a vector of atomic fractions for an alloy and
    calculates key performance metrics based on the pre-computed R2S simulation
    results stored in the element library. It calculates the interpolated dose
    rate and total gas production for the given composition and determines if
    the alloy is 'feasible' based on predefined criteria.

    Args:
        x (numpy.ndarray): A 1D array or list of atomic fractions for the
            constituent elements, in the order specified by `ELMS`.

    Returns:
        dict: A dictionary containing the input composition, calculated dose
        rates, gas production for each isotope, and a boolean 'feasible' flag.
    """
    library = get_library()
    if not library:
        raise RuntimeError("Element library is empty or could not be loaded. Run 'neutronics-calphad build-library' first.")
    
    dose = sum(x[i]*library[ELMS[i]]['dose'][...] for i in range(5))
    gases = {iso: sum(x[i]*library[ELMS[i]][f'gas/{iso}'][...]
                      for i in range(5))
             for iso in ['He3','He4','H1','H2','H3']}
    
    # Calculate total gas production
    total_he = gases['He3'] + gases['He4']
    total_h = gases['H1'] + gases['H2'] + gases['H3']
    
    feasible = (
        (dose[0] <= 1e2) and           # 14 d
        (dose[1] <= 1e-2) and          # 5 y
        (dose[2] <= 1e-2) and          # 7 y
        (dose[3] <= 1e-4) and          # 100 y
        (total_he <= CRIT['He_appm']) and
        (total_h <= CRIT['H_appm'])
    )
    return {"x":x, "dose":dose, **gases, "feasible":feasible}
