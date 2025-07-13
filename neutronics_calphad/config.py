
# neutronics_calphad/config.py

"""
Configuration profiles for different OpenMC models.
"""

# a dictionary of densities for the elements of interest in g/cm3
ELEMENT_DENSITIES = {
    'V': 6.11,
    'Cr': 7.19,
    'Ti': 4.54,
    'W': 19.35,
    'Zr': 6.51,
    'H': 1e-5,
    'Fe': 8.03,
    'Mn': 8.03,
    'Si': 8.03,
    'Ni': 8.03,
    'Mo': 8.03,
    'B': 2.5,
    'C': 2.5,
    'Li': 1.95,
    'F': 1.95,
    'Be': 1.95
}

ARC_D_SHAPE = {
    'geometry': {
        'type': 'd_shape',
        'major_radius': 330,
        'minor_radius': 113,
        'elongation': 1.8,
        'triangularity': 0.5,
        'layers': [
            {'name': 'first_wall', 'thickness': 0.2, 'material': 'tungsten'},
            {'name': 'vacuum_vessel', 'thickness': 2, 'material': 'vcrti'},
            {'name': 'blanket', 'thickness': 20, 'material': 'flibe'},
            {'name': 'tank', 'thickness': 3, 'material': 'steel'},
            {'name': 'shield', 'thickness': 30, 'material': 'boron_carbide'},
        ],
        'shield_y_extent': 600,
        'bounding_box_pad': 50.0,
    },
    'materials': {
        'tungsten': {
            'elements': {'W': 1.0},
            'density': ELEMENT_DENSITIES['W'],
            'depletable': True
        },
        'vcrti': {
            'elements': {'V': 0.9, 'Cr': 0.05, 'Ti': 0.05},
            'density': ELEMENT_DENSITIES['V'],
            'depletable': True
        },
        'flibe': {
            'elements': {'Li': 0.2857, 'F': 0.5714, 'Be': 0.1429},
            'density': ELEMENT_DENSITIES['Li'],
            'percent_type': 'ao',
            'depletable': True
        },
        'steel': {
            'elements': {'Fe': 0.95, 'Cr': 0.18, 'Mn': 0.02, 'Si': 0.01, 'Ni': 0.10, 'Mo': 0.02},
            'density': ELEMENT_DENSITIES['Fe'],
            'percent_type': 'ao',
            'depletable': True
        },
        'boron_carbide': {
            'elements': {'B': 0.8, 'C': 0.2},
            'density': ELEMENT_DENSITIES['B'],
            'percent_type': 'ao',
            'depletable': True
        },
    },
    'source': {
        'type': 'tokamak',
        'parameters': {
            'ion_density_centre': 1.09e20,
            'ion_density_pedestal': 1.09e20,
            'ion_density_peaking_factor': 1,
            'ion_density_separatrix': 3e19,
            'ion_temperature_centre': 45.9e3,
            'ion_temperature_pedestal': 6.09e3,
            'ion_temperature_separatrix': 0.1e3,
            'ion_temperature_peaking_factor': 8.06,
            'ion_temperature_beta': 6,
            'pedestal_radius_factor': 0.8,
            'mode': "H",
            'shafranov_factor': 0.0,
            'angles': (0, 1.5707963267948966), # pi/2
            'sample_seed': 42,
            'fuel': {"D": 0.5, "T": 0.5},
        }
    },
    'settings': {
        'run_mode': "fixed source",
        'batches': 10,
        'inactive': 0,
        'particles': 10000,
    }
}

SPHERICAL = {
    'geometry': {
        'type': 'spherical',
        'radius': 100,
        'layers': [
            {'name': 'first_wall', 'thickness': 2, 'material': 'tungsten'},
            {'name': 'breeder', 'thickness': 50, 'material': 'flibe'},
            {'name': 'vessel', 'thickness': 5, 'material': 'vcrti'},
        ],
        'bounding_box_pad': 10,
    },
    'materials': {
        'tungsten': {
            'elements': {'W': 1.0},
            'density': ELEMENT_DENSITIES['W'],
            'depletable': True
        },
        'flibe': {
            'elements': {'Li': 2, 'F': 4, 'Be': 1},
            'density': ELEMENT_DENSITIES['Li'],
            'percent_type': 'ao',
            'depletable': True
        },
        'vcrti': {
            'elements': {'V': 0.9, 'Cr': 0.05, 'Ti': 0.05},
            'density': ELEMENT_DENSITIES['V'],
            'depletable': True
        },
    },
    'source': {
        'type': 'point',
        'parameters': {
            'coordinate': (0, 0, 0),
            'energy': 14.1e6, # MeV
        }
    },
    'settings': {
        'run_mode': "fixed source",
        'batches': 10,
        'inactive': 0,
        'particles': 10000,
    }
} 