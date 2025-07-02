"""
Pytest configuration and fixtures for neutronics_calphad tests.
"""

import tempfile
from pathlib import Path
from typing import Dict

import h5py
import numpy as np
import pytest

from neutronics_calphad.calphad import DepletionResult, ActivationConstraints


@pytest.fixture
def sample_dose_times():
    """Sample dose times array in seconds."""
    return np.array([
        1,                    # 1 second
        3600,                 # 1 hour
        10*3600,              # 10 hours
        24*3600,              # 1 day
        7*24*3600,            # 1 week
        14*24*3600,           # 2 weeks
        30*24*3600,           # 1 month
        365*24*3600,          # 1 year
        5*365*24*3600,        # 5 years
        10*365*24*3600,       # 10 years
        100*365*24*3600       # 100 years
    ])


@pytest.fixture
def sample_dose_rates():
    """Sample dose rates array in µSv/h."""
    return np.array([
        1e10,  # 1 second
        1e9,   # 1 hour
        1e8,   # 10 hours
        1e7,   # 1 day
        1e6,   # 1 week
        1e5,   # 2 weeks
        1e4,   # 1 month
        1e3,   # 1 year
        1e2,   # 5 years
        1e1,   # 10 years
        1e0    # 100 years
    ])


@pytest.fixture
def sample_gas_production():
    """Sample gas production dictionary."""
    return {
        'He3': 1.5e15,
        'He4': 8.2e16,
        'H1': 3.1e15,
        'H2': 2.4e14,
        'H3': 1.7e13
    }


@pytest.fixture
def sample_depletion_result(sample_dose_times, sample_dose_rates, sample_gas_production):
    """Sample DepletionResult for Vanadium."""
    return DepletionResult(
        element='V',
        dose_times=sample_dose_times,
        dose_rates=sample_dose_rates,
        gas_production=sample_gas_production,
        volume=2.13e5,
        density=6.11
    )


@pytest.fixture
def sample_depletion_results(sample_dose_times, sample_gas_production):
    """Dictionary of sample depletion results for multiple elements."""
    elements = ['V', 'Cr', 'Ti', 'W', 'Zr']
    results = {}
    
    for i, element in enumerate(elements):
        # Create different dose rate profiles for each element
        dose_rates = np.array([
            1e10 * (0.5 + 0.1 * i),  # 1 second
            1e9 * (0.5 + 0.1 * i),   # 1 hour
            1e8 * (0.5 + 0.1 * i),   # 10 hours
            1e7 * (0.5 + 0.1 * i),   # 1 day
            1e6 * (0.5 + 0.1 * i),   # 1 week
            1e5 * (0.5 + 0.1 * i),   # 2 weeks
            1e4 * (0.5 + 0.1 * i),   # 1 month
            1e3 * (0.5 + 0.1 * i),   # 1 year
            1e2 * (0.5 + 0.1 * i),   # 5 years
            1e1 * (0.5 + 0.1 * i),   # 10 years
            1e0 * (0.5 + 0.1 * i)    # 100 years
        ])
        
        # Scale gas production by element
        gas_prod = {k: v * (0.5 + 0.2 * i) for k, v in sample_gas_production.items()}
        
        results[element] = DepletionResult(
            element=element,
            dose_times=sample_dose_times,
            dose_rates=dose_rates,
            gas_production=gas_prod,
            volume=2.13e5,
            density=6.11 + i * 0.5  # Different densities
        )
    
    return results


@pytest.fixture
def temp_hdf5_file():
    """Temporary HDF5 file for testing."""
    with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as f:
        yield Path(f.name)
    Path(f.name).unlink(missing_ok=True)


@pytest.fixture
def sample_hdf5_file(temp_hdf5_file, sample_dose_times, sample_dose_rates, sample_gas_production):
    """Create a sample HDF5 file with depletion data."""
    with h5py.File(temp_hdf5_file, 'w') as f:
        f.create_dataset('dose_times', data=sample_dose_times)
        f.create_dataset('dose', data=sample_dose_rates)
        for gas, atoms in sample_gas_production.items():
            f.create_dataset(f'gas/{gas}', data=atoms)
        f.attrs['element'] = 'V'
        f.attrs['volume'] = 2.13e5
        f.attrs['density'] = 6.11
    
    return temp_hdf5_file


@pytest.fixture
def sample_compositions():
    """Sample composition array for 5 elements."""
    np.random.seed(42)  # For reproducible tests
    n_samples = 100
    n_elements = 5
    
    # Generate random compositions on simplex
    x = np.random.random((n_samples, n_elements))
    compositions = x / x.sum(axis=1, keepdims=True)
    
    return compositions


@pytest.fixture
def activation_constraints():
    """Sample activation constraints."""
    return ActivationConstraints(
        dose_limits={
            '14d': 1e2,
            '5y': 1e-2,
            '7y': 1e-2,
            '100y': 1e-4
        },
        gas_limits={
            'He_appm': 1000,
            'H_appm': 500
        }
    )


@pytest.fixture
def sample_phase_data():
    """Sample phase calculation results."""
    np.random.seed(42)
    n_samples = 50
    elements = ['V', 'Cr', 'Ti', 'W', 'Zr']
    
    # Generate compositions
    x = np.random.random((n_samples, len(elements)))
    compositions = x / x.sum(axis=1, keepdims=True)
    
    # Generate phase data (simple heuristic)
    phase_counts = []
    dominant_phases = []
    single_phases = []
    
    for comp in compositions:
        max_frac = comp.max()
        if max_frac > 0.8:
            phase_count = 1
            dominant_phase = "BCC_A2"
            single_phase = True
        elif max_frac > 0.6:
            phase_count = 2
            dominant_phase = "BCC_A2"
            single_phase = False
        else:
            phase_count = 3
            dominant_phase = "BCC_A2"
            single_phase = False
            
        phase_counts.append(phase_count)
        dominant_phases.append(dominant_phase)
        single_phases.append(single_phase)
    
    return {
        'compositions': compositions,
        'phase_counts': np.array(phase_counts),
        'dominant_phases': dominant_phases,
        'single_phases': np.array(single_phases),
        'elements': elements
    } 