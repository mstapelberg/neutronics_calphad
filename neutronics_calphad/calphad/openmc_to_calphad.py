""" 
This module focues on reading and converting openmc files to calphad ready materials. 
"""
from __future__ import annotations

import json
import logging
import os
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import h5py
import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)

@dataclass
class DepletionResult:
    """Container for elemental depletion results.
    
    Attributes:
        element: Chemical symbol of the element
        dose_times: Array of cooling times in seconds
        dose_rates: Array of dose rates in µSv/h
        gas_production: Dictionary of gas isotope production (atoms)
        volume: Material volume in cm³
        density: Material density in g/cm³
    """
    element: str
    dose_times: np.ndarray
    dose_rates: np.ndarray
    gas_production: Dict[str, float]
    volume: float
    density: float
    
    @classmethod
    def from_hdf5(cls, filepath: Union[str, Path]) -> DepletionResult:
        """Load depletion results from HDF5 file.
        
        Args:
            filepath: Path to HDF5 file containing depletion results
            
        Returns:
            DepletionResult instance
        """
        filepath = Path(filepath)
        
        with h5py.File(filepath, 'r') as f:
            dose_times = f['dose_times'][:]
            dose_rates = f['dose'][:]
            
            # Get element from attributes, fallback to filename
            if 'element' in f.attrs:
                element = f.attrs['element']
                if isinstance(element, bytes):
                    element = element.decode('utf-8')
            else:
                element = filepath.stem
            
            # Get volume and density from attributes if available
            volume = float(f.attrs.get('volume', 2.13e5))
            density = float(f.attrs.get('density', 1.0))
            
            gas_production = {}
            for gas in ['He3', 'He4', 'H1', 'H2', 'H3']:
                if f'gas/{gas}' in f:
                    gas_production[gas] = float(f[f'gas/{gas}'][()])
                else:
                    gas_production[gas] = 0.0
        
        return cls(
            element=element,
            dose_times=dose_times,
            dose_rates=dose_rates,
            gas_production=gas_production,
            volume=volume,
            density=density
        )
    
    def to_hdf5(self, filepath: Union[str, Path]) -> None:
        """Save depletion results to HDF5 file.
        
        Args:
            filepath: Output HDF5 file path
        """
        with h5py.File(filepath, 'w') as f:
            f.create_dataset('dose_times', data=self.dose_times)
            f.create_dataset('dose', data=self.dose_rates)
            for gas, atoms in self.gas_production.items():
                f.create_dataset(f'gas/{gas}', data=atoms)
            f.attrs['element'] = self.element
            f.attrs['volume'] = self.volume
            f.attrs['density'] = self.density


def mix(results: Dict[str, DepletionResult], weights: List[float]) -> DepletionResult:
    """Calculate weight-averaged activation for an arbitrary alloy.
    
    Args:
        results: Dictionary mapping element symbols to their depletion results
        weights: List of atomic fractions for each element (must sum to 1)
        
    Returns:
        Mixed depletion result for the alloy composition
        
    Raises:
        ValueError: If weights don't sum to 1 or element count mismatch
    """
    elements = list(results.keys())
    weights = np.array(weights)
    
    if len(weights) != len(elements):
        raise ValueError(f"Number of weights ({len(weights)}) must match number of elements ({len(elements)})")
    
    if not np.allclose(weights.sum(), 1.0):
        raise ValueError(f"Weights must sum to 1.0, got {weights.sum()}")
    
    # Check time consistency
    first_result = next(iter(results.values()))
    dose_times = first_result.dose_times
    for element, result in results.items():
        if not np.array_equal(result.dose_times, dose_times):
            raise ValueError(f"Inconsistent time steps between elements")
    
    # Mix dose rates
    mixed_dose_rates = np.zeros_like(first_result.dose_rates)
    for i, (element, result) in enumerate(results.items()):
        mixed_dose_rates += weights[i] * result.dose_rates
    
    # Mix gas production
    mixed_gas_production = {}
    gas_species = ['He3', 'He4', 'H1', 'H2', 'H3']
    for gas in gas_species:
        mixed_gas_production[gas] = sum(
            weights[i] * result.gas_production.get(gas, 0.0)
            for i, result in enumerate(results.values())
        )
    
    # Calculate mixed properties
    mixed_volume = sum(
        weights[i] * result.volume
        for i, result in enumerate(results.values())
    )
    mixed_density = sum(
        weights[i] * result.density
        for i, result in enumerate(results.values())
    )
    
    return DepletionResult(
        element=f"Alloy({','.join(f'{e}_{w:.3f}' for e, w in zip(elements, weights))})",
        dose_times=dose_times,
        dose_rates=mixed_dose_rates,
        gas_production=mixed_gas_production,
        volume=mixed_volume,
        density=mixed_density
    )
