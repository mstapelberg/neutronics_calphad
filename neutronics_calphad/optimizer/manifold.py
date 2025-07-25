# fusion_opt/manifold.py
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from typing import List, Dict, Optional, Union, Any
from pathlib import Path
import h5py
import json
from datetime import datetime
from scipy.spatial import ConvexHull
from .evaluate_updated import evaluate_material
from dataclasses import dataclass, field

from neutronics_calphad.calphad.openmc_to_calphad import DepletionResult
from neutronics_calphad.calphad.openmc_to_calphad import mix

import logging
logger = logging.getLogger(__name__)

@dataclass
class ActivationConstraints:
    """Constraints for activation manifold construction.
    
    Attributes:
        dose_limits: Dictionary mapping time keys to dose rate limits (µSv/h)
        gas_limits: Dictionary mapping gas species to production limits (appm)
    """
    dose_limits: Dict[str, float] = field(default_factory=lambda: {
        '14d': 1e2,     # 14 days
        '5y': 1e-2,     # 5 years
        '7y': 1e-2,     # 7 years
        '100y': 1e-4    # 100 years
    })
    gas_limits: Dict[str, float] = field(default_factory=lambda: {
        'He_appm': 1000,
        'H_appm': 500
    })


class ActivationManifold:
    """Convex polytope representing feasible alloy compositions.
    
    This class builds and manages a manifold of alloy compositions that
    satisfy user-defined constraints on dose rate and gas production.
    """
    
    def __init__(self, 
                 elements: List[str],
                 constraints: Optional[ActivationConstraints] = None):
        """Initialize activation manifold.
        
        Args:
            elements: List of element symbols (e.g., ['V', 'Cr', 'Ti', 'W', 'Zr'])
            constraints: Activation constraints, uses defaults if None
        """
        self.elements = elements
        self.constraints = constraints or ActivationConstraints()
        self.feasible_compositions: Optional[np.ndarray] = None
        self.convex_hull: Optional[ConvexHull] = None
        self._metadata: Dict[str, Any] = {}
        
    def build_from_samples(self, 
                          samples: np.ndarray,
                          depletion_results: Dict[str, DepletionResult]) -> None:
        """Build manifold from sampled compositions.
        
        Args:
            samples: Array of composition samples, shape (n_samples, n_elements)
            depletion_results: Dictionary of elemental depletion results
        """
        feasible_mask = np.zeros(len(samples), dtype=bool)
        
        for i, composition in enumerate(samples):
            # Mix depletion results for this composition
            mixed_result = mix(depletion_results, composition.tolist())
            
            # Check dose constraints
            dose_ok = True
            for time_key, limit in self.constraints.dose_limits.items():
                time_idx = self._get_time_index(mixed_result.dose_times, time_key)
                if time_idx is not None and mixed_result.dose_rates[time_idx] > limit:
                    dose_ok = False
                    break
                    
            # Check gas constraints
            gas_ok = True
            total_he = mixed_result.gas_production['He3'] + mixed_result.gas_production['He4']
            total_h = sum(mixed_result.gas_production[f'H{i}'] for i in [1, 2, 3])
            
            if total_he > self.constraints.gas_limits['He_appm']:
                gas_ok = False
            if total_h > self.constraints.gas_limits['H_appm']:
                gas_ok = False
                
            feasible_mask[i] = dose_ok and gas_ok
            
        self.feasible_compositions = samples[feasible_mask]
        
        # Build convex hull if we have enough points
        if len(self.feasible_compositions) >= len(self.elements) + 1:
            try:
                self.convex_hull = ConvexHull(self.feasible_compositions)
            except Exception as e:
                # ConvexHull can fail if points are coplanar or degenerate
                logger.warning(f"Failed to build ConvexHull: {e}. Using None for convex_hull.")
                self.convex_hull = None
            
        # Store metadata
        self._metadata = {
            'n_samples': len(samples),
            'n_feasible': len(self.feasible_compositions),
            'feasibility_rate': len(self.feasible_compositions) / len(samples),
            'timestamp': datetime.now().isoformat(),
            'elements': self.elements,
            'constraints': {
                'dose_limits': self.constraints.dose_limits,
                'gas_limits': self.constraints.gas_limits
            }
        }
        
    def _get_time_index(self, times: np.ndarray, time_key: str) -> Optional[int]:
        """Get index for a time key like '14d', '5y', etc."""
        time_map = {
            '14d': 14 * 24 * 3600,
            '5y': 5 * 365.25 * 24 * 3600,
            '7y': 7 * 365.25 * 24 * 3600,
            '100y': 100 * 365.25 * 24 * 3600
        }
        
        if time_key not in time_map:
            return None
            
        target_time = time_map[time_key]
        # Find closest time
        idx = np.argmin(np.abs(times - target_time))
        return idx
        
    def contains(self, composition: np.ndarray) -> bool:
        """Check if a composition is within the feasible manifold.
        
        Args:
            composition: Array of atomic fractions
            
        Returns:
            True if composition is feasible
        """
        if self.convex_hull is None:
            return False
            
        # Use Delaunay triangulation from ConvexHull
        return self.convex_hull.find_simplex(composition) >= 0
        
    def to_hdf5(self, filepath: Union[str, Path]) -> None:
        """Save manifold to HDF5 file with metadata.
        
        Args:
            filepath: Output HDF5 file path
        """
        with h5py.File(filepath, 'w') as f:
            # Save feasible compositions
            if self.feasible_compositions is not None:
                f.create_dataset('feasible_compositions', data=self.feasible_compositions)
                
            # Save metadata
            meta_group = f.create_group('metadata')
            meta_group.attrs['timestamp'] = self._metadata.get('timestamp', '')
            meta_group.attrs['n_samples'] = self._metadata.get('n_samples', 0)
            meta_group.attrs['n_feasible'] = self._metadata.get('n_feasible', 0)
            meta_group.attrs['feasibility_rate'] = self._metadata.get('feasibility_rate', 0.0)
            meta_group.attrs['elements'] = json.dumps(self.elements)
            
            # Save constraints
            constraints_group = f.create_group('constraints')
            constraints_group.attrs['dose_limits'] = json.dumps(self.constraints.dose_limits)
            constraints_group.attrs['gas_limits'] = json.dumps(self.constraints.gas_limits)
            
            # Try to get git hash
            try:
                import subprocess
                git_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode().strip()
                meta_group.attrs['git_hash'] = git_hash
            except:
                pass
                
    @classmethod
    def from_hdf5(cls, filepath: Union[str, Path]) -> "ActivationManifold":
        """Load manifold from HDF5 file.
        
        Args:
            filepath: HDF5 file path
            
        Returns:
            ActivationManifold instance
        """
        with h5py.File(filepath, 'r') as f:
            elements = json.loads(f['metadata'].attrs['elements'])
            
            # Load constraints
            constraints = ActivationConstraints()
            if 'constraints' in f:
                constraints.dose_limits = json.loads(f['constraints'].attrs['dose_limits'])
                constraints.gas_limits = json.loads(f['constraints'].attrs['gas_limits'])
                
            manifold = cls(elements, constraints)
            
            # Load feasible compositions
            if 'feasible_compositions' in f:
                manifold.feasible_compositions = f['feasible_compositions'][:]
                if len(manifold.feasible_compositions) >= len(elements) + 1:
                    try:
                        manifold.convex_hull = ConvexHull(manifold.feasible_compositions)
                    except Exception as e:
                        logger.warning(f"Failed to rebuild ConvexHull from HDF5: {e}")
                        manifold.convex_hull = None
                    
            # Load metadata
            meta = f['metadata']
            manifold._metadata = {
                'timestamp': meta.attrs.get('timestamp', ''),
                'n_samples': meta.attrs.get('n_samples', 0),
                'n_feasible': meta.attrs.get('n_feasible', 0),
                'feasibility_rate': meta.attrs.get('feasibility_rate', 0.0),
                'git_hash': meta.attrs.get('git_hash', '')
            }
            
        return manifold

def sample_simplex(n, seed=1):
    """Generates random compositions on a 5-dimensional simplex.

    This function creates `n` random alloy compositions, where each composition
    is a vector of 5 atomic fractions that sum to 1. This is used to sample
    the design space for the V-Cr-Ti-W-Zr alloy system.

    Args:
        n (int): The number of random compositions to generate.
        seed (int, optional): A seed for the random number generator to ensure
            reproducibility. Defaults to 1.

    Returns:
        numpy.ndarray: A 2D array of shape (n, 5) where each row is a valid
            atomic composition.
    """
    rng=np.random.default_rng(seed)
    x = rng.random((n,5))
    return x/ x.sum(axis=1)[:,None]

def build_manifold(n=15000, workers=8):
    """Builds a dataset of alloy performance by sampling the design space.

    This function orchestrates the evaluation of a large number of random alloy
    compositions. It first generates the compositions using `sample_simplex`,
    then uses a process pool to evaluate the performance of each composition
    in parallel using the `evaluate` function. The results are compiled into a
    Pandas DataFrame and saved to a Parquet file.

    Args:
        n (int, optional): The total number of alloy compositions to sample and
            evaluate. Defaults to 15000.
        workers (int, optional): The number of parallel worker processes to use
            for evaluation. Defaults to 8.

    Returns:
        pandas.DataFrame: A DataFrame containing the evaluation results for
            each sampled composition.
    """
    comp = sample_simplex(n)
    with ProcessPoolExecutor(workers) as ex:
        rows = list(ex.map(evaluate_material, comp))
    df = pd.DataFrame(rows)
    df.to_parquet("manifold.parquet", compression="zstd")
    return df
