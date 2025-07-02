"""
CALPHAD integration module for neutronics-calphad.

This module provides interfaces for:
- Handling elemental depletion results from OpenMC
- Building activation constraint manifolds
- Batch equilibrium calculations using Thermo-Calc
- Bayesian optimization for single-phase region discovery
- Outlier detection and filtering
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
from scipy.spatial import ConvexHull
from sklearn.neighbors import LocalOutlierFactor

# Optional imports with graceful fallback
try:
    import torch
    from botorch.acquisition import qExpectedImprovement
    from botorch.fit import fit_gpytorch_mll
    from botorch.models import SingleTaskGP
    from gpytorch.mlls import ExactMarginalLogLikelihood
    BOTORCH_AVAILABLE = True
except ImportError:
    BOTORCH_AVAILABLE = False
    warnings.warn("BoTorch not available. Bayesian optimization features will be disabled.")

try:
    from tc_python import TCPython, ThermodynamicQuantity
    TC_AVAILABLE = True
except ImportError:
    TC_AVAILABLE = False
    warnings.warn("Thermo-Calc Python API not available. CALPHAD calculations will use stub implementation.")
except Exception:
    # Any other exception during import (like missing license, database issues, etc.)
    TC_AVAILABLE = False
    warnings.warn("Thermo-Calc Python API import failed. CALPHAD calculations will use stub implementation.")

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
    def from_hdf5(cls, filepath: Union[str, Path]) -> ActivationManifold:
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


class CALPHADBatchCalculator:
    """Batch equilibrium calculator using Thermo-Calc or stub implementation."""
    
    def __init__(self, 
                 database: str = "TCHEA7",
                 temperature: float = 823.5,
                 fixed_impurities: Optional[Dict[str, float]] = None):
        """Initialize batch calculator.
        
        Args:
            database: Thermo-Calc database name
            temperature: Calculation temperature in K
            fixed_impurities: Fixed impurity concentrations (atomic fraction)
        """
        self.database = database
        self.temperature = temperature
        # fixed impurities based on NIFS-HEAT 2
        # https://doi.org/10.1016/j.nme.2020.100782
        self.fixed_impurities = fixed_impurities or {
            'C': 290e-6,  # 290 appm
            'N': 440e-6,  # 440 appm
            'O': 470e-6   # 470 appm
        }
        
        if not TC_AVAILABLE:
            logger.warning("Using stub CALPHAD implementation")
            
    def calculate_batch(self, 
                       compositions: np.ndarray,
                       elements: List[str]) -> pd.DataFrame:
        """Calculate equilibrium for batch of compositions.
        
        Args:
            compositions: Array of compositions, shape (n_samples, n_elements)
            elements: List of element symbols (excluding impurities)
            
        Returns:
            DataFrame with columns: x_V, x_Cr, ..., phase_count, dominant_phase, single_phase
        """
        if TC_AVAILABLE:
            try:
                return self._calculate_batch_tc(compositions, elements)
            except Exception as e:
                logger.warning(f"Thermo-Calc calculation failed: {e}. Falling back to stub implementation.")
                return self._calculate_batch_stub(compositions, elements)
        else:
            return self._calculate_batch_stub(compositions, elements)
            
    def _calculate_batch_tc(self, 
                           compositions: np.ndarray,
                           elements: List[str]) -> pd.DataFrame:
        """Thermo-Calc implementation of batch calculation."""
        results = []
        
        # Add impurity elements
        all_elements = elements + list(self.fixed_impurities.keys())
        
        with TCPython() as session:
            # Setup calculation
            calc_setup = (
                session
                .set_cache_folder(f"{self.database}_cache")
                .select_database_and_elements(self.database, all_elements)
                .get_system()
                .with_single_equilibrium_calculation()
                .set_condition("T", self.temperature)
            )
            
            # Set fixed impurity conditions
            for impurity, fraction in self.fixed_impurities.items():
                calc_setup.set_condition(f"X({impurity})", fraction)
                
            # Calculate for each composition
            for comp in compositions:
                # Normalize main elements to account for impurities
                impurity_sum = sum(self.fixed_impurities.values())
                main_sum = 1.0 - impurity_sum
                normalized_comp = comp * main_sum
                
                # Set conditions for main elements (skip the balance element)
                for i in range(1, len(elements)):
                    calc_setup.set_condition(f"X({elements[i]})", normalized_comp[i])
                    
                try:
                    # Calculate equilibrium
                    result = calc_setup.calculate()
                    stable_phases = result.get_stable_phases()
                    
                    # Get phase fractions
                    phase_fractions = {}
                    for phase in stable_phases:
                        fraction = result.get_value_of(
                            ThermodynamicQuantity.mole_fraction_of_a_phase(phase)
                        )
                        phase_fractions[phase] = fraction
                        
                    # Determine dominant phase
                    dominant_phase = max(phase_fractions, key=phase_fractions.get)
                    phase_count = len(stable_phases)
                    single_phase = phase_count == 1
                    
                except Exception as e:
                    logger.warning(f"Calculation failed for composition: {comp}, error: {e}")
                    dominant_phase = "FAILED"
                    phase_count = -1
                    single_phase = False
                    phase_fractions = {}
                    
                # Build result row
                row = {f'x_{el}': comp[i] for i, el in enumerate(elements)}
                row.update({
                    'phase_count': phase_count,
                    'dominant_phase': dominant_phase,
                    'single_phase': single_phase,
                    'phases': json.dumps(phase_fractions)
                })
                results.append(row)
                
        return pd.DataFrame(results)
        
    def _calculate_batch_stub(self, 
                             compositions: np.ndarray,
                             elements: List[str]) -> pd.DataFrame:
        """Stub implementation for testing without Thermo-Calc."""
        results = []
        
        for comp in compositions:
            # Simple heuristic: single phase if largest component > 0.8
            max_fraction = comp.max()
            single_phase = max_fraction > 0.8
            
            if single_phase:
                phase_count = 1
                dominant_phase = "BCC_A2"
            else:
                phase_count = 2
                dominant_phase = "BCC_A2"
                
            row = {f'x_{el}': comp[i] for i, el in enumerate(elements)}
            row.update({
                'phase_count': phase_count,
                'dominant_phase': dominant_phase,
                'single_phase': single_phase,
                'phases': json.dumps({dominant_phase: 0.7, "LAVES": 0.3} if not single_phase else {dominant_phase: 1.0})
            })
            results.append(row)
            
        return pd.DataFrame(results)


class BayesianSearcher:
    """Bayesian optimization for single-phase region discovery."""
    
    def __init__(self,
                 elements: List[str],
                 batch_size: int = 1000,
                 convergence_threshold: float = 0.02,
                 convergence_patience: int = 5):
        """Initialize Bayesian searcher.
        
        Args:
            elements: List of element symbols
            batch_size: Number of compositions per batch
            convergence_threshold: Threshold for new discovery rate
            convergence_patience: Number of consecutive batches below threshold
        """
        if not BOTORCH_AVAILABLE:
            raise ImportError("BoTorch is required for Bayesian optimization")
            
        self.elements = elements
        self.batch_size = batch_size
        self.convergence_threshold = convergence_threshold
        self.convergence_patience = convergence_patience
        
        self.X_observed: Optional[torch.Tensor] = None
        self.y_observed: Optional[torch.Tensor] = None
        self.gp_model: Optional[SingleTaskGP] = None
        
    def suggest_next_batch(self) -> np.ndarray:
        """Suggest next batch of compositions to evaluate.
        
        Returns:
            Array of compositions, shape (batch_size, n_elements)
        """
        if self.X_observed is None:
            # Initial random sampling
            return self._sample_simplex(self.batch_size)
            
        # Fit GP model
        self._fit_gp()
        
        # Use acquisition function to suggest next batch
        candidates = self._optimize_acquisition()
        
        return candidates.numpy()
        
    def update(self, 
               compositions: np.ndarray,
               phase_counts: np.ndarray) -> Dict[str, float]:
        """Update model with new observations.
        
        Args:
            compositions: Evaluated compositions
            phase_counts: Observed phase counts
            
        Returns:
            Dictionary with discovery statistics
        """
        X_new = torch.tensor(compositions, dtype=torch.float32)
        y_new = torch.tensor(phase_counts, dtype=torch.float32).unsqueeze(-1)
        
        if self.X_observed is None:
            self.X_observed = X_new
            self.y_observed = y_new
        else:
            self.X_observed = torch.cat([self.X_observed, X_new])
            self.y_observed = torch.cat([self.y_observed, y_new])
            
        # Calculate discovery statistics
        single_phase_mask = phase_counts == 1
        n_single_phase = single_phase_mask.sum()
        discovery_rate = n_single_phase / len(phase_counts)
        
        return {
            'n_evaluated': len(self.X_observed),
            'n_single_phase_new': int(n_single_phase),
            'discovery_rate': float(discovery_rate)
        }
        
    def _sample_simplex(self, n: int) -> np.ndarray:
        """Sample compositions uniformly from simplex."""
        x = np.random.random((n, len(self.elements)))
        return x / x.sum(axis=1, keepdims=True)
        
    def _fit_gp(self) -> None:
        """Fit Gaussian Process model."""
        self.gp_model = SingleTaskGP(self.X_observed, self.y_observed)
        mll = ExactMarginalLogLikelihood(self.gp_model.likelihood, self.gp_model)
        fit_gpytorch_mll(mll)
        
    def _optimize_acquisition(self) -> torch.Tensor:
        """Optimize acquisition function to get next batch."""
        # Define bounds (simplex constraint handled separately)
        bounds = torch.tensor([[0.0] * len(self.elements),
                              [1.0] * len(self.elements)])
        
        # Use q-Expected Improvement
        acq_func = qExpectedImprovement(
            model=self.gp_model,
            best_f=self.y_observed.min(),
        )
        
        # Generate candidates on simplex
        candidates = []
        while len(candidates) < self.batch_size:
            # Sample from simplex
            x = self._sample_simplex(self.batch_size * 10)
            x_tensor = torch.tensor(x, dtype=torch.float32)
            
            # Evaluate acquisition function
            with torch.no_grad():
                acq_values = acq_func(x_tensor.unsqueeze(1))
                
            # Select top candidates
            top_indices = torch.topk(acq_values, min(self.batch_size - len(candidates), len(acq_values))).indices
            candidates.append(x_tensor[top_indices])
            
        return torch.cat(candidates)[:self.batch_size]
        
    def is_converged(self, history: List[Dict[str, float]]) -> bool:
        """Check if search has converged.
        
        Args:
            history: List of update statistics from previous batches
            
        Returns:
            True if converged
        """
        if len(history) < self.convergence_patience:
            return False
            
        recent_rates = [h['discovery_rate'] for h in history[-self.convergence_patience:]]
        return all(rate < self.convergence_threshold for rate in recent_rates)


class OutlierDetector:
    """Local Outlier Factor based outlier detection."""
    
    def __init__(self, contamination: float = 0.1):
        """Initialize outlier detector.
        
        Args:
            contamination: Expected proportion of outliers
        """
        self.contamination = contamination
        self.lof = LocalOutlierFactor(contamination=contamination)
        
    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """Detect outliers in composition data.
        
        Args:
            X: Composition array, shape (n_samples, n_features)
            
        Returns:
            Binary mask where True indicates inliers
        """
        # LOF returns -1 for outliers, 1 for inliers
        predictions = self.lof.fit_predict(X)
        return predictions == 1
        
    def filter_outliers(self, 
                       data: pd.DataFrame,
                       feature_cols: List[str],
                       keep_outliers: bool = True) -> pd.DataFrame:
        """Filter outliers from dataframe.
        
        Args:
            data: DataFrame containing composition and property data
            feature_cols: Column names to use for outlier detection
            keep_outliers: If True, mark outliers but keep them
            
        Returns:
            Filtered DataFrame with optional 'is_outlier' column
        """
        X = data[feature_cols].values
        inlier_mask = self.fit_predict(X)
        
        if keep_outliers:
            data = data.copy()
            data['is_outlier'] = ~inlier_mask
            return data
        else:
            return data[inlier_mask].copy() 