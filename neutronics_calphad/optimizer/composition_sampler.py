"""Simple composition sampler using Latin Hypercube Sampling and surrogate modeling."""

from typing import List, Optional, Dict
import numpy as np
from scipy.stats.qmc import LatinHypercube, scale
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel


class CompositionSampler:
    """Simple composition sampler using Latin Hypercube Sampling."""
    
    def __init__(self, elements: List[str], batch_size: int = 10,
                 min_compositions: Optional[Dict[str, float]] = None,
                 max_compositions: Optional[Dict[str, float]] = None):
        """Initialize the composition sampler.
        
        Parameters
        ----------
        elements : List[str]
            List of element symbols.
        batch_size : int, optional
            Batch size for sampling, by default 10.
        min_compositions : Dict[str, float], optional
            Minimum allowed fraction for each element.
        max_compositions : Dict[str, float], optional
            Maximum allowed fraction for each element.
        """
        self.elements = elements
        self.batch_size = batch_size
        self.min_vals = np.array([min_compositions.get(e, 0.0) for e in elements])
        self.max_vals = np.array([max_compositions.get(e, 1.0) for e in elements])
        self.output_names = ['dose_14', 'dose_365', 'dose_3650', 'dose_36500', 'He_appm', 'H_appm']
        self.limits = {
            'dose_14': 1e3, 
            'dose_365': 1, 
            'dose_3650': 1e-2, 
            'dose_36500': 1e-4, 
            'He_appm': 1172.2 / 2, 
            'H_appm': 1500
        }
        self.X_observed = []
        self.y_observed = []
        self.surrogates = None

    def suggest(self, num_samples: int) -> np.ndarray:
        """Suggest compositions using Latin Hypercube Sampling.
        
        Parameters
        ----------
        num_samples : int
            Number of compositions to generate.
            
        Returns
        -------
        np.ndarray
            Array of compositions, shape (num_samples, n_elements).
        """
        sampler = LatinHypercube(d=len(self.elements))
        samples = sampler.random(n=num_samples)
        # Scale to min-max constraints (but doesn't enforce sum=1)
        samples = scale(samples, self.min_vals, self.max_vals)
        # Normalize to sum to 1
        samples = samples / samples.sum(axis=1)[:, np.newaxis]
        return samples

    def update(self, compositions: np.ndarray, outputs: np.ndarray) -> None:
        """Update the sampler with new evaluation results.
        
        Parameters
        ----------
        compositions : np.ndarray
            Array of compositions that were evaluated.
        outputs : np.ndarray
            Array of outputs from the evaluations.
        """
        self.X_observed.append(compositions)
        self.y_observed.append(outputs)

    def fit_surrogates(self) -> None:
        """Fit surrogate models for all outputs."""
        if not self.X_observed:
            raise ValueError("No data available for fitting surrogates")
            
        X = np.vstack(self.X_observed)
        y = np.vstack(self.y_observed)
        self.surrogates = []
        
        for i in range(y.shape[1]):
            kernel = ConstantKernel() * RBF()
            gp = GaussianProcessRegressor(kernel=kernel, random_state=42)
            gp.fit(X, y[:, i])
            self.surrogates.append(gp)
        
        print(f"Fitted {len(self.surrogates)} surrogate models")

    def predict_feasibility(self, X: np.ndarray) -> np.ndarray:
        """Predict which compositions are feasible (satisfy all constraints).
        
        Parameters
        ----------
        X : np.ndarray
            Compositions to evaluate.
            
        Returns
        -------
        np.ndarray
            Boolean array indicating which compositions are feasible.
        """
        if self.surrogates is None:
            raise ValueError("Fit surrogates first")
            
        preds = np.column_stack([gp.predict(X) for gp in self.surrogates])
        feasible = np.all(preds < np.array([self.limits[name] for name in self.output_names]), axis=1)
        return feasible

    def get_all_evaluations(self):
        """Get all evaluated compositions and outputs.
        
        Returns
        -------
        tuple
            (compositions, outputs) arrays.
        """
        if not self.X_observed:
            return np.array([]), np.array([])
        return np.vstack(self.X_observed), np.vstack(self.y_observed)