"""Generic Bayesian optimizer for composition search."""

import numpy as np
from typing import List, Optional, Dict, Union
import warnings

try:
    import torch
    from botorch.acquisition import qLogExpectedImprovement
    from botorch.fit import fit_gpytorch_mll
    from botorch.models import SingleTaskGP
    from gpytorch.mlls import ExactMarginalLogLikelihood
    BOTORCH_AVAILABLE = True
except ImportError:
    BOTORCH_AVAILABLE = False
    warnings.warn("BoTorch not available. Bayesian optimization features will be disabled.")

class BayesianOptimizer:
    """Generic Bayesian optimizer for composition search.
    
    Parameters
    ----------
    elements : List[str]
        List of element symbols.
    batch_size : int, optional
        Number of compositions to suggest per iteration, by default 10.
    minimize : bool, optional
        Whether to minimize the objective (True) or maximize (False), by default False.
    min_compositions : Dict[str, float], optional
        Minimum allowed fraction for each element. If not specified, defaults to 0.0.
    max_compositions : Dict[str, float], optional
        Maximum allowed fraction for each element. If not specified, defaults to 1.0.
    """

    def __init__(self, elements: List[str], batch_size: int = 10, minimize: bool = False,
                 min_compositions: Optional[Dict[str, float]] = None,
                 max_compositions: Optional[Dict[str, float]] = None):
        if not BOTORCH_AVAILABLE:
            raise ImportError("BoTorch is required for Bayesian optimization")
        self.elements = elements
        self.batch_size = batch_size
        self.minimize = minimize
        
        # Set up composition constraints
        self.min_compositions = min_compositions or {}
        self.max_compositions = max_compositions or {}
        
        # Validate constraints
        self._validate_constraints()
        
        self.X_observed: Optional[torch.Tensor] = None
        self.y_observed: Optional[torch.Tensor] = None
        self.gp_model: Optional[SingleTaskGP] = None

    def suggest(self) -> np.ndarray:
        """Suggest next batch of compositions."""
        if self.X_observed is None:
            return self._sample_simplex(self.batch_size)
        self._fit_gp()
        candidates = self._optimize_acquisition()
        return candidates.numpy()

    def update(self, compositions: np.ndarray, scores: np.ndarray) -> None:
        X_new = torch.tensor(compositions, dtype=torch.float64)
        y_new = torch.tensor(scores, dtype=torch.float64).unsqueeze(-1)
        if self.minimize:
            y_new = -y_new
        if self.X_observed is None:
            self.X_observed = X_new
            self.y_observed = y_new
        else:
            self.X_observed = torch.cat([self.X_observed, X_new])
            self.y_observed = torch.cat([self.y_observed, y_new])

    def _validate_constraints(self) -> None:
        """Validate that composition constraints are feasible."""
        # Check that all constraint elements are in the elements list
        for element in self.min_compositions.keys():
            if element not in self.elements:
                raise ValueError(f"Min constraint element '{element}' not in elements list")
        for element in self.max_compositions.keys():
            if element not in self.elements:
                raise ValueError(f"Max constraint element '{element}' not in elements list")
        
        # Check that min <= max for each element
        for element in self.elements:
            min_val = self.min_compositions.get(element, 0.0)
            max_val = self.max_compositions.get(element, 1.0)
            if min_val > max_val:
                raise ValueError(f"Min constraint ({min_val}) > max constraint ({max_val}) for element '{element}'")
        
        # Check that the constraints allow for feasible compositions (sum can equal 1)
        total_min = sum(self.min_compositions.get(e, 0.0) for e in self.elements)
        total_max = sum(self.max_compositions.get(e, 1.0) for e in self.elements)
        
        if total_min > 1.0:
            raise ValueError(f"Sum of minimum constraints ({total_min:.3f}) > 1.0 - infeasible")
        if total_max < 1.0:
            raise ValueError(f"Sum of maximum constraints ({total_max:.3f}) < 1.0 - infeasible")

    def _sample_simplex(self, n: int) -> np.ndarray:
        """Sample compositions from simplex with box constraints."""
        compositions = []
        max_attempts = 10000  # Prevent infinite loops
        
        for _ in range(n):
            attempts = 0
            while attempts < max_attempts:
                # Generate random composition
                x = np.random.random(len(self.elements))
                x = x / x.sum()  # Normalize to sum to 1
                
                # Check if it satisfies constraints
                valid = True
                for i, element in enumerate(self.elements):
                    min_val = self.min_compositions.get(element, 0.0)
                    max_val = self.max_compositions.get(element, 1.0)
                    if x[i] < min_val or x[i] > max_val:
                        valid = False
                        break
                
                if valid:
                    compositions.append(x)
                    break
                    
                attempts += 1
            
            if attempts >= max_attempts:
                # Fallback: use projection method
                x = self._project_to_constraints()
                compositions.append(x)
        
        return np.array(compositions)
    
    def _project_to_constraints(self) -> np.ndarray:
        """Project a composition to satisfy constraints using a simple heuristic."""
        # Start with minimum values
        x = np.array([self.min_compositions.get(e, 0.0) for e in self.elements])
        
        # Distribute remaining mass proportionally within max constraints
        remaining = 1.0 - x.sum()
        if remaining > 0:
            # Calculate available capacity for each element
            capacity = np.array([self.max_compositions.get(e, 1.0) - x[i] 
                               for i, e in enumerate(self.elements)])
            
            # Distribute remaining mass proportionally to capacity
            if capacity.sum() > 0:
                distribution = capacity / capacity.sum() * remaining
                x += distribution
                
                # Ensure we don't exceed max constraints
                for i, element in enumerate(self.elements):
                    max_val = self.max_compositions.get(element, 1.0)
                    if x[i] > max_val:
                        excess = x[i] - max_val
                        x[i] = max_val
                        # Redistribute excess to other elements (simple approach)
                        remaining_indices = [j for j in range(len(x)) if j != i and x[j] < self.max_compositions.get(self.elements[j], 1.0)]
                        if remaining_indices:
                            for j in remaining_indices:
                                addition = excess / len(remaining_indices)
                                max_addition = self.max_compositions.get(self.elements[j], 1.0) - x[j]
                                x[j] += min(addition, max_addition)
        
        # Final normalization to ensure sum = 1
        x = x / x.sum()
        
        return x

    def _fit_gp(self) -> None:
        self.gp_model = SingleTaskGP(self.X_observed, self.y_observed)
        mll = ExactMarginalLogLikelihood(self.gp_model.likelihood, self.gp_model)
        fit_gpytorch_mll(mll)

    def _optimize_acquisition(self) -> torch.Tensor:
        """Optimize acquisition function while respecting composition constraints."""
        acq_func = qLogExpectedImprovement(model=self.gp_model, best_f=self.y_observed.min())
        candidates = []
        
        while len(candidates) < self.batch_size:
            # Sample more candidates to account for constraint filtering
            x = self._sample_simplex(self.batch_size * 20)  # Increased sampling
            x_t = torch.tensor(x, dtype=torch.float64)
            
            with torch.no_grad():
                acq_values = acq_func(x_t.unsqueeze(1))
            
            # Get top candidates that satisfy constraints
            top_indices = torch.topk(acq_values.squeeze(-1), len(acq_values)).indices
            
            # Add candidates up to batch size
            needed = self.batch_size - len(candidates)
            candidates.append(x_t[top_indices[:needed]])
            
        return torch.cat(candidates)[:self.batch_size]