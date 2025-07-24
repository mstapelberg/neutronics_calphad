import numpy as np
import pandas as pd
from typing import List, Dict, Optional
import warnings

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