import numpy as np
from typing import List, Optional
import warnings

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

class BayesianOptimizer:
    """Generic Bayesian optimizer for composition search."""

    def __init__(self, elements: List[str], batch_size: int = 10, minimize: bool = False):
        if not BOTORCH_AVAILABLE:
            raise ImportError("BoTorch is required for Bayesian optimization")
        self.elements = elements
        self.batch_size = batch_size
        self.minimize = minimize
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
        X_new = torch.tensor(compositions, dtype=torch.float32)
        y_new = torch.tensor(scores, dtype=torch.float32).unsqueeze(-1)
        if self.minimize:
            y_new = -y_new
        if self.X_observed is None:
            self.X_observed = X_new
            self.y_observed = y_new
        else:
            self.X_observed = torch.cat([self.X_observed, X_new])
            self.y_observed = torch.cat([self.y_observed, y_new])

    def _sample_simplex(self, n: int) -> np.ndarray:
        x = np.random.random((n, len(self.elements)))
        return x / x.sum(axis=1, keepdims=True)

    def _fit_gp(self) -> None:
        self.gp_model = SingleTaskGP(self.X_observed, self.y_observed)
        mll = ExactMarginalLogLikelihood(self.gp_model.likelihood, self.gp_model)
        fit_gpytorch_mll(mll)

    def _optimize_acquisition(self) -> torch.Tensor:
        acq_func = qExpectedImprovement(model=self.gp_model, best_f=self.y_observed.min())
        candidates = []
        while len(candidates) < self.batch_size:
            x = self._sample_simplex(self.batch_size * 10)
            x_t = torch.tensor(x, dtype=torch.float32)
            with torch.no_grad():
                acq_values = acq_func(x_t.unsqueeze(1))
            top = torch.topk(acq_values.squeeze(-1), min(self.batch_size - len(candidates), len(acq_values))).indices
            candidates.append(x_t[top])
        return torch.cat(candidates)[:self.batch_size]
