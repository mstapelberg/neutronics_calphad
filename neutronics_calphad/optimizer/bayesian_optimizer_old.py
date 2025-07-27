"""Generic Bayesian optimizer for composition search."""

import numpy as np
from typing import List, Optional, Dict
import warnings

try:
    import torch
    from botorch.acquisition import qLogExpectedImprovement
    from botorch.fit import fit_gpytorch_mll
    from botorch.models import SingleTaskGP, ModelListGP
    from gpytorch.mlls import ExactMarginalLogLikelihood
    from torch.distributions import Normal
    from botorch.acquisition.multi_objective.logei import qLogNoisyExpectedHypervolumeImprovement
    from botorch.optim import optimize_acqf
    from botorch.sampling import SobolQMCNormalSampler
    from botorch.utils.multi_objective.box_decompositions.non_dominated import NondominatedPartitioning
    from botorch.utils.transforms import unnormalize, normalize
    from gpytorch.likelihoods import GaussianLikelihood
    from gpytorch.constraints import Interval
    import time
    BOTORCH_AVAILABLE = True
except ImportError:
    BOTORCH_AVAILABLE = False
    warnings.warn("BoTorch not available. Bayesian optimization features will be disabled.")

from scipy.stats.qmc import LatinHypercube, scale
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel

class CompositionSampler:
    def __init__(self, elements: List[str], batch_size: int = 10,
                 min_compositions: Optional[Dict[str, float]] = None,
                 max_compositions: Optional[Dict[str, float]] = None):
        self.elements = elements
        self.batch_size = batch_size
        self.min_vals = np.array([min_compositions.get(e, 0.0) for e in elements])
        self.max_vals = np.array([max_compositions.get(e, 1.0) for e in elements])
        self.output_names = ['dose_14', 'dose_365', 'dose_3650', 'dose_36500', 'He_appm', 'H_appm']
        self.limits = {'dose_14': 1e3, 'dose_365': 1, 'dose_3650': 1e-2, 'dose_36500': 1e-4, 'He_appm': 1172.2 / 2, 'H_appm': 1500}
        self.X_observed = []
        self.y_observed = []
        self.surrogates = None

    def suggest(self, num_samples: int) -> np.ndarray:
        sampler = LatinHypercube(d=len(self.elements))
        samples = sampler.random(n=num_samples)
        # Scale to min-max constraints (but doesn't enforce sum=1)
        samples = scale(samples, self.min_vals, self.max_vals)
        # Normalize to sum to 1
        samples = samples / samples.sum(axis=1)[:, np.newaxis]
        return samples

    def update(self, compositions: np.ndarray, outputs: np.ndarray) -> None:
        self.X_observed.append(compositions)
        self.y_observed.append(outputs)

    def fit_surrogates(self) -> None:
        X = np.vstack(self.X_observed)
        y = np.vstack(self.y_observed)
        self.surrogates = []
        for i in range(y.shape[1]):
            kernel = ConstantKernel() * RBF()
            gp = GaussianProcessRegressor(kernel=kernel)
            gp.fit(X, y[:, i])
            self.surrogates.append(gp)

    def predict_feasibility(self, X: np.ndarray) -> np.ndarray:
        if self.surrogates is None:
            raise ValueError("Fit surrogates first")
        preds = np.column_stack([gp.predict(X) for gp in self.surrogates])
        feasible = np.all(preds < np.array([self.limits[name] for name in self.output_names]), axis=1)
        return feasible

    def get_all_evaluations(self):
        return np.vstack(self.X_observed), np.vstack(self.y_observed)