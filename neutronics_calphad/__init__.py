"""
Neutronics CALPHAD: Neutronics simulations for CALPHAD-based alloy optimization.

This package provides tools for running neutronics simulations of tokamak
geometries with different materials, performing activation analysis, and
optimizing alloy compositions based on nuclear performance criteria.

Main API is exposed from the following submodules:
- neutronics: core simulation, geometry, depletion, dose, config, etc.
- calphad: CALPHAD integration, alloy mixing, constraints, optimization
- utils: utilities, visualization, I/O, manifold sampling
- optimizer: evaluation and critical limits for alloy design
"""

__version__ = "0.1.0"

# --- Neutronics core API ---
from .neutronics.geometry_maker import create_model, plot_model
from .neutronics.library import run_element, build_library
from .neutronics.config import ARC_D_SHAPE, SPHERICAL
from .neutronics.time_scheduler import TimeScheduler
from .neutronics.dose import contact_dose

# --- CALPHAD API ---
# --- Utilities and Visualization ---
from .utils.visualization import (
    plot_dose_rate_vs_time,
    plot_fispact_comparison,
    plot_umap,
    plot_fispact_flux,
)

# --- Optimizer API ---
from .optimizer.evaluate import evaluate
from .optimizer.manifold import sample_simplex, build_manifold
from .optimizer.bayesian_optimizer import BayesianOptimizer

# --- Expose common constants (if needed) ---
#from .neutronics.library import ELMS, TIMES

__all__ = [
    # Neutronics core
    "create_model", "plot_model",
    "run_element", "build_library",
    "ARC_D_SHAPE", "SPHERICAL",
    "TimeScheduler", "contact_dose",
    "ELMS", "TIMES",

    # CALPHAD
    "DepletionResult", "mix", "ActivationConstraints", "ActivationManifold",
    "CALPHADBatchCalculator", "BayesianSearcher", "OutlierDetector",
    "BayesianOptimizer",

    # Utilities
    "sample_simplex", "build_manifold",

    # Visualization
    "plot_dose_rate_vs_time", "plot_fispact_comparison", "plot_umap", "plot_fispact_flux",

    # Optimizer
    "evaluate",
] 