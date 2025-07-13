"""
Neutronics CALPHAD: Neutronics simulations for CALPHAD-based alloy optimization.

This package provides tools for running neutronics simulations of tokamak
geometries with different materials, performing activation analysis, and
optimizing alloy compositions based on nuclear performance criteria.
"""

__version__ = "0.1.0"

# Import main functions for easy access
from .geometry_maker import create_model, plot_model
from .library import run_element, build_library, ELMS, TIMES
from .evaluate import evaluate
from .manifold import sample_simplex, build_manifold
from .visualization import plot_dose_rate_vs_time, plot_umap
from .config import ARC_D_SHAPE, SPHERICAL

# Import new CALPHAD module components
from .calphad import (
    DepletionResult,
    mix,
    ActivationConstraints,
    ActivationManifold,
    CALPHADBatchCalculator,
    BayesianSearcher,
    OutlierDetector
)

__all__ = [
    # Geometry
    "create_model",
    "plot_model", 
    
    # Library building
    "run_element",
    "build_library",
    "ELMS",
    "TIMES",
    
    # Evaluation
    "evaluate",
    
    # Manifold sampling
    "sample_simplex", 
    "build_manifold",
    
    # Visualization
    "plot_dose_rate_vs_time",
    "plot_umap",
    
    # Configs
    "ARC_D_SHAPE",
    "SPHERICAL",

    # CALPHAD module
    "DepletionResult",
    "mix",
    "ActivationConstraints", 
    "ActivationManifold",
    "CALPHADBatchCalculator",
    "BayesianSearcher",
    "OutlierDetector",
] 