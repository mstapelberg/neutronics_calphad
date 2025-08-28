"""Workflow utilities for composition sampling and analysis.

This subpackage provides high-level workflows to:
- Sample constrained alloy compositions
- Apply activation and ductility filters
- Run CALPHAD single-point evaluations
- Generate manifold embeddings and clustering plots
- Create ternary slice visualizations
- LightGBM-based active learning with BoTorch acquisition

All public functions are fully typed and documented.

Note: The linear impulse method has been deprecated in favor of the
LightGBM approach. See deprecated/ directory for old implementations.
"""

from __future__ import annotations

__all__ = [
    "__version__",
    # Core workflows
    "sample_and_filter",
    "embed_and_cluster_compositions", 
    "evaluate_calphad_for_valid",
    # Composition utilities
    "SamplingConstraints",
    "sample_compositions",
    "make_ternary_grid",
    # Filters
    "ActivationLimits",
    "make_activation_filter",
    "make_ductility_filter",
    "apply_filters",
    # LightGBM workflow
    "active_loop",
    "lgbm_joint_feasibility_score",
    # Batch depletion
    "BatchDepletionConfig",
    "run_batch_depletion",
]

__version__: str = "0.2.0"  # Bumped for major API change

# Import main interfaces
from .workflow import (
    sample_and_filter,
    embed_and_cluster_compositions,
    evaluate_calphad_for_valid,
)
from .composition_sampling import (
    SamplingConstraints,
    sample_compositions,
    make_ternary_grid,
)
from .filters import (
    ActivationLimits,
    make_activation_filter,
    make_ductility_filter,
    apply_filters,
)
from .lightgbm_workflow import (
    active_loop,
    lgbm_joint_feasibility_score,
)
from .batch_depletion import (
    BatchDepletionConfig,
    run_batch_depletion,
)


