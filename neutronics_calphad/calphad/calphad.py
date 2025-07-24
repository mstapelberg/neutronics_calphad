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









