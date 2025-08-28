# -*- coding: utf-8 -*-
"""Simulator implementation for LightGBM workflow.

This module provides the run_simulator function that connects the LightGBM
workflow to the existing neutronics depletion infrastructure.
"""

from __future__ import annotations

import numpy as np  # type: ignore
from pathlib import Path
from typing import List, Optional

from .batch_depletion_subprocess import (
    BatchDepletionConfig,
    run_batch_depletion_subprocess,
)

# Default configuration
DEFAULT_FLUX_MICROXS_DIR = Path("examples/analysis_results/impulse_library/library/flux_microxs")
DEFAULT_CHAIN_FILE = "/home/myless/nuclear_data/tendl21-fispact20-gefy61-chain.xml"
DEFAULT_ABS_FILE = "/home/myless/Packages/fispact/nuclear_data/decay/abs_2012"


def run_simulator(
    X_raw: np.ndarray,
    dose_times_h: List[float],
    flux_microxs_dir: Optional[Path] = None,
    chain_file: Optional[str] = None,
    abs_file: Optional[str] = None,
    output_dir: Optional[Path] = None,
    max_parallel_jobs: int = 8,
    threads_per_process: int = 4,
    **kwargs
) -> np.ndarray:
    """Run neutronics depletion simulator for compositions.
    
    This function bridges the LightGBM workflow with the existing batch
    depletion infrastructure. It takes raw compositions and returns the
    neutronics outputs needed for the LightGBM models.
    
    Args:
        X_raw: Array of shape (n, 4) with columns [Cr, Ti, W, Zr].
        dose_times_h: List of cooling times in hours for dose evaluation.
        flux_microxs_dir: Directory containing flux and microxs files.
        chain_file: OpenMC depletion chain file path.
        abs_file: FISPACT absorption file path.
        output_dir: Directory for depletion outputs.
        threads_per_process: Threads to use per depletion process.
        **kwargs: Additional arguments for BatchDepletionConfig.
        
    Returns:
        Y array of shape (n, M) where M = len(dose_times_h) + 2.
        Columns are ordered as: [dose_t1h, dose_t24h, ..., He_2y, H_2y]
    """
    if flux_microxs_dir is None:
        flux_microxs_dir = DEFAULT_FLUX_MICROXS_DIR
    if chain_file is None:
        chain_file = DEFAULT_CHAIN_FILE
    if abs_file is None:
        abs_file = DEFAULT_ABS_FILE
    if output_dir is None:
        output_dir = Path("examples/analysis_results/lightgbm_depletion")
        
    # Convert dose times from hours to days
    cooling_days = [int(h / 24) for h in dose_times_h]
    
    # Create compositions list with V as balance
    compositions = []
    for i in range(X_raw.shape[0]):
        cr, ti, w, zr = X_raw[i]
        # V = 1 - (Cr + Ti + W + Zr) - CNO_frac
        # Assuming CNO is already accounted for in the composition
        v = 1.0 - (cr + ti + w + zr)
        if v < 0:
            v = 0.0  # Clamp to avoid negative
        
        comp = {
            "V": v,
            "Cr": cr,
            "Ti": ti,
            "W": w,
            "Zr": zr
        }
        # Normalize to ensure sum = 1
        total = sum(comp.values())
        if total > 0:
            comp = {k: v/total for k, v in comp.items()}
        
        compositions.append(comp)
    
    # Configure batch depletion
    config = BatchDepletionConfig(
        flux_microxs_dir=flux_microxs_dir,
        chain_file=chain_file,
        abs_file=abs_file,
        out_dir=output_dir,
        cooling_days=cooling_days,
        threads_per_process=threads_per_process,
        max_parallel_jobs=max_parallel_jobs,
        verbose=kwargs.get('verbose', True),
        **{k: v for k, v in kwargs.items() if k not in ['max_parallel_jobs', 'verbose']}
    )
    
    # Run batch depletion using subprocess isolation
    results = run_batch_depletion_subprocess(compositions, config)
    
    # Extract outputs in the expected order
    n = len(compositions)
    m = len(dose_times_h) + 2  # dose rates + He + H
    Y = np.zeros((n, m))
    
    for i, result in enumerate(results):
        # Check for errors
        if 'error' in result:
            # Return very high values for failed runs (will fail constraints)
            Y[i, :] = 1e10
            continue
            
        # Extract dose rates at cooling times (convert days back to hours)
        dose_dict = result.get('dose_at_cooling_times', {})
        for j, h in enumerate(dose_times_h):
            days = int(h / 24)
            Y[i, j] = dose_dict.get(days, 0.0)
        
        # Extract gas production (He and H at 2 years)
        gas_dict = result.get('gas_production', {})
        Y[i, len(dose_times_h)] = gas_dict.get('He_appm', 0.0)
        Y[i, len(dose_times_h) + 1] = gas_dict.get('H_appm', 0.0)
    
    return Y


def create_simulator_from_config(
    flux_microxs_dir: Path,
    chain_file: str,
    abs_file: str,
    output_dir: Path,
    dose_times_h: List[float],
    max_parallel_jobs: int = 8,
    threads_per_process: int = 4
):
    """Create a configured simulator function for the LightGBM workflow.
    
    This factory function creates a simulator with fixed configuration,
    suitable for use in the active learning loop.
    
    Args:
        flux_microxs_dir: Directory containing flux and microxs files.
        chain_file: OpenMC depletion chain file path.
        abs_file: FISPACT absorption file path.
        output_dir: Directory for depletion outputs.
        dose_times_h: List of cooling times in hours.
        max_parallel_jobs: Maximum number of parallel depletion processes.
        threads_per_process: Threads per depletion process.
        
    Returns:
        Configured simulator function with signature run_simulator(X_raw).
    """
    def configured_simulator(X_raw: np.ndarray) -> np.ndarray:
        return run_simulator(
            X_raw=X_raw,
            dose_times_h=dose_times_h,
            flux_microxs_dir=flux_microxs_dir,
            chain_file=chain_file,
            abs_file=abs_file,
            output_dir=output_dir,
            max_parallel_jobs=max_parallel_jobs,
            threads_per_process=threads_per_process
        )
    
    return configured_simulator
