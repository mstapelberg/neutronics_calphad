# -*- coding: utf-8 -*-
"""Simulator implementation for LightGBM workflow.

This module provides the run_simulator function that connects the LightGBM
workflow to the existing neutronics depletion infrastructure.
"""

from __future__ import annotations

import numpy as np  # type: ignore
import os
from pathlib import Path
from typing import Dict, List, Optional

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
    impurity_atomic: Optional[Dict[str, float]] = None,
    impurity_wtfrac: Optional[Dict[str, float]] = None,
    cno_frac: Optional[float] = None,
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
        max_parallel_jobs: Maximum parallel depletion jobs.
        threads_per_process: Threads to use per depletion process.
        impurity_atomic: Optional mapping of impurity element to atomic fraction
            (e.g., {"C": 0.002, "N": 0.002, "O": 0.001}). If provided, takes
            precedence over ``impurity_wtfrac``.
        impurity_wtfrac: Optional mapping of impurity element to weight fraction
            (fraction or percent). Converted to atomic fractions per-composition.
        cno_frac: Optional total atomic fraction for C+N+O to reserve in the
            base composition when explicit impurities are not provided. If set and
            ``impurity_atomic``/``impurity_wtfrac`` are None, C, N, O are split
            equally.
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
    
    # Create compositions list (optionally include impurities)
    compositions: List[Dict[str, float]] = []
    for i in range(X_raw.shape[0]):
        cr, ti, w, zr = [float(x) for x in X_raw[i]]
        comp = build_depletion_composition(
            cr=cr,
            ti=ti,
            w=w,
            zr=zr,
            impurity_atomic=impurity_atomic,
            impurity_wtfrac=impurity_wtfrac,
            cno_frac=cno_frac,
        )
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
    threads_per_process: int = os.environ.get("OMP_NUM_THREADS", 4),
    impurity_atomic: Optional[Dict[str, float]] = None,
    impurity_wtfrac: Optional[Dict[str, float]] = None,
    cno_frac: Optional[float] = None,
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
        impurity_atomic: Optional impurity atomic fractions to include in all runs.
        impurity_wtfrac: Optional impurity weight fractions to include in all runs.
        cno_frac: Optional total atomic C+N+O fraction to reserve if explicit
            impurities are not provided.
        
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
            threads_per_process=threads_per_process,
            impurity_atomic=impurity_atomic,
            impurity_wtfrac=impurity_wtfrac,
            cno_frac=cno_frac,
        )
    
    return configured_simulator


def build_depletion_composition(
    *,
    cr: float,
    ti: float,
    w: float,
    zr: float,
    impurity_atomic: Optional[Dict[str, float]] = None,
    impurity_wtfrac: Optional[Dict[str, float]] = None,
    cno_frac: Optional[float] = None,
) -> Dict[str, float]:
    """Build a single depletion composition including optional impurities.
    
    Given raw alloy fractions ``cr, ti, w, zr`` (atomic fractions of non-V
    constituents in the 5-part system), this function constructs the full
    composition dictionary used for depletion simulations. Impurities can be
    supplied either as atomic fractions or as weight fractions. If neither is
    provided but ``cno_frac`` is set, C, N, and O will be added in equal
    atomic parts summing to ``cno_frac``.
    
    The main alloy elements (V, Cr, Ti, W, Zr) are renormalized to occupy the
    remaining fraction after impurities are accounted for.
    
    Args:
        cr: Chromium atomic fraction (raw input variable).
        ti: Titanium atomic fraction (raw input variable).
        w: Tungsten atomic fraction (raw input variable).
        zr: Zirconium atomic fraction (raw input variable).
        impurity_atomic: Optional mapping of impurity element -> atomic fraction.
        impurity_wtfrac: Optional mapping of impurity element -> weight fraction
            (fraction or percent). Converted per-composition.
        cno_frac: Optional total C+N+O atomic fraction if no explicit impurities
            are provided. Split equally among C, N, and O.
    
    Returns:
        Composition dict mapping element symbol to atomic fraction (sums to 1).
    """
    # Base main-element vector (before scaling down for impurities)
    v0 = 1.0 - (cr + ti + w + zr)
    if v0 < 0.0:
        v0 = 0.0
    base = {"V": float(v0), "Cr": float(cr), "Ti": float(ti), "W": float(w), "Zr": float(zr)}
    s_base = sum(base.values())
    if s_base <= 0.0:
        # Degenerate; return equal split across main elements
        base = {k: 1.0 / 5.0 for k in ("V", "Cr", "Ti", "W", "Zr")}
        s_base = 1.0

    # Determine impurity atomic fractions
    imp_atomic: Dict[str, float] = {}
    if impurity_atomic:
        for k, v in impurity_atomic.items():
            try:
                vv = float(v)
                if vv > 0.0:
                    imp_atomic[str(k)] = vv
            except Exception:
                continue
    elif impurity_wtfrac:
        # Convert weight fractions to atomic fractions for this composition
        # Atomic weights (g/mol)
        aw = {
            "V": 50.9415,
            "Cr": 51.9961,
            "Ti": 47.867,
            "W": 183.84,
            "Zr": 91.224,
            "C": 12.011,
            "N": 14.007,
            "O": 15.999,
        }
        # Normalize weight inputs to fraction
        w_imp_sum = 0.0
        for k, v in impurity_wtfrac.items():
            val = float(v)
            if val <= 0.0:
                continue
            if val > 1.0:
                val = val / 100.0
            w_imp_sum += val
        w_imp_sum = max(min(w_imp_sum, 0.99), 0.0)
        w_base = 1.0 - w_imp_sum
        # Base average atomic weight from current composition
        m_base = 0.0
        for k, frac in base.items():
            m_base += frac * aw.get(k, 1.0)
        m_base = max(m_base, 1e-12)
        n_base = w_base / m_base
        n_imp_total = 0.0
        n_imp: Dict[str, float] = {}
        for k, v in impurity_wtfrac.items():
            val = float(v)
            if val <= 0.0:
                continue
            if val > 1.0:
                val = val / 100.0
            Mi = aw.get(str(k))
            if Mi is None or Mi <= 0.0:
                continue
            ni = val / Mi
            n_imp[str(k)] = ni
            n_imp_total += ni
        n_total = max(n_base + n_imp_total, 1e-20)
        for k, ni in n_imp.items():
            imp_atomic[k] = max(ni / n_total, 0.0)
    elif cno_frac and cno_frac > 0.0:
        # Split C+N+O equally if provided without explicit breakdown
        third = float(cno_frac) / 3.0
        imp_atomic = {"C": third, "N": third, "O": third}

    # Scale main system to respect impurities, preserving alloying inputs and using V as balance.
    imp_sum = float(sum(max(v, 0.0) for v in imp_atomic.values()))
    imp_sum = max(min(imp_sum, 0.99), 0.0)
    main_sum = 1.0 - imp_sum

    # Preserve alloying elements (Cr, Ti, W, Zr) exactly as provided and set V as balance.
    cr_f = float(base["Cr"])  # keep original raw values
    ti_f = float(base["Ti"])  # keep original raw values
    w_f = float(base["W"])    # keep original raw values
    zr_f = float(base["Zr"])  # keep original raw values
    v_bal = float(main_sum - (cr_f + ti_f + w_f + zr_f))
    if v_bal < 0.0:
        # Infeasible given impurities; clamp V to 0.0 and proportionally scale down alloying to fit.
        # This path should be rare under typical constraints (sum of alloying ≤ 0.20).
        v_bal = 0.0
        s_alloy = cr_f + ti_f + w_f + zr_f
        if s_alloy > 0.0:
            scale = main_sum / s_alloy
            cr_f *= scale
            ti_f *= scale
            w_f *= scale
            zr_f *= scale
        # Recompute exact balance for numerical safety
        v_bal = max(0.0, main_sum - (cr_f + ti_f + w_f + zr_f))

    # Build main composition with V as balance
    out: Dict[str, float] = {
        "V": v_bal,
        "Cr": cr_f,
        "Ti": ti_f,
        "W": w_f,
        "Zr": zr_f,
    }

    # Add impurities; ensure closure by adjusting V minimally for rounding drift
    for k, v in imp_atomic.items():
        out[k] = out.get(k, 0.0) + max(float(v), 0.0)

    # One-pass correction: adjust V to absorb tiny drift so total sums to 1.0
    total_before = sum(out.values())
    if total_before != 1.0:
        delta = 1.0 - total_before
        out["V"] = max(0.0, out.get("V", 0.0) + delta)

    return out
