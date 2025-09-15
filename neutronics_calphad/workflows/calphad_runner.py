"""Wrappers to run CALPHAD single-point evaluations for sampled compositions.

This module provides sequential and parallel helpers to execute CALPHAD
equilibrium calculations over many compositions, returning results as
``pandas.DataFrame`` objects aligned with the input rows.
"""

from __future__ import annotations

from typing import List, Tuple, Optional, Dict, Any, Iterator

import numpy as np  # type: ignore
import pandas as pd  # type: ignore
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import os
import sys
import json
import contextlib

try:
    from tqdm import tqdm  # type: ignore
    TQDM_AVAILABLE = True
except Exception:
    TQDM_AVAILABLE = False

from neutronics_calphad.calphad.phase_calculator import CALPHADBatchCalculator
from neutronics_calphad.utils.utils import silence_stderr_fd


ELEMENTS_ORDER: Tuple[str, ...] = ("V", "Cr", "Ti", "W", "Zr")

# Atomic weights (g/mol) used for impurity conversions
ATOMIC_WEIGHTS: Dict[str, float] = {
    "V": 50.9415,
    "Cr": 51.9961,
    "Ti": 47.867,
    "W": 183.84,
    "Zr": 91.224,
    "C": 12.011,
    "N": 14.007,
    "O": 15.999,
}


def run_calphad_batch(
    compositions: pd.DataFrame,
    temperature_k: float = 873.15,
    database: str = "TCHEA8",
    fixed_impurities: Optional[Dict[str, float]] = None,
    impurity_wtfrac: Optional[Dict[str, float]] = None,
    phase_threshold: Optional[float] = None,
) -> pd.DataFrame:
    """Run CALPHAD batch for given compositions (sequential).

    Args:
        compositions: DataFrame with columns matching a subset or all of ELEMENTS_ORDER.
            Missing elements will be treated as zero. Rows will be normalized to sum 1.
        temperature_k: Equilibration temperature in Kelvin.
        database: Thermo-Calc database name.

    Returns:
        DataFrame with CALPHAD outputs merged with the input compositions.
    """
    # Align and normalize
    df = compositions.copy()
    for el in ELEMENTS_ORDER:
        if el not in df.columns:
            df[el] = 0.0
    df = df[list(ELEMENTS_ORDER)]
    total = df.sum(axis=1)
    df = df.div(total, axis=0)

    calc = CALPHADBatchCalculator(
        database=database,
        temperature=temperature_k,
        fixed_impurities=fixed_impurities,
        phase_threshold=phase_threshold if phase_threshold is not None else 0.995,
    )
    comp_array = df.values.astype(float)
    # If only weight fractions are provided, approximate atomic fractions using the mean composition
    if fixed_impurities is None and impurity_wtfrac:
        try:
            # Compute mean atomic composition across rows
            mean_comp = comp_array.mean(axis=0)
            denom = 0.0
            for i, el in enumerate(ELEMENTS_ORDER):
                denom += float(mean_comp[i]) * ATOMIC_WEIGHTS.get(el, 1.0)
            denom = max(denom, 1e-12)
            w_sum = sum(float(v) / (100.0 if float(v) > 1.0 else 1.0) for v in impurity_wtfrac.values())
            w_sum = max(min(w_sum, 0.99), 0.0)
            w_base = 1.0 - w_sum
            n_base = w_base / denom
            n_imp_total = 0.0
            n_imp: Dict[str, float] = {}
            for imp, wv in impurity_wtfrac.items():
                w = float(wv) / (100.0 if float(wv) > 1.0 else 1.0)
                Mi = ATOMIC_WEIGHTS.get(str(imp), None)
                if Mi is None or Mi <= 0:
                    continue
                ni = w / Mi
                n_imp[str(imp)] = ni
                n_imp_total += ni
            n_total = max(n_base + n_imp_total, 1e-20)
            fixed_impurities = {k: max(v / n_total, 0.0) for k, v in n_imp.items()}
        except Exception:
            fixed_impurities = None

    results = calc.calculate_batch(compositions=comp_array, elements=list(ELEMENTS_ORDER))

    # Merge: results already contain x_ columns; also attach original fractions
    merged = pd.concat([df.reset_index(drop=True), results.reset_index(drop=True)], axis=1)
    return merged



@contextlib.contextmanager
def _suppress_stdout_stderr_fd() -> Iterator[None]:
    """Temporarily suppress process-level stdout/stderr file descriptors.

    This suppresses output emitted by native libraries (e.g., JVM logs) that do
    not respect Python's logging configuration. Restores original fds on exit.
    """
    devnull = open(os.devnull, "w")
    try:
        old_out = os.dup(1)
        old_err = os.dup(2)
        os.dup2(devnull.fileno(), 1)
        os.dup2(devnull.fileno(), 2)
        yield
    finally:
        try:
            os.dup2(old_out, 1)
            os.dup2(old_err, 2)
        finally:
            try:
                os.close(old_out)
            except Exception:
                pass
            try:
                os.close(old_err)
            except Exception:
                pass
            devnull.close()


def _worker_eval_single(
    payload: Tuple[
        int,
        List[float],
        Tuple[str, ...],
        float,
        str,
        Optional[Dict[str, float]],  # fixed impurities as atomic fractions (optional)
        Optional[Dict[str, float]],  # impurities as weight fractions (optional, fraction or percent)
        float,
        bool,
    ]
) -> Tuple[int, Dict[str, Any]]:
    """Evaluate a single composition with Thermo-Calc in a separate process.

    The worker opens its own TCPython session to ensure process isolation as
    required by the Thermo-Calc API. If Thermo-Calc is not available or an
    error occurs, it returns a FAILED record for that composition.

    Args:
        payload: Tuple of (row_index, composition_vector[V,Cr,Ti,W,Zr], elements,
            temperature_k, database, fixed_impurities, phase_threshold).

    Returns:
        Tuple of (row_index, result_dict) where ``result_dict`` contains keys
        ``x_<el>`` for each element, ``phase_count``, ``dominant_phase``,
        ``single_phase``, and ``phases`` (JSON-serializable dict as a string).
    """
    idx, comp_vec, elements, temperature_k, database, fixed_impurities, impurity_wtfrac, phase_threshold, quiet = payload
    # Defer heavy import to worker context
    try:
        from tc_python import TCPython, ThermodynamicQuantity  # type: ignore
        tc_available = True
    except Exception:
        tc_available = False

    # Build base row with composition
    row: Dict[str, Any] = {f"x_{el}": float(comp_vec[i]) for i, el in enumerate(elements)}

    if not tc_available:
        # Return a FAILED marker; caller may decide to fall back to sequential stub
        row.update({
            "phase_count": -1,
            "dominant_phase": "FAILED",
            "single_phase": False,
            "phases": "{}",
        })
        return idx, row

    # Ensure numeric stability w.r.t. tiny negatives
    comp = np.asarray(comp_vec, dtype=float)
    comp = np.clip(comp, 0.0, None)

    try:
        # Silence native outputs if requested
        _silencer = _suppress_stdout_stderr_fd() if quiet else contextlib.nullcontext()
        with _silencer:
            with TCPython() as session:  # type: ignore
                # Determine impurity element set up-front
                extra_elements: List[str] = []
                if fixed_impurities:
                    extra_elements = list({str(k) for k in fixed_impurities.keys()})
                elif impurity_wtfrac:
                    extra_elements = list({str(k) for k in impurity_wtfrac.keys()})

                # Cache per database for speed; keep per-process isolation
                calc_setup = (
                    session
                    .set_cache_folder(f"{database}_cache")
                    .select_database_and_elements(database, list(elements) + extra_elements)
                    .get_system()
                    .with_single_equilibrium_calculation()
                    .set_condition("T", float(temperature_k))
                )

                # Build impurity atomic fractions (per composition if weight fractions provided)
                impurity_atomic: Dict[str, float] = {}
                if fixed_impurities:
                    impurity_atomic = {str(k): float(v) for k, v in fixed_impurities.items()}
                elif impurity_wtfrac:
                    # Convert weight fractions (or percents) to atomic fractions using this row's base composition
                    # Normalize weight to fraction units
                    wt: Dict[str, float] = {}
                    for k, v in impurity_wtfrac.items():
                        vv = float(v)
                        wt[str(k)] = vv / 100.0 if vv > 1.0 else vv
                    w_sum = max(sum(wt.values()), 0.0)
                    if w_sum >= 1.0:
                        w_sum = 0.0  # invalid; ignore to avoid negative main_sum
                    # Mass of base metals portion (arbitrary total mass = 1)
                    w_base = 1.0 - w_sum
                    # Denominator for base metals average molar mass using base atomic fractions
                    denom = 0.0
                    for i, el in enumerate(elements):
                        denom += float(comp[i]) * ATOMIC_WEIGHTS.get(el, 1.0)
                    denom = max(denom, 1e-12)
                    # Moles of base metals in 1 mass unit
                    n_base = w_base / denom
                    # Moles of impurities and total moles
                    n_imp_total = 0.0
                    n_imp: Dict[str, float] = {}
                    for imp, w in wt.items():
                        Mi = ATOMIC_WEIGHTS.get(imp, None)
                        if Mi is None or Mi <= 0:
                            continue
                        ni = w / Mi
                        n_imp[imp] = ni
                        n_imp_total += ni
                    n_total = max(n_base + n_imp_total, 1e-20)
                    for imp, ni in n_imp.items():
                        impurity_atomic[imp] = max(ni / n_total, 0.0)
                # Apply impurity atomic fractions as explicit conditions
                if impurity_atomic:
                    for imp, frac in impurity_atomic.items():
                        calc_setup.set_condition(f"X({imp})", float(frac))

                # Account for impurities by scaling main-element sum
                impurity_sum = float(sum(impurity_atomic.values())) if impurity_atomic else 0.0
                main_sum = max(1.0 - impurity_sum, 0.0)
                normalized_comp = comp * main_sum

                # Set conditions for all but the balance element (first in list)
                for i in range(1, len(elements)):
                    calc_setup.set_condition(f"X({elements[i]})", float(normalized_comp[i]))

                # Compute equilibrium
                result = calc_setup.calculate()
                stable_phases = result.get_stable_phases()

                # Collect phase fractions
                phase_fractions: Dict[str, float] = {}
                for phase in stable_phases:
                    val = result.get_value_of(ThermodynamicQuantity.mole_fraction_of_a_phase(phase))  # type: ignore
                    try:
                        phase_fractions[str(phase)] = float(val)
                    except Exception:
                        continue

                dominant_phase = max(phase_fractions, key=phase_fractions.get) if phase_fractions else "NONE"
                phase_count = int(len(phase_fractions))
                dom_frac = float(phase_fractions.get(dominant_phase, 0.0))
                single_phase = bool(dom_frac >= float(phase_threshold))

                row.update({
                    "phase_count": phase_count,
                    "dominant_phase": dominant_phase,
                    "single_phase": single_phase,
                    "phases": json.dumps(phase_fractions),  # type: ignore[name-defined]
                })
                return idx, row

    except Exception:
        row.update({
            "phase_count": -1,
            "dominant_phase": "FAILED",
            "single_phase": False,
            "phases": "{}",
        })
        return idx, row


def run_calphad_batch_parallel(
    compositions: pd.DataFrame,
    temperature_k: float = 873.15,
    database: str = "TCHEA8",
    fixed_impurities: Optional[Dict[str, float]] = None,
    impurity_wtfrac: Optional[Dict[str, float]] = None,
    phase_threshold: Optional[float] = None,
    num_workers: int = 32,
    chunksize: int = 1,
    show_progress: bool = True,
    batch_size: Optional[int] = None,
    quiet_workers: bool = True,
    progress_desc: Optional[str] = None,
) -> pd.DataFrame:
    """Run CALPHAD batch in parallel using process-based workers.

    Distributes individual equilibrium calculations across multiple processes.
    Falls back to sequential if Thermo-Calc is unavailable or num_workers < 2.

    Args:
        compositions: DataFrame whose columns are a subset/superset of ELEMENTS_ORDER.
        temperature_k: Equilibration temperature in Kelvin.
        database: Thermo-Calc database name.
        fixed_impurities: Optional fixed impurity fractions (atomic basis).
        impurity_wtfrac: Optional impurity weight fractions (e.g., for total mass balance).
        phase_threshold: Dominant phase minimum fraction (0-1) for single_phase (default 0.995).
        num_workers: Number of worker processes.
        chunksize: Chunksize used by the executor.
        show_progress: Show tqdm progress bars if available.
        batch_size: Optional outer batching of payloads.
        quiet_workers: If True, install fd-level stderr filter in each worker
                       (or inherit /dev/null for stderr if the filter is unavailable).

    Returns:
        DataFrame containing input compositions and CALPHAD outputs (row order preserved).
    """
    import os
    import contextlib
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor

    # Normalize and align composition columns first
    df = compositions.copy()
    for el in ELEMENTS_ORDER:
        if el not in df.columns:
            df[el] = 0.0
    df = df[list(ELEMENTS_ORDER)]
    total = df.sum(axis=1)
    df = df.div(total, axis=0)

    # Fast exits (be sure to forward impurity_wtfrac now)
    if num_workers is None or num_workers < 2 or len(df) == 0:
        return run_calphad_batch(
            compositions=df,
            temperature_k=temperature_k,
            database=database,
            fixed_impurities=fixed_impurities,
            impurity_wtfrac=impurity_wtfrac,
            phase_threshold=phase_threshold,
        )

    # Build payloads per-row
    elements: Tuple[str, ...] = ELEMENTS_ORDER
    phase_thr = float(phase_threshold) if phase_threshold is not None else 0.995
    payloads: List[
        Tuple[int, List[float], Tuple[str, ...], float, str, Optional[Dict[str, float]], Optional[Dict[str, float]], float, bool]
    ] = []
    for idx, row in enumerate(df.itertuples(index=False, name=None)):
        payloads.append(
            (
                idx,
                list(row),
                elements,
                float(temperature_k),
                str(database),
                fixed_impurities,
                impurity_wtfrac,
                phase_thr,
                bool(quiet_workers),
            )
        )

    # Avoid oversubscription in child processes
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

    results_buffer: List[Optional[Dict[str, Any]]] = [None] * len(payloads)
    max_workers = min(int(num_workers), len(payloads))

    def _iter_batches(seq: List[Any], size: int) -> Iterator[List[Any]]:
        """Yield successive lists of at most ``size`` items from ``seq``."""
        for i in range(0, len(seq), size):
            yield seq[i : i + size]

    # Prefer 'spawn' start method for deterministic worker init across platforms
    ctx = mp.get_context("spawn")

    # -------------------- Robust stderr filtering for workers -----------------
    # Try to use fd-level filter as a pool initializer; otherwise, silence
    # stderr during pool creation so spawned workers inherit /dev/null for fd=2.
    endf_filter_initializer = None
    initargs: Tuple[Any, ...] = ()
    filter_patterns = [
        r"LTT\s*\(?3\)?\s*for elastic scattering.*Legendre only",
        r"GNDS naming convention",
        r"cross_sections",
    ]
    suppress_prefixes = ["n-00"]  # ENDF/GNDS line prefixes like n-001_H_002...

    if quiet_workers:
        try:
            # Must be a top-level function (picklable) to serve as initializer
            from .utils.openmc_noise import install_endf_stderr_filter as _install_endf_stderr_filter  # type: ignore

            endf_filter_initializer = _install_endf_stderr_filter
            initargs = (filter_patterns, suppress_prefixes)

            # Also install in the parent so any local ENDF parsing is quiet
            try:
                _install_endf_stderr_filter(patterns=filter_patterns, suppress_prefixes=suppress_prefixes)
            except Exception:
                pass
        except Exception:
            endf_filter_initializer = None
            initargs = ()

    def _pool_outer_cm():
        # If we don't have the robust filter initializer, silence fd=2 *before*
        # the pool spawns so children inherit /dev/null for stderr.
        if quiet_workers and endf_filter_initializer is None:
            return silence_stderr_fd()
        return contextlib.nullcontext()

    # ------------------------------ Dispatch ----------------------------------
    if batch_size is None or batch_size <= 0 or batch_size >= len(payloads):
        # Single batch over all payloads
        try:
            with _pool_outer_cm():
                with ProcessPoolExecutor(
                    max_workers=max_workers,
                    mp_context=ctx,
                    initializer=endf_filter_initializer,
                    initargs=initargs,
                ) as executor:
                    iterator = executor.map(_worker_eval_single, payloads, chunksize=chunksize)
                    if show_progress and TQDM_AVAILABLE:
                        desc = progress_desc or "CALPHAD calculations"
                        iterator = tqdm(
                            iterator,
                            total=len(payloads),
                            desc=desc,
                            unit="comp",
                            dynamic_ncols=True,
                            leave=True,
                            file=sys.stdout,  # ensure visible even if stderr is filtered
                        )  # type: ignore
                    for idx, row_dict in iterator:
                        results_buffer[idx] = row_dict
        except Exception:
            # Robust fallback: run sequentially if the pool crashes
            seq_df = run_calphad_batch(
                compositions=df,
                temperature_k=temperature_k,
                database=database,
                fixed_impurities=fixed_impurities,
                impurity_wtfrac=impurity_wtfrac,
                phase_threshold=phase_threshold,
            )
            # Fill buffer with rows matching worker schema: x_* and phase fields only
            for i in range(seq_df.shape[0]):
                r = seq_df.iloc[i]
                row_dict = {f"x_{el}": float(r.get(f"x_{el}", 0.0)) for el in elements}
                row_dict.update(
                    {
                        "phase_count": int(r.get("phase_count", 0)),
                        "dominant_phase": str(r.get("dominant_phase", "NONE")),
                        "single_phase": bool(r.get("single_phase", False)),
                        "phases": str(r.get("phases", "{}")),
                    }
                )
                results_buffer[i] = row_dict
    else:
        # Process in batches; show a batch-level bar and an inner comp bar
        num_batches = (len(payloads) + batch_size - 1) // batch_size
        outer = range(num_batches)
        if show_progress and TQDM_AVAILABLE:
            outer = tqdm(outer, total=num_batches, desc="CALPHAD batches", unit="batch")  # type: ignore
        for b in outer:
            start = b * batch_size
            end = min(start + batch_size, len(payloads))
            batch = payloads[start:end]
            try:
                with _pool_outer_cm():
                    with ProcessPoolExecutor(
                        max_workers=min(max_workers, len(batch)),
                        mp_context=ctx,
                        initializer=endf_filter_initializer,
                        initargs=initargs,
                    ) as executor:
                        iterator = executor.map(_worker_eval_single, batch, chunksize=chunksize)
                        if show_progress and TQDM_AVAILABLE:
                            base = progress_desc or "CALPHAD"
                            iterator = tqdm(
                                iterator,
                                total=len(batch),
                                desc=f"{base} batch {b+1}/{num_batches}",
                                unit="comp",
                                dynamic_ncols=True,
                                leave=True,
                                file=sys.stdout,
                            )  # type: ignore
                        for local_idx, row_dict in iterator:
                            results_buffer[local_idx] = row_dict
            except Exception:
                # If a batch crashes, fall back to sequential for just this batch
                batch_df = pd.DataFrame([p[1] for p in batch], columns=list(ELEMENTS_ORDER))
                seq_res = run_calphad_batch(
                    compositions=batch_df,
                    temperature_k=temperature_k,
                    database=database,
                    fixed_impurities=fixed_impurities,
                    impurity_wtfrac=impurity_wtfrac,
                    phase_threshold=phase_threshold,
                )
                # Write results back to the buffer in worker schema
                for local_offset in range(seq_res.shape[0]):
                    r = seq_res.iloc[local_offset]
                    row_dict = {f"x_{el}": float(r.get(f"x_{el}", 0.0)) for el in elements}
                    row_dict.update(
                        {
                            "phase_count": int(r.get("phase_count", 0)),
                            "dominant_phase": str(r.get("dominant_phase", "NONE")),
                            "single_phase": bool(r.get("single_phase", False)),
                            "phases": str(r.get("phases", "{}")),
                        }
                    )
                    results_buffer[start + local_offset] = row_dict

    # If any worker reported FAILED due to missing Thermo-Calc, fall back to sequential
    if any(
        r is not None and r.get("dominant_phase") == "FAILED" and r.get("phase_count") == -1
        for r in results_buffer
        if r is not None
    ):
        return run_calphad_batch(
            compositions=df,
            temperature_k=temperature_k,
            database=database,
            fixed_impurities=fixed_impurities,
            impurity_wtfrac=impurity_wtfrac,
            phase_threshold=phase_threshold,
        )

    # Materialize results DataFrame and merge with input alignment
    results_rows = [r for r in results_buffer if r is not None]
    results_df = pd.DataFrame(results_rows)
    merged = pd.concat([df.reset_index(drop=True), results_df.reset_index(drop=True)], axis=1)
    return merged


