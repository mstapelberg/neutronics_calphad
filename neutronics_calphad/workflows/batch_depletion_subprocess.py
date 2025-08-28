"""Subprocess-based batch depletion runner with proper process isolation.

This implementation uses subprocess calls to avoid OpenMC nuclear data loading
conflicts, similar to the successful GNU parallel approach.

Typical usage:

    from pathlib import Path
    from neutronics_calphad.workflows.batch_depletion_subprocess import (
        BatchDepletionConfig, run_batch_depletion_subprocess
    )

    cfg = BatchDepletionConfig(
        flux_microxs_dir=Path("examples/analysis_results/impulse_library/library/flux_microxs"),
        max_parallel_jobs=8,
        threads_per_process=4,
    )

    compositions = [
        {"V": 0.76, "Cr": 0.06, "Ti": 0.06, "W": 0.06, "Zr": 0.06},
        {"V": 0.90, "Cr": 0.05, "Ti": 0.05},
    ]

    results = run_batch_depletion_subprocess(compositions, cfg)
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np  # type: ignore


def _json_safe(obj: Any) -> Any:
    """Recursively convert objects to JSON-serializable forms.

    - Path -> str
    - numpy scalars -> Python scalars
    - dict/list/tuple -> recurse
    """
    try:
        from numpy import generic as _np_generic  # type: ignore
    except Exception:  # pragma: no cover
        _np_generic = ()  # type: ignore

    if isinstance(obj, Path):
        return str(obj)
    if _np_generic and isinstance(obj, _np_generic):  # type: ignore[arg-type]
        return obj.item()
    # numpy arrays -> lists
    try:
        import numpy as _np  # type: ignore
        if isinstance(obj, _np.ndarray):  # type: ignore[attr-defined]
            return obj.tolist()
    except Exception:
        pass
    # torch tensors -> lists
    try:
        import torch  # type: ignore
        if isinstance(obj, torch.Tensor):  # type: ignore[attr-defined]
            return obj.detach().cpu().tolist()
    except Exception:
        pass
    # sets -> lists
    if isinstance(obj, set):
        return list(obj)
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [ _json_safe(v) for v in obj ]
    return obj

def composition_hash(
    composition: Dict[str, float],
    precision: int = 4
) -> str:
    """Generate hash for composition with specified precision.
    
    Args:
        composition: Dict mapping elements to atomic fractions
        precision: Decimal places for rounding
        
    Returns:
        Hex string hash
    """
    import hashlib
    
    # Sort by element and round
    rounded = {}
    for elem in sorted(composition.keys()):
        val = round(composition[elem], precision)
        if val > 0:  # Only include non-zero
            rounded[elem] = val
    
    # Create string representation
    comp_str = json.dumps(rounded, sort_keys=True)
    
    # Return hash
    return hashlib.md5(comp_str.encode()).hexdigest()[:16]


@dataclass
class BatchDepletionConfig:
    """Configuration for subprocess-based batch depletion.
    
    Attributes:
        flux_microxs_dir: Directory containing flux and microxs files
        chain_file: Path to OpenMC depletion chain file
        abs_file: Path to FISPACT ABS data directory
        out_dir: Base directory for run outputs
        depletable_cell: Name of the geometry cell to replace
        geometry_config: Geometry config dict
        particles: Number of OpenMC particles
        power_mw: Fusion power in MW
        torus_to_sphere_ratio: Ratio for source normalization
        irradiation_time: Irradiation duration string
        irradiation_steps: Number of irradiation substeps
        cooling_days: Cooling time points in days
        max_parallel_jobs: Maximum parallel subprocess jobs
        threads_per_process: OpenMP threads per subprocess
        python_executable: Python executable to use
        runner_script: Path to single_depletion_runner.py
        verbose: Enable verbose output
    """
    flux_microxs_dir: Path
    chain_file: Optional[str] = None
    abs_file: str = "/home/myless/Packages/fispact/nuclear_data/decay/abs_2012"
    out_dir: Path = Path("examples/analysis_results/batch_depletion_subprocess")
    depletable_cell: str = "vessel"
    geometry_config: Dict[str, Any] = field(default_factory=dict)
    particles: int = 10000
    power_mw: float = 500.0
    torus_to_sphere_ratio: float = 1 / 4.03
    irradiation_time: str = "2 years"
    irradiation_steps: int = 24
    cooling_days: List[int] = field(default_factory=lambda: [30, 365, 5 * 365, 36500])
    max_parallel_jobs: int = 8
    threads_per_process: int = 4
    python_executable: str = sys.executable
    runner_script: Optional[Path] = None
    verbose: bool = True
    # Mitigations for I/O contention during startup
    stagger_start_s: float = 0.0  # random sleep up to this many seconds before starting
    prewarm_chain: bool = False   # optionally read chain file to warm OS page cache
    prewarm_abs: bool = False     # optionally touch ABS files to warm OS page cache
    debug_dose_dump: bool = False # write per-run dose debug artifacts

    def __post_init__(self) -> None:
        """Resolve paths and set defaults."""
        if self.runner_script is None:
            # Find the runner script relative to this file
            this_dir = Path(__file__).parent
            self.runner_script = this_dir / "single_depletion_runner.py"
        
        if self.chain_file is None:
            self.chain_file = os.environ.get(
                "OPENMC_CHAIN_FILE",
                "/home/myless/nuclear_data/tendl21-fispact20-gefy61-chain.xml"
            )
        
        if not self.geometry_config:
            from neutronics_calphad.neutronics.config import SPHERICAL
            self.geometry_config = SPHERICAL


def _run_subprocess_depletion(
    composition: Dict[str, float],
    comp_idx: int,
    config: BatchDepletionConfig,
    config_file: Path
) -> Tuple[int, Dict[str, Any]]:
    """Run a single depletion as a subprocess.
    
    Args:
        composition: Composition dict
        comp_idx: Index in the batch
        config: Batch configuration
        config_file: Path to config JSON file
        
    Returns:
        Tuple of (comp_idx, result_dict)
    """
    comp_id = composition_hash(composition)
    # Use a unique run directory per job to avoid HDF5 file lock collisions
    run_id = uuid.uuid4().hex[:8]
    comp_dir = config.out_dir / f"comp_{comp_id}" / f"run_{run_id}"
    
    # Build subprocess command
    cmd = [
        config.python_executable,
        str(config.runner_script),
        "--composition", json.dumps(composition),
        "--output-dir", str(comp_dir),
        "--config", str(config_file)
    ]
    
    # Set environment for thread control
    env = os.environ.copy()
    env.update({
        "OMP_NUM_THREADS": str(config.threads_per_process),
        "OPENBLAS_NUM_THREADS": str(config.threads_per_process),
        "MKL_NUM_THREADS": str(config.threads_per_process),
        "NUMEXPR_NUM_THREADS": str(config.threads_per_process),
        "PYTHONUNBUFFERED": "1",
        "OPENMC_LOG_LEVEL": "ERROR",
        # HDF5 file locking can fail under high concurrency / WSL+NTFS; disable locks safely for read-mostly workloads
        "HDF5_USE_FILE_LOCKING": "FALSE",
    })
    
    # Run subprocess
    start_time = time.time()
    if config.verbose:
        print(f"[{comp_idx}] Starting: {comp_id} (run {run_id})")
    
    try:
        # Use subprocess.run directly with proper argument handling
        # This avoids bash command parsing issues with JSON strings
        result = subprocess.run(
            cmd,
            env=env,
            capture_output=True,
            text=True,
            timeout=1800  # 30 minute timeout
        )
        
        duration = time.time() - start_time
        
        if result.returncode != 0:
            error_msg = result.stderr or "Unknown error"
            if config.verbose:
                print(f"[{comp_idx}] FAILED after {duration:.1f}s: {error_msg}")
            
            return comp_idx, {
                "composition": composition,
                "composition_id": comp_id,
                "error": error_msg,
                "duration_s": duration
            }
        
        # Load result
        result_file = comp_dir / "depletion_result.json"
        if not result_file.exists():
            return comp_idx, {
                "composition": composition,
                "composition_id": comp_id,
                "error": "Result file not found",
                "duration_s": duration
            }
        
        with open(result_file, 'r') as f:
            result_data = json.load(f)
        # Normalize dose keys to int (JSON keys are strings)
        raw_dose = result_data.get("dose_at_cooling_times", {}) or {}
        try:
            dose_norm = {int(k): float(v) for k, v in raw_dose.items()}
        except Exception:
            dose_norm = raw_dose
        
        if config.verbose:
            print(f"[{comp_idx}] SUCCESS after {duration:.1f}s: {comp_id}")
        
        return comp_idx, {
            "composition": composition,
            "composition_id": comp_id,
            "gas_production": result_data.get("gas_production", {}),
            "dose_at_cooling_times": dose_norm,
            "out_dir": str(comp_dir),
            "duration_s": duration
        }
        
    except subprocess.TimeoutExpired:
        if config.verbose:
            print(f"[{comp_idx}] TIMEOUT after 1800s: {comp_id}")
        return comp_idx, {
            "composition": composition,
            "composition_id": comp_id,
            "error": "Subprocess timeout after 30 minutes",
            "duration_s": 1800
        }
    except Exception as e:
        if config.verbose:
            print(f"[{comp_idx}] ERROR: {str(e)}")
        return comp_idx, {
            "composition": composition,
            "composition_id": comp_id,
            "error": str(e),
            "duration_s": time.time() - start_time
        }


def run_batch_depletion_subprocess(
    compositions: List[Dict[str, float]],
    config: BatchDepletionConfig,
) -> List[Dict[str, Any]]:
    """Run batch depletion using subprocess isolation.
    
    This approach avoids OpenMC nuclear data loading conflicts by running
    each depletion in a separate process, similar to GNU parallel.
    
    Args:
        compositions: List of composition dicts
        config: Batch configuration
        
    Returns:
        List of result dictionaries
    """
    if not compositions:
        return []
    
    # Prepare output directory
    config.out_dir.mkdir(parents=True, exist_ok=True)
    
    # Write config to temporary file
    config_dict = {
        'flux_microxs_dir': str(config.flux_microxs_dir),
        'chain_file': config.chain_file,
        'abs_file': config.abs_file,
        'depletable_cell': config.depletable_cell,
        'geometry_config': config.geometry_config,
        'particles': config.particles,
        'power_mw': config.power_mw,
        'torus_to_sphere_ratio': config.torus_to_sphere_ratio,
        'irradiation_time': config.irradiation_time,
        'irradiation_steps': config.irradiation_steps,
        'cooling_days': config.cooling_days,
        'stagger_start_s': config.stagger_start_s,
        'prewarm_chain': config.prewarm_chain,
        'prewarm_abs': config.prewarm_abs,
        'debug_dose_dump': config.debug_dose_dump,
    }
    
    config_file = config.out_dir / "batch_config.json"
    with open(config_file, 'w') as f:
        json.dump(_json_safe(config_dict), f, indent=2)
    
    print(f"\nRunning {len(compositions)} compositions with up to {config.max_parallel_jobs} parallel jobs")
    print(f"Each job uses {config.threads_per_process} threads")
    print(f"Output directory: {config.out_dir}")
    
    start_time = time.time()
    results: List[Dict[str, Any]] = [{}] * len(compositions)  # Pre-allocate
    
    # Use ThreadPoolExecutor to manage subprocess calls
    # (threads are fine here since we're just managing subprocesses)
    with ThreadPoolExecutor(max_workers=config.max_parallel_jobs) as executor:
        # Submit all jobs
        futures = {
            executor.submit(
                _run_subprocess_depletion,
                comp,
                i,
                config,
                config_file
            ): i
            for i, comp in enumerate(compositions)
        }
        
        # Process completed jobs
        completed = 0
        failed = 0
        
        for future in as_completed(futures):
            idx = futures[future]
            try:
                comp_idx, result = future.result()
                results[comp_idx] = result
                
                if 'error' in result:
                    failed += 1
                    
                completed += 1
                
                # Progress update
                if config.verbose and completed % 10 == 0:
                    elapsed = time.time() - start_time
                    rate = completed / elapsed
                    eta = (len(compositions) - completed) / rate if rate > 0 else 0
                    print(f"\nProgress: {completed}/{len(compositions)} "
                          f"({failed} failed) - "
                          f"Rate: {rate:.2f} comps/s - "
                          f"ETA: {eta/60:.1f} min")
                    
            except Exception as e:
                print(f"Unexpected error processing future: {e}")
                results[idx] = {
                    "composition": compositions[idx],
                    "composition_id": composition_hash(compositions[idx]),
                    "error": f"Future processing error: {str(e)}"
                }
                failed += 1
                completed += 1
    
    # Final summary
    total_time = time.time() - start_time
    success_count = len([r for r in results if 'error' not in r])
    
    print(f"\n{'='*60}")
    print(f"Batch depletion complete!")
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"Successful: {success_count}/{len(compositions)}")
    print(f"Failed: {failed}/{len(compositions)}")
    print(f"Average time per composition: {total_time/len(compositions):.1f}s")
    print(f"{'='*60}\n")
    
    # Save summary
    summary_file = config.out_dir / "batch_summary.json"
    with open(summary_file, 'w') as f:
        # Keep summary minimal and JSON-safe; reference the config file path instead of embedding the dict
        summary_payload = {
            'n_compositions': len(compositions),
            'n_success': success_count,
            'n_failed': failed,
            'total_time_s': total_time,
            'config_file': str(config_file),
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())
        }
        json.dump(summary_payload, f, indent=2)
    
    return results
