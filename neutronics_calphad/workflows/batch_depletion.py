"""Parallel batch depletion runner using precomputed flux and MicroXS.

Provides a simple, reusable API to execute many independent depletion runs in
parallel, with process-level parallelism and per-process thread limiting for
best throughput on shared-memory machines.

Typical usage:

    from pathlib import Path
    from neutronics_calphad.workflows.batch_depletion import (
        BatchDepletionConfig, run_batch_depletion
    )

    cfg = BatchDepletionConfig(
        flux_microxs_dir=Path("examples/analysis_results/impulse_library/library/flux_microxs"),
        threads_available=32,
        threads_per_process=4,
    )

    compositions = [
        {"V": 0.76, "Cr": 0.06, "Ti": 0.06, "W": 0.06, "Zr": 0.06},
        {"V": 0.90, "Cr": 0.05, "Ti": 0.05},
    ]

    results = run_batch_depletion(compositions, cfg)

The function returns a list of result dicts with composition, dose rates at
configured cooling days, and gas production.
"""

from __future__ import annotations

import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np  # type: ignore
import openmc  # type: ignore

from neutronics_calphad.neutronics.config import SPHERICAL
from neutronics_calphad.neutronics.depletion import run_independent_depletion
from neutronics_calphad.neutronics.geometry_maker import create_model
from neutronics_calphad.neutronics.time_scheduler import TimeScheduler
from neutronics_calphad.optimizer.parsers import parse_openmc_results
from neutronics_calphad.utils.io import create_material
# Import moved to inline function to avoid dependency on deprecated module


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
    import json
    
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
    """Configuration for batch independent depletion.

    Attributes:
        flux_microxs_dir: Directory containing ``flux_spectrum_1102.txt`` and
            ``microxs_1102.csv`` to reuse for depletion.
        chain_file: Path to OpenMC depletion chain file. Defaults from
            ``OPENMC_CHAIN_FILE`` or ``openmc.config['chain_file']``.
        abs_file: Path to FISPACT ABS data directory used by dose parser.
        out_dir: Base directory for run outputs (one subdir per composition).
        depletable_cell: Name of the geometry cell to replace (default: 'vessel').
        geometry_config: Geometry config dict used by ``create_model``.
        particles: Number of OpenMC particles for model settings.
        power_mw: Fusion power in MW (for source-rate normalization).
        torus_to_sphere_ratio: Ratio to convert tokamak source to spherical model.
        irradiation_time: Irradiation duration string (e.g., '2 years').
        irradiation_steps: Number of irradiation substeps.
        cooling_days: Cooling time points in days for dose extraction.
        threads_available: Total threads available on the machine.
        threads_per_process: Threads to allocate per worker process.
        max_workers: Cap on processes; defaults to threads_available // threads_per_process.
    """

    flux_microxs_dir: Path
    chain_file: Optional[str] = None
    abs_file: str = "/home/myless/Packages/fispact/nuclear_data/decay/abs_2012"
    out_dir: Path = Path("examples/analysis_results/batch_depletion")
    depletable_cell: str = "vessel"
    geometry_config: Dict[str, Any] = field(default_factory=lambda: SPHERICAL)
    particles: int = 10000
    power_mw: float = 500.0
    torus_to_sphere_ratio: float = 1 / 4.03
    irradiation_time: str = "2 years"
    irradiation_steps: int = 24
    cooling_days: List[int] = None  # type: ignore[assignment]
    threads_available: int = 32
    threads_per_process: int = 4
    max_workers: Optional[int] = None

    def __post_init__(self) -> None:
        """Fill defaults for optional fields after initialization."""
        if self.cooling_days is None:
            self.cooling_days = [30, 365, 5 * 365, 36500]


def _set_openmc_paths_from_env(chain_file: Optional[str]) -> str:
    """Resolve and set OpenMC chain path using environment/defaults.

    Args:
        chain_file: Optional explicit path.

    Returns:
        Resolved chain file path.
    """
    resolved_chain = chain_file or os.environ.get(
        "OPENMC_CHAIN_FILE",
        "/home/myless/nuclear_data/tendl21-fispact20-gefy61-chain.xml",
    )
    openmc.config["chain_file"] = resolved_chain
    openmc.config["cross_sections"] = os.environ.get(
        "OPENMC_CROSS_SECTIONS",
        "/home/myless/nuclear_data/tendl-2021-hdf5/cross_sections.xml",
    )
    return resolved_chain


def _limit_threads(n: int) -> None:
    """Limit threads in BLAS/OpenMP-backed libraries for a worker process.
    
    Also installs aggressive warning suppression to filter OpenMC noise.

    Args:
        n: Number of threads per worker process.
    """
    # Set thread limits for numerical libraries
    os.environ.update(
        {
            "OMP_NUM_THREADS": str(n),
            "OPENBLAS_NUM_THREADS": str(n),
            "MKL_NUM_THREADS": str(n),
            "VECLIB_MAXIMUM_THREADS": str(n),
            "NUMEXPR_NUM_THREADS": str(n),
        }
    )
    
    # Set OpenMC environment variables BEFORE importing OpenMC to suppress data loading warnings
    os.environ["OPENMC_SUPPRESS_WARNINGS"] = "1"
    os.environ["OPENMC_LOG_LEVEL"] = "ERROR"
    os.environ["OPENMC_QUIET"] = "1"
    
    try:
        from threadpoolctl import threadpool_limits  # type: ignore

        threadpool_limits(limits=n)
    except Exception:
        pass
    
    # Install aggressive warning suppression for OpenMC noise
    try:
        from neutronics_calphad.utils.utils import install_global_output_filter, permanently_redirect_stderr_to_null
        
        # Filter out all the annoying n-00* warnings and LTT elastic scattering messages
        install_global_output_filter(
            patterns=[
                "LTT (3) for elastic scattering, using Legendre only",
                "LTT(3) for elastic scattering, using Legendre only", 
                "GNDS naming convention",
                "cross_sections",
                "Warning, LTT",
                "using Legendre only",
            ],
            suppress_prefixes=["n-00"]  # This catches all the n-001_H_002.endf, n-006_C_012.endf, etc.
        )
        
        # For maximum suppression, also redirect OS-level stderr to /dev/null
        # This catches C/C++ library writes that bypass Python's sys.stderr
        permanently_redirect_stderr_to_null()
        
        # Also suppress Python warnings at the module level
        import warnings
        warnings.filterwarnings("ignore", category=UserWarning)
        warnings.filterwarnings("ignore", message=".*LTT.*")
        warnings.filterwarnings("ignore", message=".*elastic scattering.*")
        warnings.filterwarnings("ignore", message=".*Legendre only.*")
        
    except Exception:
        # If warning suppression fails, continue without it
        pass


def _load_flux_and_microxs(flux_dir: Path) -> Tuple[np.ndarray, openmc.deplete.MicroXS]:
    """Load flux spectrum and MicroXS from the provided directory.

    Args:
        flux_dir: Directory containing the flux and MicroXS files.

    Returns:
        Tuple of (flux_spectrum, microxs).
    """
    flux_file = flux_dir / "flux_spectrum_1102.txt"
    microxs_file = flux_dir / "microxs_1102.csv"
    if not flux_file.exists():
        raise FileNotFoundError(f"Flux file not found: {flux_file}")
    if not microxs_file.exists():
        raise FileNotFoundError(f"MicroXS file not found: {microxs_file}")

    flux = np.loadtxt(flux_file, comments="#", usecols=1)
    microxs = openmc.deplete.MicroXS.from_csv(microxs_file)
    return flux, microxs


def _deplete_one(
    comp_idx: int,
    composition: Dict[str, float],
    cfg: BatchDepletionConfig,
    chain_file: str,
) -> Tuple[int, Dict[str, Any]]:
    """Worker: run a single independent depletion and parse outputs.

    Args:
        comp_idx: Index of the composition in the input list.
        composition: Mapping of element to atomic fraction (sums to 1.0).
        cfg: BatchDepletionConfig instance.
        chain_file: Resolved chain file path.

    Returns:
        Tuple of (comp_idx, result_dict).
    """
    _limit_threads(cfg.threads_per_process)

    flux, microxs = _load_flux_and_microxs(cfg.flux_microxs_dir)

    # Build model, set material in vessel cell
    model = create_model(config=cfg.geometry_config)
    model.settings.particles = cfg.particles

    mat_name = f"batch_comp_{composition_hash(composition)}"
    material = create_material(composition, mat_name)
    material.depletable = True
    vessel_cell = model.geometry.get_cells_by_name(cfg.depletable_cell)[0]
    material.volume = getattr(vessel_cell.fill, "volume", 2.13e5) or 2.13e5
    vessel_cell.fill = material

    # Schedule
    mev_to_j = 1.602176634e-13
    source_rate = (
        cfg.power_mw * 1e6 / (17.6 * mev_to_j) * cfg.torus_to_sphere_ratio
    )
    scheduler = TimeScheduler(
        irradiation_time=cfg.irradiation_time,
        cooling_times=[f"{d} days" for d in cfg.cooling_days],
        source_rate=source_rate,
        irradiation_steps=cfg.irradiation_steps,
    )
    timesteps, source_rates = scheduler.get_timesteps_and_source_rates()

    # Output dir per composition
    comp_dir = cfg.out_dir / f"comp_{composition_hash(composition)}"
    comp_dir.mkdir(parents=True, exist_ok=True)

    # Run depletion
    results = run_independent_depletion(
        model=model,
        depletable_cell=cfg.depletable_cell,
        microxs=microxs,
        flux=flux,
        chain_file=chain_file,
        timesteps=timesteps,
        source_rates=source_rates,
        outdir=comp_dir,
    )

    parsed = parse_openmc_results(
        results=results,
        chain_file=chain_file,
        abs_file=cfg.abs_file,
        cooling_days=cfg.cooling_days,
    )

    return comp_idx, {
        "composition": composition,
        "composition_id": composition_hash(composition),
        "gas_production": parsed.get("gas_production", {}),
        "dose_at_cooling_times": parsed.get("dose_at_cooling_times", {}),
        "out_dir": str(comp_dir),
    }


def run_batch_depletion(
    compositions: List[Dict[str, float]],
    config: BatchDepletionConfig,
) -> List[Dict[str, Any]]:
    """Run many independent depletions in parallel with thread-capped workers.

    Args:
        compositions: List of composition dicts that sum to 1.0.
        config: BatchDepletionConfig with flux/Xs paths and execution settings.

    Returns:
        List of result dictionaries keyed by 'composition', 'composition_id',
        'gas_production', 'dose_at_cooling_times', and 'out_dir'.
    """
    if not compositions:
        return []

    # Resolve OpenMC chain/XS
    chain_file = _set_openmc_paths_from_env(config.chain_file)

    # Prepare output root
    config.out_dir.mkdir(parents=True, exist_ok=True)

    # Determine worker pool size - limit to avoid I/O contention
    max_by_threads = max(1, config.threads_available // max(1, config.threads_per_process))
    n_workers = min(len(compositions), config.max_workers or max_by_threads)
    
    # Limit workers to avoid overwhelming the system with nuclear data loading
    n_workers = min(n_workers, 2)  # Maximum 2 parallel processes to avoid I/O contention
    
    print(f"Running {len(compositions)} compositions with {n_workers} workers ({config.threads_per_process} threads each)")
    print(f"Note: Limited to 2 workers to avoid nuclear data loading contention")

    start = time.time()
    results: List[Dict[str, Any]] = []

    # Use spawn to avoid OpenMP fork issues
    import multiprocessing as mp

    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=n_workers,
        mp_context=ctx,
        initializer=_limit_threads,
        initargs=(config.threads_per_process,),
    ) as ex:
        futs = {
            ex.submit(_deplete_one, i, comp, config, chain_file): i
            for i, comp in enumerate(compositions)
        }
        completed = 0
        for fut in as_completed(futs):
            try:
                _idx, row = fut.result()
                results.append(row)
                completed += 1
                print(f"Completed {completed}/{len(compositions)} compositions")
                
                # Check for errors
                if 'error' in row:
                    print(f"  - Composition {_idx} failed: {row['error']}")
                
            except Exception as e:
                # Capture failure as a result row for traceability
                i = futs[fut]
                error_row = {
                    "composition": compositions[i],
                    "composition_id": composition_hash(compositions[i]),
                    "error": str(e),
                }
                results.append(error_row)
                completed += 1
                print(f"Completed {completed}/{len(compositions)} compositions (with error)")
                print(f"  - Composition {i} failed: {str(e)}")

    elapsed = time.time() - start
    print(
        f"Batch depletion: {len(compositions)} comps across {n_workers} workers "
        f"({config.threads_per_process} threads each) in {elapsed:.1f}s"
    )
    return results


