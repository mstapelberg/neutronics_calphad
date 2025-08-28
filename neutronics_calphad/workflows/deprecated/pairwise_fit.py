"""Pairwise correction fitting utilities for impulse depletion surrogate.

Builds small 2D residual surfaces for element pairs using a grid of two-element
impulses and fits a quadratic form per metric:

    Δy_ij(xi,xj) ≈ c_ij xi xj + ai xi^2 + aj xj^2

Metrics supported:
- Dose at cooling times (per configured days)
- Gas (He_appm, H_appm)

Outputs a JSON dictionary keyed by pair name (e.g., "Cr+W").
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Any

import json
import os
import time
from contextlib import suppress
from datetime import datetime
import numpy as np  # type: ignore
import openmc  # type: ignore

from .impulse_depletion import ImpulseLibrary
from .impulse_depletion import composition_hash
from neutronics_calphad.neutronics.config import SPHERICAL  # type: ignore
from neutronics_calphad.neutronics.geometry_maker import create_model  # type: ignore
from neutronics_calphad.neutronics.flux import get_flux_and_microxs  # type: ignore
from neutronics_calphad.neutronics.depletion import run_independent_depletion  # type: ignore
from neutronics_calphad.neutronics.depletion import validate_depletion_results  # type: ignore
from neutronics_calphad.neutronics.time_scheduler import TimeScheduler  # type: ignore
from neutronics_calphad.optimizer.parsers import parse_openmc_results  # type: ignore
from neutronics_calphad.utils.io import create_material  # type: ignore


@dataclass
class PairwiseFitConfig:
    """Configuration for pairwise correction fitting.

    Attributes:
        pairs: List of element pairs to fit (e.g., [("V","W"), ("Cr","W")]).
        grid_fracs: Grid of atomic fractions to sample for each element in the pair.
        cooling_days: Cooling days for dose metrics.
        use_full_depletion: If True, compute targets with full depletion runs; if False,
            use the impulse synthesis (will yield near-zero residuals).
        dose_terms: Polynomial terms for dose correction fit. "full" → [xi*xj, xi^2, xj^2],
            "cross_only" → only xi*xj (sets ai=aj=0).
        ridge_lambda: Ridge regularization strength (λ ≥ 0) applied to all fits.
        gas_model: How to fit gas corrections. "additive" → fit absolute Δ; "fractional"
            → fit (target-linear)/max(linear, eps) and apply multiplicatively in synth.
        gas_terms: Polynomial terms for gas correction fit (same options as dose_terms).
    """
    pairs: List[Tuple[str, str]]
    grid_fracs: List[float]
    cooling_days: List[int]
    use_full_depletion: bool = True
    dose_terms: str = "full"
    ridge_lambda: float = 0.0
    gas_model: str = "fractional"
    gas_terms: str = "cross_only"


def _build_design_matrix(xi: np.ndarray, xj: np.ndarray) -> np.ndarray:
    """Design matrix for quadratic cross terms [xi*xj, xi^2, xj^2]."""
    return np.stack([xi * xj, xi * xi, xj * xj], axis=1)


def _normalize_composition(comp: Dict[str, float], eps: float = 1e-12) -> Dict[str, float]:
    """Normalize a composition to sum to 1.0 with small-value filtering.

    Parameters
    ----------
    comp : Dict[str, float]
        Mapping of element symbol to atomic fraction.
    eps : float
        Threshold for dropping tiny/negative entries.

    Returns
    -------
    Dict[str, float]
        New mapping scaled to sum exactly to 1.0.
    """
    filtered: Dict[str, float] = {k: max(0.0, float(v)) for k, v in comp.items() if abs(float(v)) > eps}
    total = sum(filtered.values())
    if total <= 0.0:
        raise ValueError("Composition has non-positive total fraction after filtering")
    return {k: v / total for k, v in filtered.items()}


def _fit_quadratic_residual(
    xi: np.ndarray,
    xj: np.ndarray,
    dy: np.ndarray,
    ridge_lambda: float = 0.0,
) -> Tuple[float, float, float]:
    """Fit dy ≈ c*xi*xj + ai*xi^2 + aj*xj^2 with optional ridge regularization.

    Parameters
    ----------
    xi, xj : np.ndarray
        Element fractions arrays.
    dy : np.ndarray
        Residual vector to fit.
    ridge_lambda : float
        Non-negative ridge regularization coefficient (λ).
    """
    A = _build_design_matrix(xi, xj)
    if ridge_lambda and ridge_lambda > 0.0:
        AtA = A.T @ A
        Atb = A.T @ dy
        coef = np.linalg.solve(AtA + ridge_lambda * np.eye(AtA.shape[0]), Atb)
    else:
        coef, _, _, _ = np.linalg.lstsq(A, dy, rcond=None)
    c, ai, aj = coef.tolist()
    return float(c), float(ai), float(aj)


def _ensure_flux_microxs(library_dir: Path, chain_file: str) -> Tuple[Path, Path]:
    """Ensure flux and microxs files exist; return their paths.

    Parameters
    ----------
    library_dir : Path
        Impulse library directory.
    chain_file : str
        OpenMC depletion chain file path.

    Returns
    -------
    Tuple[Path, Path]
        (flux_file, microxs_file)
    """
    flux_dir = library_dir / "flux_microxs"
    flux_dir.mkdir(exist_ok=True)
    flux_file = flux_dir / "flux_spectrum_1102.txt"
    microxs_file = flux_dir / "microxs_1102.csv"
    if not (flux_file.exists() and microxs_file.exists()):
        lock_file = flux_dir / ".generate.lock"
        # Attempt to acquire a simple file lock to avoid concurrent generation
        acquired = False
        try:
            fd = os.open(str(lock_file), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
            acquired = True
        except FileExistsError:
            acquired = False

        if acquired:
            try:
                model = create_model(config=SPHERICAL)
                model.settings.particles = 10000
                get_flux_and_microxs(
                    model,
                    chain_file=chain_file,
                    group_structure='UKAEA-1102',
                    outdir=flux_dir,
                )
            finally:
                with suppress(Exception):
                    os.remove(lock_file)
        else:
            # Wait for another process to finish generation
            for _ in range(600):  # up to ~10 minutes
                if flux_file.exists() and microxs_file.exists():
                    break
                time.sleep(1.0)
    return flux_file, microxs_file


def _run_full_depletion_for_composition(
    library: ImpulseLibrary,
    composition: Dict[str, float],
    cooling_days: List[int],
) -> Dict[str, Any]:
    """Run a full depletion calculation for a given composition and parse outputs.

    Uses the same geometry, chain, and flux/microxs bundle as the impulse library.

    Parameters
    ----------
    library : ImpulseLibrary
        Loaded impulse library.
    composition : Dict[str, float]
        Composition mapping that must sum to 1.0.
    cooling_days : List[int]
        Cooling days at which dose will be extracted.

    Returns
    -------
    Dict[str, Any]
        Dictionary with keys 'dose_at_cooling_times' and 'gas_production'.
    """
    # Normalize composition defensively
    composition = _normalize_composition(composition)

    # Resolve inputs from metadata
    chain_file = str(library.metadata.get('chain_file', openmc.config.get('chain_file')))
    abs_file = str(library.metadata.get('abs_file', '/home/myless/Packages/fispact/nuclear_data/decay/abs_2012'))

    # Ensure flux/microxs
    flux_file, microxs_file = _ensure_flux_microxs(library.library_dir, chain_file)  # type: ignore[attr-defined]

    # Cache key and paths
    comp_id = composition_hash(composition)
    days_key = "-".join(str(d) for d in sorted(cooling_days))
    cache_dir = library.library_dir / "pairwise_full"  # type: ignore[attr-defined]
    cache_dir.mkdir(exist_ok=True)
    cache_json = cache_dir / f"target_{comp_id}_{days_key}.json"
    # Unique run directory to avoid parallel write collisions
    run_dir = cache_dir / "runs" / f"{comp_id}_{days_key}"
    run_dir.mkdir(parents=True, exist_ok=True)

    if cache_json.exists():
        try:
            with open(cache_json, 'r') as f:
                cached = json.load(f)
            if set(map(int, cached.get('cooling_days', []))) == set(cooling_days):
                print(f"✓ Loading from cache: {cache_json}")
                return {
                    'dose_at_cooling_times': cached.get('dose_at_cooling_times', {}),
                    'gas_production': cached.get('gas_production', {}),
                }
        except Exception as e:
            print(f"⚠ Cache read failed: {e}")
    else:
        print(f"⚠ Cache not found: {cache_json}")
        print(f"  Composition: {composition}")
        print(f"  Cooling days: {cooling_days}")
        print(f"  Comp ID: {comp_id}")
        print(f"  Days key: {days_key}")

    # Load flux and microxs
    flux = [np.loadtxt(str(flux_file), comments='#', usecols=1)]
    microxs = openmc.deplete.MicroXS.from_csv(str(microxs_file))  # type: ignore[attr-defined]

    # Create model and material
    model = create_model(config=SPHERICAL)
    model.settings.particles = 10000
    material_name = "pair_fit_comp"
    material = create_material(composition, material_name)
    material.depletable = True
    vessel_cell = model.geometry.get_cells_by_name('vessel')[0]
    # Copy volume from existing vessel material if present
    if getattr(vessel_cell.fill, 'volume', None):
        material.volume = vessel_cell.fill.volume
    else:
        material.volume = 2.13e5
    vessel_cell.fill = material

    # Schedule (match library build defaults)
    POWER_MW = 500.0
    TORUS_TO_SPHERE_VOLUME_RATIO = 1 / 4.03
    FUSION_POWER_MEV = 17.6
    MEV_TO_J = 1.602176634e-13
    source_rate = POWER_MW * 1e6 / (FUSION_POWER_MEV * MEV_TO_J) * TORUS_TO_SPHERE_VOLUME_RATIO

    scheduler = TimeScheduler(
        irradiation_time='2 years',
        cooling_times=[f"{d} days" for d in cooling_days],
        source_rate=source_rate,
        irradiation_steps=24,
    )
    timesteps, source_rates = scheduler.get_timesteps_and_source_rates()

    # Progress tracking
    progress_path = run_dir / "progress.json"
    started_at = datetime.utcnow().isoformat() + 'Z'
    try:
        with open(progress_path, 'w') as f:
            json.dump({
                'status': 'running',
                'started_at': started_at,
                'cooling_days': cooling_days,
                'composition': dict(sorted(composition.items(), key=lambda kv: kv[0])),
            }, f, indent=2)
    except Exception:
        pass

    # Persist composition input for traceability
    with suppress(Exception):
        with open(run_dir / 'composition.json', 'w') as f:
            json.dump({
                'composition': dict(sorted(composition.items(), key=lambda kv: kv[0])),
                'cooling_days': cooling_days,
                'chain_file': chain_file,
            }, f, indent=2)

    # Run depletion into a unique directory for this composition
    t0 = time.time()
    try:
        results = run_independent_depletion(
            model=model,
            depletable_cell='vessel',
            microxs=microxs,
            flux=flux,
            chain_file=chain_file,
            timesteps=timesteps,
            source_rates=source_rates,
            outdir=run_dir,
        )
        # Print timestep info for debugging
        with suppress(Exception):
            expected_len = len(timesteps)
            actual_len = len(results)
            print(f"Timesteps: expected={expected_len}, actual={actual_len}")
            print(f"Expected timesteps (delta): {timesteps}")
            # Extract delta timesteps from cumulative times
            actual_cumulative = [results[i].time[0] for i in range(len(results))]
            actual_deltas = [actual_cumulative[i] - actual_cumulative[i-1] if i > 0 else actual_cumulative[i] for i in range(len(actual_cumulative))]
            print(f"Actual timesteps (delta): {actual_deltas}")
            print(f"Actual timesteps (cumulative): {actual_cumulative}")
            with open(run_dir / 'validation.json', 'w') as vf:
                json.dump({
                    'expected_steps': expected_len, 
                    'actual_steps': actual_len,
                    'timestep_diff': actual_len - expected_len,
                    'expected_timesteps_delta': timesteps,
                    'actual_timesteps_delta': actual_deltas,
                    'actual_timesteps_cumulative': [float(x) for x in actual_cumulative],
                }, vf, indent=2)
    except Exception as e:
        with suppress(Exception):
            with open(progress_path, 'w') as f:
                json.dump({
                    'status': 'failed',
                    'started_at': started_at,
                    'finished_at': datetime.utcnow().isoformat() + 'Z',
                    'duration_s': time.time() - t0,
                    'cooling_days': cooling_days,
                    'composition': dict(sorted(composition.items(), key=lambda kv: kv[0])),
                    'error': str(e),
                }, f, indent=2)
        raise

    # Parse
    try:
        parsed = parse_openmc_results(
            results=results,
            chain_file=chain_file,
            abs_file=abs_file,
            cooling_days=cooling_days,
        )
    except Exception as e:
        with suppress(Exception):
            with open(progress_path, 'w') as f:
                json.dump({
                    'status': 'failed_parse',
                    'started_at': started_at,
                    'finished_at': datetime.utcnow().isoformat() + 'Z',
                    'duration_s': time.time() - t0,
                    'cooling_days': cooling_days,
                    'composition': dict(sorted(composition.items(), key=lambda kv: kv[0])),
                    'error': str(e),
                }, f, indent=2)
        raise
    out = {
        'dose_at_cooling_times': parsed.get('dose_at_cooling_times', {}),
        'gas_production': parsed.get('gas_production', {}),
    }
    # Save cache (include composition and run metadata). Write atomically to avoid corruption.
    try:
        tmp_path = cache_json.with_suffix('.json.tmp')
        payload: Dict[str, Any] = {
            'cooling_days': cooling_days,
            'composition': dict(sorted(composition.items(), key=lambda kv: kv[0])),
            'metadata': {
                'generated_at': datetime.utcnow().isoformat() + 'Z',
                'base_element': (library.metadata.get('base_element') if isinstance(library.metadata, dict) else None) or 'V',
                'run_dir': str(run_dir),
                'results_h5': str(run_dir / 'depletion_results.h5'),
                'timesteps': timesteps,
            },
            **out,
        }
        with open(tmp_path, 'w') as f:
            json.dump(payload, f, indent=2)
        os.replace(tmp_path, cache_json)
    except Exception:
        pass

    # Update progress file to succeeded
    with suppress(Exception):
        with open(progress_path, 'w') as f:
            json.dump({
                'status': 'succeeded',
                'started_at': started_at,
                'finished_at': datetime.utcnow().isoformat() + 'Z',
                'duration_s': time.time() - t0,
                'cooling_days': cooling_days,
                'composition': dict(sorted(composition.items(), key=lambda kv: kv[0])),
                'output_json': str(cache_json),
            }, f, indent=2)
    return out


def fit_pairwise_corrections(
    library: ImpulseLibrary,
    cfg: PairwiseFitConfig,
) -> Dict[str, Dict[str, float]]:
    """Fit pairwise corrections for specified element pairs.

    For each pair (e1,e2) and for each grid point (xi,xj), compute a target via
    full depletion (if enabled) and the linear prediction via impulse synthesis.
    The residual Δy is then fit to a small quadratic surface.

    Returns a dict suitable for saving to library.pairwise_corrections.
    """
    corrections: Dict[str, Dict[str, float]] = {}

    # Build base grid once
    fracs = np.array(cfg.grid_fracs, dtype=float)

    for e1, e2 in cfg.pairs:
        # Determine base element and whether this pair includes it
        base_el = library.metadata.get('base_element', 'V') if isinstance(library.metadata, dict) else 'V'
        includes_base = (e1 == base_el) or (e2 == base_el)
        pair_key = "+".join(sorted([e1, e2]))
        corr: Dict[str, float] = {}

        # Build local grids for this pair
        if includes_base:
            # Use 1D grid: x_other varies in fracs, x_base = 1 - x_other
            # For V+W: xi_local = [0.0025, 0.005, ..., 0.20], xj_local = [0.9975, 0.995, ..., 0.80]
            xi_local = fracs.copy()
            xj_local = 1.0 - xi_local
        else:
            # Full 2D grid
            XI, XJ = np.meshgrid(fracs, fracs, indexing='ij')
            xi_local = XI.ravel()
            xj_local = XJ.ravel()

        n_pts = xi_local.size

        # Construct arrays for targets and linear parts
        dose_targets: Dict[int, np.ndarray] = {d: np.zeros(n_pts, dtype=float) for d in cfg.cooling_days}
        dose_linear: Dict[int, np.ndarray] = {d: np.zeros(n_pts, dtype=float) for d in cfg.cooling_days}
        he_target = np.zeros(n_pts, dtype=float)
        he_linear = np.zeros(n_pts, dtype=float)
        h_target = np.zeros(n_pts, dtype=float)
        h_linear = np.zeros(n_pts, dtype=float)

        valid_indices: List[int] = []
        xi_eff_list: List[float] = []
        xj_eff_list: List[float] = []

        for k in range(n_pts):
            # Build valid composition that sums to 1.0
            if includes_base:
                # Pair is (base_el, other)
                # For V+W: xi_local[k] is the W fraction, xj_local[k] is the V fraction
                if e1 == base_el:  # e1 is V, e2 is W
                    x_other = float(xi_local[k])  # W fraction
                    v_frac = float(xj_local[k])   # V fraction (1 - x_other)
                else:  # e1 is W, e2 is V
                    x_other = float(xj_local[k])  # W fraction
                    v_frac = float(xi_local[k])   # V fraction (1 - x_other)
                comp = {base_el: v_frac, (e2 if e1 == base_el else e1): x_other}
                xi_eff = float(comp.get(e1, 0.0))
                xj_eff = float(comp.get(e2, 0.0))
            else:
                v_frac = max(0.0, 1.0 - float(xi_local[k]) - float(xj_local[k]))
                comp = {base_el: v_frac, e1: float(xi_local[k]), e2: float(xj_local[k])}
                xi_eff = float(xi_local[k])
                xj_eff = float(xj_local[k])

            # Normalize composition to avoid tiny sum errors
            comp = _normalize_composition(comp)
            if not np.isclose(sum(comp.values()), 1.0):
                continue
            if v_frac <= 0.0:
                continue

            # Compute target (full depletion) or impulse-only (fast)
            if cfg.use_full_depletion:
                target = _run_full_depletion_for_composition(library, comp, cfg.cooling_days)
            else:
                target = library.synthesize(comp, cooling_days=cfg.cooling_days, disable_pairwise=True)  # type: ignore[assignment]

            # Linear part: impulse-only prediction for the full composition
            lin_pred = library.synthesize(comp, cooling_days=cfg.cooling_days, disable_pairwise=True)

            # Record target and linear values
            for d in cfg.cooling_days:
                dose_targets[d][k] = float(target['dose_at_cooling_times'].get(d, 0.0))
                dose_linear[d][k] = float(lin_pred['dose_at_cooling_times'].get(d, 0.0))
            he_target[k] = float(target['gas_production'].get('He_appm', 0.0))
            he_linear[k] = float(lin_pred['gas_production'].get('He_appm', 0.0))
            h_target[k] = float(target['gas_production'].get('H_appm', 0.0))
            h_linear[k] = float(lin_pred['gas_production'].get('H_appm', 0.0))

            valid_indices.append(k)
            xi_eff_list.append(xi_eff)
            xj_eff_list.append(xj_eff)

        if not valid_indices:
            print(f"Warning: No valid compositions found for pair {e1}+{e2}")
            continue

        valid_xi = np.array(xi_eff_list, dtype=float)
        valid_xj = np.array(xj_eff_list, dtype=float)

        # Filter the arrays to valid indices
        for d in cfg.cooling_days:
            dose_targets[d] = dose_targets[d][valid_indices]
            dose_linear[d] = dose_linear[d][valid_indices]
        he_target = he_target[valid_indices]
        he_linear = he_linear[valid_indices]
        h_target = h_target[valid_indices]
        h_linear = h_linear[valid_indices]

        # Fit dose corrections per cooling day
        for d in cfg.cooling_days:
            dy = dose_targets[d] - dose_linear[d]
            c, ai, aj = _fit_quadratic_residual(valid_xi, valid_xj, dy, ridge_lambda=cfg.ridge_lambda)
            if cfg.dose_terms == "cross_only":
                ai, aj = 0.0, 0.0
            corr[f'dose_{int(d)}d_c'] = c
            corr[f'dose_{int(d)}d_ai'] = ai
            corr[f'dose_{int(d)}d_aj'] = aj

        # Fit gas corrections (additive or fractional)
        eps = 1e-12
        if cfg.gas_model == "fractional":
            # Fit fractional residuals r = (target - linear) / max(linear, eps)
            r_he = (he_target - he_linear) / np.maximum(he_linear, eps)
            c, ai, aj = _fit_quadratic_residual(valid_xi, valid_xj, r_he, ridge_lambda=cfg.ridge_lambda)
            if cfg.gas_terms == "cross_only":
                ai, aj = 0.0, 0.0
            corr['He_appm_frac_c'] = c
            corr['He_appm_frac_ai'] = ai
            corr['He_appm_frac_aj'] = aj

            r_h = (h_target - h_linear) / np.maximum(h_linear, eps)
            c, ai, aj = _fit_quadratic_residual(valid_xi, valid_xj, r_h, ridge_lambda=cfg.ridge_lambda)
            if cfg.gas_terms == "cross_only":
                ai, aj = 0.0, 0.0
            corr['H_appm_frac_c'] = c
            corr['H_appm_frac_ai'] = ai
            corr['H_appm_frac_aj'] = aj
        else:
            # Additive model (legacy)
            dy_he = he_target - he_linear
            c, ai, aj = _fit_quadratic_residual(valid_xi, valid_xj, dy_he, ridge_lambda=cfg.ridge_lambda)
            if cfg.gas_terms == "cross_only":
                ai, aj = 0.0, 0.0
            corr['He_appm_c'] = c
            corr['He_appm_ai'] = ai
            corr['He_appm_aj'] = aj

            dy_h = h_target - h_linear
            c, ai, aj = _fit_quadratic_residual(valid_xi, valid_xj, dy_h, ridge_lambda=cfg.ridge_lambda)
            if cfg.gas_terms == "cross_only":
                ai, aj = 0.0, 0.0
            corr['H_appm_c'] = c
            corr['H_appm_ai'] = ai
            corr['H_appm_aj'] = aj

        corrections[pair_key] = corr

    return corrections


def save_pairwise_corrections(library_dir: Path, corrections: Dict[str, Dict[str, float]]) -> Path:
    """Persist pairwise corrections to JSON alongside the library.

    Parameters
    ----------
    library_dir : Path
        Impulse library directory.
    corrections : Dict[str, Dict[str, float]]
        Corrections dictionary keyed by pair name.

    Returns
    -------
    Path
        Path to the saved JSON file.
    """
    out = Path(library_dir) / "pairwise_corrections.json"
    merged: Dict[str, Dict[str, float]] = {}
    if out.exists():
        try:
            with open(out, 'r') as f:
                existing = json.load(f)
            if isinstance(existing, dict):
                merged.update(existing)
        except Exception:
            pass
    # Update/overwrite with new ones
    merged.update(corrections)
    with open(out, 'w') as f:
        json.dump(merged, f, indent=2)
    return out

