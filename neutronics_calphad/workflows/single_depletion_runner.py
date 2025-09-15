#!/usr/bin/env python3
"""Standalone script to run depletion for a single composition.

This script is designed to be called as a subprocess to provide proper
process isolation and avoid OpenMC nuclear data loading conflicts.

Usage:
    python single_depletion_runner.py --composition '{"V":0.8,"Cr":0.05,"Ti":0.05,"W":0.05,"Zr":0.05}' \
        --output-dir results/comp_123 --config config.json

The script writes results to output_dir/depletion_result.json
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np  # type: ignore
import openmc  # type: ignore

# Add project root to path if needed
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from neutronics_calphad.neutronics.config import SPHERICAL  # noqa: E402
from neutronics_calphad.neutronics.depletion import run_independent_depletion  # noqa: E402
from neutronics_calphad.neutronics.geometry_maker import create_model  # noqa: E402
from neutronics_calphad.neutronics.time_scheduler import TimeScheduler  # noqa: E402
from neutronics_calphad.optimizer.parsers import parse_openmc_results  # noqa: E402
from neutronics_calphad.utils.io import create_material  # noqa: E402


def setup_openmc_paths(config: Dict[str, Any]) -> None:
    """Set up OpenMC configuration from config dict."""
    chain_file = config.get('chain_file') or os.environ.get(
        'OPENMC_CHAIN_FILE',
        '/home/myless/nuclear_data/tendl21-fispact20-gefy61-chain.xml'
    )
    cross_sections = config.get('cross_sections') or os.environ.get(
        'OPENMC_CROSS_SECTIONS',
        '/home/myless/nuclear_data/tendl-2021-hdf5/cross_sections.xml'
    )
    
    openmc.config['chain_file'] = chain_file
    openmc.config['cross_sections'] = cross_sections
    
    # Suppress OpenMC warnings
    os.environ['OPENMC_LOG_LEVEL'] = 'ERROR'
    # Optional: further suppress noisy output
    os.environ.setdefault('OPENMC_SUPPRESS_WARNINGS', '1')
    os.environ.setdefault('OPENMC_QUIET', '1')


def _round_composition_for_output(
    composition: Dict[str, float],
    alloy_decimals: int = 3,
    impurity_decimals: int = 6,
    v_decimals: int = 6,
) -> Dict[str, float]:
    """Round composition values for stable JSON output.

    Cr, Ti, W, Zr are rounded to ``alloy_decimals`` and impurities
    (C, N, O) to ``impurity_decimals``. V is set as the balance after
    rounding others and then rounded to ``v_decimals``. A final small
    correction is applied to V to enforce closure within rounding.

    Args:
        composition: Element-to-fraction mapping (sums to ≈ 1.0).
        alloy_decimals: Decimal places for Cr, Ti, W, Zr.
        impurity_decimals: Decimal places for C, N, O.
        v_decimals: Decimal places for V.

    Returns:
        Rounded composition dict that sums to 1.0 within rounding.
    """
    alloy_keys = {"Cr", "Ti", "W", "Zr"}
    impurity_keys = {"C", "N", "O"}

    rounded: Dict[str, float] = {}

    # Round all elements except V first
    for elem, frac in composition.items():
        if elem == "V":
            continue
        try:
            val = float(frac)
        except Exception:
            continue
        if elem in alloy_keys:
            rounded[elem] = round(val, alloy_decimals)
        elif elem in impurity_keys:
            rounded[elem] = round(val, impurity_decimals)
        else:
            # Default rounding for any other trace species
            rounded[elem] = round(val, impurity_decimals)

    # Compute V as balance and round
    others_sum = sum(rounded.values())
    v_balance = max(0.0, 1.0 - others_sum)
    rounded["V"] = round(v_balance, v_decimals)

    # Final correction to enforce exact closure within rounding resolution
    total = sum(rounded.values())
    if abs(total - 1.0) > 1e-12:
        delta = 1.0 - total
        rounded["V"] = round(rounded["V"] + delta, v_decimals)

    return rounded


def run_single_depletion(
    composition: Dict[str, float],
    output_dir: Path,
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """Run depletion for a single composition.
    
    Args:
        composition: Dict mapping elements to atomic fractions
        output_dir: Directory to save outputs
        config: Configuration dict with all parameters
        
    Returns:
        Result dict with gas_production and dose_at_cooling_times
    """
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    # In case multiple processes are writing in parallel on WSL/NTFS, relax HDF5 file locking
    os.environ.setdefault('HDF5_USE_FILE_LOCKING', 'FALSE')
    
    # Optional startup staggering to avoid I/O spikes across many concurrent jobs
    stagger = float(config.get('stagger_start_s', 0.0) or 0.0)
    if stagger > 0.0:
        import random, time as _t
        _t.sleep(random.uniform(0.0, stagger))

    # Prewarm chain/ABS to warm page cache if desired (avoid heavy seeks on many workers)
    try:
        if bool(config.get('prewarm_chain', False)):
            _cf = Path(openmc.config['chain_file'])
            if _cf.exists():
                with open(_cf, 'rb') as _f:
                    _f.read(1024 * 1024)
        if bool(config.get('prewarm_abs', False)):
            _af = Path(str(config.get('abs_file', '')))
            if _af.exists():
                # Touch a few files in the directory if it's a directory path
                if _af.is_dir():
                    for i, p in enumerate(sorted(_af.glob('**/*'))):
                        if p.is_file():
                            with open(p, 'rb') as _f:
                                _ = _f.read(4096)
                        if i > 10:
                            break
                else:
                    with open(_af, 'rb') as _f:
                        _ = _f.read(1024 * 1024)
    except Exception:
        pass

    # Load flux and microxs
    flux_dir = Path(config['flux_microxs_dir'])
    flux_file = flux_dir / "flux_spectrum_1102.txt"
    microxs_file = flux_dir / "microxs_1102.csv"
    
    if not flux_file.exists():
        raise FileNotFoundError(f"Flux file not found: {flux_file}")
    if not microxs_file.exists():
        raise FileNotFoundError(f"MicroXS file not found: {microxs_file}")
    
    flux = np.loadtxt(flux_file, comments='#', usecols=1)
    microxs = openmc.deplete.MicroXS.from_csv(microxs_file)
    
    # Create model and set material
    model = create_model(config=config.get('geometry_config', SPHERICAL))
    model.settings.particles = config.get('particles', 10000)
    
    # Round composition for stability and ensure V is exact balance BEFORE creating material
    composition = _round_composition_for_output(composition)
    # Create material
    mat_name = f"comp_{output_dir.name}"
    material = create_material(composition, mat_name)
    material.depletable = True
    
    # Set vessel material
    vessel_cell = model.geometry.get_cells_by_name(config.get('depletable_cell', 'vessel'))[0]
    material.volume = getattr(vessel_cell.fill, 'volume', 2.13e5) or 2.13e5
    vessel_cell.fill = material
    
    # Set up time schedule
    mev_to_j = 1.602176634e-13
    source_rate = (
        config.get('power_mw', 500) * 1e6 / 
        (17.6 * mev_to_j) * 
        config.get('torus_to_sphere_ratio', 1/4.03)
    )
    
    scheduler = TimeScheduler(
        irradiation_time=config.get('irradiation_time', '2 years'),
        cooling_times=[f"{d} days" for d in config.get('cooling_days', [30, 365, 5*365, 36500])],
        source_rate=source_rate,
        irradiation_steps=config.get('irradiation_steps', 24)
    )
    timesteps, source_rates = scheduler.get_timesteps_and_source_rates()
    
    # Run depletion
    results = run_independent_depletion(
        model=model,
        depletable_cell=config.get('depletable_cell', 'vessel'),
        microxs=microxs,
        flux=[flux],
        chain_file=config['chain_file'],
        timesteps=timesteps,
        source_rates=source_rates,
        outdir=output_dir
    )
    
    # Parse results
    parsed = parse_openmc_results(
        results=results,
        chain_file=config['chain_file'],
        abs_file=config.get('abs_file', '/home/myless/Packages/fispact/nuclear_data/decay/abs_2012'),
        cooling_days=config.get('cooling_days', [30, 365, 5*365, 36500])
    )

    # Robust dose extraction fallback if keys missing or zeros returned
    try:
        dose_dict = parsed.get('dose_at_cooling_times', {}) or {}
        need_days = list(config.get('cooling_days', [30, 365, 5*365, 36500]))
        missing = [d for d in need_days if d not in dose_dict or not isinstance(dose_dict.get(d), (int, float))]
        # Determine irradiation end from scheduler used above (cumulative timesteps)
        # results[i].time[0] is cumulative time in seconds at end of step i
        cum_times = [results[i].time[0] for i in range(len(results))] if len(results) else []
        irrad_end_s = float(cum_times[-1]) if cum_times else 0.0

        # Trigger fallback if missing, or if all zero AND post-irradiation sampling would expect non-zero
        if missing or all(abs(float(v)) < 1e-20 for v in dose_dict.values()):
            from neutronics_calphad.neutronics.dose import contact_dose  # type: ignore
            times_s, dose_dicts = contact_dose(
                results=results,
                chain_file=config['chain_file'],
                abs_file=config.get('abs_file', '/home/myless/Packages/fispact/nuclear_data/decay/abs_2012')
            )
            # Build dense series and sample requested cooling points (after irradiation end)
            # Use irrad_end_s directly from results; if zero, still sample indices safely
            # Map total dose at nearest times to expected cooling days
            import numpy as _np
            total_dose = _np.array([sum(d.values()) for d in dose_dicts], dtype=float)
            time_arr = _np.array(times_s, dtype=float)
            fixed = {}
            for days in need_days:
                t_abs = irrad_end_s + float(days) * 24.0 * 3600.0
                idx = int(_np.argmin(_np.abs(time_arr - t_abs))) if time_arr.size else 0
                fixed[int(days)] = float(total_dose[idx]) if total_dose.size else 0.0
            parsed['dose_at_cooling_times'] = fixed

        # Optional debug dump for dose timeline
        if bool(config.get('debug_dose_dump', False)):
            try:
                from neutronics_calphad.neutronics.dose import contact_dose  # type: ignore
                times_s, dose_dicts = contact_dose(
                    results=results,
                    chain_file=config['chain_file'],
                    abs_file=config.get('abs_file', '/home/myless/Packages/fispact/nuclear_data/decay/abs_2012')
                )
                import numpy as _np
                total_dose = [float(sum(d.values())) for d in dose_dicts]
                dbg = {
                    'irrad_end_s': irrad_end_s,
                    'times_s': list(map(float, times_s)),
                    'total_dose': total_dose,
                    'dose_at_cooling_times': {int(k): float(v) for k, v in (parsed.get('dose_at_cooling_times') or {}).items()}
                }
                with open(output_dir / 'dose_debug.json', 'w') as _f:
                    json.dump(dbg, _f, indent=2)
            except Exception:
                pass
    except Exception:
        # keep parsed as-is if fallback fails
        pass
    
    return {
        'composition': _round_composition_for_output(composition),
        'gas_production': parsed.get('gas_production', {}),
        'dose_at_cooling_times': parsed.get('dose_at_cooling_times', {}),
        'output_dir': str(output_dir)
    }


def main() -> None:
    """Main entry point for subprocess execution."""
    parser = argparse.ArgumentParser(description='Run single depletion calculation')
    parser.add_argument('--composition', type=str, required=True,
                        help='JSON string of composition dict')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Output directory path')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to configuration JSON file')
    
    args = parser.parse_args()
    
    try:
        # Parse inputs
        composition = json.loads(args.composition)
        output_dir = Path(args.output_dir)
        
        # Load config
        with open(args.config, 'r') as f:
            config = json.load(f)
        
        # Set up OpenMC
        setup_openmc_paths(config)
        
        # Run depletion
        result = run_single_depletion(composition, output_dir, config)
        
        # Save result
        result_file = output_dir / 'depletion_result.json'
        with open(result_file, 'w') as f:
            json.dump(result, f, indent=2)
        
        # Print success for logging
        print(f"SUCCESS: {output_dir.name}")
        
    except Exception as e:
        # Save error
        error_file = Path(args.output_dir) / 'error.json'
        error_file.parent.mkdir(parents=True, exist_ok=True)
        with open(error_file, 'w') as f:
            json.dump({'error': str(e), 'composition': args.composition}, f)
        
        print(f"FAILED: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
