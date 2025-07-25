"""Data parsers for extracting standardized results from neutronics and CALPHAD calculations."""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Union
import logging
import json

logger = logging.getLogger(__name__)


def parse_openmc_results(
    results,  # openmc.deplete.Results
    chain_file: str,
    abs_file: str
) -> Dict[str, Any]:
    """
    Parse OpenMC depletion results to extract dose rates and gas production data.
    
    Parameters
    ----------
    results : openmc.deplete.Results
        Depletion results object.
    chain_file : str
        Path to the depletion chain file.
    abs_file : str
        Path to the absorption cross-section file.
        
    Returns
    -------
    Dict[str, Any]
        Dictionary containing raw extracted data:
        - dose_at_cooling_times: Dict mapping cooling days to dose rates (Sv/h/kg)
        - gas_production: Dict mapping gas species to production rates (appm)
        - times_s: Time series data
        - source_rates: Source rate data
        - final_irr_time: End of irradiation time
        - cool_start_time: Start of cooling time
    """
    from neutronics_calphad.neutronics.dose import contact_dose
    from neutronics_calphad.neutronics.depletion import extract_gas_production
    
    # Compute full dose-time series
    times_s, dose_dicts = contact_dose(results=results, chain_file=chain_file, abs_file=abs_file)
    
    # Compute gas production (appm)
    gas_production = extract_gas_production(results)
    
    # Identify end of irradiation (last non-zero source rate)
    source_rates = results.get_source_rates()
    final_idx = np.nonzero(source_rates)[0][-1]
    final_irr_time = times_s[final_idx]
    
    # Build a total-dose lookup
    total_dose = {t: sum(d.values()) for t, d in zip(times_s, dose_dicts)}
    
    # Locate the first cooling step
    cool_start_idx = final_idx + 1
    if cool_start_idx >= len(times_s):
        raise RuntimeError("No post-irradiation time steps available for cooling-time checks.")
    
    cool_start_time = times_s[cool_start_idx]
    
    # Build cooling-time array and corresponding doses
    cool_times = times_s[cool_start_idx:] - cool_start_time
    
    # Extract dose rates at standard cooling times
    standard_cooling_days = [14, 365, 3650, 36500]  # 14 days, 1 year, 10 years, 100 years
    dose_at_cooling_times = {}
    
    for days_after in standard_cooling_days:
        target_s = days_after * 24 * 3600
        # first index in cool_times >= target_s
        rel_idx = np.searchsorted(cool_times, target_s, side='left')
        if rel_idx >= len(cool_times):
            rel_idx = len(cool_times) - 1
        
        abs_idx = cool_start_idx + rel_idx
        t_actual = times_s[abs_idx]
        rate_actual = total_dose[t_actual]
        
        dose_at_cooling_times[days_after] = rate_actual
    
    return {
        'dose_at_cooling_times': dose_at_cooling_times,
        'gas_production': gas_production,
        'times_s': times_s.tolist(),
        'source_rates': source_rates.tolist(),
        'final_irr_time': final_irr_time,
        'cool_start_time': cool_start_time,
        'total_dose_lookup': {str(t): v for t, v in total_dose.items()}
    }


def parse_calphad_results(
    phase_results: pd.DataFrame,
    phase_fraction_threshold: float = 0.001
) -> Dict[str, Any]:
    """
    Parse CALPHAD calculation results to extract phase fraction data.
    
    Parameters
    ----------
    phase_results : pd.DataFrame
        DataFrame from CALPHADBatchCalculator.calculate_batch() containing phase data.
        Expected columns: phase_count, dominant_phase, single_phase, phases (JSON string)
    phase_fraction_threshold : float, optional
        Minimum phase fraction to consider. Phases below this threshold are ignored.
        Default is 0.001.
        
    Returns
    -------
    Dict[str, Any]
        Dictionary containing:
        - phase_fractions: List of dicts with phase fractions for each composition
        - filtered_phase_counts: Array of phase counts after filtering
        - dominant_phases: List of dominant phase names
        - single_phase_flags: Array of boolean flags for single-phase compositions
    """
    n_compositions = len(phase_results)
    filtered_phase_counts = np.zeros(n_compositions, dtype=int)
    phase_fractions = []
    dominant_phases = []
    single_phase_flags = np.zeros(n_compositions, dtype=bool)
    
    for i, row in phase_results.iterrows():
        try:
            # Parse phase fractions from JSON
            raw_phase_fractions = json.loads(row['phases'])
            
            # Filter phases by minimum fraction threshold
            significant_phases = {
                phase: fraction 
                for phase, fraction in raw_phase_fractions.items() 
                if fraction >= phase_fraction_threshold
            }
            
            filtered_phase_counts[i] = len(significant_phases)
            phase_fractions.append(significant_phases)
            
            # Get dominant phase (phase with highest fraction)
            if significant_phases:
                dominant_phase = max(significant_phases, key=significant_phases.get)
                dominant_phases.append(dominant_phase)
                single_phase_flags[i] = len(significant_phases) == 1
            else:
                dominant_phases.append("NONE")
                single_phase_flags[i] = False
                
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Failed to process composition {i}: {e}")
            filtered_phase_counts[i] = -1
            phase_fractions.append({})
            dominant_phases.append("ERROR")
            single_phase_flags[i] = False
    
    return {
        'phase_fractions': phase_fractions,
        'filtered_phase_counts': filtered_phase_counts,
        'dominant_phases': dominant_phases,
        'single_phase_flags': single_phase_flags,
        'phase_fraction_threshold': phase_fraction_threshold
    }


def check_constraints(
    openmc_data: Optional[Dict[str, Any]] = None,
    calphad_data: Optional[Dict[str, Any]] = None,
    dose_limits: Optional[Dict[int, float]] = None,
    gas_limits: Optional[Dict[str, float]] = None,
    phase_limits: Optional[Dict[str, float]] = None
) -> Dict[str, Any]:
    """
    Check if parsed data satisfies various constraints.
    
    Parameters
    ----------
    openmc_data : Dict[str, Any], optional
        Parsed OpenMC data from parse_openmc_results()
    calphad_data : Dict[str, Any], optional
        Parsed CALPHAD data from parse_calphad_results()
    dose_limits : Dict[int, float], optional
        Dictionary mapping cooling days to maximum allowed dose rates (Sv/h/kg)
    gas_limits : Dict[str, float], optional
        Dictionary mapping gas species to maximum allowed production (appm)
    phase_limits : Dict[str, float], optional
        Dictionary mapping phase names to maximum allowed fractions
        
    Returns
    -------
    Dict[str, Any]
        Dictionary containing constraint satisfaction results:
        - satisfy_dose: Boolean indicating if all dose limits are met
        - satisfy_gas: Boolean indicating if all gas limits are met
        - satisfy_phases: Array of booleans for each composition (CALPHAD only)
        - violations: Dict with detailed violation information
    """
    results = {
        'satisfy_dose': True,
        'satisfy_gas': True,
        'satisfy_phases': None,
        'violations': {}
    }
    
    # Default limits
    default_dose_limits = {14: 1e3, 365: 1, 3650: 1e-2, 36500: 1e-4}
    default_gas_limits = {"He_appm": 500, "H_appm": 1000}
    default_phase_limits = {'C15_LAVES': 0.001, 'C14_LAVES': 0.001, 'SIGMA': 0.001}
    
    dose_limits = dose_limits or default_dose_limits
    gas_limits = gas_limits or default_gas_limits
    phase_limits = phase_limits or default_phase_limits
    
    # Check OpenMC constraints
    if openmc_data is not None:
        dose_violations = {}
        gas_violations = {}
        
        # Check dose limits
        for days, limit in dose_limits.items():
            actual = openmc_data['dose_at_cooling_times'].get(days, 0.0)
            if actual > limit:
                results['satisfy_dose'] = False
                dose_violations[f'{days}_days'] = {
                    'actual': actual,
                    'limit': limit,
                    'ratio': actual / limit
                }
        
        # Check gas limits
        for gas, limit in gas_limits.items():
            actual = openmc_data['gas_production'].get(gas, 0.0)
            if actual > limit:
                results['satisfy_gas'] = False
                gas_violations[gas] = {
                    'actual': actual,
                    'limit': limit,
                    'ratio': actual / limit
                }
        
        results['violations']['dose'] = dose_violations
        results['violations']['gas'] = gas_violations
    
    # Check CALPHAD constraints
    if calphad_data is not None:
        n_compositions = len(calphad_data['phase_fractions'])
        satisfy_phases = np.ones(n_compositions, dtype=bool)
        phase_violations = []
        
        for i, phase_fractions in enumerate(calphad_data['phase_fractions']):
            violations = []
            for phase, fraction in phase_fractions.items():
                limit = phase_limits.get(phase, np.inf)
                if fraction > limit:
                    satisfy_phases[i] = False
                    violations.append({
                        'phase': phase,
                        'fraction': fraction,
                        'limit': limit,
                        'ratio': fraction / limit
                    })
            phase_violations.append(violations)
        
        results['satisfy_phases'] = satisfy_phases
        results['violations']['phases'] = phase_violations
    
    return results


def dict_to_array(comp_dict: Dict[str, float], elements: List[str]) -> np.ndarray:
    """Convert composition dictionary to numpy array.
    
    Parameters
    ----------
    comp_dict : Dict[str, float]
        Composition dictionary with element symbols as keys.
    elements : List[str]
        Ordered list of element symbols.
        
    Returns
    -------
    np.ndarray
        Composition array with values ordered according to elements list.
    """
    return np.array([comp_dict.get(el, 0.0) for el in elements])


def array_to_dict(comp_array: np.ndarray, elements: List[str]) -> Dict[str, float]:
    """Convert composition array to dictionary.
    
    Parameters
    ----------
    comp_array : np.ndarray
        Composition array.
    elements : List[str]
        Ordered list of element symbols corresponding to array indices.
        
    Returns
    -------
    Dict[str, float]
        Composition dictionary with element symbols as keys.
    """
    return {el: float(comp_array[i]) for i, el in enumerate(elements)} 