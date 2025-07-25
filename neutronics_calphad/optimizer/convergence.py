"""Convergence checking utilities for Bayesian optimization."""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import logging

logger = logging.getLogger(__name__)


def check_prediction_convergence(
    predicted_values: Dict[str, np.ndarray],
    limits: Dict[str, float],
    tolerance: float = 0.05,
    min_samples: int = 10
) -> Dict[str, Union[bool, float, Dict]]:
    """
    Check if Bayesian optimization has converged based on prediction accuracy relative to limits.
    
    Parameters
    ----------
    predicted_values : Dict[str, np.ndarray]
        Dictionary mapping target names to arrays of predicted values.
        E.g., {'dose_14d': [1e1, 2e-1, ...], 'gas_he': [100, 250, ...]}
    limits : Dict[str, float]
        Dictionary mapping target names to limit values.
        E.g., {'dose_14d': 1e2, 'gas_he': 500}
    tolerance : float, optional
        Relative tolerance for convergence (e.g., 0.05 = 5%), by default 0.05
    min_samples : int, optional
        Minimum number of samples required before checking convergence, by default 10
        
    Returns
    -------
    Dict[str, Union[bool, float, Dict]]
        Dictionary containing:
        - converged: Overall convergence status (bool)
        - worst_error: Maximum relative error across all targets (float)
        - target_errors: Dict of relative errors for each target
        - target_convergence: Dict of convergence status for each target
        - samples_available: Number of samples available
    """
    if not predicted_values:
        return {
            'converged': False,
            'worst_error': float('inf'),
            'target_errors': {},
            'target_convergence': {},
            'samples_available': 0,
            'reason': 'No predicted values provided'
        }
    
    # Check minimum samples requirement
    n_samples = len(next(iter(predicted_values.values())))
    if n_samples < min_samples:
        return {
            'converged': False,
            'worst_error': float('inf'),
            'target_errors': {},
            'target_convergence': {},
            'samples_available': n_samples,
            'reason': f'Insufficient samples ({n_samples} < {min_samples})'
        }
    
    target_errors = {}
    target_convergence = {}
    all_converged = True
    worst_error = 0.0
    
    for target_name, predictions in predicted_values.items():
        if target_name not in limits:
            logger.warning(f"No limit specified for target '{target_name}', skipping convergence check")
            continue
            
        limit = limits[target_name]
        if limit == 0:
            logger.warning(f"Zero limit for target '{target_name}', using absolute tolerance")
            # For zero limits, use absolute tolerance
            relative_error = np.abs(predictions).max() / (tolerance * 1.0)  # Assume scale of 1.0
        else:
            # Calculate relative error as maximum distance from limit (normalized by limit)
            relative_error = np.abs(predictions - limit).max() / abs(limit)
        
        target_errors[target_name] = relative_error
        target_converged = relative_error <= tolerance
        target_convergence[target_name] = target_converged
        
        if not target_converged:
            all_converged = False
        
        worst_error = max(worst_error, relative_error)
    
    return {
        'converged': all_converged,
        'worst_error': worst_error,
        'target_errors': target_errors,
        'target_convergence': target_convergence,
        'samples_available': n_samples,
        'reason': 'Converged' if all_converged else f'Worst error: {worst_error:.1%} > {tolerance:.1%}'
    }


def check_constraint_boundary_convergence(
    predicted_values: Dict[str, np.ndarray],
    limits: Dict[str, float],
    boundary_tolerance: float = 0.1,
    boundary_fraction: float = 0.1
) -> Dict[str, Union[bool, float, Dict]]:
    """
    Check convergence based on how well the model predicts constraint boundaries.
    
    This function checks if the surrogate model can accurately predict which compositions
    will satisfy constraints vs. which will violate them.
    
    Parameters
    ----------
    predicted_values : Dict[str, np.ndarray]
        Dictionary mapping target names to arrays of predicted values.
    limits : Dict[str, float]
        Dictionary mapping target names to limit values.
    boundary_tolerance : float, optional
        Tolerance for boundary predictions (relative to limit), by default 0.1
    boundary_fraction : float, optional
        Fraction of predictions that should be near boundaries for meaningful check, by default 0.1
        
    Returns
    -------
    Dict[str, Union[bool, float, Dict]]
        Dictionary containing boundary convergence results.
    """
    boundary_results = {}
    all_boundaries_converged = True
    
    for target_name, predictions in predicted_values.items():
        if target_name not in limits:
            continue
            
        limit = limits[target_name]
        
        # Find predictions near the boundary
        if limit > 0:
            boundary_range = abs(limit) * boundary_tolerance
            near_boundary = np.abs(predictions - limit) <= boundary_range
        else:
            boundary_range = boundary_tolerance
            near_boundary = np.abs(predictions) <= boundary_range
        
        n_near_boundary = np.sum(near_boundary)
        boundary_fraction_actual = n_near_boundary / len(predictions)
        
        # Check if we have enough boundary samples and they're well-predicted
        boundary_converged = (
            boundary_fraction_actual >= boundary_fraction and
            n_near_boundary >= 3  # Minimum for meaningful statistics
        )
        
        boundary_results[target_name] = {
            'converged': boundary_converged,
            'near_boundary_count': n_near_boundary,
            'boundary_fraction': boundary_fraction_actual,
            'boundary_range': boundary_range
        }
        
        if not boundary_converged:
            all_boundaries_converged = False
    
    return {
        'converged': all_boundaries_converged,
        'target_boundaries': boundary_results,
        'reason': 'Boundary convergence achieved' if all_boundaries_converged else 'Insufficient boundary sampling'
    }


def check_composition_space_coverage(
    compositions: np.ndarray,
    elements: List[str],
    min_coverage_per_element: float = 0.2
) -> Dict[str, Union[bool, float, Dict]]:
    """
    Check if the composition space has been adequately explored.
    
    Parameters
    ----------
    compositions : np.ndarray
        Array of compositions, shape (n_samples, n_elements)
    elements : List[str]
        List of element symbols
    min_coverage_per_element : float, optional
        Minimum range coverage for each element (0-1), by default 0.2
        
    Returns
    -------
    Dict[str, Union[bool, float, Dict]]
        Dictionary containing coverage results.
    """
    if len(compositions) == 0:
        return {
            'converged': False,
            'reason': 'No compositions provided',
            'element_coverage': {}
        }
    
    element_coverage = {}
    all_elements_covered = True
    
    for i, element in enumerate(elements):
        element_fractions = compositions[:, i]
        min_val = element_fractions.min()
        max_val = element_fractions.max()
        coverage = max_val - min_val
        
        element_converged = coverage >= min_coverage_per_element
        element_coverage[element] = {
            'converged': element_converged,
            'coverage': coverage,
            'min': min_val,
            'max': max_val
        }
        
        if not element_converged:
            all_elements_covered = False
    
    return {
        'converged': all_elements_covered,
        'element_coverage': element_coverage,
        'reason': 'Space coverage adequate' if all_elements_covered else 'Insufficient space exploration'
    }


def overall_convergence_check(
    predicted_values: Dict[str, np.ndarray],
    compositions: np.ndarray,
    elements: List[str],
    limits: Dict[str, float],
    tolerance: float = 0.05,
    min_samples: int = 20,
    check_boundaries: bool = True,
    check_coverage: bool = True
) -> Dict[str, Union[bool, Dict]]:
    """
    Comprehensive convergence check combining multiple criteria.
    
    Parameters
    ----------
    predicted_values : Dict[str, np.ndarray]
        Dictionary mapping target names to arrays of predicted values.
    compositions : np.ndarray
        Array of compositions, shape (n_samples, n_elements)
    elements : List[str]
        List of element symbols
    limits : Dict[str, float]
        Dictionary mapping target names to limit values.
    tolerance : float, optional
        Relative tolerance for prediction convergence, by default 0.05
    min_samples : int, optional
        Minimum number of samples required, by default 20
    check_boundaries : bool, optional
        Whether to check boundary convergence, by default True
    check_coverage : bool, optional
        Whether to check composition space coverage, by default True
        
    Returns
    -------
    Dict[str, Union[bool, Dict]]
        Dictionary containing overall convergence results.
    """
    results = {
        'converged': False,
        'prediction_convergence': {},
        'boundary_convergence': {},
        'coverage_convergence': {},
        'summary': {}
    }
    
    # Check prediction convergence
    pred_conv = check_prediction_convergence(
        predicted_values, limits, tolerance, min_samples
    )
    results['prediction_convergence'] = pred_conv
    
    # Check boundary convergence if requested
    if check_boundaries:
        boundary_conv = check_constraint_boundary_convergence(predicted_values, limits)
        results['boundary_convergence'] = boundary_conv
    
    # Check space coverage if requested
    if check_coverage:
        coverage_conv = check_composition_space_coverage(compositions, elements)
        results['coverage_convergence'] = coverage_conv
    
    # Determine overall convergence
    prediction_ok = pred_conv['converged']
    boundary_ok = not check_boundaries or results['boundary_convergence'].get('converged', True)
    coverage_ok = not check_coverage or results['coverage_convergence'].get('converged', True)
    
    overall_converged = prediction_ok and boundary_ok and coverage_ok
    
    # Create summary
    summary = {
        'prediction_converged': prediction_ok,
        'boundary_converged': boundary_ok,
        'coverage_converged': coverage_ok,
        'samples_evaluated': pred_conv.get('samples_available', 0),
        'worst_prediction_error': pred_conv.get('worst_error', float('inf'))
    }
    
    if overall_converged:
        summary['message'] = "Bayesian optimization has converged"
    else:
        failed_checks = []
        if not prediction_ok:
            failed_checks.append("prediction accuracy")
        if not boundary_ok:
            failed_checks.append("boundary detection")
        if not coverage_ok:
            failed_checks.append("space coverage")
        summary['message'] = f"Convergence failed: {', '.join(failed_checks)}"
    
    results['converged'] = overall_converged
    results['summary'] = summary
    
    return results 