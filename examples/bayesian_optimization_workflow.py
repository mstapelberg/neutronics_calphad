"""Example workflow for Bayesian optimization with neutronics and CALPHAD evaluation."""

# NEXT STEPS:
# 1) Modify BayesianOptimizer to predict raw values (dose rates, gas production) instead of scores
# 2) Add multi-target GP modeling for separate dose/gas/phase predictions
# 3) Implement acquisition functions that directly target constraint boundaries
# 4) The architecture now supports your goal of creating a 5D composition manifold that 
# clearly delineates acceptable vs. unacceptable compositions for both neutronics and CALPHAD constraints!

import numpy as np
import pandas as pd
from typing import Dict, List, Any
import logging

# Import our modules
from neutronics_calphad.optimizer.bayesian_optimizer import BayesianOptimizer
from neutronics_calphad.calphad.phase_calculator import CALPHADBatchCalculator
from neutronics_calphad.optimizer.parsers import (
    parse_openmc_results, 
    parse_calphad_results, 
    check_constraints,
    dict_to_array,
    array_to_dict
)
from neutronics_calphad.optimizer.convergence import overall_convergence_check

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def example_workflow():
    """
    Example workflow showing how to use Bayesian optimization for composition search.
    
    This demonstrates the intended architecture:
    1. Bayesian optimizer suggests compositions
    2. External workflow runs calculations (OpenMC, CALPHAD)
    3. Parsers extract standardized data from results
    4. Constraints are checked against limits
    5. Process repeats until convergence
    """
    
    # Define the system
    elements = ['V', 'Cr', 'Ti', 'W', 'Zr']
    
    # Set up constraints/limits
    dose_limits = {
        14: 1e3,      # 14 days → 1000 Sv/h/kg
        365: 1,    # 1 year → 1 Sv/h/kg
        3650: 1e-2,   # 10 years → 0.01 Sv/h/kg
        36500: 1e-4   # 100 years → 0.0001 Sv/h/kg
    }
    
    gas_limits = {
        "He_appm": 550,
        "H_appm": 1250
    }
    
    phase_limits = {
        'C15_LAVES': 0.001,
        'C14_LAVES': 0.001,
        'SIGMA': 0.001
    }
    
    # Initialize Bayesian optimizer
    # Note: This will need to be modified to handle multiple targets and parsers
    optimizer = BayesianOptimizer(
        elements=elements,
        batch_size=10,
        minimize=False  # We want to maximize the probability of meeting constraints
    )
    
    # Initialize CALPHAD calculator
    calphad_calc = CALPHADBatchCalculator(
        database="TCHEA7",
        temperature=823.5
    )
    
    # Optimization loop
    iteration = 0
    max_iterations = 50
    converged = False
    
    # Storage for tracking progress
    all_compositions = []
    all_neutronics_data = []
    all_calphad_data = []
    all_scores = []
    
    while not converged and iteration < max_iterations:
        logger.info(f"=== Iteration {iteration + 1} ===")
        
        # 1. Get composition suggestions from Bayesian optimizer
        if iteration == 0:
            # Initial random sampling
            comp_arrays = optimizer.suggest()
        else:
            # Update optimizer with previous results and get new suggestions
            # Convert scores to format expected by optimizer
            combined_scores = np.array([
                score.get('combined_score', 0.0) for score in all_scores[-optimizer.batch_size:]
            ])
            
            # Update optimizer
            last_compositions = np.array(all_compositions[-optimizer.batch_size:])
            optimizer.update(last_compositions, combined_scores)
            
            # Get new suggestions
            comp_arrays = optimizer.suggest()
        
        # Convert compositions to dictionaries for calculations
        comp_dicts = [array_to_dict(comp, elements) for comp in comp_arrays]
        all_compositions.extend(comp_arrays)
        
        logger.info(f"Evaluating {len(comp_dicts)} compositions...")
        
        # 2. Run CALPHAD calculations
        logger.info("Running CALPHAD calculations...")
        phase_results = calphad_calc.calculate_batch(comp_arrays, elements)
        calphad_data = parse_calphad_results(phase_results)
        all_calphad_data.append(calphad_data)
        
        # 3. Simulate OpenMC calculations (for this example, we'll mock them)
        logger.info("Simulating OpenMC calculations...")
        neutronics_data = simulate_openmc_calculations(comp_dicts)
        all_neutronics_data.append(neutronics_data)
        
        # 4. Check constraints for each composition
        batch_scores = []
        for i in range(len(comp_dicts)):
            # Extract data for this composition
            comp_neutronics = {
                'dose_at_cooling_times': {
                    days: neutronics_data['dose_at_cooling_times'][days][i]
                    for days in neutronics_data['dose_at_cooling_times']
                },
                'gas_production': {
                    gas: neutronics_data['gas_production'][gas][i]
                    for gas in neutronics_data['gas_production']
                }
            }
            
            comp_calphad = {
                'phase_fractions': [calphad_data['phase_fractions'][i]],
                'filtered_phase_counts': calphad_data['filtered_phase_counts'][i:i+1],
                'dominant_phases': [calphad_data['dominant_phases'][i]],
                'single_phase_flags': calphad_data['single_phase_flags'][i:i+1]
            }
            
            # Check constraints
            constraint_results = check_constraints(
                openmc_data=comp_neutronics,
                calphad_data=comp_calphad,
                dose_limits=dose_limits,
                gas_limits=gas_limits,
                phase_limits=phase_limits
            )
            
            # Calculate combined score
            # For this example, simple approach: 1.0 if all constraints satisfied, 0.0 otherwise
            all_satisfied = (
                constraint_results['satisfy_dose'] and
                constraint_results['satisfy_gas'] and
                constraint_results['satisfy_phases'][0] if constraint_results['satisfy_phases'] is not None else True
            )
            
            score_dict = {
                'combined_score': 1.0 if all_satisfied else 0.0,
                'satisfy_dose': constraint_results['satisfy_dose'],
                'satisfy_gas': constraint_results['satisfy_gas'],
                'satisfy_phases': constraint_results['satisfy_phases'][0] if constraint_results['satisfy_phases'] is not None else True,
                'composition': comp_dicts[i],
                'violations': constraint_results['violations']
            }
            
            batch_scores.append(score_dict)
        
        all_scores.extend(batch_scores)
        
        # 5. Check convergence
        # For convergence, we need to accumulate predicted values
        # This is a simplified version - in practice, you'd extract predictions from the GP model
        if iteration >= 5:  # Start checking after a few iterations
            # Mock predicted values for convergence check
            predicted_values = {
                'dose_14': np.array([score['composition']['V'] * 1e1 for score in all_scores[-20:]]),  # Mock prediction
                'gas_he': np.array([score['composition']['Cr'] * 100 for score in all_scores[-20:]])   # Mock prediction
            }
            
            recent_compositions = np.array(all_compositions[-20:])
            
            convergence_limits = {
                'dose_14': dose_limits[14],
                'gas_he': gas_limits['He_appm']
            }
            
            conv_results = overall_convergence_check(
                predicted_values=predicted_values,
                compositions=recent_compositions,
                elements=elements,
                limits=convergence_limits,
                tolerance=0.05,
                min_samples=15
            )
            
            logger.info(f"Convergence check: {conv_results['summary']['message']}")
            converged = conv_results['converged']
        
        # 6. Log progress
        n_satisfied = sum(1 for score in batch_scores if score['combined_score'] > 0.5)
        logger.info(f"Batch results: {n_satisfied}/{len(batch_scores)} compositions satisfy all constraints")
        
        total_satisfied = sum(1 for score in all_scores if score['combined_score'] > 0.5)
        logger.info(f"Overall: {total_satisfied}/{len(all_scores)} compositions satisfy constraints")
        
        iteration += 1
    
    # Final results
    logger.info(f"\n=== Optimization Complete ===")
    logger.info(f"Total iterations: {iteration}")
    logger.info(f"Total compositions evaluated: {len(all_scores)}")
    
    successful_compositions = [
        score for score in all_scores if score['combined_score'] > 0.5
    ]
    logger.info(f"Successful compositions: {len(successful_compositions)}")
    
    if successful_compositions:
        logger.info("\nBest compositions:")
        for i, comp in enumerate(successful_compositions[:5]):  # Show top 5
            comp_str = ", ".join([f"{el}: {comp['composition'][el]:.3f}" for el in elements])
            logger.info(f"  {i+1}. {comp_str}")
    
    return {
        'converged': converged,
        'iterations': iteration,
        'all_compositions': all_compositions,
        'all_scores': all_scores,
        'successful_compositions': successful_compositions
    }


def simulate_openmc_calculations(compositions: List[Dict[str, float]]) -> Dict[str, Dict[int, List[float]]]:
    """
    Simulate OpenMC calculations for example purposes.
    
    In reality, this would run actual OpenMC depletion calculations.
    """
    n_comps = len(compositions)
    
    # Generate mock dose rates and gas production
    # These would come from actual OpenMC calculations
    simulated_data = {
        'dose_at_cooling_times': {},
        'gas_production': {}
    }
    
    # Mock dose rates (roughly inversely correlated with V content for variety)
    for days in [14, 365, 3650, 36500]:
        base_dose = {14: 1e2, 365: 1e-2, 3650: 1e-2, 36500: 1e-4}[days]
        # Add some variation based on composition
        doses = []
        for comp in compositions:
            v_fraction = comp.get('V', 0.5)
            # Higher V → lower activation (simplified model)
            dose_multiplier = (1.5 - v_fraction) + np.random.normal(0, 0.2)
            dose_multiplier = max(0.1, dose_multiplier)  # Keep positive
            doses.append(base_dose * dose_multiplier)
        
        simulated_data['dose_at_cooling_times'][days] = doses
    
    # Mock gas production
    for gas in ['He_appm', 'H_appm']:
        base_production = {'He_appm': 200, 'H_appm': 400}[gas]
        productions = []
        for comp in compositions:
            cr_fraction = comp.get('Cr', 0.1)
            # Higher Cr → higher gas production (simplified model)
            gas_multiplier = (1 + cr_fraction) + np.random.normal(0, 0.3)
            gas_multiplier = max(0.1, gas_multiplier)
            productions.append(base_production * gas_multiplier)
        
        simulated_data['gas_production'][gas] = productions
    
    return simulated_data


if __name__ == "__main__":
    # Run the example
    results = example_workflow()
    
    print(f"\n=== Final Summary ===")
    print(f"Converged: {results['converged']}")
    print(f"Iterations: {results['iterations']}")
    print(f"Success rate: {len(results['successful_compositions'])}/{len(results['all_scores'])} = "
          f"{100 * len(results['successful_compositions']) / len(results['all_scores']):.1f}%") 