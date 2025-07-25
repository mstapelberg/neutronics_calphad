"""Example script demonstrating Bayesian optimization driven composition search.

This script mirrors ``run_fispact_depletion.py`` but performs the material
exploration in a loop using a simple Bayesian optimizer.  At each iteration a
batch of candidate compositions is suggested, depleted with the independent
operator, evaluated using :func:`evaluate_material`, and the results fed back
to the optimizer.

The search terminates after a fixed number of iterations.  Custom gas and dose
criteria can be passed in ``critical_limits`` and ``dose_limits`` dictionaries.
"""

from __future__ import annotations

import os
import json
from datetime import datetime
from typing import Dict, List, Any
from pathlib import Path

import openmc
import openmc.deplete
import numpy as np

from neutronics_calphad.neutronics.config import SPHERICAL
from neutronics_calphad.neutronics.geometry_maker import create_model
from neutronics_calphad.neutronics.depletion import run_independent_depletion
from neutronics_calphad.neutronics.time_scheduler import TimeScheduler
from neutronics_calphad.optimizer.evaluate_updated import evaluate_material
from neutronics_calphad.utils.io import material_string, create_material
from neutronics_calphad.optimizer.bayesian_optimizer import BayesianOptimizer
from neutronics_calphad.neutronics.flux import get_flux_and_microxs

# -----------------------------------------------------------------------------
# OpenMC configuration (update paths to your local data as needed)
openmc.config['chain_file'] = '/home/myless/nuclear_data/tendl21-fispact20-gefy61-chain.xml'
openmc.config['cross_sections'] = '/home/myless/nuclear_data/tendl-2021-hdf5/cross_sections.xml'

# Load chain once for efficiency
openmc.deplete.Chain.from_xml(openmc.config['chain_file'])

# -----------------------------------------------------------------------------
# Geometry and flux preparation
model = create_model(config=SPHERICAL)
model.settings.particles = 1000
cells = list(model.geometry.get_all_cells().values())

# Create necessary directories
microxs_and_flux_dir = os.path.join('restarted_bayesian_optimizer', 'microxs_and_flux')
depletion_results_dir = os.path.join('restarted_bayesian_optimizer', 'depletion_results')
os.makedirs(microxs_and_flux_dir, exist_ok=True)
os.makedirs(depletion_results_dir, exist_ok=True)

if os.path.exists(os.path.join(microxs_and_flux_dir, 'microxs_1102.csv')) and os.path.exists(os.path.join(microxs_and_flux_dir, 'flux_spectrum_1102.txt')):
    print("Using existing microxs and flux")
    flux = [np.loadtxt(os.path.join(microxs_and_flux_dir, 'flux_spectrum_1102.txt'), comments='#', usecols=1)]
    microxs = openmc.deplete.MicroXS.from_csv(os.path.join(microxs_and_flux_dir, 'microxs_1102.csv'))
else:
    print("Calculating microxs and flux")
    flux_path, microxs_path = get_flux_and_microxs(
        model,
        chain_file=openmc.config['chain_file'],
        group_structure='UKAEA-1102',
        outdir=microxs_and_flux_dir,
    )
    flux = [np.loadtxt(flux_path, comments='#', usecols=1)]
    microxs = openmc.deplete.MicroXS.from_csv(microxs_path)

# Index of vessel cell to replace material
VESSEL_CELL = next(i for i, c in enumerate(cells) if c.name == 'vessel')

# Store original number of materials to prevent memory accumulation
ORIGINAL_MATERIALS_COUNT = len(model.materials)

# -----------------------------------------------------------------------------
# Time scheduler for irradiation / cooling
POWER_MW = 500
FUSION_POWER_MEV = 17.6
MEV_TO_J = 1.602176634e-13
SOURCE_RATE = POWER_MW * 1e6 / (FUSION_POWER_MEV * MEV_TO_J)

scheduler = TimeScheduler(
    irradiation_time='1 year',
    cooling_times=['2 weeks', '1 year', '10 years', '100 years'],
    source_rate=SOURCE_RATE,
    irradiation_steps=12,
)

TIMESTEPS, SOURCES = scheduler.get_timesteps_and_source_rates()

# -----------------------------------------------------------------------------
# Optimization setup
ELEMENTS = ['V', 'Cr', 'Ti', 'W', 'Zr']

# Define composition constraints (optional)
# Example: V-based alloy with reasonable element limits
MIN_COMPOSITIONS = {
    'V': 0.70,    # V should be at least 70% (base element)
    # Other elements can go to 0 if not specified
}

MAX_COMPOSITIONS = {
    'V': 0.95,    # V at most 95%
    'Cr': 0.20,   # Cr at most 20%
    'Ti': 0.15,   # Ti at most 15%
    'W': 0.20,    # W at most 20% 
    'Zr': 0.1,   # Zr at most 5%
}

# Create optimizer with composition constraints
# Note: To run unconstrained, use: optimizer = BayesianOptimizer(ELEMENTS, batch_size=5, minimize=False)
optimizer = BayesianOptimizer(
    ELEMENTS, 
    batch_size=5, 
    minimize=False,
    min_compositions=MIN_COMPOSITIONS,
    max_compositions=MAX_COMPOSITIONS
)

print("Using composition constraints:")
print(f"  Min compositions: {MIN_COMPOSITIONS}")
print(f"  Max compositions: {MAX_COMPOSITIONS}")

# temporarily set He to be 1 micron grain size limit based on Gilbert
# set the hydrogen limit to be 1200 appm based on loomisResponseUnirradiatedNeutronirradiated1991
CRIT_LIMITS = {"He_appm": 1172.2/2, "H_appm": 1500}
DOSE_LIMITS = {14: 1e3, 365: 1, 3650: 1e-2, 36500: 1e-4}

N_ITERATIONS = 5  # demonstration only

# -----------------------------------------------------------------------------
# Results storage
optimization_results = {
    'metadata': {
        'timestamp': datetime.now().isoformat(),
        'elements': ELEMENTS,
        'critical_limits': CRIT_LIMITS,
        'dose_limits': DOSE_LIMITS,
        'n_iterations': N_ITERATIONS,
        'batch_size': optimizer.batch_size,
        'power_mw': POWER_MW,
        'irradiation_time': '1 year',
        'cooling_times': ['2 weeks', '1 year', '10 years', '100 years']
    },
    'iterations': []
}

def evaluate_material_detailed(
    results: openmc.deplete.Results,
    chain_file: str,
    abs_file: str,
    critical_gas_limits: dict,
    dose_limits: dict,
    score: str = "continuous"
) -> Dict[str, Any]:
    """
    Evaluate a depletion result and return detailed results including dose rates and gas production.
    
    Parameters
    ----------
    results : openmc.deplete.Results
        Depletion results object.
    chain_file : str
        Path to the depletion chain file.
    abs_file : str
        Path to the absorption cross-section file.
    critical_gas_limits : dict
        Dictionary of critical gas limits.
    dose_limits : dict
        Dictionary of dose rate limits.
    score : str, optional
        Scoring strategy to use, by default "continuous".
        
    Returns
    -------
    Dict[str, Any]
        Dictionary containing detailed evaluation results.
    """
    from neutronics_calphad.neutronics.dose import contact_dose
    from neutronics_calphad.neutronics.depletion import extract_gas_production
    
    # Compute full dose-time series
    times_s, dose_dicts = contact_dose(results=results, chain_file=chain_file, abs_file=abs_file)
    
    # Compute gas production (appm)
    gas_production_rates = extract_gas_production(results)
    
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
    cool_doses = [total_dose[t] for t in times_s[cool_start_idx:]]
    
    # Check each cooling-time limit
    satisfy_dose = True
    dose_at_limit = {}
    for days_after, limit in dose_limits.items():
        target_s = days_after * 24 * 3600
        # first index in cool_times >= target_s
        rel_idx = np.searchsorted(cool_times, target_s, side='left')
        if rel_idx >= len(cool_times):
            rel_idx = len(cool_times) - 1
        
        abs_idx = cool_start_idx + rel_idx
        t_actual = times_s[abs_idx]
        cool_actual = cool_times[rel_idx]
        rate_actual = total_dose[t_actual]
        
        dose_at_limit[days_after] = rate_actual
        if rate_actual > limit:
            satisfy_dose = False
    
    # Check gas-production limits
    satisfy_gas = True
    for gas, produced in gas_production_rates.items():
        limit = critical_gas_limits.get(gas, np.inf)
        if produced > limit:
            satisfy_gas = False
    
    # Compute score
    if score == "ternary":
        final_score = 1.0 if (satisfy_dose and satisfy_gas) else 0.5 if (satisfy_dose or satisfy_gas) else 0.0
    elif score == 'continuous':
        dose_scores = [max(0.0, 1 - dose_at_limit[d] / dose_limits[d]) for d in dose_limits]
        gas_scores = [max(0.0, 1 - gas_production_rates.get(g, 0.0) / critical_gas_limits.get(g, np.inf)) for g in critical_gas_limits]
        final_score = float((np.mean(dose_scores) + np.mean(gas_scores)) / 2)
    else:
        raise ValueError(f"Invalid score mode: {score}")
    
    return {
        'score': final_score,
        'satisfy_dose': satisfy_dose,
        'satisfy_gas': satisfy_gas,
        'dose_at_limit': dose_at_limit,
        'gas_production_rates': gas_production_rates,
        'times_s': times_s.tolist(),
        'total_dose': {str(t): v for t, v in total_dose.items()},
        'final_irr_time': final_irr_time,
        'cool_start_time': cool_start_time,
        'source_rates': source_rates.tolist()
    }

for iteration in range(N_ITERATIONS):
    print(f"\n=== Iteration {iteration+1} ===")
    batch = optimizer.suggest()
    
    iteration_results = {
        'iteration': iteration + 1,
        'materials': []
    }
    
    scores: List[float] = []
    for comp in batch:
        comp_dict = dict(zip(ELEMENTS, comp))
        mat_name = material_string(comp_dict, 'V')
        material = create_material(comp_dict, mat_name)
        material.depletable = True
        
        # Clean up: Remove any previously added materials to prevent memory accumulation
        while len(model.materials) > ORIGINAL_MATERIALS_COUNT:
            model.materials.pop()
        
        # Replace vessel material
        model.materials.append(material)
        vessel_cell = model.geometry.get_cells_by_name('vessel')[0]
        material.volume = next(m.volume for m in model.materials if m.name == 'vcrtiwzr')
        vessel_cell.fill = material
        
        # Run depletion
        results = run_independent_depletion(
            model=model,
            depletable_cell='vessel',
            microxs=microxs,
            flux=flux,
            chain_file=openmc.config['chain_file'],
            timesteps=TIMESTEPS,
            source_rates=SOURCES,
            outdir=os.path.join('bayesian_optimizer', 'depletion_results', mat_name),
        )
        
        # Get detailed evaluation results
        detailed_results = evaluate_material_detailed(
            results=results,
            chain_file=openmc.config['chain_file'],
            abs_file='/home/myless/Packages/fispact/nuclear_data/decay/abs_2012',
            critical_gas_limits=CRIT_LIMITS,
            dose_limits=DOSE_LIMITS,
            score='continuous',
        )
        
        scores.append(detailed_results['score'])
        
        # Store material results
        material_result = {
            'material_name': mat_name,
            'composition': comp_dict,
            'composition_array': comp.tolist(),
            'score': detailed_results['score'],
            'satisfy_dose': detailed_results['satisfy_dose'],
            'satisfy_gas': detailed_results['satisfy_gas'],
            'dose_rates': detailed_results['dose_at_limit'],
            'gas_production': detailed_results['gas_production_rates'],
            'times_s': detailed_results['times_s'],
            'total_dose': detailed_results['total_dose'],
            'final_irr_time': detailed_results['final_irr_time'],
            'cool_start_time': detailed_results['cool_start_time'],
            'source_rates': detailed_results['source_rates']
        }
        
        iteration_results['materials'].append(material_result)
        print(f"{mat_name}: score={detailed_results['score']}")
    
    optimizer.update(batch, np.array(scores))
    optimization_results['iterations'].append(iteration_results)

# Save results to JSON file
output_file = Path('bayesian_optimizer', 'bayesian_optimization_results.json')
with open(output_file, 'w') as f:
    json.dump(optimization_results, f, indent=2, default=str)

print(f"\nOptimization complete")
print(f"Results saved to: {output_file.absolute()}")

# Print summary
print(f"\n=== Optimization Summary ===")
print(f"Total materials evaluated: {sum(len(iter_data['materials']) for iter_data in optimization_results['iterations'])}")
print(f"Best score: {max(max(mat['score'] for mat in iter_data['materials']) for iter_data in optimization_results['iterations']):.4f}")
print(f"Worst score: {min(min(mat['score'] for mat in iter_data['materials']) for iter_data in optimization_results['iterations']):.4f}")

# Find best material
best_material = None
best_score = -1
for iter_data in optimization_results['iterations']:
    for material in iter_data['materials']:
        if material['score'] > best_score:
            best_score = material['score']
            best_material = material

if best_material:
    print(f"\nBest material: {best_material['material_name']}")
    print(f"Composition: {best_material['composition']}")
    print(f"Score: {best_material['score']:.4f}")
    print(f"Satisfies dose limits: {best_material['satisfy_dose']}")
    print(f"Satisfies gas limits: {best_material['satisfy_gas']}")
    print(f"Gas production: {best_material['gas_production']}")
    print(f"Dose rates: {best_material['dose_rates']}")

# Clean up: Restore original materials list
while len(model.materials) > ORIGINAL_MATERIALS_COUNT:
    model.materials.pop()
print(f"\nRestored model to original state with {len(model.materials)} materials")