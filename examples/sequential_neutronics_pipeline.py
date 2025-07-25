"""Restartable sequential materials design pipeline: Neutronics → CALPHAD → Ductility.

This script implements the sequential approach:
1. Stage 1: Neutronics-only Bayesian optimization to define activation manifold
2. Stage 2: CALPHAD calculations on neutronics-passing compositions
3. Stage 3: Ductility/DBTT modeling on phase-stable compositions
4. Restartable at any stage with JSON persistence

Based on run_restarted_bo_search.py architecture.
"""

from __future__ import annotations

import os
import json
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
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
from neutronics_calphad.calphad.phase_calculator import CALPHADBatchCalculator
from neutronics_calphad.optimizer.convergence import overall_convergence_check

# -----------------------------------------------------------------------------
# Configuration
PIPELINE_NAME = "sequential_materials_pipeline"
RESULTS_DIR = PIPELINE_NAME
NEUTRONICS_RESULTS_FILE = os.path.join(RESULTS_DIR, "neutronics_optimization_results.json")
CALPHAD_RESULTS_FILE = os.path.join(RESULTS_DIR, "calphad_results.json")
DUCTILITY_RESULTS_FILE = os.path.join(RESULTS_DIR, "ductility_results.json")
PIPELINE_STATE_FILE = os.path.join(RESULTS_DIR, "pipeline_state.json")

# Create results directory
os.makedirs(RESULTS_DIR, exist_ok=True)

# OpenMC configuration
openmc.config['chain_file'] = '/home/myless/nuclear_data/tendl21-fispact20-gefy61-chain.xml'
openmc.config['cross_sections'] = '/home/myless/nuclear_data/tendl-2021-hdf5/cross_sections.xml'
openmc.deplete.Chain.from_xml(openmc.config['chain_file'])

# System definition
ELEMENTS = ['V', 'Cr', 'Ti', 'W', 'Zr']

# Composition constraints for V-based alloy
MIN_COMPOSITIONS = {'V': 0.70}
MAX_COMPOSITIONS = {'V': 0.95, 'Cr': 0.20, 'Ti': 0.15, 'W': 0.20, 'Zr': 0.1}

# Neutronics limits
CRIT_LIMITS = {"He_appm": 1172.2/2, "H_appm": 1500}
DOSE_LIMITS = {14: 1e3, 365: 1, 3650: 1e-2, 36500: 1e-4}

# CALPHAD limits
PHASE_LIMITS = {'C15_LAVES': 0.001, 'C14_LAVES': 0.001, 'SIGMA': 0.001, 'CHI': 0.001, 'MU': 0.001}

# Pipeline parameters
NEUTRONICS_MAX_ITERATIONS = 30
NEUTRONICS_BATCH_SIZE = 8
NEUTRONICS_CONVERGENCE_TOLERANCE = 0.05


class PipelineState:
    """Manages pipeline state and restart capability."""
    
    def __init__(self, state_file: str):
        self.state_file = state_file
        self.state = self._load_state()
    
    def _load_state(self) -> Dict[str, Any]:
        """Load pipeline state from JSON file."""
        if os.path.exists(self.state_file):
            with open(self.state_file, 'r') as f:
                return json.load(f)
        else:
            return {
                'current_stage': 'neutronics',
                'neutronics_completed': False,
                'calphad_completed': False,
                'ductility_completed': False,
                'neutronics_iteration': 0,
                'timestamp': datetime.now().isoformat()
            }
    
    def save_state(self):
        """Save current state to JSON file."""
        self.state['last_updated'] = datetime.now().isoformat()
        with open(self.state_file, 'w') as f:
            json.dump(self.state, f, indent=2)
    
    def update_stage(self, stage: str, completed: bool = False):
        """Update current stage and completion status."""
        self.state['current_stage'] = stage
        self.state[f'{stage}_completed'] = completed
        self.save_state()
    
    def get_current_stage(self) -> str:
        """Get current pipeline stage."""
        return self.state['current_stage']
    
    def is_stage_completed(self, stage: str) -> bool:
        """Check if a stage is completed."""
        return self.state.get(f'{stage}_completed', False)


def load_neutronics_results() -> Optional[Dict[str, Any]]:
    """Load neutronics optimization results if they exist."""
    if os.path.exists(NEUTRONICS_RESULTS_FILE):
        with open(NEUTRONICS_RESULTS_FILE, 'r') as f:
            return json.load(f)
    return None


def save_neutronics_results(results: Dict[str, Any]):
    """Save neutronics optimization results."""
    with open(NEUTRONICS_RESULTS_FILE, 'w') as f:
        json.dump(results, f, indent=2, default=str)


def extract_neutronics_passing_compositions(results: Dict[str, Any], 
                                           score_threshold: float = 0.7) -> List[Dict[str, float]]:
    """Extract compositions that pass neutronics constraints."""
    passing_compositions = []
    
    for iteration in results['iterations']:
        for material in iteration['materials']:
            if (material['score'] >= score_threshold and 
                material['satisfy_dose'] and 
                material['satisfy_gas']):
                passing_compositions.append(material['composition'])
    
    return passing_compositions


def create_restartable_optimizer(existing_results: Optional[Dict[str, Any]]) -> Tuple[BayesianOptimizer, int]:
    """Create Bayesian optimizer with optional restart from existing results."""
    optimizer = BayesianOptimizer(
        ELEMENTS,
        batch_size=NEUTRONICS_BATCH_SIZE,
        minimize=False,
        min_compositions=MIN_COMPOSITIONS,
        max_compositions=MAX_COMPOSITIONS
    )
    
    start_iteration = 0
    
    if existing_results:
        print("Restarting from existing neutronics results...")
        
        # Reconstruct training data
        all_compositions = []
        all_scores = []
        
        for iteration in existing_results['iterations']:
            for material in iteration['materials']:
                comp_array = material['composition_array']
                score = material['score']
                all_compositions.append(comp_array)
                all_scores.append(score)
        
        if all_compositions:
            # Update optimizer with existing data
            compositions_array = np.array(all_compositions)
            scores_array = np.array(all_scores)
            optimizer.update(compositions_array, scores_array)
            
            start_iteration = len(existing_results['iterations'])
            print(f"Loaded {len(all_compositions)} previous evaluations")
            print(f"Starting from iteration {start_iteration + 1}")
    
    return optimizer, start_iteration


def setup_openmc_model():
    """Set up OpenMC model and get flux/microxs data."""
    model = create_model(config=SPHERICAL)
    model.settings.particles = 10000
    cells = list(model.geometry.get_all_cells().values())
    
    # Create necessary directories
    microxs_and_flux_dir = os.path.join(RESULTS_DIR, 'microxs_and_flux')
    os.makedirs(microxs_and_flux_dir, exist_ok=True)
    
    microxs_file = os.path.join(microxs_and_flux_dir, 'microxs_1102.csv')
    flux_file = os.path.join(microxs_and_flux_dir, 'flux_spectrum_1102.txt')
    
    if os.path.exists(microxs_file) and os.path.exists(flux_file):
        print("Using existing microxs and flux")
        flux = [np.loadtxt(flux_file, comments='#', usecols=1)]
        microxs = openmc.deplete.MicroXS.from_csv(microxs_file)
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
    
    # Time scheduler
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
    
    timesteps, sources = scheduler.get_timesteps_and_source_rates()
    
    return model, flux, microxs, timesteps, sources


def evaluate_material_detailed(results, chain_file: str, abs_file: str, 
                             critical_gas_limits: dict, dose_limits: dict,
                             score: str = "continuous") -> Dict[str, Any]:
    """Detailed material evaluation with full data extraction."""
    from neutronics_calphad.neutronics.dose import contact_dose
    from neutronics_calphad.neutronics.depletion import extract_gas_production
    
    # Compute full dose-time series
    times_s, dose_dicts = contact_dose(results=results, chain_file=chain_file, abs_file=abs_file)
    
    # Compute gas production (appm)
    gas_production_rates = extract_gas_production(results)
    
    # Identify end of irradiation
    source_rates = results.get_source_rates()
    final_idx = np.nonzero(source_rates)[0][-1]
    final_irr_time = times_s[final_idx]
    
    # Build total-dose lookup
    total_dose = {t: sum(d.values()) for t, d in zip(times_s, dose_dicts)}
    
    # Locate cooling step
    cool_start_idx = final_idx + 1
    if cool_start_idx >= len(times_s):
        raise RuntimeError("No post-irradiation time steps available")
    
    cool_start_time = times_s[cool_start_idx]
    cool_times = times_s[cool_start_idx:] - cool_start_time
    
    # Check dose limits
    satisfy_dose = True
    dose_at_limit = {}
    for days_after, limit in dose_limits.items():
        target_s = days_after * 24 * 3600
        rel_idx = np.searchsorted(cool_times, target_s, side='left')
        if rel_idx >= len(cool_times):
            rel_idx = len(cool_times) - 1
        
        abs_idx = cool_start_idx + rel_idx
        t_actual = times_s[abs_idx]
        rate_actual = total_dose[t_actual]
        
        dose_at_limit[days_after] = rate_actual
        if rate_actual > limit:
            satisfy_dose = False
    
    # Check gas limits
    satisfy_gas = True
    for gas, produced in gas_production_rates.items():
        limit = critical_gas_limits.get(gas, np.inf)
        if produced > limit:
            satisfy_gas = False
    
    # Compute score
    if score == "continuous":
        dose_scores = [max(0.0, 1 - dose_at_limit[d] / dose_limits[d]) for d in dose_limits]
        gas_scores = [max(0.0, 1 - gas_production_rates.get(g, 0.0) / critical_gas_limits.get(g, np.inf)) for g in critical_gas_limits]
        final_score = float((np.mean(dose_scores) + np.mean(gas_scores)) / 2)
    else:
        final_score = 1.0 if (satisfy_dose and satisfy_gas) else 0.0
    
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


def run_neutronics_optimization_stage(pipeline_state: PipelineState) -> Dict[str, Any]:
    """Run Stage 1: Neutronics-only Bayesian optimization."""
    print("=== STAGE 1: NEUTRONICS OPTIMIZATION ===")
    
    # Load existing results if available
    existing_results = load_neutronics_results()
    
    # Check if already completed
    if pipeline_state.is_stage_completed('neutronics') and existing_results:
        print("Neutronics stage already completed. Loading existing results...")
        return existing_results
    
    # Set up OpenMC
    model, flux, microxs, timesteps, sources = setup_openmc_model()
    vessel_cell_idx = next(i for i, c in enumerate(model.geometry.get_all_cells().values()) if c.name == 'vessel')
    original_materials_count = len(model.materials)
    
    # Create/restart optimizer
    optimizer, start_iteration = create_restartable_optimizer(existing_results)
    
    # Initialize results structure
    if existing_results:
        optimization_results = existing_results
    else:
        optimization_results = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'elements': ELEMENTS,
                'critical_limits': CRIT_LIMITS,
                'dose_limits': DOSE_LIMITS,
                'composition_constraints': {
                    'min': MIN_COMPOSITIONS,
                    'max': MAX_COMPOSITIONS
                },
                'stage': 'neutronics_only',
                'max_iterations': NEUTRONICS_MAX_ITERATIONS,
                'batch_size': NEUTRONICS_BATCH_SIZE,
                'convergence_tolerance': NEUTRONICS_CONVERGENCE_TOLERANCE
            },
            'iterations': []
        }
    
    # Optimization loop
    for iteration in range(start_iteration, NEUTRONICS_MAX_ITERATIONS):
        print(f"\n=== Neutronics Iteration {iteration + 1} ===")
        
        # Get suggestions
        batch = optimizer.suggest()
        
        iteration_results = {
            'iteration': iteration + 1,
            'materials': []
        }
        
        scores = []
        for comp in batch:
            comp_dict = dict(zip(ELEMENTS, comp))
            mat_name = material_string(comp_dict, 'V')
            material = create_material(comp_dict, mat_name)
            material.depletable = True
            
            # Clean up materials
            while len(model.materials) > original_materials_count:
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
                timesteps=timesteps,
                source_rates=sources,
                outdir=os.path.join(RESULTS_DIR, 'neutronics_depletion', mat_name),
            )
            
            # Evaluate
            detailed_results = evaluate_material_detailed(
                results=results,
                chain_file=openmc.config['chain_file'],
                abs_file='/home/myless/Packages/fispact/nuclear_data/decay/abs_2012',
                critical_gas_limits=CRIT_LIMITS,
                dose_limits=DOSE_LIMITS,
                score='continuous',
            )
            
            scores.append(detailed_results['score'])
            
            # Store results
            material_result = {
                'material_name': mat_name,
                'composition': comp_dict,
                'composition_array': comp.tolist(),
                'score': detailed_results['score'],
                'satisfy_dose': detailed_results['satisfy_dose'],
                'satisfy_gas': detailed_results['satisfy_gas'],
                'dose_rates': detailed_results['dose_at_limit'],
                'gas_production': detailed_results['gas_production_rates'],
                'raw_neutronics_data': {
                    'times_s': detailed_results['times_s'],
                    'total_dose': detailed_results['total_dose'],
                    'final_irr_time': detailed_results['final_irr_time'],
                    'cool_start_time': detailed_results['cool_start_time'],
                    'source_rates': detailed_results['source_rates']
                }
            }
            
            iteration_results['materials'].append(material_result)
            print(f"  {mat_name}: score={detailed_results['score']:.3f}, "
                  f"dose_ok={detailed_results['satisfy_dose']}, "
                  f"gas_ok={detailed_results['satisfy_gas']}")
        
        # Update optimizer
        optimizer.update(batch, np.array(scores))
        optimization_results['iterations'].append(iteration_results)
        
        # Save progress
        save_neutronics_results(optimization_results)
        pipeline_state.state['neutronics_iteration'] = iteration + 1
        pipeline_state.save_state()
        
        # Check convergence
        if iteration >= 5:  # Start checking after some iterations
            # Extract predicted values for convergence check
            # This is simplified - in practice, you'd extract from GP model
            all_scores = []
            all_compositions = []
            for iter_data in optimization_results['iterations'][-10:]:  # Last 10 iterations
                for mat in iter_data['materials']:
                    all_scores.append(mat['score'])
                    all_compositions.append(mat['composition_array'])
            
            if len(all_scores) >= 20:  # Minimum for convergence check
                # Mock convergence check - replace with real GP predictions
                predicted_values = {
                    'dose_14': np.array([score * DOSE_LIMITS[14] for score in all_scores[-20:]]),
                    'gas_he': np.array([score * CRIT_LIMITS['He_appm'] for score in all_scores[-20:]])
                }
                
                conv_limits = {'dose_14': DOSE_LIMITS[14], 'gas_he': CRIT_LIMITS['He_appm']}
                
                conv_results = overall_convergence_check(
                    predicted_values=predicted_values,
                    compositions=np.array(all_compositions[-20:]),
                    elements=ELEMENTS,
                    limits=conv_limits,
                    tolerance=NEUTRONICS_CONVERGENCE_TOLERANCE,
                    min_samples=15
                )
                
                print(f"Convergence: {conv_results['summary']['message']}")
                if conv_results['converged']:
                    print("Neutronics optimization converged!")
                    break
        
        # Print iteration summary
        n_passing = sum(1 for mat in iteration_results['materials'] 
                       if mat['satisfy_dose'] and mat['satisfy_gas'])
        print(f"Iteration summary: {n_passing}/{len(iteration_results['materials'])} "
              f"compositions pass neutronics constraints")
    
    # Mark neutronics stage as completed
    pipeline_state.update_stage('neutronics', completed=True)
    
    # Clean up
    while len(model.materials) > original_materials_count:
        model.materials.pop()
    
    return optimization_results


def run_calphad_stage(pipeline_state: PipelineState, neutronics_results: Dict[str, Any]) -> Dict[str, Any]:
    """Run Stage 2: CALPHAD calculations on neutronics-passing compositions."""
    print("\n=== STAGE 2: CALPHAD PHASE ANALYSIS ===")
    
    # Check if already completed
    if pipeline_state.is_stage_completed('calphad') and os.path.exists(CALPHAD_RESULTS_FILE):
        print("CALPHAD stage already completed. Loading existing results...")
        with open(CALPHAD_RESULTS_FILE, 'r') as f:
            return json.load(f)
    
    # Extract neutronics-passing compositions
    passing_compositions = extract_neutronics_passing_compositions(neutronics_results, score_threshold=0.7)
    
    print(f"Found {len(passing_compositions)} compositions that pass neutronics constraints")
    
    if not passing_compositions:
        print("No compositions pass neutronics constraints. Skipping CALPHAD stage.")
        return {'compositions': [], 'results': []}
    
    # Initialize CALPHAD calculator
    calphad_calc = CALPHADBatchCalculator(
        database="TCHEA7",
        temperature=823.5
    )
    
    # Convert to arrays for batch calculation
    comp_arrays = np.array([[comp[el] for el in ELEMENTS] for comp in passing_compositions])
    
    # Run CALPHAD calculations
    print("Running CALPHAD calculations...")
    phase_results = calphad_calc.calculate_batch(comp_arrays, ELEMENTS)
    
    # Process results
    calphad_results = {
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'n_compositions': len(passing_compositions),
            'phase_limits': PHASE_LIMITS,
            'elements': ELEMENTS,
            'calphad_database': "TCHEA7",
            'temperature': 823.5
        },
        'compositions': passing_compositions,
        'phase_results': phase_results.to_dict(orient='records'),
        'phase_analysis': []
    }
    
    # Analyze each composition
    n_phase_passing = 0
    for i, (comp, phase_row) in enumerate(zip(passing_compositions, phase_results.itertuples())):
        # Parse phase data
        import json as json_module
        phase_fractions = json_module.loads(phase_row.phases)
        
        # Filter by threshold
        significant_phases = {
            phase: fraction for phase, fraction in phase_fractions.items()
            if fraction >= 0.001  # Phase fraction threshold
        }
        
        # Check phase limits
        phase_violations = []
        satisfies_phase_limits = True
        
        for phase, fraction in significant_phases.items():
            limit = PHASE_LIMITS.get(phase, np.inf)
            if fraction > limit:
                phase_violations.append({
                    'phase': phase,
                    'fraction': fraction,
                    'limit': limit
                })
                satisfies_phase_limits = False
        
        if satisfies_phase_limits:
            n_phase_passing += 1
        
        analysis = {
            'composition': comp,
            'phase_count': len(significant_phases),
            'dominant_phase': phase_row.dominant_phase,
            'single_phase': phase_row.single_phase,
            'significant_phases': significant_phases,
            'satisfies_phase_limits': satisfies_phase_limits,
            'phase_violations': phase_violations
        }
        
        calphad_results['phase_analysis'].append(analysis)
    
    # Save results
    with open(CALPHAD_RESULTS_FILE, 'w') as f:
        json.dump(calphad_results, f, indent=2, default=str)
    
    pipeline_state.update_stage('calphad', completed=True)
    
    print(f"CALPHAD analysis complete:")
    print(f"  {len(passing_compositions)} compositions analyzed")
    print(f"  {n_phase_passing} compositions pass phase constraints")
    print(f"  Success rate: {100 * n_phase_passing / len(passing_compositions):.1f}%")
    
    return calphad_results


def main():
    """Main pipeline execution."""
    print("=== SEQUENTIAL MATERIALS DESIGN PIPELINE ===")
    
    # Initialize pipeline state
    pipeline_state = PipelineState(PIPELINE_STATE_FILE)
    
    current_stage = pipeline_state.get_current_stage()
    print(f"Current pipeline stage: {current_stage}")
    
    # Stage 1: Neutronics optimization
    if current_stage in ['neutronics'] or not pipeline_state.is_stage_completed('neutronics'):
        neutronics_results = run_neutronics_optimization_stage(pipeline_state)
    else:
        neutronics_results = load_neutronics_results()
    
    # Stage 2: CALPHAD analysis
    if current_stage in ['neutronics', 'calphad'] or not pipeline_state.is_stage_completed('calphad'):
        calphad_results = run_calphad_stage(pipeline_state, neutronics_results)
    
    # Stage 3: Future stages (ductility, etc.)
    # TODO: Implement ductility modeling stage
    
    print("\n=== PIPELINE SUMMARY ===")
    total_evaluated = sum(len(iter_data['materials']) for iter_data in neutronics_results['iterations'])
    neutronics_passing = len(extract_neutronics_passing_compositions(neutronics_results))
    
    print(f"Total compositions evaluated: {total_evaluated}")
    print(f"Neutronics-passing compositions: {neutronics_passing}")
    print(f"Neutronics success rate: {100 * neutronics_passing / total_evaluated:.1f}%")
    
    if pipeline_state.is_stage_completed('calphad'):
        with open(CALPHAD_RESULTS_FILE, 'r') as f:
            calphad_data = json.load(f)
        phase_passing = sum(1 for analysis in calphad_data['phase_analysis'] 
                           if analysis['satisfies_phase_limits'])
        print(f"Phase-stable compositions: {phase_passing}")
        print(f"Overall success rate: {100 * phase_passing / total_evaluated:.1f}%")


if __name__ == "__main__":
    main() 