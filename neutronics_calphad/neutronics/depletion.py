"""
Depletion and transmutation calculation utilities.

This module contains functions for running neutron activation calculations,
handling burnup and cooling phases, and extracting gas production data.
"""
import numpy as np
import openmc
import openmc.deplete
from pathlib import Path
from typing import List, Dict, Any
import os

def run_independent_depletion(model: openmc.Model, 
                             depletable_cell: str,
                             microxs: openmc.deplete.MicroXS, 
                             flux: np.ndarray, 
                             chain_file: str, 
                             timesteps: List[float], 
                             source_rates: List[float],
                             outdir: Path) -> openmc.deplete.Results:
    """Run depletion calculation with IndependentOperator using collapsed cross sections.
    
    Args:
        model: The OpenMC model with initial compositions.
        collapsed_microxs: The collapsed cross-sections object.
        flux: The flux spectrum array.
        chain_file: Path to the depletion chain file.
        timesteps: List of irradiation/cooling durations in seconds.
        power: Fusion power in Watts.
        outdir: Directory to save depletion results.
        
    Returns:
        The depletion results object.
        
    Raises:
        ValueError: If no depletable materials are found or vcrti material is missing.
    """

    
    cell = model.geometry.get_cells_by_name(depletable_cell)[0]
    # gets the first material object in the dictionary
    material = next(iter(cell.get_all_materials().values())) 

    
    print(f"Using material for depletion: {material.name}")
    
    # Create IndependentOperator with the collapsed cross sections
    # Only pass the material we have flux/XS data for
    op = openmc.deplete.IndependentOperator(
        materials=[material],      # Only the material we calculated flux/XS for
        fluxes=[flux],            # List of flux arrays (one per material domain)
        micros=[microxs],      # List of MicroXS objects (one per material domain)
        chain_file=str(chain_file),
        normalization_mode='source-rate',
        reduce_chain_level=8             # More aggressive chain reduction to avoid small concentrations
    )
    
    
    # Use CECM integrator with tighter tolerance to handle numerical issues
    integrator = openmc.deplete.PredictorIntegrator(
        op, 
        timesteps, 
        source_rates=source_rates,
        timestep_units='s'
    )
    
    # Set solver options to handle negative densities
    # These settings help with numerical stability
    integrator.solver_kwargs = {
        'rtol': 1e-4,        # Relative tolerance (slightly relaxed)
        'atol': 1e-12,       # Absolute tolerance for small concentrations
        'max_step': 1e6,     # Maximum step size in seconds
        'method': 'BDF'      # Backward differentiation formula for stiff problems
    }
    
    print("Running depletion with enhanced numerical stability settings...")
    integrator.integrate(path=str(os.path.join(outdir, "depletion_results.h5")))
    
    return openmc.deplete.Results(os.path.join(outdir, "depletion_results.h5"))


def extract_gas_production(results: openmc.deplete.Results) -> Dict[str, float]:
    """Extract gas production data from depletion results.
    
    Args:
        results: Depletion results object.
        
    Returns:
        Dictionary mapping gas categories to appm (atomic parts per million) after irradiation.
        Returns 'He_appm' (sum of He3 + He4) and 'H_appm' (sum of H1 + H2 + H3).
    """
    gases = {}
    gas_species = ['He3', 'He4', 'H1', 'H2', 'H3']
    
    # Get the material id from the results
    material_id = list(results[0].index_mat.keys())[0]

    print(f"Gas production analysis:")
    print(f"  - Material ID: {material_id}")
    print(f"  - Number of time steps in results: {len(results)}")

    # Get the final irradiation time step based on the source rates from the results 
    # it will be the last non-zero source rate
    source_rates = results.get_source_rates()
    final_source_rate_index = np.nonzero(source_rates)[0][-1]
    
    # Get total number of atoms in the material at the final timestep
    # We need to get all nuclides and sum their atoms
    try:
        # Get the material at the final timestep
        final_material = results[final_source_rate_index].get_material(str(material_id))
        # Get all nuclides in the material
        all_nuclides = final_material.get_nuclides()
        
        # Sum all atoms for each nuclide
        total_atoms = 0.0
        for nuclide in all_nuclides:
            times, numbers = results.get_atoms(material_id, nuclide)
            total_atoms += numbers[final_source_rate_index]
        
        print(f"  - Total atoms in material: {total_atoms:.2e}")
    except Exception as e:
        print(f"  - Warning: Could not get total atoms: {e}")
        total_atoms = 1.0  # Fallback to avoid division by zero
    
    # Initialize individual gas appm values
    individual_gases = {}
    
    for gas in gas_species:
        try:
            times, numbers = results.get_atoms(material_id, gas)
            gas_atoms = numbers[final_source_rate_index]
            
            # Convert to appm (atomic parts per million)
            gas_appm = (gas_atoms / total_atoms) * 1e6 if total_atoms > 0 else 0.0
            
            individual_gases[gas] = gas_appm
            print(f"  - {gas}: {gas_atoms:.2e} atoms = {gas_appm:.2f} appm")
            
        except Exception as e:
            print(f"  - Warning: Could not get {gas} atoms: {e}")
            individual_gases[gas] = 0.0
    
    # Combine into categories expected by the evaluation function
    gases['He_appm'] = individual_gases.get('He3', 0.0) + individual_gases.get('He4', 0.0)
    gases['H_appm'] = individual_gases.get('H1', 0.0) + individual_gases.get('H2', 0.0) + individual_gases.get('H3', 0.0)
    
    print(f"  - Total He appm: {gases['He_appm']:.2f}")
    print(f"  - Total H appm: {gases['H_appm']:.2f}")

    return gases


    


def analyze_neutron_activation(results: openmc.deplete.Results, 
                              material_id: int, 
                              element: str) -> Dict[str, Any]:
    """Analyze neutron activation effectiveness and products.
    
    Args:
        results: Depletion results object.
        material_id: ID of the material to analyze.
        element: Element symbol being analyzed.
        
    Returns:
        Dictionary containing activation analysis results.
    """
    analysis = {}
    
    print(f"\nNeutron activation analysis for {element}")
    print(f"  - Material ID: {material_id}")
    
    try:
        # Get material after irradiation (step 0 is initial, step 1 is after irradiation)
        initial_mat = results[0].get_material(str(material_id))
        irradiated_mat = results[1].get_material(str(material_id))
        
        # Check total activity after irradiation
        total_activity = irradiated_mat.get_activity()  # Bq
        analysis['total_activity_bq'] = total_activity
        print(f"  - Total activity after irradiation: {total_activity:.2e} Bq")
        
        # Check decay heat
        decay_heat = irradiated_mat.get_decay_heat()  # W
        analysis['decay_heat_w'] = decay_heat
        print(f"  - Decay heat after irradiation: {decay_heat:.2e} W")
        
        # List major activation products
        nuclide_list = list(irradiated_mat.get_nuclides())
        analysis['num_nuclides'] = len(nuclide_list)
        analysis['nuclides'] = nuclide_list
        print(f"  - Number of nuclides after irradiation: {len(nuclide_list)}")
        print(f"  - Nuclides present: {nuclide_list[:10]}...")  # Show first 10
        
        # Just list the nuclides for now (individual activity calculation needs different API)
        print(f"  - Major activation products: {nuclide_list[:5]}")  # Show first 5
        
        # Material density information if available
        initial_density = getattr(initial_mat, 'density', None)
        irradiated_density = getattr(irradiated_mat, 'density', None)
        
        if initial_density is not None:
            analysis['initial_density_g_cm3'] = initial_density
            print(f"  - Initial material density: {initial_density:.2f} g/cm³")
        if irradiated_density is not None:
            analysis['irradiated_density_g_cm3'] = irradiated_density
            print(f"  - Irradiated material density: {irradiated_density:.2f} g/cm³")
            
    except Exception as e:
        print(f"  - Error analyzing activation: {e}")
        import traceback
        traceback.print_exc()
        analysis['error'] = str(e)
    
    return analysis


def validate_depletion_results(results: openmc.deplete.Results, 
                              expected_timesteps: int) -> bool:
    """Validate depletion results for consistency and completeness.
    
    Args:
        results: Depletion results object to validate.
        expected_timesteps: Expected number of time steps.
        
    Returns:
        True if validation passes, False otherwise.
    """
    print(f"Validating depletion results:")
    print(f"  - Expected timesteps: {expected_timesteps}")
    print(f"  - Actual timesteps: {len(results)}")
    
    if len(results) != expected_timesteps:
        print(f"  ❌ FAIL: Timestep count mismatch")
        return False
    
    # Check that we have at least initial and post-irradiation states
    if len(results) < 2:
        print(f"  ❌ FAIL: Need at least 2 timesteps (initial + post-irradiation)")
        return False
    
    try:
        # Test basic result access
        material_ids = list(results[0].index_mat.keys())
        if not material_ids:
            print(f"  ❌ FAIL: No materials found in results")
            return False
        
        # Test activity calculation
        test_mat = results[1].get_material(str(material_ids[0]))
        test_activity = test_mat.get_activity()
        
        if test_activity < 0:
            print(f"  ❌ FAIL: Negative activity detected")
            return False
        
        print(f"  ✅ PASS: Depletion results validation successful")
        return True
        
    except Exception as e:
        print(f"  ❌ FAIL: Error accessing results: {e}")
        return False 