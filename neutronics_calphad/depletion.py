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

from .flux import calculate_actual_flux, E_PER_FUSION_eV, UNITS_EV_TO_J


def run_independent_depletion(model: openmc.Model, 
                             collapsed_microxs: openmc.deplete.MicroXS, 
                             flux: np.ndarray, 
                             chain_file: str, 
                             timesteps: List[float], 
                             power: float, 
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
    # Calculate actual flux from fusion power
    actual_flux = calculate_actual_flux(flux, power)
    
    # Find the vcrti material (the one we calculated flux/XS for)
    vcrti_material = None
    for mat in model.materials:
        if mat.name == 'vcrti' and mat.depletable:
            vcrti_material = mat
            break
    
    if vcrti_material is None:
        # Fallback to first depletable material
        depletable_mats = [m for m in model.materials if m.depletable]
        if not depletable_mats:
            raise ValueError("No depletable materials found")
        vcrti_material = depletable_mats[0]
        print(f"Warning: vcrti material not found, using {vcrti_material.name}")
    
    print(f"Using material for depletion: {vcrti_material.name}")
    
    # Create IndependentOperator with the collapsed cross sections
    # Only pass the material we have flux/XS data for
    op = openmc.deplete.IndependentOperator(
        materials=[vcrti_material],      # Only the material we calculated flux/XS for
        fluxes=[actual_flux],            # List of flux arrays (one per material domain)
        micros=[collapsed_microxs],      # List of MicroXS objects (one per material domain)
        chain_file=str(chain_file),
        normalization_mode='source-rate',
        reduce_chain=True,               # Enable chain reduction
        reduce_chain_level=8             # More aggressive chain reduction to avoid small concentrations
    )
    
    # Set up time steps and source rates (irradiation followed by cooling)
    source_rate = power / (E_PER_FUSION_eV * UNITS_EV_TO_J)
    source_rates = [source_rate] + [0.0] * (len(timesteps) - 1)  # Irradiation then cooling
    
    # Use CECM integrator with tighter tolerance to handle numerical issues
    integrator = openmc.deplete.CECMIntegrator(
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
    integrator.integrate(path=str(outdir / "depletion_results.h5"))
    
    return openmc.deplete.Results(outdir / "depletion_results.h5")


def extract_gas_production(results: openmc.deplete.Results, 
                          material_id: int) -> Dict[str, float]:
    """Extract gas production data from depletion results.
    
    Args:
        results: Depletion results object.
        material_id: ID of the material to analyze.
        
    Returns:
        Dictionary mapping gas species to atom counts after irradiation.
    """
    gases = {}
    gas_species = ['He3', 'He4', 'H1', 'H2', 'H3']
    
    print(f"Gas production analysis:")
    print(f"  - Material ID: {material_id}")
    print(f"  - Number of time steps in results: {len(results)}")
    
    for gas in gas_species:
        try:
            # Use Results.get_atoms() method - this is the correct API
            times, atom_counts = results.get_atoms(material_id, gas, nuc_units="atoms")
            print(f"  - {gas}: Found {len(atom_counts)} time points")
            if len(atom_counts) > 0:
                print(f"    Time 0 (initial): {atom_counts[0]:.2e} atoms")
                if len(atom_counts) > 1:
                    print(f"    Time 1 (after irradiation): {atom_counts[1]:.2e} atoms")
                    gases[gas] = atom_counts[1]  # After irradiation, not initial
                else:
                    gases[gas] = atom_counts[0]
            else:
                gases[gas] = 0.0
                print(f"    No data found for {gas}")
            print(f"DEBUG: {gas} atoms after irradiation: {gases[gas]:.2e}")
        except Exception as e:
            print(f"Warning: Could not get {gas} atoms from results: {e}")
            # Try alternative approach - check if the nuclide exists at all
            try:
                # Check if this gas nuclide is present in any time step
                for i in range(len(results)):
                    mat = results[i].get_material(str(material_id))
                    nuclides = list(mat.get_nuclides())
                    if gas in nuclides:
                        print(f"  - {gas} found in time step {i}: {nuclides}")
                        break
                else:
                    print(f"  - {gas} not found in any time step")
            except:
                pass
            gases[gas] = 0.0
    
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