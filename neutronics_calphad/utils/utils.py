"""
Utility functions for neutronics calculations.

This module contains helper functions, constants, and utilities
that are shared across different parts of the neutronics_calphad package.
"""
from typing import List, Dict, Any
import openmc


# Common time constants
TIMES = [
    1,                    # 1 second
    3600,                 # 1 hour
    10*3600,              # 10 hours
    24*3600,              # 1 day
    7*24*3600,            # 1 week
    14*24*3600,           # 2 weeks
    30*24*3600,           # 1 month (30 days)
    60*24*3600,           # 2 months (60 days)
    365*24*3600,          # 1 year
    5*365*24*3600,        # 5 years
    10*365*24*3600,       # 10 years
    25*365*24*3600,       # 25 years
    100*365*24*3600       # 100 years
]

# Element list for batch processing
ELMS = ['V', 'Cr', 'Ti', 'W', 'Zr']

# Constants for FISPACT-like calculations
E_PER_FUSION_eV = 17.6e6  # eV per D-T fusion event
UNITS_EV_TO_J = 1.60218e-19


def get_material_by_name(materials: List[openmc.Material], 
                        name: str) -> openmc.Material:
    """Find a material by name in a list of materials.
    
    Args:
        materials: List of OpenMC materials to search.
        name: Name of the material to find.
        
    Returns:
        The material with the specified name.
        
    Raises:
        ValueError: If no material with the given name is found.
    """
    for material in materials:
        if material.name == name:
            return material
    raise ValueError(f"Material with name '{name}' not found")


def format_time_label(time_seconds: float) -> str:
    """Convert time in seconds to a human-readable label.
    
    Args:
        time_seconds: Time in seconds.
        
    Returns:
        Human-readable time label.
    """
    time_labels = {
        1: "1 second",
        3600: "1 hour",
        10*3600: "10 hours",
        24*3600: "1 day",
        7*24*3600: "1 week",
        14*24*3600: "2 weeks",
        30*24*3600: "1 month",
        60*24*3600: "2 months",
        365*24*3600: "1 year",
        5*365*24*3600: "5 years",
        10*365*24*3600: "10 years",
        25*365*24*3600: "25 years",
        100*365*24*3600: "100 years"
    }
    
    return time_labels.get(time_seconds, f"{time_seconds} s")


def validate_environment_variables() -> Dict[str, Any]:
    """Validate that required environment variables are set.
    
    Returns:
        Dictionary containing validation results and paths.
    """
    import os
    
    validation = {
        'valid': True,
        'issues': [],
        'paths': {}
    }
    
    # Check for OpenMC chain file
    chain_file = os.environ.get('OPENMC_CHAIN_FILE')
    if not chain_file:
        validation['valid'] = False
        validation['issues'].append("OPENMC_CHAIN_FILE environment variable not set")
    else:
        validation['paths']['chain_file'] = chain_file
        if not os.path.exists(chain_file):
            validation['valid'] = False
            validation['issues'].append(f"Chain file not found: {chain_file}")
    
    # Check for OpenMC cross sections
    xs_file = os.environ.get('OPENMC_CROSS_SECTIONS')
    if not xs_file:
        validation['valid'] = False
        validation['issues'].append("OPENMC_CROSS_SECTIONS environment variable not set")
    else:
        validation['paths']['cross_sections'] = xs_file
        if not os.path.exists(xs_file):
            validation['valid'] = False
            validation['issues'].append(f"Cross sections file not found: {xs_file}")
    
    return validation


def print_simulation_header(element: str, 
                           workflow: str, 
                           power: float) -> None:
    """Print a formatted header for simulation runs.
    
    Args:
        element: Element symbol being simulated.
        workflow: Workflow type (r2s, fispact_path_a, etc.).
        power: Fusion power in Watts.
    """
    print(f"\n{'='*80}")
    print(f" NEUTRONICS SIMULATION")
    print(f"{'='*80}")
    print(f" Element: {element}")
    print(f" Workflow: {workflow}")
    print(f" Power: {power/1e6:.1f} MW")
    print(f"{'='*80}")


def print_simulation_summary(element: str, 
                           workflow: str, 
                           dose_rates: List[float],
                           gases: Dict[str, float]) -> None:
    """Print a summary of simulation results.
    
    Args:
        element: Element symbol that was simulated.
        workflow: Workflow type that was used.
        dose_rates: List of calculated dose rates.
        gases: Dictionary of gas production results.
    """
    print(f"\n{'='*60}")
    print(f" SIMULATION SUMMARY - {element} ({workflow})")
    print(f"{'='*60}")
    
    # Dose rate summary
    if dose_rates:
        print(f" Dose rates calculated: {len(dose_rates)} time points")
        max_dose = max(dose_rates) if dose_rates else 0
        print(f" Maximum dose rate: {max_dose:.2e} µSv/h")
        
        # Check if all doses are zero
        if all(d == 0 for d in dose_rates):
            print(f" ⚠️  WARNING: All dose rates are zero")
        elif max_dose > 1e15:  # Very high dose rates
            print(f" ⚠️  WARNING: Very high dose rates detected")
    else:
        print(f" ❌ No dose rates calculated")
    
    # Gas production summary
    print(f"\n Gas production:")
    for gas, atoms in gases.items():
        if atoms > 0:
            print(f"   {gas}: {atoms:.2e} atoms")
        else:
            print(f"   {gas}: None detected")
    
    print(f"{'='*60}")


def check_openmc_version() -> Dict[str, Any]:
    """Check OpenMC version and compatibility.
    
    Returns:
        Dictionary containing version information and compatibility status.
    """
    import openmc
    
    version_info = {
        'version': openmc.__version__,
        'compatible': True,
        'warnings': []
    }
    
    # Check for minimum version requirements
    try:
        version_parts = openmc.__version__.split('.')
        major = int(version_parts[0])
        minor = int(version_parts[1]) if len(version_parts) > 1 else 0
        
        # Require OpenMC 0.13+ for best depletion support
        if major == 0 and minor < 13:
            version_info['compatible'] = False
            version_info['warnings'].append(
                f"OpenMC version {openmc.__version__} may not support all depletion features. "
                "Consider upgrading to 0.13+ for best results."
            )
    except (ValueError, IndexError):
        version_info['warnings'].append(
            f"Could not parse OpenMC version: {openmc.__version__}"
        )
    
    # Check for required modules
    required_modules = ['openmc.deplete', 'openmc.lib']
    for module in required_modules:
        try:
            parts = module.split('.')
            obj = openmc
            for part in parts[1:]:
                obj = getattr(obj, part)
        except AttributeError:
            version_info['compatible'] = False
            version_info['warnings'].append(f"Required module not available: {module}")
    
    return version_info


def create_progress_tracker(total_elements: int) -> Dict[str, Any]:
    """Create a progress tracking object for batch simulations.
    
    Args:
        total_elements: Total number of elements to process.
        
    Returns:
        Dictionary containing progress tracking information.
    """
    return {
        'total': total_elements,
        'completed': 0,
        'failed': 0,
        'current': None,
        'start_time': None,
        'results': {}
    }


def update_progress_tracker(tracker: Dict[str, Any], 
                          element: str, 
                          success: bool,
                          error_msg: str = None) -> None:
    """Update progress tracking information.
    
    Args:
        tracker: Progress tracking dictionary.
        element: Element that was just processed.
        success: Whether the element processing succeeded.
        error_msg: Error message if processing failed.
    """
    tracker['completed'] += 1
    if not success:
        tracker['failed'] += 1
    
    tracker['results'][element] = {
        'success': success,
        'error': error_msg
    }
    
    # Print progress
    completion_rate = tracker['completed'] / tracker['total'] * 100
    print(f"\nProgress: {tracker['completed']}/{tracker['total']} ({completion_rate:.1f}%)")
    if tracker['failed'] > 0:
        print(f"  - Failed: {tracker['failed']}")


def validate_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Validate a simulation configuration.
    
    Args:
        config: Configuration dictionary to validate.
        
    Returns:
        Dictionary containing validation results.
    """
    validation = {
        'valid': True,
        'errors': [],
        'warnings': []
    }
    
    # Check required top-level keys
    required_keys = ['geometry', 'materials']
    for key in required_keys:
        if key not in config:
            validation['valid'] = False
            validation['errors'].append(f"Missing required config key: {key}")
    
    # Check materials configuration
    if 'materials' in config:
        materials = config['materials']
        if not isinstance(materials, dict):
            validation['valid'] = False
            validation['errors'].append("'materials' must be a dictionary")
        else:
            # Check for vcrti material (commonly used)
            if 'vcrti' not in materials:
                validation['warnings'].append("No 'vcrti' material found in config")
            
            # Validate each material
            for mat_name, mat_config in materials.items():
                if not isinstance(mat_config, dict):
                    validation['errors'].append(f"Material '{mat_name}' config must be a dictionary")
                    continue
                
                # Check required material properties
                if 'elements' not in mat_config:
                    validation['errors'].append(f"Material '{mat_name}' missing 'elements'")
                if 'density' not in mat_config:
                    validation['errors'].append(f"Material '{mat_name}' missing 'density'")
    
    # Check geometry configuration
    if 'geometry' in config:
        geometry = config['geometry']
        if not isinstance(geometry, dict):
            validation['valid'] = False
            validation['errors'].append("'geometry' must be a dictionary")
    
    return validation 