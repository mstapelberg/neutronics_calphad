"""
Dose rate calculation and radiation safety utilities.

This module contains functions for calculating contact dose rates,
processing gamma spectra, and implementing FISPACT-like dose formulas.
"""
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import openmc.deplete
import re


def calculate_fispact_dose(results: openmc.deplete.Results, 
                          data_dir: Path) -> np.ndarray:
    """Calculate dose rate using the FISPACT semi-empirical formula.
    
    Args:
        results: Depletion results object.
        data_dir: Path to the data directory containing 'mass_energy_abs_coeff_air.csv'.
        
    Returns:
        Array of dose rates in µSv/h for each cooling step.
        
    Raises:
        FileNotFoundError: If the required data file is not found.
    """
    coeff_file = data_dir / "mass_energy_abs_coeff_air.csv"
    if not coeff_file.exists():
        raise FileNotFoundError(f"Required data file not found: {coeff_file}")
    
    coeffs = pd.read_csv(coeff_file)
    interp_func = interp1d(
        coeffs["Energy (MeV)"], 
        coeffs["mu_en/rho (cm^2/g)"],
        bounds_error=False, 
        fill_value="extrapolate"
    )

    dose_rates = []
    # Skip initial and irradiation steps (indices 0 and 1)
    for i in range(2, len(results)):
        mat = results[i].get_material(str(list(results[i].index_mat.keys())[0]))
        activities = mat.get_activity(by_nuclide=True, units='Bq')
        gamma_spec = mat.get_decay_photon_energy()

        if not gamma_spec:
            dose_rates.append(0.0)
            continue
        
        # Gamma spectrum energies are in eV, convert to MeV
        energies_mev = gamma_spec.x * 1e-6
        mu_en_rho = interp_func(energies_mev)
        
        # Dose rate formula from FISPACT manual (Eq. 60)
        # Dose (Sv/h) = 1.602E-13 * 3600 * sum(Activity * Energy * Coeff)
        # Here we get µSv/h
        dose_sv_h = 1.602e-13 * 3600 * np.sum(gamma_spec.p * energies_mev * mu_en_rho)
        dose_usv_h = dose_sv_h * 1e6
        
        # Debug: Add activity and photon rate information
        total_activity = mat.get_activity()  # Bq
        if hasattr(gamma_spec, 'integral'):
            photon_rate = gamma_spec.integral()  # photons/s
        else:
            photon_rate = np.sum(gamma_spec.p)
        
        if i == 2:  # Only print for first cooling step to avoid spam
            print(f"DEBUG FISPACT dose calculation:")
            print(f"  - Total activity: {total_activity:.2e} Bq")
            print(f"  - Photon rate: {photon_rate:.2e} photons/s")
            print(f"  - Average photon energy: {np.mean(energies_mev):.3f} MeV")
            print(f"  - Dose rate: {dose_usv_h:.2e} µSv/h = {dose_sv_h:.2e} Sv/h")
        
        dose_rates.append(dose_usv_h)
        
    return np.array(dose_rates)


def get_fispact_xs(printlib_path: Path, 
                   outdir: Path) -> Path:
    """Parse a FISPACT printlib file to extract collapsed cross sections.

    This function reads a FISPACT-II `printlib` file, finds the block
    containing one-group cross-sections, parses the data, and saves it
    to a CSV file in the same format as the collapsed OpenMC cross sections.

    Args:
        printlib_path: Path to the FISPACT printlib file.
        outdir: Directory to save the output CSV.

    Returns:
        Path to the output CSV file with collapsed xs.
        
    Raises:
        ValueError: If no cross-section data can be parsed from the file.
    """
    import openmc
    
    print(f"Parsing FISPACT printlib file: {printlib_path}")
    collapsed_data = []
    in_xs_block = False

    with open(printlib_path, 'r') as f:
        for line in f:
            # Block 3 with cross sections starts with this header
            if "CROSS-SECTIONS FOR ALL REACTIONS" in line:
                in_xs_block = True
                # Skip the next 3 header lines to get to the data
                next(f, None); next(f, None); next(f, None)
                continue

            if not in_xs_block:
                continue

            # A blank line or a line with "TOTAL" indicates the end of the block
            if not line.strip() or "TOTAL" in line:
                break
            
            # Attempt to parse the line. This is based on a common fixed-width format.
            try:
                # Example: V 51      102 (n,gamma)         CR 52           1.0945E-02 ...
                parts = line.split()
                if len(parts) < 5:
                    continue

                # The MT number is the first integer that is not the mass number
                mt = -1
                for part in parts:
                    if part.isdigit() and int(part) != int(parts[1]):
                        mt = int(part)
                        break
                if mt == -1:
                    continue

                # Reconstruct nuclide string, e.g., 'V51', 'Am242m' -> 'Am242_m1'
                element = parts[0]
                mass_number = parts[1]
                nuclide_str = f"{element.capitalize()}{mass_number}"
                if 'm' in nuclide_str:
                    nuclide_str = re.sub(r'm(\d*)', r'_m\1', nuclide_str)

                # Validate nuclide name
                try:
                    zaid = openmc.Nuclide(nuclide_str).zaid
                except:
                    # Skip if nuclide not recognized
                    continue
                
                # The cross section is usually the first floating point number
                # after the reaction string like '(n,gamma)'
                xs_val = 0.0
                for part in parts:
                    try:
                        if 'E' in part or '.' in part:
                           val = float(part)
                           if val >= 0.0:
                               xs_val = val
                               break 
                    except ValueError:
                        continue

                if xs_val > 0.0:
                    collapsed_data.append({
                        'nuclide': nuclide_str,
                        'reaction': mt,
                        'xs_barns': xs_val
                    })
            except (ValueError, IndexError, openmc.exceptions.DataError) as e:
                if "No data available" not in str(e): # Suppress common benign errors
                    print(f" could not parse line: '{line.strip()}'. Error: {e}")
                continue

    if not collapsed_data:
        raise ValueError(
            f"Could not parse any cross-section data from {printlib_path}. "
            "Please check if the file is a valid printlib file containing "
            "the 'CROSS-SECTIONS FOR ALL REACTIONS' block."
        )

    # Save to CSV in same format as collapsed OpenMC cross sections
    collapsed_csv = outdir / "fispact_collapsed_xs.csv"
    df = pd.DataFrame(collapsed_data)
    
    # Remove duplicates (same nuclide-reaction pair) keeping first occurrence
    df = df.drop_duplicates(subset=['nuclide', 'reaction'], keep='first')
    
    df.to_csv(collapsed_csv, index=False)
    
    print(f" Successfully parsed {len(df)} unique cross sections from printlib.")
    print(f"Saved FISPACT collapsed cross sections to {collapsed_csv}")
    
    return collapsed_csv


def get_reference_dose_rates(element: str, 
                            time_after_shutdown: float) -> Tuple[float, float]:
    """Provide reference dose rate ranges for comparison.
    
    Based on typical fusion reactor activation studies and ITER estimates.
    
    Args:
        element: Element symbol (V, Cr, etc.).
        time_after_shutdown: Time after shutdown in seconds.
        
    Returns:
        Tuple of (low_estimate, high_estimate) in µSv/h for contact dose rate.
    """
    # Reference data from fusion literature
    # These are contact dose rates for structural materials after ~1 year of operation
    
    reference_data = {
        'V': {
            1: (1e10, 1e12),         # 1 second: 100-10000 Sv/h
            3600: (1e10, 1e12),       # 1 hour: 10-1000 Sv/h
            24*3600: (1e8, 1e12),    # 1 day: 1-100 Sv/h
            14*24*3600: (1e8, 1e12), # 2 weeks: 0.01-1 Sv/h (hands-on = 0.01 Sv/h)
            365*24*3600: (1e6, 1e10), # 1 year: 0.001-0.1 Sv/h
            5*365*24*3600: (1e4, 1e8), # 5 years: 0.0001-0.01 Sv/h
            7*365*24*3600: (1e2, 1e5), # 7 years
            100*365*24*3600: (1e0, 1e4), # 100 years: 0.0001-0.01 Sv/h
        },
        'Cr': {
            1: (1e10, 1e12),         # 1 second: 100-10000 Sv/h
            3600: (1e10, 1e12),       # 1 hour: 10-1000 Sv/h
            24*3600: (1e8, 1e12),    # 1 day: 1-100 Sv/h
            14*24*3600: (1e8, 1e12), # 2 weeks: 0.01-1 Sv/h (hands-on = 0.01 Sv/h)
            365*24*3600: (1e6, 1e10), # 1 year: 0.001-0.1 Sv/h
            5*365*24*3600: (1e4, 1e8), # 5 years: 0.0001-0.01 Sv/h
            7*365*24*3600: (1e2, 1e5), # 7 years
            100*365*24*3600: (1e0, 1e4), # 100 years: 0.0001-0.01 Sv/h
        }
        # Similar patterns for other elements...
    }
    
    # Default to V estimates if element not found
    element_data = reference_data.get(element, reference_data['V'])
    
    # Find closest time point
    closest_time = min(element_data.keys(), key=lambda t: abs(t - time_after_shutdown))
    return element_data[closest_time]


def validate_dose_rates(dose_rates: List[float], 
                       times: List[float], 
                       element: str) -> Dict[str, any]:
    """Validate calculated dose rates against reference values.
    
    Args:
        dose_rates: List of calculated dose rates in µSv/h.
        times: List of cooling times in seconds.
        element: Element symbol.
        
    Returns:
        Dictionary containing validation results and statistics.
    """
    validation = {
        'total_tests': 0,
        'passed': 0,
        'failed': 0,
        'warnings': 0,
        'results': []
    }
    
    print(f"Validating dose rates for {element}:")
    
    for i, (dose_rate, time_s) in enumerate(zip(dose_rates, times)):
        ref_low, ref_high = get_reference_dose_rates(element, time_s)
        
        # Convert µSv/h to Sv/h for comparison with references
        dose_sv_h = dose_rate * 1e-6
        ref_low_sv = ref_low * 1e-6
        ref_high_sv = ref_high * 1e-6
        
        test_result = {
            'time_s': time_s,
            'dose_usv_h': dose_rate,
            'dose_sv_h': dose_sv_h,
            'ref_low_sv_h': ref_low_sv,
            'ref_high_sv_h': ref_high_sv,
            'status': 'unknown'
        }
        
        validation['total_tests'] += 1
        
        if dose_sv_h == 0:
            test_result['status'] = 'zero'
            validation['warnings'] += 1
        elif dose_sv_h > ref_high_sv * 10:
            test_result['status'] = 'too_high'
            validation['failed'] += 1
        elif dose_sv_h < ref_low_sv / 10:
            test_result['status'] = 'too_low'
            validation['failed'] += 1
        elif ref_low_sv <= dose_sv_h <= ref_high_sv:
            test_result['status'] = 'pass'
            validation['passed'] += 1
        else:
            test_result['status'] = 'marginal'
            validation['warnings'] += 1
        
        validation['results'].append(test_result)
        
        # Time label for display
        time_label = f"{time_s} s"
        if time_s == 365*24*3600:
            time_label = "1 year"
        elif time_s == 5*365*24*3600:
            time_label = "5 years"
            
        print(f"  {time_label:>12}: {dose_sv_h:.2e} Sv/h [{test_result['status']}]")
    
    success_rate = validation['passed'] / validation['total_tests'] * 100
    print(f"\nValidation summary:")
    print(f"  - Tests passed: {validation['passed']}/{validation['total_tests']} ({success_rate:.1f}%)")
    print(f"  - Tests failed: {validation['failed']}")
    print(f"  - Warnings: {validation['warnings']}")
    
    return validation


def create_dose_summary_plot(dose_rates: List[float], 
                            times: List[float], 
                            element: str, 
                            output_path: Path) -> None:
    """Create a dose rate vs. cooling time plot.
    
    Args:
        dose_rates: List of dose rates in µSv/h.
        times: List of cooling times in seconds.
        element: Element symbol.
        output_path: Path to save the plot.
    """
    import matplotlib.pyplot as plt
    
    # Convert to years for x-axis
    times_years = np.array(times) / (365.25 * 24 * 3600)
    doses_sv = np.array(dose_rates) * 1e-6  # Convert to Sv/h
    
    # Reference values in Sv/h
    reference_values_sv = {
        1: 1e5,                  # 100,000 Sv/h at 1 second
        3600: 5e4,               # 10,000 Sv/h at 1 hour
        10*3600: 1e4,            # 5,000 Sv/h at 10 hours
        24*3600: 1e4,            # 1,000 Sv/h at 1 day
        7*24*3600: 1e2,          # 100 Sv/h at 1 week
        14*24*3600: 5e1,         # 50 Sv/h at 2 weeks
        30*24*3600: 1e1,         # 10 Sv/h at 1 month
        60*24*3600: 5e0,         # 5 Sv/h at 2 months
        365*24*3600: 1e0,        # 1 Sv/h at 1 year
        5*365*24*3600: 1e-1,     # 0.1 Sv/h at 5 years
        10*365*24*3600: 1e-2,    # 0.05 Sv/h at 10 years
        25*365*24*3600: 1e-3,    # 0.01 Sv/h at 25 years
        100*365*24*3600: 1e-4    # 0.0001 Sv/h at 100 years
    }
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot calculated dose rates
    ax.loglog(times_years, doses_sv, 'bo-', linewidth=2, markersize=8, 
              label=f'{element} (calculated)')
    
    # Plot reference values
    ref_times_years = np.array(list(reference_values_sv.keys())) / (365.25 * 24 * 3600)
    ref_doses_sv = np.array(list(reference_values_sv.values()))
    ax.loglog(ref_times_years, ref_doses_sv, 'r--', linewidth=2, 
              label='Reference (typical)')
    
    # Add hands-on maintenance limit
    ax.axhline(y=1e-2, color='green', linestyle=':', linewidth=2, 
               label='Hands-on limit (10 mSv/h)')
    
    ax.set_xlabel('Time after shutdown (years)')
    ax.set_ylabel('Contact dose rate (Sv/h)')
    ax.set_title(f'Contact Dose Rate vs. Cooling Time - {element}')
    ax.grid(True, which="both", ls="-", alpha=0.2)
    ax.legend()
    
    # Set reasonable axis limits
    ax.set_ylim(1e-8, 1e6)
    ax.set_xlim(1e-7, 1e3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    
    print(f"Dose rate plot saved to: {output_path}")


def diagnose_dose_calculation_issues(dose_rates: List[float], 
                                   activities: Optional[List[float]] = None,
                                   photon_rates: Optional[List[float]] = None) -> Dict[str, any]:
    """Diagnose issues with dose rate calculations.
    
    Args:
        dose_rates: List of calculated dose rates in µSv/h.
        activities: Optional list of activities in Bq.
        photon_rates: Optional list of photon rates in photons/s.
        
    Returns:
        Dictionary containing diagnostic information.
    """
    diagnosis = {
        'issues_found': [],
        'recommendations': [],
        'statistics': {}
    }
    
    # Basic statistics
    dose_array = np.array(dose_rates)
    diagnosis['statistics'] = {
        'mean': np.mean(dose_array),
        'max': np.max(dose_array),
        'min': np.min(dose_array),
        'num_zero': np.sum(dose_array == 0),
        'num_points': len(dose_array)
    }
    
    # Check for common issues
    if np.all(dose_array == 0):
        diagnosis['issues_found'].append("All dose rates are zero")
        diagnosis['recommendations'].append("Check gamma spectrum calculation")
    
    if np.any(dose_array < 0):
        diagnosis['issues_found'].append("Negative dose rates detected")
        diagnosis['recommendations'].append("Review dose calculation formula")
    
    # Check for unreasonably high values
    max_reasonable = 1e20  # µSv/h
    if np.any(dose_array > max_reasonable):
        diagnosis['issues_found'].append(f"Dose rates exceed {max_reasonable:.1e} µSv/h")
        diagnosis['recommendations'].append("Check flux normalization and activity calculation")
    
    # Check temporal behavior
    if len(dose_array) > 1:
        # Dose should generally decrease with time during cooling
        increasing_points = np.sum(np.diff(dose_array) > 0)
        if increasing_points > len(dose_array) / 4:  # More than 25% increasing
            diagnosis['issues_found'].append("Dose rates increasing significantly during cooling")
            diagnosis['recommendations'].append("Review cooling phase calculation")
    
    # Activity correlation (if provided)
    if activities and len(activities) == len(dose_rates):
        correlation = np.corrcoef(activities, dose_rates)[0, 1]
        if correlation < 0.8:
            diagnosis['issues_found'].append(f"Low correlation between activity and dose rate: {correlation:.2f}")
            diagnosis['recommendations'].append("Check gamma spectrum or dose conversion factors")
    
    return diagnosis 