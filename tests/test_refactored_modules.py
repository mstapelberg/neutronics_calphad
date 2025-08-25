"""
Test the refactored modules to ensure functionality is preserved.

This module tests the newly created flux, depletion, dose, and utils modules
to ensure that the refactoring didn't break existing functionality.
"""
import pytest
import numpy as np
from pathlib import Path
import tempfile
from typing import Dict, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from _pytest.capture import CaptureFixture
    from _pytest.fixtures import FixtureRequest
    from _pytest.logging import LogCaptureFixture
    from _pytest.monkeypatch import MonkeyPatch
    from pytest_mock.plugin import MockerFixture

from neutronics_calphad.flux import validate_flux_normalization, calculate_actual_flux
from neutronics_calphad.depletion import validate_depletion_results
from neutronics_calphad.dose import get_reference_dose_rates, validate_dose_rates, diagnose_dose_calculation_issues
from neutronics_calphad.utils import (
    get_material_by_name, format_time_label, validate_environment_variables, 
    validate_config, check_openmc_version, TIMES, ELMS
)


class TestFluxModule:
    """Test the flux calculation module."""
    
    def test_validate_flux_normalization_good_flux(self) -> None:
        """Test flux validation with reasonable flux values."""
        # Good flux: per-source-neutron values much less than 1
        good_flux = np.array([0.001, 0.002, 0.0015, 0.0005])
        
        result = validate_flux_normalization(good_flux)
        assert result is True
    
    def test_validate_flux_normalization_bad_flux(self) -> None:
        """Test flux validation with unreasonable flux values."""
        # Bad flux: per-source-neutron values > 1 (the problem we're seeing)
        bad_flux = np.array([5.0, 3.2, 7.1, 0.8])  # Total = 16.1 >> 1
        
        result = validate_flux_normalization(bad_flux)
        assert result is False
    
    def test_calculate_actual_flux(self) -> None:
        """Test actual flux calculation from per-source-neutron flux."""
        flux_per_neutron = np.array([0.001, 0.002, 0.001])  # Reasonable values
        power = 500e6  # 500 MW
        
        actual_flux = calculate_actual_flux(flux_per_neutron, power)
        
        # Check that scaling is correct
        total_flux_per_neutron = np.sum(flux_per_neutron)
        total_actual_flux = np.sum(actual_flux)
        
        # Should scale by the neutron source rate
        expected_neutron_rate = power / (17.6e6 * 1.60218e-19)
        expected_total_flux = total_flux_per_neutron * expected_neutron_rate
        
        assert abs(total_actual_flux - expected_total_flux) < expected_total_flux * 0.01  # 1% tolerance


class TestDepletionModule:
    """Test the depletion calculation module."""
    
    def test_validate_depletion_results_good(self, mocker: 'MockerFixture') -> None:
        """Test depletion results validation with good results."""
        # Mock a good depletion results object
        mock_results = mocker.MagicMock()
        mock_results.__len__.return_value = 14  # Expected number of timesteps
        
        # Mock material access
        mock_results.__getitem__.return_value.index_mat = {1: 'test_material'}
        mock_material = mocker.MagicMock()
        mock_material.get_activity.return_value = 1e15  # Positive activity
        mock_results.__getitem__.return_value.get_material.return_value = mock_material
        
        result = validate_depletion_results(mock_results, 14)
        assert result is True
    
    def test_validate_depletion_results_wrong_timesteps(self, mocker: 'MockerFixture') -> None:
        """Test depletion results validation with wrong timestep count."""
        mock_results = mocker.MagicMock()
        mock_results.__len__.return_value = 10  # Wrong number
        
        result = validate_depletion_results(mock_results, 14)
        assert result is False


class TestDoseModule:
    """Test the dose calculation module."""
    
    def test_get_reference_dose_rates(self) -> None:
        """Test reference dose rate lookup."""
        # Test known reference values
        low, high = get_reference_dose_rates('V', 365*24*3600)  # 1 year
        
        assert low > 0
        assert high > low
        assert low < 1e12  # Should be reasonable values in µSv/h
        assert high < 1e15
    
    def test_validate_dose_rates(self) -> None:
        """Test dose rate validation against references."""
        # Test with reasonable dose rates
        dose_rates = [1e12, 5e11, 1e11, 1e10]  # µSv/h, decreasing
        times = [3600, 24*3600, 365*24*3600, 5*365*24*3600]  # 1h, 1d, 1y, 5y
        
        validation = validate_dose_rates(dose_rates, times, 'V')
        
        assert validation['total_tests'] == 4
        assert validation['total_tests'] == validation['passed'] + validation['failed'] + validation['warnings']
    
    def test_diagnose_dose_calculation_issues_high_doses(self) -> None:
        """Test dose calculation diagnostics with unreasonably high doses."""
        # Extremely high dose rates (like we're seeing in the bug)
        high_doses = [1e18, 1e17, 1e16, 1e15]  # µSv/h
        
        diagnosis = diagnose_dose_calculation_issues(high_doses)
        
        assert len(diagnosis['issues_found']) > 0
        assert 'exceed' in ' '.join(diagnosis['issues_found']).lower()
        assert len(diagnosis['recommendations']) > 0
    
    def test_diagnose_dose_calculation_issues_zero_doses(self) -> None:
        """Test dose calculation diagnostics with all zero doses."""
        zero_doses = [0.0, 0.0, 0.0, 0.0]
        
        diagnosis = diagnose_dose_calculation_issues(zero_doses)
        
        assert 'zero' in ' '.join(diagnosis['issues_found']).lower()


class TestUtilsModule:
    """Test the utilities module."""
    
    def test_get_material_by_name_success(self) -> None:
        """Test successful material lookup by name."""
        import openmc
        
        # Create mock materials
        mat1 = openmc.Material(name='mat1')
        mat2 = openmc.Material(name='vcrti')
        mat3 = openmc.Material(name='mat3')
        materials = [mat1, mat2, mat3]
        
        found_mat = get_material_by_name(materials, 'vcrti')
        assert found_mat.name == 'vcrti'
        assert found_mat is mat2
    
    def test_get_material_by_name_not_found(self) -> None:
        """Test material lookup failure."""
        import openmc
        
        mat1 = openmc.Material(name='mat1')
        materials = [mat1]
        
        with pytest.raises(ValueError, match="Material with name 'missing' not found"):
            get_material_by_name(materials, 'missing')
    
    def test_format_time_label(self) -> None:
        """Test time label formatting."""
        # Test known values
        assert format_time_label(3600) == "1 hour"
        assert format_time_label(365*24*3600) == "1 year"
        assert format_time_label(5*365*24*3600) == "5 years"
        
        # Test unknown value
        assert format_time_label(12345) == "12345 s"
    
    def test_validate_environment_variables(self, monkeypatch: 'MonkeyPatch') -> None:
        """Test environment variable validation."""
        # Test with missing variables
        monkeypatch.delenv('OPENMC_CHAIN_FILE', raising=False)
        monkeypatch.delenv('OPENMC_CROSS_SECTIONS', raising=False)
        
        validation = validate_environment_variables()
        assert validation['valid'] is False
        assert len(validation['issues']) >= 2
        
        # Test with set variables (but files may not exist)
        monkeypatch.setenv('OPENMC_CHAIN_FILE', '/fake/path/chain.xml')
        monkeypatch.setenv('OPENMC_CROSS_SECTIONS', '/fake/path/xs.xml')
        
        validation = validate_environment_variables()
        # Should be invalid because files don't exist
        assert validation['valid'] is False
        assert 'chain.xml' in str(validation['issues'])
    
    def test_validate_config_good(self) -> None:
        """Test configuration validation with good config."""
        good_config = {
            'geometry': {
                'type': 'sphere',
                'major_radius': 330.0,
                'minor_radius': 113.0
            },
            'materials': {
                'vcrti': {
                    'elements': {'V': 1.0},
                    'density': 6.11
                }
            }
        }
        
        validation = validate_config(good_config)
        assert validation['valid'] is True
        assert len(validation['errors']) == 0
    
    def test_validate_config_missing_keys(self) -> None:
        """Test configuration validation with missing required keys."""
        bad_config = {
            'geometry': {'type': 'sphere'}
            # Missing 'materials'
        }
        
        validation = validate_config(bad_config)
        assert validation['valid'] is False
        assert any('materials' in error for error in validation['errors'])
    
    def test_check_openmc_version(self) -> None:
        """Test OpenMC version checking."""
        version_info = check_openmc_version()
        
        assert 'version' in version_info
        assert 'compatible' in version_info
        assert isinstance(version_info['warnings'], list)
    
    def test_constants_exist(self) -> None:
        """Test that important constants are defined."""
        assert isinstance(TIMES, list)
        assert len(TIMES) > 0
        assert all(isinstance(t, (int, float)) for t in TIMES)
        
        assert isinstance(ELMS, list)
        assert len(ELMS) > 0
        assert all(isinstance(e, str) for e in ELMS)


class TestModuleIntegration:
    """Test integration between the refactored modules."""
    
    def test_flux_to_dose_workflow(self) -> None:
        """Test that flux calculations can feed into dose calculations."""
        # Test the conceptual workflow without actually running OpenMC
        
        # Step 1: Mock flux calculation result
        mock_flux = np.array([0.001, 0.002, 0.001])  # Reasonable per-source-neutron flux
        power = 500e6
        
        # Step 2: Validate flux
        flux_valid = validate_flux_normalization(mock_flux)
        assert flux_valid is True
        
        # Step 3: Calculate actual flux
        actual_flux = calculate_actual_flux(mock_flux, power)
        assert np.sum(actual_flux) > 1e13  # Should be reasonable fusion reactor flux
        
        # Step 4: Mock dose calculation (we'd use actual_flux in real calculation)
        mock_dose_rates = [1e12, 5e11, 1e11]  # µSv/h
        mock_times = [3600, 24*3600, 365*24*3600]
        
        # Step 5: Validate dose rates
        dose_validation = validate_dose_rates(mock_dose_rates, mock_times, 'V')
        assert dose_validation['total_tests'] == 3
    
    def test_error_propagation(self) -> None:
        """Test that errors in one module are properly handled by others."""
        # Test with bad flux that should cause issues downstream
        bad_flux = np.array([15.0, 10.0, 5.0])  # Total >> 1, problematic
        
        # Flux validation should catch this
        flux_valid = validate_flux_normalization(bad_flux)
        assert flux_valid is False
        
        # If we proceed anyway, dose rates would be too high
        actual_flux = calculate_actual_flux(bad_flux, 500e6)
        
        # Simulate the high dose rates this would cause
        # (scaling factor would be ~15x too high)
        mock_reasonable_doses = [1e11, 5e10, 1e10]
        mock_high_doses = [d * 15 for d in mock_reasonable_doses]
        
        # Dose diagnostics should catch this
        diagnosis = diagnose_dose_calculation_issues(mock_high_doses)
        assert len(diagnosis['issues_found']) > 0


@pytest.mark.integration
class TestRefactoredLibraryCompatibility:
    """Test that the refactored modules are compatible with the original library."""
    
    def test_import_compatibility(self) -> None:
        """Test that all necessary functions can be imported."""
        # Test imports from all modules
        from neutronics_calphad.flux import get_flux_and_microxs, collapse_cross_sections
        from neutronics_calphad.depletion import run_independent_depletion, extract_gas_production
        from neutronics_calphad.dose import calculate_fispact_dose, get_fispact_xs
        from neutronics_calphad.utils import get_material_by_name, TIMES
        
        # All imports should work without errors
        assert callable(get_flux_and_microxs)
        assert callable(run_independent_depletion)
        assert callable(calculate_fispact_dose)
        assert callable(get_material_by_name)
    
    def test_constants_consistency(self) -> None:
        """Test that constants are consistent between modules."""
        from neutronics_calphad.flux import E_PER_FUSION_eV as flux_constant
        from neutronics_calphad.depletion import E_PER_FUSION_eV as depletion_constant
        from neutronics_calphad.utils import E_PER_FUSION_eV as utils_constant
        
        # All should be the same value
        assert flux_constant == depletion_constant == utils_constant
        assert flux_constant == 17.6e6 