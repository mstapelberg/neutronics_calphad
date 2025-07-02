"""
Test cases for DepletionResult class and mix function.
"""

import numpy as np
import pytest
import h5py

from neutronics_calphad.calphad import DepletionResult, mix


class TestDepletionResult:
    """Test cases for DepletionResult class."""
    
    def test_depletion_result_creation(self, sample_depletion_result):
        """Test basic DepletionResult creation."""
        result = sample_depletion_result
        
        assert result.element == 'V'
        assert len(result.dose_times) == 11
        assert len(result.dose_rates) == 11
        assert len(result.gas_production) == 5
        assert result.volume > 0
        assert result.density > 0
        
    def test_depletion_result_gas_production(self, sample_depletion_result):
        """Test gas production data structure."""
        result = sample_depletion_result
        expected_gases = ['He3', 'He4', 'H1', 'H2', 'H3']
        
        assert all(gas in result.gas_production for gas in expected_gases)
        assert all(result.gas_production[gas] >= 0 for gas in expected_gases)
        
    def test_from_hdf5(self, sample_hdf5_file):
        """Test loading DepletionResult from HDF5 file."""
        result = DepletionResult.from_hdf5(sample_hdf5_file)
        
        assert result.element == 'V'
        assert len(result.dose_times) == 11
        assert len(result.dose_rates) == 11
        assert len(result.gas_production) == 5
        
        # Check that all expected gas species are present
        expected_gases = ['He3', 'He4', 'H1', 'H2', 'H3']
        assert all(gas in result.gas_production for gas in expected_gases)
        
    def test_to_hdf5(self, sample_depletion_result, temp_hdf5_file):
        """Test saving DepletionResult to HDF5 file."""
        result = sample_depletion_result
        result.to_hdf5(temp_hdf5_file)
        
        # Verify the file was created and contains expected data
        with h5py.File(temp_hdf5_file, 'r') as f:
            assert 'dose_times' in f
            assert 'dose' in f
            assert f.attrs['element'] == 'V'
            assert f.attrs['volume'] == result.volume
            assert f.attrs['density'] == result.density
            
            # Check gas production data
            for gas in ['He3', 'He4', 'H1', 'H2', 'H3']:
                assert f'gas/{gas}' in f
                
    def test_roundtrip_hdf5(self, sample_depletion_result, temp_hdf5_file):
        """Test save and load roundtrip to HDF5."""
        original = sample_depletion_result
        original.to_hdf5(temp_hdf5_file)
        loaded = DepletionResult.from_hdf5(temp_hdf5_file)
        
        assert loaded.element == original.element
        np.testing.assert_array_equal(loaded.dose_times, original.dose_times)
        np.testing.assert_array_equal(loaded.dose_rates, original.dose_rates)
        assert loaded.volume == original.volume
        assert loaded.density == original.density
        
        for gas in original.gas_production:
            assert loaded.gas_production[gas] == original.gas_production[gas]


class TestMixFunction:
    """Test cases for the mix function."""
    
    def test_mix_equal_weights(self, sample_depletion_results):
        """Test mixing with equal weights."""
        elements = list(sample_depletion_results.keys())
        weights = [0.2] * 5  # Equal weights for 5 elements
        
        mixed = mix(sample_depletion_results, weights)
        
        assert mixed.element.startswith("Alloy(")
        assert len(mixed.dose_times) == 11
        assert len(mixed.dose_rates) == 11
        assert len(mixed.gas_production) == 5
        
        # Mixed dose rates should be average of input rates
        expected_dose = np.mean([r.dose_rates for r in sample_depletion_results.values()], axis=0)
        np.testing.assert_allclose(mixed.dose_rates, expected_dose, rtol=1e-10)
        
    def test_mix_single_element(self, sample_depletion_results):
        """Test mixing with single element (weight = 1)."""
        element = 'V'
        results = {element: sample_depletion_results[element]}
        weights = [1.0]
        
        mixed = mix(results, weights)
        original = sample_depletion_results[element]
        
        np.testing.assert_array_equal(mixed.dose_times, original.dose_times)
        np.testing.assert_array_equal(mixed.dose_rates, original.dose_rates)
        assert mixed.volume == original.volume
        assert mixed.density == original.density
        
        for gas in original.gas_production:
            assert mixed.gas_production[gas] == original.gas_production[gas]
            
    def test_mix_binary_alloy(self, sample_depletion_results):
        """Test mixing binary alloy."""
        elements = ['V', 'Cr']
        results = {el: sample_depletion_results[el] for el in elements}
        weights = [0.7, 0.3]
        
        mixed = mix(results, weights)
        
        # Check that mixing is linear
        expected_dose = (0.7 * sample_depletion_results['V'].dose_rates + 
                        0.3 * sample_depletion_results['Cr'].dose_rates)
        np.testing.assert_allclose(mixed.dose_rates, expected_dose, rtol=1e-10)
        
    def test_mix_weights_validation(self, sample_depletion_results):
        """Test weight validation in mix function."""
        elements = list(sample_depletion_results.keys())
        
        # Test weights don't sum to 1
        with pytest.raises(ValueError, match="Weights must sum to 1.0"):
            mix(sample_depletion_results, [0.1, 0.2, 0.3, 0.2, 0.1])
            
        # Test wrong number of weights
        with pytest.raises(ValueError, match="Number of weights"):
            mix(sample_depletion_results, [0.5, 0.5])
            
    def test_mix_time_consistency(self, sample_depletion_results):
        """Test that mix function checks time consistency."""
        # Create inconsistent time data
        inconsistent_result = DepletionResult(
            element='Inconsistent',
            dose_times=np.array([1, 2, 3]),  # Different times
            dose_rates=np.array([1e6, 1e5, 1e4]),
            gas_production={'He3': 0, 'He4': 0, 'H1': 0, 'H2': 0, 'H3': 0},
            volume=1e5,
            density=5.0
        )
        
        results = {'V': sample_depletion_results['V'], 'Bad': inconsistent_result}
        weights = [0.5, 0.5]
        
        with pytest.raises(ValueError, match="Inconsistent time steps"):
            mix(results, weights)
            
    def test_mix_gas_production(self, sample_depletion_results):
        """Test gas production mixing."""
        elements = ['V', 'Cr']
        results = {el: sample_depletion_results[el] for el in elements}
        weights = [0.6, 0.4]
        
        mixed = mix(results, weights)
        
        # Check gas mixing for each species
        for gas in ['He3', 'He4', 'H1', 'H2', 'H3']:
            expected_gas = (0.6 * results['V'].gas_production[gas] + 
                           0.4 * results['Cr'].gas_production[gas])
            assert abs(mixed.gas_production[gas] - expected_gas) < 1e-10
            
    def test_mix_properties_weighted_average(self, sample_depletion_results):
        """Test that volume and density are weighted averages."""
        elements = ['V', 'Cr', 'Ti']
        results = {el: sample_depletion_results[el] for el in elements}
        weights = [0.5, 0.3, 0.2]
        
        mixed = mix(results, weights)
        
        expected_volume = sum(w * results[el].volume for w, el in zip(weights, elements))
        expected_density = sum(w * results[el].density for w, el in zip(weights, elements))
        
        assert abs(mixed.volume - expected_volume) < 1e-10
        assert abs(mixed.density - expected_density) < 1e-10
        
    def test_mix_zero_weights(self, sample_depletion_results):
        """Test mixing with some zero weights."""
        elements = ['V', 'Cr', 'Ti', 'W', 'Zr']
        results = sample_depletion_results
        weights = [1.0, 0.0, 0.0, 0.0, 0.0]  # Only V
        
        mixed = mix(results, weights)
        original = sample_depletion_results['V']
        
        # Should be identical to pure V
        np.testing.assert_array_equal(mixed.dose_rates, original.dose_rates)
        for gas in original.gas_production:
            assert mixed.gas_production[gas] == original.gas_production[gas] 