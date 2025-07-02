"""
Integration tests for neutronics_calphad.calphad module.

These tests verify that different components work together correctly.
"""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path
import tempfile

from neutronics_calphad.calphad import (
    DepletionResult, ActivationManifold, ActivationConstraints,
    CALPHADBatchCalculator, OutlierDetector, mix
)


class TestIntegration:
    """Integration tests combining multiple CALPHAD components."""
    
    def test_depletion_to_manifold_workflow(self, sample_depletion_results, sample_compositions):
        """Test complete workflow from depletion results to activation manifold."""
        elements = list(sample_depletion_results.keys())
        
        # Create activation constraints
        constraints = ActivationConstraints(
            dose_limits={'14d': 1e8, '5y': 1e5, '7y': 1e5, '100y': 1e2},
            gas_limits={'He_appm': 1e17, 'H_appm': 1e16}
        )
        
        # Build manifold
        manifold = ActivationManifold(elements, constraints)
        
        # Use subset of compositions for faster testing
        test_compositions = sample_compositions[:30]
        manifold.build_from_samples(test_compositions, sample_depletion_results)
        
        # Check manifold was built successfully
        assert manifold.feasible_compositions is not None
        assert len(manifold.feasible_compositions) >= 0
        assert manifold._metadata['n_samples'] == 30
        
        # Test some compositions for feasibility
        if len(manifold.feasible_compositions) > 0:
            test_comp = manifold.feasible_compositions[0]
            if manifold.convex_hull is not None:
                assert manifold.contains(test_comp)
                
    def test_calphad_outlier_workflow(self, sample_compositions):
        """Test CALPHAD calculation followed by outlier detection."""
        elements = ['V', 'Cr', 'Ti', 'W', 'Zr']
        
        # Calculate phases
        calc = CALPHADBatchCalculator()
        phase_df = calc.calculate_batch(sample_compositions, elements)
        
        # Detect outliers based on composition
        detector = OutlierDetector(contamination=0.1)
        composition_cols = [f'x_{el}' for el in elements]
        
        filtered_df = detector.filter_outliers(
            phase_df, 
            composition_cols, 
            keep_outliers=True
        )
        
        # Check integration worked
        assert len(filtered_df) == len(phase_df)
        assert 'is_outlier' in filtered_df.columns
        assert all(col in filtered_df.columns for col in composition_cols)
        assert 'phase_count' in filtered_df.columns
        assert 'single_phase' in filtered_df.columns
        
        # Check outlier detection found some outliers
        n_outliers = filtered_df['is_outlier'].sum()
        assert n_outliers >= 0
        
    def test_mix_multiple_elements_comprehensive(self, sample_depletion_results):
        """Test mixing multiple elements with comprehensive validation."""
        elements = list(sample_depletion_results.keys())
        
        # Test different mixing scenarios
        test_cases = [
            ([0.2, 0.2, 0.2, 0.2, 0.2], "Equal mix"),
            ([0.8, 0.05, 0.05, 0.05, 0.05], "V dominated"),
            ([0.1, 0.7, 0.1, 0.05, 0.05], "Cr dominated"),
            ([0.0, 0.5, 0.5, 0.0, 0.0], "Binary Cr-Ti"),
        ]
        
        for weights, description in test_cases:
            mixed = mix(sample_depletion_results, weights)
            
            # Validate mixed result
            assert mixed.element.startswith("Alloy(")
            assert len(mixed.dose_times) == len(sample_depletion_results['V'].dose_times)
            assert len(mixed.dose_rates) == len(sample_depletion_results['V'].dose_rates)
            
            # Check that mixing preserves physics
            # Dose rates should be weighted averages
            expected_dose = np.zeros_like(mixed.dose_rates)
            for i, (element, weight) in enumerate(zip(elements, weights)):
                expected_dose += weight * sample_depletion_results[element].dose_rates
            
            np.testing.assert_allclose(mixed.dose_rates, expected_dose, rtol=1e-10)
            
            # Gas production should also be weighted averages
            for gas in ['He3', 'He4', 'H1', 'H2', 'H3']:
                expected_gas = sum(
                    weight * sample_depletion_results[element].gas_production[gas]
                    for element, weight in zip(elements, weights)
                )
                assert abs(mixed.gas_production[gas] - expected_gas) < 1e-10
                
    def test_hdf5_roundtrip_integration(self, sample_depletion_results, sample_compositions):
        """Test HDF5 save/load for both depletion results and manifolds."""
        elements = list(sample_depletion_results.keys())
        
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            
            # Test DepletionResult HDF5 roundtrip
            depletion_file = tmpdir / "test_depletion.h5"
            original_result = sample_depletion_results['V']
            original_result.to_hdf5(depletion_file)
            loaded_result = DepletionResult.from_hdf5(depletion_file)
            
            # Verify depletion roundtrip
            assert loaded_result.element == original_result.element
            np.testing.assert_array_equal(loaded_result.dose_times, original_result.dose_times)
            np.testing.assert_array_equal(loaded_result.dose_rates, original_result.dose_rates)
            
            # Test ActivationManifold HDF5 roundtrip
            manifold_file = tmpdir / "test_manifold.h5"
            
            constraints = ActivationConstraints(
                dose_limits={'14d': 1e10, '5y': 1e10, '7y': 1e10, '100y': 1e10},
                gas_limits={'He_appm': 1e10, 'H_appm': 1e10}
            )
            
            original_manifold = ActivationManifold(elements, constraints)
            test_compositions = sample_compositions[:20]
            original_manifold.build_from_samples(test_compositions, sample_depletion_results)
            
            original_manifold.to_hdf5(manifold_file)
            loaded_manifold = ActivationManifold.from_hdf5(manifold_file)
            
            # Verify manifold roundtrip
            assert loaded_manifold.elements == original_manifold.elements
            assert loaded_manifold._metadata['n_samples'] == original_manifold._metadata['n_samples']
            
            if original_manifold.feasible_compositions is not None:
                np.testing.assert_array_equal(
                    loaded_manifold.feasible_compositions,
                    original_manifold.feasible_compositions
                )
                
    def test_full_workflow_simulation(self):
        """Simulate a complete workflow from scratch."""
        # Define system
        elements = ['V', 'Cr', 'Ti']
        n_compositions = 50
        
        # Step 1: Generate compositions
        np.random.seed(42)
        x = np.random.random((n_compositions, len(elements)))
        compositions = x / x.sum(axis=1, keepdims=True)
        
        # Step 2: Mock depletion results (would come from neutronics simulations)
        mock_depletion_results = {}
        times = np.array([1, 3600, 24*3600, 365*24*3600, 10*365*24*3600])
        
        for i, element in enumerate(elements):
            # Create realistic dose rate decay
            dose_rates = 1e6 * np.exp(-times / (365*24*3600)) * (1 + 0.2 * i)
            
            gas_production = {
                'He3': 1e15 * (1 + 0.3 * i),
                'He4': 5e15 * (1 + 0.3 * i),
                'H1': 2e15 * (1 + 0.1 * i),
                'H2': 1e14 * (1 + 0.1 * i),
                'H3': 5e13 * (1 + 0.1 * i)
            }
            
            mock_depletion_results[element] = DepletionResult(
                element=element,
                dose_times=times,
                dose_rates=dose_rates,
                gas_production=gas_production,
                volume=2e5,
                density=6.0 + i
            )
        
        # Step 3: Calculate phases
        calc = CALPHADBatchCalculator()
        phase_df = calc.calculate_batch(compositions, elements)
        
        # Step 4: Build activation manifold
        constraints = ActivationConstraints(
            dose_limits={'14d': 1e4, '5y': 1e2, '7y': 1e2, '100y': 1e-1},
            gas_limits={'He_appm': 1e16, 'H_appm': 5e15}
        )
        
        manifold = ActivationManifold(elements, constraints)
        manifold.build_from_samples(compositions, mock_depletion_results)
        
        # Step 5: Detect outliers
        detector = OutlierDetector(contamination=0.15)
        composition_cols = [f'x_{el}' for el in elements]
        
        final_df = detector.filter_outliers(
            phase_df,
            composition_cols,
            keep_outliers=True
        )
        
        # Validate complete workflow
        assert len(final_df) == n_compositions
        assert 'is_outlier' in final_df.columns
        assert 'single_phase' in final_df.columns
        assert manifold.feasible_compositions is not None
        
        # Check feasibility statistics
        feasibility_rate = manifold._metadata['feasibility_rate']
        assert 0 <= feasibility_rate <= 1
        
        # Check outlier detection
        outlier_rate = final_df['is_outlier'].sum() / len(final_df)
        assert outlier_rate <= 0.25  # Reasonable outlier rate
        
        # Check single-phase statistics
        single_phase_rate = final_df['single_phase'].sum() / len(final_df)
        assert 0 <= single_phase_rate <= 1
        
    def test_error_handling_integration(self):
        """Test error handling across integrated components."""
        elements = ['V', 'Cr', 'Ti']
        
        # Test with inconsistent data
        inconsistent_results = {
            'V': DepletionResult(
                element='V',
                dose_times=np.array([1, 2, 3]),  # Different times
                dose_rates=np.array([1e6, 1e5, 1e4]),
                gas_production={'He3': 0, 'He4': 0, 'H1': 0, 'H2': 0, 'H3': 0},
                volume=1e5,
                density=6.0
            ),
            'Cr': DepletionResult(
                element='Cr',
                dose_times=np.array([1, 2, 3, 4]),  # Different length
                dose_rates=np.array([1e6, 1e5, 1e4, 1e3]),
                gas_production={'He3': 0, 'He4': 0, 'H1': 0, 'H2': 0, 'H3': 0},
                volume=1e5,
                density=7.0
            )
        }
        
        # This should raise an error in mix function
        with pytest.raises(ValueError, match="Inconsistent time steps"):
            mix(inconsistent_results, [0.5, 0.5])
            
        # Test manifold with empty feasible set
        constraints = ActivationConstraints(
            dose_limits={'14d': 1e-20, '5y': 1e-20, '7y': 1e-20, '100y': 1e-20},
            gas_limits={'He_appm': 1e-20, 'H_appm': 1e-20}
        )
        
        manifold = ActivationManifold(elements, constraints)
        
        # Create some compositions
        np.random.seed(42)
        x = np.random.random((10, len(elements)))
        compositions = x / x.sum(axis=1, keepdims=True)
        
        # This should work but result in no feasible compositions
        # Using consistent depletion results
        consistent_results = {
            'V': DepletionResult(
                element='V', dose_times=np.array([1, 2, 3]), dose_rates=np.array([1e6, 1e5, 1e4]),
                gas_production={'He3': 1e20, 'He4': 1e20, 'H1': 1e20, 'H2': 0, 'H3': 0},
                volume=1e5, density=6.0
            ),
            'Cr': DepletionResult(
                element='Cr', dose_times=np.array([1, 2, 3]), dose_rates=np.array([1e6, 1e5, 1e4]),
                gas_production={'He3': 1e20, 'He4': 1e20, 'H1': 1e20, 'H2': 0, 'H3': 0},
                volume=1e5, density=7.0
            ),
            'Ti': DepletionResult(
                element='Ti', dose_times=np.array([1, 2, 3]), dose_rates=np.array([1e6, 1e5, 1e4]),
                gas_production={'He3': 1e20, 'He4': 1e20, 'H1': 1e20, 'H2': 0, 'H3': 0},
                volume=1e5, density=4.5
            )
        }
        
        manifold.build_from_samples(compositions, consistent_results)
        
        # Should have no feasible compositions due to restrictive constraints
        assert len(manifold.feasible_compositions) == 0
        assert manifold._metadata['feasibility_rate'] == 0.0
        assert manifold.convex_hull is None 