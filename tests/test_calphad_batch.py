"""
Test cases for CALPHADBatchCalculator class.
"""

import json
import numpy as np
import pandas as pd
import pytest

from neutronics_calphad.calphad import CALPHADBatchCalculator


class TestCALPHADBatchCalculator:
    """Test cases for CALPHADBatchCalculator class."""
    
    def test_calculator_initialization_default(self):
        """Test default initialization."""
        calc = CALPHADBatchCalculator()
        
        assert calc.database == "TCNI12"
        assert calc.temperature == 800.0
        assert 'C' in calc.fixed_impurities
        assert 'N' in calc.fixed_impurities
        assert 'O' in calc.fixed_impurities
        
        # Check default impurity levels (250 appm each)
        assert calc.fixed_impurities['C'] == 250e-6
        assert calc.fixed_impurities['N'] == 250e-6
        assert calc.fixed_impurities['O'] == 250e-6
        
    def test_calculator_initialization_custom(self):
        """Test custom initialization."""
        custom_impurities = {'C': 100e-6, 'N': 150e-6}
        calc = CALPHADBatchCalculator(
            database="CUSTOM_DB",
            temperature=1000.0,
            fixed_impurities=custom_impurities
        )
        
        assert calc.database == "CUSTOM_DB"
        assert calc.temperature == 1000.0
        assert calc.fixed_impurities == custom_impurities
        
    def test_stub_calculation_single_composition(self):
        """Test stub calculation with single composition."""
        calc = CALPHADBatchCalculator()
        
        # Create a single composition
        elements = ['V', 'Cr', 'Ti', 'W', 'Zr']
        composition = np.array([[0.7, 0.1, 0.1, 0.05, 0.05]])
        
        result_df = calc.calculate_batch(composition, elements)
        
        # Check DataFrame structure
        assert len(result_df) == 1
        expected_columns = [f'x_{el}' for el in elements] + [
            'phase_count', 'dominant_phase', 'single_phase', 'phases'
        ]
        assert all(col in result_df.columns for col in expected_columns)
        
        # Check composition columns
        for i, element in enumerate(elements):
            assert result_df[f'x_{element}'].iloc[0] == composition[0, i]
            
        # Check phase data types
        assert isinstance(result_df['phase_count'].iloc[0], (int, np.integer))
        assert isinstance(result_df['dominant_phase'].iloc[0], str)
        assert isinstance(result_df['single_phase'].iloc[0], (bool, np.bool_))
        assert isinstance(result_df['phases'].iloc[0], str)
        
        # Check that phases string is valid JSON
        phases_dict = json.loads(result_df['phases'].iloc[0])
        assert isinstance(phases_dict, dict)
        
    def test_stub_calculation_multiple_compositions(self, sample_compositions):
        """Test stub calculation with multiple compositions."""
        calc = CALPHADBatchCalculator()
        elements = ['V', 'Cr', 'Ti', 'W', 'Zr']
        
        # Use first 10 sample compositions
        compositions = sample_compositions[:10]
        
        result_df = calc.calculate_batch(compositions, elements)
        
        # Check DataFrame size
        assert len(result_df) == 10
        
        # Check all rows have valid data
        for _, row in result_df.iterrows():
            # Check composition sums to approximately 1
            composition_sum = sum(row[f'x_{el}'] for el in elements)
            assert abs(composition_sum - 1.0) < 1e-10
            
            # Check phase count is positive
            assert row['phase_count'] > 0
            
            # Check dominant phase is not empty
            assert len(row['dominant_phase']) > 0
            
            # Check phases JSON is valid
            phases_dict = json.loads(row['phases'])
            assert isinstance(phases_dict, dict)
            assert len(phases_dict) > 0
            
    def test_stub_single_phase_heuristic(self):
        """Test that stub implementation uses correct single-phase heuristic."""
        calc = CALPHADBatchCalculator()
        elements = ['V', 'Cr', 'Ti']
        
        # Create compositions where one element dominates (>0.8)
        single_phase_comps = np.array([
            [0.85, 0.075, 0.075],  # V dominant
            [0.92, 0.04, 0.04]
        ])
        
        result_df = calc.calculate_batch(single_phase_comps, elements)
        
        # All should be predicted as single phase
        assert all(result_df['single_phase'])
        assert all(result_df['phase_count'] == 1)
        
        # Create compositions with no dominant element
        multi_phase_comps = np.array([
            [0.4, 0.3, 0.3],   # No clear dominant
            [0.5, 0.25, 0.25], # V slightly dominant but < 0.8
        ])
        
        result_df = calc.calculate_batch(multi_phase_comps, elements)
        
        # Should be predicted as multi-phase
        assert all(~result_df['single_phase'])
        assert all(result_df['phase_count'] > 1)
        
    def test_composition_validation(self, sample_compositions):
        """Test that compositions are properly handled."""
        calc = CALPHADBatchCalculator()
        elements = ['V', 'Cr', 'Ti', 'W', 'Zr']
        
        # Test with sample compositions
        result_df = calc.calculate_batch(sample_compositions, elements)
        
        # Check that all compositions in result sum to 1
        for _, row in result_df.iterrows():
            comp_sum = sum(row[f'x_{el}'] for el in elements)
            assert abs(comp_sum - 1.0) < 1e-10
            
    def test_different_element_lists(self):
        """Test with different element combinations."""
        calc = CALPHADBatchCalculator()
        
        # Test with 3 elements
        elements_3 = ['V', 'Cr', 'Ti']
        comp_3 = np.array([[0.5, 0.3, 0.2]])
        
        result_3 = calc.calculate_batch(comp_3, elements_3)
        assert len(result_3.columns) == len(elements_3) + 4  # compositions + 4 phase columns
        
        # Test with 4 elements
        elements_4 = ['V', 'Cr', 'Ti', 'W']
        comp_4 = np.array([[0.4, 0.3, 0.2, 0.1]])
        
        result_4 = calc.calculate_batch(comp_4, elements_4)
        assert len(result_4.columns) == len(elements_4) + 4
        
    def test_reproducibility(self):
        """Test that stub calculations are reproducible."""
        calc = CALPHADBatchCalculator()
        elements = ['V', 'Cr', 'Ti']
        
        composition = np.array([[0.6, 0.25, 0.15]])
        
        # Run calculation twice
        result1 = calc.calculate_batch(composition, elements)
        result2 = calc.calculate_batch(composition, elements)
        
        # Results should be identical
        pd.testing.assert_frame_equal(result1, result2)
        
    def test_edge_compositions(self):
        """Test with edge case compositions."""
        calc = CALPHADBatchCalculator()
        elements = ['V', 'Cr', 'Ti']
        
        # Pure element compositions
        pure_comps = np.array([
            [1.0, 0.0, 0.0],  # Pure V
            [0.0, 1.0, 0.0],  # Pure Cr
            [0.0, 0.0, 1.0],  # Pure Ti
        ])
        
        result_df = calc.calculate_batch(pure_comps, elements)
        
        # Pure compositions should be single phase
        assert all(result_df['single_phase'])
        assert all(result_df['phase_count'] == 1)
        
        # Nearly equal compositions
        equal_comp = np.array([[0.333, 0.333, 0.334]])
        
        result_equal = calc.calculate_batch(equal_comp, elements)
        
        # Equal composition should be multi-phase (no element > 0.8)
        assert not result_equal['single_phase'].iloc[0]
        assert result_equal['phase_count'].iloc[0] > 1
        
    def test_phases_json_structure(self):
        """Test that phases JSON has correct structure."""
        calc = CALPHADBatchCalculator()
        elements = ['V', 'Cr']
        
        compositions = np.array([
            [0.9, 0.1],   # Single phase
            [0.6, 0.4],   # Multi phase
        ])
        
        result_df = calc.calculate_batch(compositions, elements)
        
        for _, row in result_df.iterrows():
            phases_dict = json.loads(row['phases'])
            
            # Check that it's a valid dictionary
            assert isinstance(phases_dict, dict)
            
            # Check that phase fractions are positive
            for phase, fraction in phases_dict.items():
                assert isinstance(phase, str)
                assert isinstance(fraction, (int, float))
                assert fraction > 0
                
            # Check that fractions approximately sum to 1
            total_fraction = sum(phases_dict.values())
            assert abs(total_fraction - 1.0) < 1e-6
            
            # Check consistency with phase_count
            assert len(phases_dict) == row['phase_count']
            
    def test_empty_compositions(self):
        """Test behavior with empty composition array."""
        calc = CALPHADBatchCalculator()
        elements = ['V', 'Cr', 'Ti']
        
        empty_comps = np.array([]).reshape(0, 3)
        
        result_df = calc.calculate_batch(empty_comps, elements)
        
        # Should return empty DataFrame with correct columns
        expected_columns = [f'x_{el}' for el in elements] + [
            'phase_count', 'dominant_phase', 'single_phase', 'phases'
        ]
        assert len(result_df) == 0
        assert all(col in result_df.columns for col in expected_columns)
        
    def test_large_batch(self):
        """Test with larger batch of compositions."""
        calc = CALPHADBatchCalculator()
        elements = ['V', 'Cr', 'Ti', 'W', 'Zr']
        
        # Generate larger batch
        np.random.seed(42)
        n_samples = 100
        x = np.random.random((n_samples, len(elements)))
        compositions = x / x.sum(axis=1, keepdims=True)
        
        result_df = calc.calculate_batch(compositions, elements)
        
        # Check size
        assert len(result_df) == n_samples
        
        # Check that all required columns are present
        expected_columns = [f'x_{el}' for el in elements] + [
            'phase_count', 'dominant_phase', 'single_phase', 'phases'
        ]
        assert all(col in result_df.columns for col in expected_columns)
        
        # Check data validity
        assert all(result_df['phase_count'] > 0)
        assert all(result_df['dominant_phase'].str.len() > 0)
        
    def test_impurity_handling(self):
        """Test that impurity levels are properly configured."""
        # Test default impurities
        calc_default = CALPHADBatchCalculator()
        assert len(calc_default.fixed_impurities) == 3
        
        # Test custom impurities
        custom_impurities = {
            'C': 500e-6,   # 500 appm
            'N': 100e-6,   # 100 appm
            'O': 200e-6,   # 200 appm
            'S': 50e-6     # 50 appm sulfur
        }
        
        calc_custom = CALPHADBatchCalculator(fixed_impurities=custom_impurities)
        assert calc_custom.fixed_impurities == custom_impurities
        assert len(calc_custom.fixed_impurities) == 4 