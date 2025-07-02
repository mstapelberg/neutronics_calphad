"""
Test cases for ActivationConstraints and ActivationManifold classes.
"""

import json
import numpy as np
import pytest
import h5py
from scipy.spatial import ConvexHull

from neutronics_calphad.calphad import ActivationConstraints, ActivationManifold, mix


class TestActivationConstraints:
    """Test cases for ActivationConstraints class."""
    
    def test_default_constraints(self):
        """Test default constraint values."""
        constraints = ActivationConstraints()
        
        # Check default dose limits
        assert '14d' in constraints.dose_limits
        assert '5y' in constraints.dose_limits
        assert '7y' in constraints.dose_limits
        assert '100y' in constraints.dose_limits
        
        # Check default gas limits
        assert 'He_appm' in constraints.gas_limits
        assert 'H_appm' in constraints.gas_limits
        
        # Check values are reasonable
        assert constraints.dose_limits['14d'] == 1e2
        assert constraints.dose_limits['100y'] == 1e-4
        assert constraints.gas_limits['He_appm'] == 1000
        assert constraints.gas_limits['H_appm'] == 500
        
    def test_custom_constraints(self):
        """Test custom constraint values."""
        custom_dose = {'14d': 1e3, '5y': 1e-1}
        custom_gas = {'He_appm': 2000, 'H_appm': 1000}
        
        constraints = ActivationConstraints(
            dose_limits=custom_dose,
            gas_limits=custom_gas
        )
        
        assert constraints.dose_limits == custom_dose
        assert constraints.gas_limits == custom_gas


class TestActivationManifold:
    """Test cases for ActivationManifold class."""
    
    def test_manifold_initialization(self, activation_constraints):
        """Test basic manifold initialization."""
        elements = ['V', 'Cr', 'Ti', 'W', 'Zr']
        manifold = ActivationManifold(elements, activation_constraints)
        
        assert manifold.elements == elements
        assert manifold.constraints == activation_constraints
        assert manifold.feasible_compositions is None
        assert manifold.convex_hull is None
        
    def test_manifold_with_default_constraints(self):
        """Test manifold with default constraints."""
        elements = ['V', 'Cr', 'Ti']
        manifold = ActivationManifold(elements)
        
        assert manifold.elements == elements
        assert manifold.constraints is not None
        assert isinstance(manifold.constraints, ActivationConstraints)
        
    def test_get_time_index(self, activation_constraints):
        """Test time index lookup."""
        elements = ['V', 'Cr']
        manifold = ActivationManifold(elements, activation_constraints)
        
        # Create sample times
        times = np.array([
            1, 3600, 24*3600, 14*24*3600,
            365*24*3600, 5*365*24*3600, 100*365*24*3600
        ])
        
        # Test exact matches
        assert manifold._get_time_index(times, '14d') == 3
        assert manifold._get_time_index(times, '5y') == 5
        
        # Test invalid key
        assert manifold._get_time_index(times, 'invalid') is None
        
    def test_build_from_samples_all_feasible(self, sample_depletion_results, activation_constraints):
        """Test building manifold when all samples are feasible."""
        elements = list(sample_depletion_results.keys())
        manifold = ActivationManifold(elements, activation_constraints)
        
        # Create very permissive constraints
        permissive_constraints = ActivationConstraints(
            dose_limits={'14d': 1e10, '5y': 1e10, '7y': 1e10, '100y': 1e10},
            gas_limits={'He_appm': 1e10, 'H_appm': 1e10}
        )
        manifold.constraints = permissive_constraints
        
        # Generate a few sample compositions
        np.random.seed(42)
        n_samples = 20
        x = np.random.random((n_samples, len(elements)))
        samples = x / x.sum(axis=1, keepdims=True)
        
        manifold.build_from_samples(samples, sample_depletion_results)
        
        # All should be feasible with permissive constraints
        assert len(manifold.feasible_compositions) == n_samples
        assert manifold._metadata['feasibility_rate'] == 1.0
        assert manifold.convex_hull is not None
        
    def test_build_from_samples_none_feasible(self, sample_depletion_results, activation_constraints):
        """Test building manifold when no samples are feasible."""
        elements = list(sample_depletion_results.keys())
        manifold = ActivationManifold(elements, activation_constraints)
        
        # Create very restrictive constraints
        restrictive_constraints = ActivationConstraints(
            dose_limits={'14d': 1e-10, '5y': 1e-10, '7y': 1e-10, '100y': 1e-10},
            gas_limits={'He_appm': 1e-10, 'H_appm': 1e-10}
        )
        manifold.constraints = restrictive_constraints
        
        # Generate sample compositions
        np.random.seed(42)
        n_samples = 10
        x = np.random.random((n_samples, len(elements)))
        samples = x / x.sum(axis=1, keepdims=True)
        
        manifold.build_from_samples(samples, sample_depletion_results)
        
        # None should be feasible with restrictive constraints
        assert len(manifold.feasible_compositions) == 0
        assert manifold._metadata['feasibility_rate'] == 0.0
        assert manifold.convex_hull is None
        
    def test_build_from_samples_partial_feasible(self, sample_depletion_results):
        """Test building manifold with partial feasibility."""
        elements = list(sample_depletion_results.keys())
        
        # Create moderate constraints that should allow some compositions
        moderate_constraints = ActivationConstraints(
            dose_limits={'14d': 1e8, '5y': 1e5, '7y': 1e5, '100y': 1e2},
            gas_limits={'He_appm': 1e17, 'H_appm': 1e16}
        )
        manifold = ActivationManifold(elements, moderate_constraints)
        
        # Generate sample compositions
        np.random.seed(42)
        n_samples = 50
        x = np.random.random((n_samples, len(elements)))
        samples = x / x.sum(axis=1, keepdims=True)
        
        manifold.build_from_samples(samples, sample_depletion_results)
        
        # Should have some feasible compositions
        assert 0 < len(manifold.feasible_compositions) < n_samples
        assert 0 < manifold._metadata['feasibility_rate'] < 1.0
        
    def test_contains_method(self, sample_depletion_results, activation_constraints):
        """Test the contains method for checking composition feasibility."""
        elements = list(sample_depletion_results.keys())
        manifold = ActivationManifold(elements, activation_constraints)
        
        # Build manifold with permissive constraints
        permissive_constraints = ActivationConstraints(
            dose_limits={'14d': 1e10, '5y': 1e10, '7y': 1e10, '100y': 1e10},
            gas_limits={'He_appm': 1e10, 'H_appm': 1e10}
        )
        manifold.constraints = permissive_constraints
        
        # Generate samples
        np.random.seed(42)
        n_samples = 20
        x = np.random.random((n_samples, len(elements)))
        samples = x / x.sum(axis=1, keepdims=True)
        
        manifold.build_from_samples(samples, sample_depletion_results)
        
        if manifold.convex_hull is not None:
            # Test points that should be inside
            for i in range(min(5, len(samples))):
                test_point = samples[i]
                # Since all points were feasible, they should be in the convex hull
                assert manifold.contains(test_point)
                
            # Test centroid (should be inside convex hull)
            centroid = np.mean(samples, axis=0)
            assert manifold.contains(centroid)
        else:
            # If no convex hull, contains should return False
            test_point = samples[0]
            assert not manifold.contains(test_point)
            
    def test_metadata_creation(self, sample_depletion_results, activation_constraints):
        """Test metadata creation during manifold building."""
        elements = list(sample_depletion_results.keys())
        manifold = ActivationManifold(elements, activation_constraints)
        
        # Generate samples
        np.random.seed(42)
        n_samples = 30
        x = np.random.random((n_samples, len(elements)))
        samples = x / x.sum(axis=1, keepdims=True)
        
        manifold.build_from_samples(samples, sample_depletion_results)
        
        # Check metadata exists and has correct structure
        assert 'n_samples' in manifold._metadata
        assert 'n_feasible' in manifold._metadata
        assert 'feasibility_rate' in manifold._metadata
        assert 'timestamp' in manifold._metadata
        assert 'elements' in manifold._metadata
        assert 'constraints' in manifold._metadata
        
        # Check values
        assert manifold._metadata['n_samples'] == n_samples
        assert manifold._metadata['elements'] == elements
        
    def test_to_hdf5(self, sample_depletion_results, activation_constraints, temp_hdf5_file):
        """Test saving manifold to HDF5."""
        elements = list(sample_depletion_results.keys())
        manifold = ActivationManifold(elements, activation_constraints)
        
        # Build manifold
        np.random.seed(42)
        n_samples = 20
        x = np.random.random((n_samples, len(elements)))
        samples = x / x.sum(axis=1, keepdims=True)
        
        # Use permissive constraints for testing
        permissive_constraints = ActivationConstraints(
            dose_limits={'14d': 1e10, '5y': 1e10, '7y': 1e10, '100y': 1e10},
            gas_limits={'He_appm': 1e10, 'H_appm': 1e10}
        )
        manifold.constraints = permissive_constraints
        manifold.build_from_samples(samples, sample_depletion_results)
        
        # Save to HDF5
        manifold.to_hdf5(temp_hdf5_file)
        
        # Verify file structure
        with h5py.File(temp_hdf5_file, 'r') as f:
            assert 'feasible_compositions' in f
            assert 'metadata' in f
            assert 'constraints' in f
            
            # Check metadata attributes
            meta = f['metadata']
            assert 'timestamp' in meta.attrs
            assert 'n_samples' in meta.attrs
            assert 'n_feasible' in meta.attrs
            assert 'elements' in meta.attrs
            
            # Check constraints
            constraints = f['constraints']
            assert 'dose_limits' in constraints.attrs
            assert 'gas_limits' in constraints.attrs
            
    def test_from_hdf5(self, sample_depletion_results, activation_constraints, temp_hdf5_file):
        """Test loading manifold from HDF5."""
        elements = list(sample_depletion_results.keys())
        original_manifold = ActivationManifold(elements, activation_constraints)
        
        # Build and save manifold
        np.random.seed(42)
        n_samples = 15
        x = np.random.random((n_samples, len(elements)))
        samples = x / x.sum(axis=1, keepdims=True)
        
        # Use permissive constraints
        permissive_constraints = ActivationConstraints(
            dose_limits={'14d': 1e10, '5y': 1e10, '7y': 1e10, '100y': 1e10},
            gas_limits={'He_appm': 1e10, 'H_appm': 1e10}
        )
        original_manifold.constraints = permissive_constraints
        original_manifold.build_from_samples(samples, sample_depletion_results)
        original_manifold.to_hdf5(temp_hdf5_file)
        
        # Load manifold
        loaded_manifold = ActivationManifold.from_hdf5(temp_hdf5_file)
        
        # Check that loaded manifold matches original
        assert loaded_manifold.elements == original_manifold.elements
        assert loaded_manifold.constraints.dose_limits == original_manifold.constraints.dose_limits
        assert loaded_manifold.constraints.gas_limits == original_manifold.constraints.gas_limits
        
        if original_manifold.feasible_compositions is not None:
            np.testing.assert_array_equal(
                loaded_manifold.feasible_compositions,
                original_manifold.feasible_compositions
            )
            
    def test_roundtrip_hdf5(self, sample_depletion_results, activation_constraints, temp_hdf5_file):
        """Test complete save/load roundtrip."""
        elements = list(sample_depletion_results.keys())
        original = ActivationManifold(elements, activation_constraints)
        
        # Build manifold
        np.random.seed(42)
        n_samples = 25
        x = np.random.random((n_samples, len(elements)))
        samples = x / x.sum(axis=1, keepdims=True)
        
        # Use permissive constraints
        permissive_constraints = ActivationConstraints(
            dose_limits={'14d': 1e10, '5y': 1e10, '7y': 1e10, '100y': 1e10},
            gas_limits={'He_appm': 1e10, 'H_appm': 1e10}
        )
        original.constraints = permissive_constraints
        original.build_from_samples(samples, sample_depletion_results)
        
        # Save and load
        original.to_hdf5(temp_hdf5_file)
        loaded = ActivationManifold.from_hdf5(temp_hdf5_file)
        
        # Compare key attributes
        assert loaded.elements == original.elements
        assert loaded._metadata['n_samples'] == original._metadata['n_samples']
        assert loaded._metadata['n_feasible'] == original._metadata['n_feasible']
        
        if original.feasible_compositions is not None and len(original.feasible_compositions) > 0:
            np.testing.assert_array_equal(
                loaded.feasible_compositions,
                original.feasible_compositions
            )
            
            # Test contains method works on loaded manifold
            if loaded.convex_hull is not None:
                test_point = original.feasible_compositions[0]
                assert loaded.contains(test_point)
                
    def test_empty_manifold_hdf5(self, temp_hdf5_file):
        """Test saving/loading empty manifold."""
        elements = ['V', 'Cr']
        constraints = ActivationConstraints()
        manifold = ActivationManifold(elements, constraints)
        
        # Don't build the manifold, just save empty one
        manifold.to_hdf5(temp_hdf5_file)
        loaded = ActivationManifold.from_hdf5(temp_hdf5_file)
        
        assert loaded.elements == elements
        assert loaded.feasible_compositions is None
        assert loaded.convex_hull is None 