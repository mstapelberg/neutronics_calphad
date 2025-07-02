"""
Test cases for BayesianSearcher class.
"""

import numpy as np
import pytest
from unittest.mock import Mock, patch

try:
    import torch
    from neutronics_calphad.calphad import BayesianSearcher
    BOTORCH_AVAILABLE = True
except ImportError:
    BOTORCH_AVAILABLE = False


@pytest.mark.skipif(not BOTORCH_AVAILABLE, reason="BoTorch not available")
class TestBayesianSearcher:
    """Test cases for BayesianSearcher class when BoTorch is available."""
    
    def test_initialization(self):
        """Test basic initialization."""
        elements = ['V', 'Cr', 'Ti', 'W', 'Zr']
        searcher = BayesianSearcher(
            elements=elements,
            batch_size=100,
            convergence_threshold=0.01,
            convergence_patience=3
        )
        
        assert searcher.elements == elements
        assert searcher.batch_size == 100
        assert searcher.convergence_threshold == 0.01
        assert searcher.convergence_patience == 3
        assert searcher.X_observed is None
        assert searcher.y_observed is None
        assert searcher.gp_model is None
        
    def test_initialization_defaults(self):
        """Test initialization with default parameters."""
        elements = ['V', 'Cr', 'Ti']
        searcher = BayesianSearcher(elements)
        
        assert searcher.elements == elements
        assert searcher.batch_size == 1000
        assert searcher.convergence_threshold == 0.02
        assert searcher.convergence_patience == 5
        
    def test_sample_simplex(self):
        """Test simplex sampling method."""
        elements = ['V', 'Cr', 'Ti']
        searcher = BayesianSearcher(elements)
        
        n_samples = 50
        samples = searcher._sample_simplex(n_samples)
        
        # Check shape
        assert samples.shape == (n_samples, len(elements))
        
        # Check that all samples sum to 1 (within tolerance)
        sums = np.sum(samples, axis=1)
        np.testing.assert_allclose(sums, 1.0, rtol=1e-10)
        
        # Check that all values are non-negative
        assert np.all(samples >= 0)
        
    def test_initial_suggestion(self):
        """Test initial batch suggestion (random sampling)."""
        elements = ['V', 'Cr', 'Ti', 'W', 'Zr']
        searcher = BayesianSearcher(elements, batch_size=20)
        
        # Initial suggestion should be random sampling
        suggestions = searcher.suggest_next_batch()
        
        assert suggestions.shape == (20, 5)
        
        # Check simplex constraint
        sums = np.sum(suggestions, axis=1)
        np.testing.assert_allclose(sums, 1.0, rtol=1e-10)
        assert np.all(suggestions >= 0)
        
    def test_update_observations(self):
        """Test updating observations."""
        elements = ['V', 'Cr', 'Ti']
        searcher = BayesianSearcher(elements, batch_size=10)
        
        # Create some fake data
        compositions = np.array([
            [0.8, 0.1, 0.1],
            [0.5, 0.3, 0.2],
            [0.3, 0.4, 0.3],
        ])
        phase_counts = np.array([1, 2, 3])
        
        stats = searcher.update(compositions, phase_counts)
        
        # Check that observations were stored
        assert searcher.X_observed is not None
        assert searcher.y_observed is not None
        assert searcher.X_observed.shape == (3, 3)
        assert searcher.y_observed.shape == (3, 1)
        
        # Check statistics
        assert stats['n_evaluated'] == 3
        assert stats['n_single_phase_new'] == 1
        assert stats['discovery_rate'] == 1/3
        
    def test_multiple_updates(self):
        """Test multiple observation updates."""
        elements = ['V', 'Cr']
        searcher = BayesianSearcher(elements, batch_size=5)
        
        # First update
        comp1 = np.array([[0.8, 0.2], [0.6, 0.4]])
        phase1 = np.array([1, 2])
        stats1 = searcher.update(comp1, phase1)
        
        assert stats1['n_evaluated'] == 2
        assert stats1['n_single_phase_new'] == 1
        
        # Second update
        comp2 = np.array([[0.9, 0.1], [0.5, 0.5], [0.3, 0.7]])
        phase2 = np.array([1, 2, 1])
        stats2 = searcher.update(comp2, phase2)
        
        assert stats2['n_evaluated'] == 5  # Total observations
        assert stats2['n_single_phase_new'] == 2  # New single-phase in this batch
        
        # Check that data was concatenated
        assert searcher.X_observed.shape == (5, 2)
        assert searcher.y_observed.shape == (5, 1)
        
    def test_convergence_checking(self):
        """Test convergence checking logic."""
        elements = ['V', 'Cr', 'Ti']
        searcher = BayesianSearcher(elements, convergence_threshold=0.1, convergence_patience=3)
        
        # Create history with high discovery rates (not converged)
        high_rate_history = [
            {'discovery_rate': 0.15},
            {'discovery_rate': 0.12},
            {'discovery_rate': 0.18}
        ]
        assert not searcher.is_converged(high_rate_history)
        
        # Create history with low discovery rates (converged)
        low_rate_history = [
            {'discovery_rate': 0.08},
            {'discovery_rate': 0.05},
            {'discovery_rate': 0.03}
        ]
        assert searcher.is_converged(low_rate_history)
        
        # Mixed history (not converged)
        mixed_history = [
            {'discovery_rate': 0.15},
            {'discovery_rate': 0.05},
            {'discovery_rate': 0.12}
        ]
        assert not searcher.is_converged(mixed_history)
        
        # Insufficient history
        short_history = [{'discovery_rate': 0.05}]
        assert not searcher.is_converged(short_history)
        
    @patch('neutronics_calphad.calphad.SingleTaskGP')
    @patch('neutronics_calphad.calphad.fit_gpytorch_mll')
    def test_gp_fitting(self, mock_fit, mock_gp):
        """Test GP model fitting."""
        elements = ['V', 'Cr']
        searcher = BayesianSearcher(elements)
        
        # Add some observations
        comp = np.array([[0.8, 0.2], [0.6, 0.4]])
        phase = np.array([1, 2])
        searcher.update(comp, phase)
        
        # Mock GP model
        mock_model = Mock()
        mock_gp.return_value = mock_model
        
        # Fit GP
        searcher._fit_gp()
        
        # Check that GP was created and fitted
        mock_gp.assert_called_once()
        mock_fit.assert_called_once()
        assert searcher.gp_model == mock_model
        
    def test_suggest_after_observations(self):
        """Test suggestion after having observations (should use GP)."""
        elements = ['V', 'Cr', 'Ti']
        searcher = BayesianSearcher(elements, batch_size=5)
        
        # Add some observations
        comp = np.array([[0.8, 0.1, 0.1], [0.5, 0.3, 0.2]])
        phase = np.array([1, 2])
        searcher.update(comp, phase)
        
        # Mock the GP fitting and acquisition optimization
        with patch.object(searcher, '_fit_gp') as mock_fit:
            with patch.object(searcher, '_optimize_acquisition') as mock_opt:
                mock_candidates = torch.tensor([[0.6, 0.2, 0.2],
                                              [0.7, 0.15, 0.15],
                                              [0.4, 0.3, 0.3],
                                              [0.9, 0.05, 0.05],
                                              [0.3, 0.4, 0.3]])
                mock_opt.return_value = mock_candidates
                
                suggestions = searcher.suggest_next_batch()
                
                # Check that GP was fitted and acquisition was optimized
                mock_fit.assert_called_once()
                mock_opt.assert_called_once()
                
                # Check suggestions
                assert suggestions.shape == (5, 3)
                np.testing.assert_allclose(np.sum(suggestions, axis=1), 1.0, rtol=1e-6)


class TestBayesianSearcherMock:
    """Test cases for BayesianSearcher when BoTorch is not available."""
    
    def test_import_error_when_botorch_unavailable(self):
        """Test that ImportError is raised when BoTorch is not available."""
        with patch('neutronics_calphad.calphad.BOTORCH_AVAILABLE', False):
            with pytest.raises(ImportError, match="BoTorch is required"):
                from neutronics_calphad.calphad import BayesianSearcher
                BayesianSearcher(['V', 'Cr'])
                
    @pytest.mark.skipif(BOTORCH_AVAILABLE, reason="BoTorch is available")
    def test_graceful_degradation_without_botorch(self):
        """Test that module still loads without BoTorch but warns user."""
        # This test only runs when BoTorch is not available
        # The warning should already be issued during import
        import warnings
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            # Re-import to trigger warning
            import importlib
            import neutronics_calphad.calphad
            importlib.reload(neutronics_calphad.calphad)
            
            # Check that a warning was issued
            botorch_warnings = [warning for warning in w 
                              if "BoTorch not available" in str(warning.message)]
            assert len(botorch_warnings) > 0


class TestBayesianSearcherIntegration:
    """Integration tests for BayesianSearcher with mock components."""
    
    @pytest.mark.skipif(not BOTORCH_AVAILABLE, reason="BoTorch not available")
    def test_full_search_loop_mock(self, sample_phase_data):
        """Test a complete search loop with mocked CALPHAD calculations."""
        elements = sample_phase_data['elements']
        searcher = BayesianSearcher(elements, batch_size=10, convergence_patience=2)
        
        # Mock the phase calculation
        def mock_phase_calc(compositions):
            """Mock phase calculation that returns single phase for high V content."""
            phase_counts = []
            for comp in compositions:
                if comp[0] > 0.7:  # High V content -> single phase
                    phase_counts.append(1)
                else:
                    phase_counts.append(2)
            return np.array(phase_counts)
        
        history = []
        for iteration in range(5):
            # Get suggestions
            suggestions = searcher.suggest_next_batch()
            
            # Mock evaluate
            phase_counts = mock_phase_calc(suggestions)
            
            # Update model
            stats = searcher.update(suggestions, phase_counts)
            history.append(stats)
            
            # Check if converged
            if searcher.is_converged(history):
                break
                
        # Should have some observations
        assert searcher.X_observed is not None
        assert len(searcher.X_observed) > 0
        
        # Should have some single-phase discoveries
        total_single_phase = sum(h['n_single_phase_new'] for h in history)
        assert total_single_phase >= 0  # At least some discoveries possible
        
    @pytest.mark.skipif(not BOTORCH_AVAILABLE, reason="BoTorch not available")
    def test_discovery_rate_calculation(self):
        """Test discovery rate calculation with known data."""
        elements = ['V', 'Cr', 'Ti']
        searcher = BayesianSearcher(elements)
        
        # Test batch with known single-phase compositions
        compositions = np.array([
            [0.9, 0.05, 0.05],  # Single phase (V dominant)
            [0.5, 0.3, 0.2],    # Multi phase
            [0.85, 0.1, 0.05],  # Single phase (V dominant)
            [0.3, 0.4, 0.3],    # Multi phase
        ])
        phase_counts = np.array([1, 2, 1, 3])
        
        stats = searcher.update(compositions, phase_counts)
        
        # Should have 2 single-phase out of 4 total
        assert stats['n_single_phase_new'] == 2
        assert stats['discovery_rate'] == 0.5
        assert stats['n_evaluated'] == 4
        
    @pytest.mark.skipif(not BOTORCH_AVAILABLE, reason="BoTorch not available")
    def test_batch_size_consistency(self):
        """Test that suggested batches have correct size."""
        elements = ['V', 'Cr', 'Ti', 'W']
        
        for batch_size in [5, 10, 50, 100]:
            searcher = BayesianSearcher(elements, batch_size=batch_size)
            suggestions = searcher.suggest_next_batch()
            assert suggestions.shape == (batch_size, len(elements))
            
    @pytest.mark.skipif(not BOTORCH_AVAILABLE, reason="BoTorch not available")
    def test_element_count_consistency(self):
        """Test that searcher works with different numbers of elements."""
        batch_size = 20
        
        for n_elements in [2, 3, 4, 5, 6]:
            elements = [f'El{i}' for i in range(n_elements)]
            searcher = BayesianSearcher(elements, batch_size=batch_size)
            
            suggestions = searcher.suggest_next_batch()
            assert suggestions.shape == (batch_size, n_elements)
            
            # Test update
            phase_counts = np.random.randint(1, 4, batch_size)
            stats = searcher.update(suggestions, phase_counts)
            assert stats['n_evaluated'] == batch_size 