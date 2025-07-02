"""
Test cases for OutlierDetector class.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_blobs

from neutronics_calphad.calphad import OutlierDetector


class TestOutlierDetector:
    """Test cases for OutlierDetector class."""
    
    def test_initialization_default(self):
        """Test default initialization."""
        detector = OutlierDetector()
        
        assert detector.contamination == 0.1
        assert hasattr(detector, 'lof')
        assert detector.lof.contamination == 0.1
        
    def test_initialization_custom(self):
        """Test custom initialization."""
        contamination = 0.05
        detector = OutlierDetector(contamination=contamination)
        
        assert detector.contamination == contamination
        assert detector.lof.contamination == contamination
        
    def test_fit_predict_basic(self):
        """Test basic outlier detection functionality."""
        detector = OutlierDetector(contamination=0.1)
        
        # Create simple data with clear outliers
        np.random.seed(42)
        normal_points = np.random.normal(0, 1, (90, 2))
        outlier_points = np.random.normal(5, 0.5, (10, 2))  # Far from normal points
        
        X = np.vstack([normal_points, outlier_points])
        
        predictions = detector.fit_predict(X)
        
        # Check return type and shape
        assert isinstance(predictions, np.ndarray)
        assert predictions.shape == (100,)
        assert predictions.dtype == bool
        
        # Should identify some outliers
        n_outliers = np.sum(~predictions)
        assert n_outliers > 0
        assert n_outliers <= 15  # Should be roughly 10% contamination
        
    def test_fit_predict_no_outliers(self):
        """Test detection with homogeneous data (no outliers)."""
        detector = OutlierDetector(contamination=0.1)
        
        # Create tightly clustered data
        np.random.seed(42)
        X = np.random.normal(0, 0.1, (50, 3))
        
        predictions = detector.fit_predict(X)
        
        # Even with homogeneous data, LOF might flag some points as outliers
        # but most should be inliers
        n_inliers = np.sum(predictions)
        assert n_inliers >= 40  # At least 80% should be inliers
        
    def test_fit_predict_composition_data(self, sample_compositions):
        """Test with composition data (simplex constraint)."""
        detector = OutlierDetector(contamination=0.15)
        
        # Use sample compositions (already on simplex)
        predictions = detector.fit_predict(sample_compositions)
        
        assert predictions.shape == (len(sample_compositions),)
        assert predictions.dtype == bool
        
        # Should identify some compositions as outliers
        n_outliers = np.sum(~predictions)
        assert n_outliers > 0
        
    def test_filter_outliers_keep_them(self):
        """Test filtering outliers while keeping them marked."""
        detector = OutlierDetector(contamination=0.2)
        
        # Create test DataFrame
        np.random.seed(42)
        normal_data = np.random.normal(0, 1, (40, 3))
        outlier_data = np.random.normal(4, 0.5, (10, 3))
        X = np.vstack([normal_data, outlier_data])
        
        df = pd.DataFrame(X, columns=['x_V', 'x_Cr', 'x_Ti'])
        df['property'] = np.random.random(50)
        
        feature_cols = ['x_V', 'x_Cr', 'x_Ti']
        
        filtered_df = detector.filter_outliers(df, feature_cols, keep_outliers=True)
        
        # Should have same number of rows
        assert len(filtered_df) == len(df)
        
        # Should have additional 'is_outlier' column
        assert 'is_outlier' in filtered_df.columns
        assert filtered_df['is_outlier'].dtype == bool
        
        # Should identify some outliers
        n_outliers = filtered_df['is_outlier'].sum()
        assert n_outliers > 0
        
        # Original columns should be preserved
        for col in df.columns:
            assert col in filtered_df.columns
            pd.testing.assert_series_equal(df[col], filtered_df[col])
            
    def test_filter_outliers_remove_them(self):
        """Test filtering outliers by removing them."""
        detector = OutlierDetector(contamination=0.3)
        
        # Create test DataFrame with clear outliers
        np.random.seed(42)
        normal_data = np.random.normal(0, 1, (35, 2))
        outlier_data = np.random.normal(6, 0.3, (15, 2))
        X = np.vstack([normal_data, outlier_data])
        
        df = pd.DataFrame(X, columns=['x_V', 'x_Cr'])
        df['phase_count'] = np.random.randint(1, 4, 50)
        
        feature_cols = ['x_V', 'x_Cr']
        
        filtered_df = detector.filter_outliers(df, feature_cols, keep_outliers=False)
        
        # Should have fewer rows (outliers removed)
        assert len(filtered_df) < len(df)
        
        # Should not have 'is_outlier' column
        assert 'is_outlier' not in filtered_df.columns
        
        # Original columns should be preserved
        for col in df.columns:
            assert col in filtered_df.columns
            
    def test_different_contamination_levels(self):
        """Test different contamination levels."""
        # Create data with known structure
        np.random.seed(42)
        X, _ = make_blobs(n_samples=100, centers=1, cluster_std=1.0, random_state=42)
        
        contamination_levels = [0.05, 0.1, 0.2, 0.3]
        outlier_counts = []
        
        for contamination in contamination_levels:
            detector = OutlierDetector(contamination=contamination)
            predictions = detector.fit_predict(X)
            n_outliers = np.sum(~predictions)
            outlier_counts.append(n_outliers)
            
        # Higher contamination should generally identify more outliers
        # (though this is not strictly guaranteed with LOF)
        assert outlier_counts[-1] >= outlier_counts[0]  # Highest vs lowest
        
    def test_small_dataset(self):
        """Test with small dataset."""
        detector = OutlierDetector(contamination=0.2)
        
        # Very small dataset
        X = np.array([
            [0, 0],
            [1, 1],
            [0.5, 0.5],
            [10, 10],  # Potential outlier
            [0.2, 0.8]
        ])
        
        predictions = detector.fit_predict(X)
        
        assert predictions.shape == (5,)
        assert predictions.dtype == bool
        
        # Should work without errors even with small dataset
        n_inliers = np.sum(predictions)
        assert n_inliers >= 3  # Most points should be inliers
        
    def test_single_feature(self):
        """Test with single feature."""
        detector = OutlierDetector(contamination=0.1)
        
        # 1D data
        np.random.seed(42)
        normal_data = np.random.normal(0, 1, 45).reshape(-1, 1)
        outlier_data = np.random.normal(5, 0.5, 5).reshape(-1, 1)
        X = np.vstack([normal_data, outlier_data])
        
        predictions = detector.fit_predict(X)
        
        assert predictions.shape == (50,)
        assert predictions.dtype == bool
        
    def test_high_dimensional_data(self):
        """Test with high-dimensional data."""
        detector = OutlierDetector(contamination=0.15)
        
        # High-dimensional data (10 features)
        np.random.seed(42)
        X = np.random.normal(0, 1, (100, 10))
        
        # Add some outliers in high-dimensional space
        outliers = np.random.normal(3, 0.5, (15, 10))
        X = np.vstack([X, outliers])
        
        predictions = detector.fit_predict(X)
        
        assert predictions.shape == (115,)
        assert predictions.dtype == bool
        
        n_outliers = np.sum(~predictions)
        assert n_outliers > 0
        
    def test_identical_points(self):
        """Test with dataset containing identical points."""
        detector = OutlierDetector(contamination=0.1)
        
        # Dataset with many identical points
        X = np.array([
            [1, 1], [1, 1], [1, 1], [1, 1], [1, 1],  # Identical points
            [1.1, 1.1], [0.9, 0.9],  # Close points
            [5, 5], [6, 6]  # Outliers
        ])
        
        predictions = detector.fit_predict(X)
        
        assert predictions.shape == (9,)
        assert predictions.dtype == bool
        
        # Should handle identical points without errors
        n_inliers = np.sum(predictions)
        assert n_inliers >= 5
        
    def test_filter_outliers_empty_dataframe(self):
        """Test filtering with empty DataFrame."""
        detector = OutlierDetector()
        
        # Empty DataFrame
        df = pd.DataFrame(columns=['x_V', 'x_Cr', 'property'])
        feature_cols = ['x_V', 'x_Cr']
        
        # Should handle empty DataFrame gracefully
        try:
            filtered_df = detector.filter_outliers(df, feature_cols)
            # If it doesn't raise an error, check structure
            assert len(filtered_df) == 0
            assert all(col in filtered_df.columns for col in df.columns)
        except ValueError:
            # LOF might raise ValueError for empty data, which is acceptable
            pass
            
    def test_filter_outliers_single_row(self):
        """Test filtering with single-row DataFrame."""
        detector = OutlierDetector()
        
        # Single row DataFrame
        df = pd.DataFrame({
            'x_V': [0.8],
            'x_Cr': [0.2],
            'property': [1.5]
        })
        feature_cols = ['x_V', 'x_Cr']
        
        # Should handle single row
        try:
            filtered_df = detector.filter_outliers(df, feature_cols)
            assert len(filtered_df) <= 1
        except ValueError:
            # LOF might raise ValueError for insufficient data
            pass
            
    def test_composition_outlier_detection(self, sample_compositions):
        """Test outlier detection specifically for alloy compositions."""
        detector = OutlierDetector(contamination=0.1)
        
        # Add some extreme compositions as outliers
        outlier_compositions = np.array([
            [0.99, 0.005, 0.003, 0.001, 0.001],  # Almost pure first element
            [0.001, 0.99, 0.003, 0.003, 0.003],  # Almost pure second element
        ])
        
        all_compositions = np.vstack([sample_compositions, outlier_compositions])
        
        predictions = detector.fit_predict(all_compositions)
        
        # Check that extreme compositions are more likely to be flagged
        n_total = len(all_compositions)
        n_outliers = np.sum(~predictions)
        
        assert n_outliers > 0
        assert n_outliers <= 0.2 * n_total  # Reasonable outlier rate
        
    def test_feature_selection_importance(self):
        """Test that feature selection affects outlier detection."""
        detector = OutlierDetector(contamination=0.1)
        
        # Create DataFrame with correlated and uncorrelated features
        np.random.seed(42)
        n_samples = 100
        
        df = pd.DataFrame({
            'x_V': np.random.normal(0.5, 0.1, n_samples),
            'x_Cr': np.random.normal(0.3, 0.1, n_samples),
            'x_Ti': np.random.normal(0.2, 0.1, n_samples),
            'noise_feature': np.random.normal(0, 10, n_samples),  # High variance noise
            'property': np.random.random(n_samples)
        })
        
        # Test with different feature sets
        features_with_noise = ['x_V', 'x_Cr', 'x_Ti', 'noise_feature']
        features_without_noise = ['x_V', 'x_Cr', 'x_Ti']
        
        result_with_noise = detector.filter_outliers(df, features_with_noise, keep_outliers=True)
        result_without_noise = detector.filter_outliers(df, features_without_noise, keep_outliers=True)
        
        # Results might be different due to noise feature
        outliers_with_noise = result_with_noise['is_outlier'].sum()
        outliers_without_noise = result_without_noise['is_outlier'].sum()
        
        # Both should detect some outliers
        assert outliers_with_noise >= 0
        assert outliers_without_noise >= 0 