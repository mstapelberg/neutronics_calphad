"""Analysis utilities for Gaussian process-based materials optimization results from the sequential pipeline.

This module provides comprehensive analysis and visualization tools for the sequential materials
design pipeline results, including UMAP dimensionality reduction for composition space exploration.
"""

import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional, Tuple, Union
from pathlib import Path
import seaborn as sns

# Optional imports for advanced visualization
try:
    import umap
    from sklearn.cluster import KMeans, DBSCAN
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    from sklearn.manifold import TSNE
    import hdbscan
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    print("Warning: UMAP and/or sklearn not available. Advanced visualization will be skipped.")

# Optional imports for interactive plots
try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("Warning: Plotly not available. Interactive plots will be skipped.")


def load_pipeline_results(results_dir: str) -> Dict[str, Any]:
    """
    Load all results from a sequential pipeline run.
    
    Parameters
    ----------
    results_dir : str
        Directory containing pipeline results.
        
    Returns
    -------
    Dict[str, Any]
        Dictionary containing all pipeline results.
    """
    results = {}
    
    # Load neutronics results
    neutronics_file = os.path.join(results_dir, "neutronics_optimization_results.json")
    if os.path.exists(neutronics_file):
        with open(neutronics_file, 'r') as f:
            results['neutronics'] = json.load(f)
    
    # Load CALPHAD results
    calphad_file = os.path.join(results_dir, "calphad_results.json")
    if os.path.exists(calphad_file):
        with open(calphad_file, 'r') as f:
            results['calphad'] = json.load(f)
    
    # Load pipeline state
    state_file = os.path.join(results_dir, "pipeline_state.json")
    if os.path.exists(state_file):
        with open(state_file, 'r') as f:
            results['pipeline_state'] = json.load(f)
    
    return results


def extract_optimization_history(neutronics_results: Dict[str, Any]) -> pd.DataFrame:
    """
    Extract optimization history as a pandas DataFrame.
    
    Parameters
    ----------
    neutronics_results : Dict[str, Any]
        Neutronics optimization results.
        
    Returns
    -------
    pd.DataFrame
        DataFrame with columns: iteration, material_name, composition columns, 
        satisfy_dose, satisfy_gas, dose_rates, gas_production.
    """
    records = []
    
    for iteration_data in neutronics_results['iterations']:
        iteration = iteration_data['iteration']
        
        for material in iteration_data['materials']:
            record = {
                'iteration': iteration,
                'material_name': material['material_name'],
                'satisfy_dose': material['satisfy_dose'],
                'satisfy_gas': material['satisfy_gas']
            }
            
            # Add composition
            for element, fraction in material['composition'].items():
                record[f'comp_{element}'] = fraction
            
            # Add dose rates
            for days, dose_rate in material['dose_rates'].items():
                record[f'dose_{days}d'] = dose_rate
            
            # Add gas production
            for gas, production in material['gas_production'].items():
                record[f'gas_{gas}'] = production
            
            # Add neutronics outputs if available
            if 'neutronics_outputs' in material:
                outputs = material['neutronics_outputs']
                record['dose_14d_raw'] = outputs[0]
                record['dose_365d_raw'] = outputs[1]
                record['dose_3650d_raw'] = outputs[2]
                record['dose_36500d_raw'] = outputs[3]
                record['He_appm_raw'] = outputs[4]
                record['H_appm_raw'] = outputs[5]
            
            records.append(record)
    
    return pd.DataFrame(records)


def extract_good_compositions(neutronics_results: Dict[str, Any]) -> np.ndarray:
    """
    Extract the good compositions found by the Gaussian process surrogate.
    
    Parameters
    ----------
    neutronics_results : Dict[str, Any]
        Neutronics optimization results.
        
    Returns
    -------
    np.ndarray
        Array of good compositions, shape (n_good, n_elements).
    """
    if 'good_compositions' in neutronics_results:
        return np.array(neutronics_results['good_compositions'])
    else:
        return np.array([])


def analyze_convergence(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyze optimization convergence for Gaussian process sampling.
    
    Parameters
    ----------
    df : pd.DataFrame
        Optimization history DataFrame.
        
    Returns
    -------
    Dict[str, Any]
        Convergence analysis results.
    """
    # Calculate running statistics
    df = df.sort_values('iteration')
    
    # Success rate by iteration
    success_rate_by_iter = df.groupby('iteration').agg({
        'satisfy_dose': 'mean',
        'satisfy_gas': 'mean'
    }).round(3)
    
    # Overall statistics
    total_materials = len(df)
    dose_success_rate = df['satisfy_dose'].mean()
    gas_success_rate = df['satisfy_gas'].mean()
    combined_success_rate = (df['satisfy_dose'] & df['satisfy_gas']).mean()
    
    # Analyze all dose rates if available
    dose_analysis = {}
    dose_cols_raw = [col for col in df.columns if col.startswith('dose_') and col.endswith('_raw')]
    dose_cols_regular = [col for col in df.columns if col.startswith('dose_') and not col.endswith('_raw') and col != 'dose_rates']
    
    # Use raw if available, otherwise use regular
    dose_cols = dose_cols_raw if dose_cols_raw else dose_cols_regular
    
    if dose_cols:
        dose_limits = {14: 1e3, 365: 1, 3650: 1e-2, 36500: 1e-4}
        
        for col in dose_cols:
            # Extract days from column name
            if col.endswith('_raw'):
                days = int(col.replace('dose_', '').replace('d_raw', ''))
            else:
                days = int(col.replace('dose_', '').replace('d', ''))
            limit = dose_limits.get(days, float('inf'))
            
            dose_analysis[f'dose_{days}d'] = {
                'mean': df[col].mean(),
                'min': df[col].min(),
                'max': df[col].max(),
                'std': df[col].std(),
                'limit': limit,
                'success_rate': (df[col] <= limit).mean(),
                'best_composition': df.loc[df[col].idxmin()].to_dict() if len(df) > 0 else None
            }
    
    # Analyze gas production
    gas_analysis = {}
    he_col = None
    h_col = None
    
    if 'He_appm_raw' in df.columns:
        he_col = 'He_appm_raw'
    elif 'He_appm' in df.columns:
        he_col = 'He_appm'
    
    if 'H_appm_raw' in df.columns:
        h_col = 'H_appm_raw'
    elif 'H_appm' in df.columns:
        h_col = 'H_appm'
    
    if he_col is not None:
        gas_analysis['He_appm'] = {
            'mean': df[he_col].mean(),
            'min': df[he_col].min(),
            'max': df[he_col].max(),
            'std': df[he_col].std(),
            'limit': 1172.2/2,
            'success_rate': (df[he_col] <= 1172.2/2).mean(),
            'best_composition': df.loc[df[he_col].idxmin()].to_dict() if len(df) > 0 else None
        }
    
    if h_col is not None:
        gas_analysis['H_appm'] = {
            'mean': df[h_col].mean(),
            'min': df[h_col].min(),
            'max': df[h_col].max(),
            'std': df[h_col].std(),
            'limit': 1500,
            'success_rate': (df[h_col] <= 1500).mean(),
            'best_composition': df.loc[df[h_col].idxmin()].to_dict() if len(df) > 0 else None
        }
    
    # Create comprehensive performance metric
    if dose_cols:
        # Normalize all metrics by their limits
        normalized_metrics = []
        
        for col in dose_cols:
            if col.endswith('_raw'):
                days = int(col.replace('dose_', '').replace('d_raw', ''))
            else:
                days = int(col.replace('dose_', '').replace('d', ''))
            limit = dose_limits.get(days, float('inf'))
            if limit != float('inf'):
                normalized_metrics.append(df[col] / limit)
        
        if he_col is not None:
            normalized_metrics.append(df[he_col] / (1172.2/2))
        
        if h_col is not None:
            normalized_metrics.append(df[h_col] / 1500)
        
        if normalized_metrics:
            df['performance_score'] = sum(normalized_metrics) / len(normalized_metrics)
            
            best_overall = df.loc[df['performance_score'].idxmin()]
            best_dose = df[df['satisfy_dose']].loc[df[df['satisfy_dose']]['performance_score'].idxmin()] if df['satisfy_dose'].any() else None
            best_gas = df[df['satisfy_gas']].loc[df[df['satisfy_gas']]['performance_score'].idxmin()] if df['satisfy_gas'].any() else None
        else:
            best_overall = None
            best_dose = None
            best_gas = None
    else:
        best_overall = None
        best_dose = None
        best_gas = None
    
    return {
        'total_materials_evaluated': total_materials,
        'dose_success_rate': dose_success_rate,
        'gas_success_rate': gas_success_rate,
        'combined_success_rate': combined_success_rate,
        'success_rate_by_iteration': success_rate_by_iter.to_dict(),
        'dose_analysis': dose_analysis,
        'gas_analysis': gas_analysis,
        'best_compositions': {
            'overall': best_overall.to_dict() if best_overall is not None else None,
            'best_dose': best_dose.to_dict() if best_dose is not None else None,
            'best_gas': best_gas.to_dict() if best_gas is not None else None
        }
    }


def create_umap_visualization(
    df: pd.DataFrame, 
    good_compositions: Optional[np.ndarray] = None,
    elements: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    interactive: bool = False
) -> Optional[go.Figure]:
    """
    Create UMAP visualization of the composition space.
    
    Parameters
    ----------
    df : pd.DataFrame
        Optimization history DataFrame.
    good_compositions : Optional[np.ndarray]
        Array of good compositions from surrogate prediction.
    elements : Optional[List[str]]
        List of element symbols.
    save_path : Optional[str]
        Path to save the plot.
    interactive : bool
        Whether to create interactive plotly plot.
        
    Returns
    -------
    Optional[go.Figure]
        Plotly figure if interactive=True, otherwise None.
    """
    if not UMAP_AVAILABLE:
        print("UMAP not available. Skipping UMAP visualization.")
        return None
    
    # Extract composition data
    comp_cols = [col for col in df.columns if col.startswith('comp_')]
    if not comp_cols:
        print("No composition columns found in DataFrame.")
        return None
    
    if elements is None:
        elements = [col.replace('comp_', '') for col in comp_cols]
    
    # Prepare data for UMAP
    composition_data = df[comp_cols].values
    
    # Add good compositions if available
    if good_compositions is not None and len(good_compositions) > 0:
        composition_data = np.vstack([composition_data, good_compositions])
        is_good_comp = np.concatenate([
            np.zeros(len(df), dtype=bool),
            np.ones(len(good_compositions), dtype=bool)
        ])
    else:
        is_good_comp = np.zeros(len(df), dtype=bool)
    
    # Standardize the data
    scaler = StandardScaler()
    composition_scaled = scaler.fit_transform(composition_data)
    
    # Apply UMAP for dimensionality reduction
    reducer = umap.UMAP(
        n_components=2, 
        random_state=42,
        n_neighbors=15,
        min_dist=0.1,
        metric='euclidean'
    )
    embedding = reducer.fit_transform(composition_scaled)
    
    # Apply HDBSCAN clustering for better manifold detection
    try:
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=5,
            min_samples=3,
            cluster_selection_epsilon=0.1,
            cluster_selection_method='eom'
        )
        clusters = clusterer.fit_predict(composition_scaled)
        
        # Also apply K-means for comparison
        kmeans = KMeans(n_clusters=5, random_state=42)
        kmeans_clusters = kmeans.fit_predict(composition_scaled)
        
        # Identify acceptable materials manifold
        acceptable_mask = np.zeros(len(composition_data), dtype=bool)
        if len(df) > 0:
            acceptable_mask[:len(df)] = (df['satisfy_dose'] & df['satisfy_gas']).values
        
        # Find clusters with high concentration of acceptable materials
        acceptable_clusters = []
        for cluster_id in np.unique(clusters):
            if cluster_id == -1:  # Noise points
                continue
            cluster_mask = clusters == cluster_id
            cluster_acceptable = acceptable_mask[cluster_mask]
            if len(cluster_acceptable) > 0:
                acceptable_fraction = cluster_acceptable.sum() / len(cluster_acceptable)
                if acceptable_fraction > 0.3:  # More than 30% acceptable
                    acceptable_clusters.append(cluster_id)
        
    except Exception as e:
        print(f"HDBSCAN clustering failed, falling back to K-means: {e}")
        clusters = KMeans(n_clusters=5, random_state=42).fit_predict(composition_scaled)
        kmeans_clusters = clusters.copy()
        acceptable_clusters = []
        acceptable_mask = np.zeros(len(composition_data), dtype=bool)
    
    if interactive and PLOTLY_AVAILABLE:
        return _create_interactive_umap_plot(
            embedding, df, good_compositions, is_good_comp, clusters, kmeans_clusters, 
            acceptable_clusters, acceptable_mask, elements
        )
    else:
        _create_static_umap_plot(
            embedding, df, good_compositions, is_good_comp, clusters, kmeans_clusters,
            acceptable_clusters, acceptable_mask, elements, save_path
        )
        return None


def _create_static_umap_plot(
    embedding: np.ndarray,
    df: pd.DataFrame,
    good_compositions: Optional[np.ndarray],
    is_good_comp: np.ndarray,
    clusters: np.ndarray,
    kmeans_clusters: np.ndarray,
    acceptable_clusters: List[int],
    acceptable_mask: np.ndarray,
    elements: List[str],
    save_path: Optional[str]
) -> None:
    """Create static matplotlib UMAP plot."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: UMAP colored by constraint satisfaction
    ax1 = axes[0, 0]
    n_evaluated = len(df)
    
    # Plot good compositions first (background) if available
    if good_compositions is not None and len(good_compositions) > 0:
        ax1.scatter(embedding[n_evaluated:, 0], embedding[n_evaluated:, 1], 
                   c='purple', alpha=0.1, s=10, marker='s', label='Predicted Good')
    
    # Plot evaluated compositions on top
    colors = []
    for _, row in df.iterrows():
        if row['satisfy_dose'] and row['satisfy_gas']:
            colors.append('green')
        elif row['satisfy_dose']:
            colors.append('orange')
        elif row['satisfy_gas']:
            colors.append('blue')
        else:
            colors.append('red')
    
    scatter1 = ax1.scatter(embedding[:n_evaluated, 0], embedding[:n_evaluated, 1], 
                          c=colors, alpha=0.8, s=40)
    
    ax1.set_xlabel('UMAP Dimension 1')
    ax1.set_ylabel('UMAP Dimension 2')
    ax1.set_title('Composition Space (Constraint Satisfaction)')
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green', label='Both satisfied'),
        Patch(facecolor='orange', label='Dose only'),
        Patch(facecolor='blue', label='Gas only'),
        Patch(facecolor='red', label='Neither satisfied')
    ]
    if good_compositions is not None and len(good_compositions) > 0:
        legend_elements.append(Patch(facecolor='purple', label='Predicted Good'))
    ax1.legend(handles=legend_elements, loc='best')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: UMAP colored by HDBSCAN clusters with acceptable manifold highlighting
    ax2 = axes[0, 1]
    
    # Create custom colormap for clusters
    unique_clusters = np.unique(clusters)
    n_clusters = len(unique_clusters)
    colors_clusters = plt.cm.tab10(np.linspace(0, 1, n_clusters))
    
    # Plot all points
    for i, cluster_id in enumerate(unique_clusters):
        mask = clusters == cluster_id
        if cluster_id == -1:  # Noise points
            color = 'gray'
            alpha = 0.3
            s = 20
        elif cluster_id in acceptable_clusters:  # Acceptable clusters
            color = 'lime'
            alpha = 0.8
            s = 40
        else:  # Other clusters
            color = colors_clusters[i]
            alpha = 0.6
            s = 30
        
        ax2.scatter(embedding[mask, 0], embedding[mask, 1], 
                   c=color, alpha=alpha, s=s, 
                   label=f'C{cluster_id}' if cluster_id != -1 else 'Noise')
    
    ax2.set_xlabel('UMAP Dimension 1')
    ax2.set_ylabel('UMAP Dimension 2')
    ax2.set_title('Composition Space (HDBSCAN Clusters)\nLime = Acceptable Manifolds')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: UMAP colored by dose rate (if available)
    ax3 = axes[1, 0]
    dose_col = None
    if 'dose_14d_raw' in df.columns:
        dose_col = 'dose_14d_raw'
    elif 'dose_14d' in df.columns:
        dose_col = 'dose_14d'
    
    if dose_col is not None:
        dose_values = df[dose_col].values
        scatter = ax3.scatter(embedding[:n_evaluated, 0], embedding[:n_evaluated, 1], 
                             c=dose_values, cmap='viridis', alpha=0.7, s=30)
        ax3.set_xlabel('UMAP Dimension 1')
        ax3.set_ylabel('UMAP Dimension 2')
        ax3.set_title('Composition Space (14-day Dose Rate)')
        plt.colorbar(scatter, ax=ax3, label='Dose Rate (Sv/h)')
    else:
        ax3.text(0.5, 0.5, 'Dose rate data not available', 
                ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title('Dose Rate Data Not Available')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: UMAP colored by He production (if available)
    ax4 = axes[1, 1]
    he_col = None
    if 'He_appm_raw' in df.columns:
        he_col = 'He_appm_raw'
    elif 'He_appm' in df.columns:
        he_col = 'He_appm'
    
    if he_col is not None:
        he_values = df[he_col].values
        scatter = ax4.scatter(embedding[:n_evaluated, 0], embedding[:n_evaluated, 1], 
                             c=he_values, cmap='plasma', alpha=0.7, s=30)
        ax4.set_xlabel('UMAP Dimension 1')
        ax4.set_ylabel('UMAP Dimension 2')
        ax4.set_title('Composition Space (He Production)')
        plt.colorbar(scatter, ax=ax4, label='He Production (appm)')
    else:
        ax4.text(0.5, 0.5, 'He production data not available', 
                ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title('He Production Data Not Available')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"UMAP plot saved to {save_path}")
    
    plt.show()


def _create_interactive_umap_plot(
    embedding: np.ndarray,
    df: pd.DataFrame,
    good_compositions: Optional[np.ndarray],
    is_good_comp: np.ndarray,
    clusters: np.ndarray,
    elements: List[str]
) -> go.Figure:
    """Create interactive plotly UMAP plot."""
    n_evaluated = len(df)
    
    # Create hover text for evaluated compositions
    hover_text_evaluated = []
    for _, row in df.iterrows():
        comp_text = '<br>'.join([f"{el}: {row[f'comp_{el}']:.3f}" for el in elements])
        status = "✓ Both" if row['satisfy_dose'] and row['satisfy_gas'] else \
                "✓ Dose" if row['satisfy_dose'] else \
                "✓ Gas" if row['satisfy_gas'] else "✗ None"
        hover_text_evaluated.append(f"<b>{row['material_name']}</b><br>{comp_text}<br>Status: {status}")
    
    # Create hover text for good compositions
    hover_text_good = []
    if good_compositions is not None and len(good_compositions) > 0:
        for i, comp in enumerate(good_compositions):
            comp_text = '<br>'.join([f"{el}: {comp[j]:.3f}" for j, el in enumerate(elements)])
            hover_text_good.append(f"<b>Predicted Good {i+1}</b><br>{comp_text}")
    
    # Create traces
    traces = []
    
    # Evaluated compositions
    colors_evaluated = []
    for _, row in df.iterrows():
        if row['satisfy_dose'] and row['satisfy_gas']:
            colors_evaluated.append('green')
        elif row['satisfy_dose']:
            colors_evaluated.append('orange')
        elif row['satisfy_gas']:
            colors_evaluated.append('blue')
        else:
            colors_evaluated.append('red')
    
    traces.append(go.Scatter(
        x=embedding[:n_evaluated, 0],
        y=embedding[:n_evaluated, 1],
        mode='markers',
        marker=dict(color=colors_evaluated, size=8, opacity=0.7),
        text=hover_text_evaluated,
        hoverinfo='text',
        name='Evaluated Compositions',
        showlegend=True
    ))
    
    # Good compositions
    if good_compositions is not None and len(good_compositions) > 0:
        traces.append(go.Scatter(
            x=embedding[n_evaluated:, 0],
            y=embedding[n_evaluated:, 1],
            mode='markers',
            marker=dict(color='purple', size=6, opacity=0.5, symbol='square'),
            text=hover_text_good,
            hoverinfo='text',
            name='Predicted Good Compositions',
            showlegend=True
        ))
    
    # Create layout
    layout = go.Layout(
        title='UMAP Visualization of Composition Space',
        xaxis=dict(title='UMAP Dimension 1'),
        yaxis=dict(title='UMAP Dimension 2'),
        hovermode='closest',
        width=800,
        height=600
    )
    
    return go.Figure(data=traces, layout=layout)


def plot_optimization_progress(df: pd.DataFrame, save_path: Optional[str] = None) -> None:
    """
    Plot optimization progress over iterations for Gaussian process sampling.
    
    Parameters
    ----------
    df : pd.DataFrame
        Optimization history DataFrame.
    save_path : Optional[str]
        Path to save the plot.
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Success rates over iterations
    iteration_stats = df.groupby('iteration').agg({
        'satisfy_dose': 'mean',
        'satisfy_gas': 'mean'
    })
    
    ax1 = axes[0, 0]
    iterations = iteration_stats.index
    ax1.plot(iterations, iteration_stats['satisfy_dose'], 'r-o', label='Dose success rate', markersize=4)
    ax1.plot(iterations, iteration_stats['satisfy_gas'], 'b-s', label='Gas success rate', markersize=4)
    combined_success = df.groupby('iteration').apply(lambda x: (x['satisfy_dose'] & x['satisfy_gas']).mean())
    ax1.plot(iterations, combined_success, 'g-^', label='Combined success rate', markersize=4)
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Success Rate')
    ax1.set_title('Constraint Satisfaction Rates')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1.1)
    
    # Plot 2: Composition space exploration (V vs Cr)
    ax2 = axes[0, 1]
    scatter = ax2.scatter(df['comp_V'], df['comp_Cr'], 
                         c=df['iteration'], cmap='viridis', 
                         alpha=0.7, s=30)
    ax2.set_xlabel('V fraction')
    ax2.set_ylabel('Cr fraction')
    ax2.set_title('Composition Space Exploration (V vs Cr)')
    plt.colorbar(scatter, ax=ax2, label='Iteration')
    
    # Plot 3: Dose vs Gas constraints
    ax3 = axes[1, 0]
    # Use first dose limit and He gas for plotting
    dose_col = [col for col in df.columns if col.startswith('dose_')][0]
    gas_col = [col for col in df.columns if col.startswith('gas_') and 'He' in col][0]
    
    # Color by constraint satisfaction
    colors = []
    for _, row in df.iterrows():
        if row['satisfy_dose'] and row['satisfy_gas']:
            colors.append('green')
        elif row['satisfy_dose']:
            colors.append('orange')
        elif row['satisfy_gas']:
            colors.append('blue')
        else:
            colors.append('red')
    
    ax3.scatter(df[dose_col], df[gas_col], c=colors, alpha=0.7, s=30)
    ax3.set_xlabel(dose_col.replace('_', ' ').title())
    ax3.set_ylabel(gas_col.replace('_', ' ').title())
    ax3.set_title('Dose vs Gas Production')
    ax3.set_yscale('log')
    ax3.set_xscale('log')
    
    # Add legend for colors
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green', label='Both satisfied'),
        Patch(facecolor='orange', label='Dose only'),
        Patch(facecolor='blue', label='Gas only'),
        Patch(facecolor='red', label='Neither satisfied')
    ]
    ax3.legend(handles=legend_elements, loc='best')
    
    # Plot 4: Performance metrics over iterations (if available)
    ax4 = axes[1, 1]
    # Check for both raw and non-raw column names
    dose_col = None
    he_col = None
    
    if 'dose_14d_raw' in df.columns:
        dose_col = 'dose_14d_raw'
    elif 'dose_14d' in df.columns:
        dose_col = 'dose_14d'
    
    if 'He_appm_raw' in df.columns:
        he_col = 'He_appm_raw'
    elif 'He_appm' in df.columns:
        he_col = 'He_appm'
    
    if dose_col is not None and he_col is not None:
        iteration_performance = df.groupby('iteration').agg({
            dose_col: ['mean', 'min'],
            he_col: ['mean', 'min']
        })
        
        # Plot dose rates
        ax4_twin = ax4.twinx()
        line1 = ax4.plot(iterations, iteration_performance[(dose_col, 'min')], 
                        'b-o', label='Min 14d dose', markersize=4)
        line2 = ax4_twin.plot(iterations, iteration_performance[(he_col, 'min')], 
                             'r-s', label='Min He production', markersize=4)
        
        ax4.set_xlabel('Iteration')
        ax4.set_ylabel('Dose Rate (Sv/h)', color='blue')
        ax4_twin.set_ylabel('He Production (appm)', color='red')
        ax4.set_title('Best Performance Over Iterations')
        ax4.grid(True, alpha=0.3)
        
        # Combine legends
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax4.legend(lines, labels, loc='upper right')
    else:
        ax4.text(0.5, 0.5, 'Performance data not available', 
                ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title('Performance Data Not Available')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Optimization progress plot saved to {save_path}")
    
    plt.show()


def plot_composition_space_exploration(df: pd.DataFrame, save_path: Optional[str] = None) -> None:
    """
    Plot comprehensive composition space exploration showing all V-X binaries.
    
    Parameters
    ----------
    df : pd.DataFrame
        Optimization history DataFrame.
    save_path : Optional[str]
        Path to save the plot.
    """
    # Extract composition columns
    comp_cols = [col for col in df.columns if col.startswith('comp_')]
    elements = [col.replace('comp_', '') for col in comp_cols]
    
    # Find V index
    v_idx = None
    for i, element in enumerate(elements):
        if element == 'V':
            v_idx = i
            break
    
    if v_idx is None:
        print("V element not found in composition data")
        return
    
    # Create subplots for all V-X binaries
    n_other_elements = len(elements) - 1
    n_cols = 2  # Changed to 2 columns for 2x2 grid
    n_rows = 2  # Fixed to 2 rows for 2x2 grid
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 10))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    # Color by constraint satisfaction
    colors = []
    for _, row in df.iterrows():
        if row['satisfy_dose'] and row['satisfy_gas']:
            colors.append('green')
        elif row['satisfy_dose']:
            colors.append('orange')
        elif row['satisfy_gas']:
            colors.append('blue')
        else:
            colors.append('red')
    
    plot_idx = 0
    for i, element in enumerate(elements):
        if element == 'V':
            continue
        
        row = plot_idx // n_cols
        col = plot_idx % n_cols
        
        if n_rows == 1:
            ax = axes[col]
        else:
            ax = axes[row, col]
        
        # Plot V vs this element
        ax.scatter(df[f'comp_V'], df[f'comp_{element}'], 
                  c=colors, alpha=0.7, s=30)
        ax.set_xlabel('V fraction')
        ax.set_ylabel(f'{element} fraction')
        ax.set_title(f'V vs {element}')
        ax.grid(True, alpha=0.3)
        
        # Add success rate annotation for this specific binary
        # Calculate success rate for compositions in this V-X range
        v_values = df[f'comp_V'].values
        x_values = df[f'comp_{element}'].values
        
        # Find compositions in the main V-X region (not extreme values)
        v_range = (v_values >= 0.7) & (v_values <= 0.95)
        x_range = (x_values >= 0.0) & (x_values <= 0.2)  # Reasonable range for other elements
        
        in_range = v_range & x_range
        if in_range.sum() > 0:
            range_colors = [colors[i] for i in range(len(colors)) if in_range[i]]
            successful_in_range = sum(1 for c in range_colors if c == 'green')
            success_rate = successful_in_range / len(range_colors) if len(range_colors) > 0 else 0
        else:
            # Fallback to overall success rate
            successful = sum(1 for c in colors if c == 'green')
            total = len(colors)
            success_rate = successful / total if total > 0 else 0
        
        ax.text(0.05, 0.95, f'Success: {success_rate:.1%}', 
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plot_idx += 1
    
    # Hide unused subplots
    for i in range(plot_idx, n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        if n_rows == 1:
            axes[col].set_visible(False)
        else:
            axes[row, col].set_visible(False)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green', label='Both satisfied'),
        Patch(facecolor='orange', label='Dose only'),
        Patch(facecolor='blue', label='Gas only'),
        Patch(facecolor='red', label='Neither satisfied')
    ]
    
    # Place legend in the last visible subplot
    last_plot_idx = plot_idx - 1
    last_row = last_plot_idx // n_cols
    last_col = last_plot_idx % n_cols
    if n_rows == 1:
        axes[last_col].legend(handles=legend_elements, loc='upper right')
    else:
        axes[last_row, last_col].legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Composition space exploration plot saved to {save_path}")
    
    plt.show()


def plot_dose_rate_analysis(df: pd.DataFrame, save_path: Optional[str] = None) -> None:
    """
    Plot comprehensive dose rate analysis for all time points.
    
    Parameters
    ----------
    df : pd.DataFrame
        Optimization history DataFrame.
    save_path : Optional[str]
        Path to save the plot.
    """
    # Find all dose rate columns (both raw and non-raw)
    dose_cols_raw = [col for col in df.columns if col.startswith('dose_') and col.endswith('_raw')]
    dose_cols_regular = [col for col in df.columns if col.startswith('dose_') and not col.endswith('_raw') and col != 'dose_rates']
    
    # Use raw if available, otherwise use regular
    dose_cols = dose_cols_raw if dose_cols_raw else dose_cols_regular
    
    if not dose_cols:
        print("No dose rate data found")
        return
    
    # Extract time points
    time_points = []
    for col in dose_cols:
        if col.endswith('_raw'):
            days = int(col.replace('dose_', '').replace('d_raw', ''))
        else:
            days = int(col.replace('dose_', '').replace('d', ''))
        time_points.append(days)
    
    # Sort by time
    sorted_indices = np.argsort(time_points)
    dose_cols = [dose_cols[i] for i in sorted_indices]
    time_points = [time_points[i] for i in sorted_indices]
    
    # Create subplots
    n_doses = len(dose_cols)
    n_cols = 2
    n_rows = (n_doses + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    # Color by constraint satisfaction
    colors = []
    for _, row in df.iterrows():
        if row['satisfy_dose'] and row['satisfy_gas']:
            colors.append('green')
        elif row['satisfy_dose']:
            colors.append('orange')
        elif row['satisfy_gas']:
            colors.append('blue')
        else:
            colors.append('red')
    
    for i, (col, days) in enumerate(zip(dose_cols, time_points)):
        row = i // n_cols
        col_idx = i % n_cols
        
        if n_rows == 1:
            ax = axes[col_idx]
        else:
            ax = axes[row, col_idx]
        
        # Plot dose rate vs iteration
        ax.scatter(df['iteration'], df[col], c=colors, alpha=0.7, s=30)
        ax.set_xlabel('Iteration')
        ax.set_ylabel(f'Dose Rate ({days}d) (Sv/h)')
        ax.set_title(f'{days}-day Dose Rate')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        
        # Add limit line
        dose_limits = {14: 1e3, 365: 1, 3650: 1e-2, 36500: 1e-4}
        if days in dose_limits:
            limit = dose_limits[days]
            ax.axhline(y=limit, color='red', linestyle='--', alpha=0.7, label=f'Limit: {limit:.0e}')
            ax.legend()
        
        # Add statistics
        mean_dose = df[col].mean()
        min_dose = df[col].min()
        success_rate = (df[col] <= dose_limits.get(days, float('inf'))).mean()
        
        stats_text = f'Mean: {mean_dose:.2e}\nMin: {min_dose:.2e}\nSuccess: {success_rate:.1%}'
        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Hide unused subplots
    for i in range(n_doses, n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        if n_rows == 1:
            axes[col].set_visible(False)
        else:
            axes[row, col].set_visible(False)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Dose rate analysis plot saved to {save_path}")
    
    plt.show()


def analyze_composition_space(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyze composition space coverage and patterns.
    
    Parameters
    ----------
    df : pd.DataFrame
        Optimization history DataFrame.
        
    Returns
    -------
    Dict[str, Any]
        Composition space analysis results.
    """
    # Extract composition columns
    comp_cols = [col for col in df.columns if col.startswith('comp_')]
    elements = [col.replace('comp_', '') for col in comp_cols]
    
    # Calculate ranges and coverage
    composition_stats = {}
    for col, element in zip(comp_cols, elements):
        composition_stats[element] = {
            'min': df[col].min(),
            'max': df[col].max(),
            'range': df[col].max() - df[col].min(),
            'mean': df[col].mean(),
            'std': df[col].std()
        }
    
    # Successful vs unsuccessful compositions
    successful = df[df['satisfy_dose'] & df['satisfy_gas']]
    unsuccessful = df[~(df['satisfy_dose'] & df['satisfy_gas'])]
    
    successful_stats = {}
    unsuccessful_stats = {}
    
    for col, element in zip(comp_cols, elements):
        if len(successful) > 0:
            successful_stats[element] = {
                'mean': successful[col].mean(),
                'std': successful[col].std(),
                'min': successful[col].min(),
                'max': successful[col].max()
            }
        
        if len(unsuccessful) > 0:
            unsuccessful_stats[element] = {
                'mean': unsuccessful[col].mean(),
                'std': unsuccessful[col].std(),
                'min': unsuccessful[col].min(),
                'max': unsuccessful[col].max()
            }
    
    return {
        'overall_composition_stats': composition_stats,
        'successful_composition_stats': successful_stats,
        'unsuccessful_composition_stats': unsuccessful_stats,
        'n_successful': len(successful),
        'n_unsuccessful': len(unsuccessful),
        'elements': elements
    }


def create_composition_heatmap(df: pd.DataFrame, save_path: Optional[str] = None) -> None:
    """
    Create a correlation heatmap between composition and performance.
    
    Parameters
    ----------
    df : pd.DataFrame
        Optimization history DataFrame.
    save_path : Optional[str]
        Path to save the plot.
    """
    # Select relevant columns
    comp_cols = [col for col in df.columns if col.startswith('comp_')]
    performance_cols = ['satisfy_dose', 'satisfy_gas']
    dose_cols = [col for col in df.columns if col.startswith('dose_')]
    gas_cols = [col for col in df.columns if col.startswith('gas_')]
    
    selected_cols = comp_cols + performance_cols + dose_cols[:2] + gas_cols[:2]  # Limit for readability
    correlation_data = df[selected_cols].corr()
    
    plt.figure(figsize=(12, 10))
    mask = np.triu(np.ones_like(correlation_data, dtype=bool))
    
    sns.heatmap(correlation_data, 
                mask=mask,
                annot=True, 
                cmap='RdBu_r', 
                center=0,
                square=True,
                fmt='.2f',
                cbar_kws={'label': 'Correlation'})
    
    plt.title('Composition-Performance Correlation Matrix')
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Correlation heatmap saved to {save_path}")
    
    plt.show()


def analyze_calphad_results(calphad_results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze CALPHAD phase stability results.
    
    Parameters
    ----------
    calphad_results : Dict[str, Any]
        CALPHAD results dictionary.
        
    Returns
    -------
    Dict[str, Any]
        CALPHAD analysis results.
    """
    if 'phase_analysis' not in calphad_results:
        return {'error': 'No phase analysis data found'}
    
    phase_analysis = calphad_results['phase_analysis']
    
    # Basic statistics
    n_compositions = len(phase_analysis)
    n_phase_passing = sum(1 for analysis in phase_analysis 
                         if analysis.get('satisfies_phase_limits', False))
    
    # Phase count distribution
    phase_counts = [analysis.get('phase_count', 0) for analysis in phase_analysis]
    phase_count_dist = pd.Series(phase_counts).value_counts().sort_index()
    
    # Dominant phase analysis
    dominant_phases = [analysis.get('dominant_phase', 'Unknown') for analysis in phase_analysis]
    dominant_phase_dist = pd.Series(dominant_phases).value_counts()
    
    # Single phase analysis
    single_phase_count = sum(1 for analysis in phase_analysis 
                           if analysis.get('single_phase', False))
    
    # Phase violation analysis
    all_violations = []
    for analysis in phase_analysis:
        violations = analysis.get('phase_violations', [])
        all_violations.extend(violations)
    
    violation_summary = {}
    for violation in all_violations:
        phase = violation.get('phase', 'Unknown')
        if phase not in violation_summary:
            violation_summary[phase] = 0
        violation_summary[phase] += 1
    
    return {
        'n_compositions_analyzed': n_compositions,
        'n_phase_passing': n_phase_passing,
        'phase_success_rate': n_phase_passing / n_compositions if n_compositions > 0 else 0,
        'phase_count_distribution': phase_count_dist.to_dict(),
        'dominant_phase_distribution': dominant_phase_dist.to_dict(),
        'single_phase_count': single_phase_count,
        'single_phase_fraction': single_phase_count / n_compositions if n_compositions > 0 else 0,
        'phase_violation_summary': violation_summary
    }


def generate_analysis_report(results_dir: str, output_file: Optional[str] = None) -> Dict[str, Any]:
    """
    Generate a comprehensive analysis report.
    
    Parameters
    ----------
    results_dir : str
        Directory containing pipeline results.
    output_file : Optional[str]
        Path to save the JSON report.
        
    Returns
    -------
    Dict[str, Any]
        Complete analysis report.
    """
    # Load results
    results = load_pipeline_results(results_dir)
    
    if 'neutronics' not in results:
        raise ValueError("No neutronics results found")
    
    # Extract optimization history
    df = extract_optimization_history(results['neutronics'])
    
    # Extract good compositions
    good_compositions = extract_good_compositions(results['neutronics'])
    
    # Perform analyses
    convergence_analysis = analyze_convergence(df)
    composition_analysis = analyze_composition_space(df)
    
    # Create report
    report = {
        'metadata': {
            'analysis_timestamp': pd.Timestamp.now().isoformat(),
            'results_directory': str(results_dir),
            'pipeline_metadata': results['neutronics'].get('metadata', {})
        },
        'convergence_analysis': convergence_analysis,
        'composition_analysis': composition_analysis,
        'pipeline_state': results.get('pipeline_state', {}),
        'raw_data_summary': {
            'total_iterations': df['iteration'].max() if len(df) > 0 else 0,
            'total_materials': len(df),
            'elements_analyzed': composition_analysis.get('elements', []),
            'good_compositions_found': len(good_compositions)
        }
    }
    
    # Add CALPHAD analysis if available
    if 'calphad' in results:
        calphad_analysis = analyze_calphad_results(results['calphad'])
        report['calphad_analysis'] = calphad_analysis
        
        # Calculate overall pipeline success
        n_phase_passing = calphad_analysis['n_phase_passing']
        total_evaluated = len(df)
        report['overall_pipeline_success'] = {
            'total_evaluated': total_evaluated,
            'neutronics_passing': convergence_analysis['total_materials_evaluated'] * convergence_analysis['combined_success_rate'],
            'phase_passing': n_phase_passing,
            'overall_success_rate': n_phase_passing / total_evaluated if total_evaluated > 0 else 0
        }
    
    # Save report
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        print(f"Analysis report saved to {output_file}")
    
    return report


def quick_analysis(results_dir: str, create_umap: bool = True, interactive: bool = False) -> None:
    """
    Perform a quick analysis of pipeline results.
    
    Parameters
    ----------
    results_dir : str
        Path to results directory.
    create_umap : bool
        Whether to create UMAP visualizations.
    interactive : bool
        Whether to create interactive plots.
    """
    results_path = Path(results_dir)
    
    print(f"=== Quick Analysis: {results_dir} ===")
    
    # Generate report
    report = generate_analysis_report(results_path)
    
    # Print summary
    conv = report['convergence_analysis']
    comp = report['composition_analysis']
    
    print(f"\nOptimization Summary:")
    print(f"  Total materials evaluated: {conv['total_materials_evaluated']}")
    print(f"  Dose success rate: {conv['dose_success_rate']:.1%}")
    print(f"  Gas success rate: {conv['gas_success_rate']:.1%}")
    print(f"  Combined success rate: {conv['combined_success_rate']:.1%}")
    
    print(f"\nComposition Space:")
    print(f"  Elements: {comp['elements']}")
    print(f"  Successful compositions: {comp['n_successful']}")
    print(f"  Unsuccessful compositions: {comp['n_unsuccessful']}")
    print(f"  Good compositions found: {report['raw_data_summary']['good_compositions_found']}")
    
    if 'calphad_analysis' in report:
        calphad = report['calphad_analysis']
        print(f"\nCALPHAD Analysis:")
        print(f"  Compositions analyzed: {calphad['n_compositions_analyzed']}")
        print(f"  Phase-stable compositions: {calphad['n_phase_passing']}")
        print(f"  Phase success rate: {calphad['phase_success_rate']:.1%}")
        
        if 'overall_pipeline_success' in report:
            overall = report['overall_pipeline_success']
            print(f"  Overall pipeline success: {overall['overall_success_rate']:.1%}")
    
    # Create plots
    results = load_pipeline_results(results_path)
    df = extract_optimization_history(results['neutronics'])
    good_compositions = extract_good_compositions(results['neutronics'])
    
    print(f"\nGenerating plots...")
    
    # Basic plots
    plot_optimization_progress(df, save_path=os.path.join(results_path, "optimization_progress.png"))
    create_composition_heatmap(df, save_path=os.path.join(results_path, "composition_heatmap.png"))
    
    # Comprehensive composition space exploration
    print("Creating composition space exploration plots...")
    plot_composition_space_exploration(df, save_path=os.path.join(results_path, "composition_space_exploration.png"))
    
    # Dose rate analysis
    print("Creating dose rate analysis plots...")
    plot_dose_rate_analysis(df, save_path=os.path.join(results_path, "dose_rate_analysis.png"))
    
    # UMAP visualization
    if create_umap and UMAP_AVAILABLE:
        print("Creating UMAP visualization...")
        elements = comp['elements']
        umap_fig = create_umap_visualization(
            df, good_compositions, elements,
            save_path=os.path.join(results_path, "umap_visualization.png"),
            interactive=interactive
        )
        
        if interactive and umap_fig is not None:
            umap_fig.write_html(os.path.join(results_path, "umap_interactive.html"))
            print("Interactive UMAP plot saved as umap_interactive.html")
    
    print(f"Analysis complete!")


if __name__ == "__main__":
    # Example usage
    import sys
    if len(sys.argv) > 1:
        results_dir = sys.argv[1]
        create_umap = '--no-umap' not in sys.argv
        interactive = '--interactive' in sys.argv
        quick_analysis(results_dir, create_umap=create_umap, interactive=interactive)
    else:
        print("Usage: python bo_results_analyzer.py <results_directory> [--no-umap] [--interactive]")
        print("Example: python bo_results_analyzer.py sequential_materials_pipeline_run_1 --interactive") 