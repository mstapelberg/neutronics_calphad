"""Utility script to analyze Bayesian optimization results from JSON file.

This script loads the optimization results and provides various analysis and
visualization capabilities for the material compositions, scores, dose rates,
and gas production data.

Dependencies:
- Required: numpy, pandas, matplotlib
- Optional: umap-learn, scikit-learn (for advanced composition clustering)
  Install with: pip install umap-learn scikit-learn
"""

from __future__ import annotations

import json
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime

# Optional imports for UMAP clustering
try:
    import umap
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    print("Warning: UMAP and/or sklearn not available. Clustering visualization will be skipped.")


def load_optimization_results(filepath: str) -> Dict[str, Any]:
    """Load optimization results from JSON file.
    
    Parameters
    ----------
    filepath : str
        Path to the JSON file containing optimization results.
        
    Returns
    -------
    Dict[str, Any]
        Dictionary containing the optimization results.
    """
    with open(filepath, 'r') as f:
        results = json.load(f)
    return results


def create_materials_dataframe(results: Dict[str, Any]) -> pd.DataFrame:
    """Convert optimization results to a pandas DataFrame for analysis.
    
    Parameters
    ----------
    results : Dict[str, Any]
        Optimization results dictionary.
        
    Returns
    -------
    pd.DataFrame
        DataFrame containing all material evaluations.
    """
    materials_data = []
    
    for iteration in results['iterations']:
        iter_num = iteration['iteration']
        for material in iteration['materials']:
            row = {
                'iteration': iter_num,
                'material_name': material['material_name'],
                'score': material['score'],
                'satisfy_dose': material['satisfy_dose'],
                'satisfy_gas': material['satisfy_gas'],
                'He_appm': material['gas_production'].get('He_appm', 0.0),
                'H_appm': material['gas_production'].get('H_appm', 0.0),
                'dose_14d': material['dose_rates'].get('14', 0.0),
                'dose_365d': material['dose_rates'].get('365', 0.0),
                'dose_3650d': material['dose_rates'].get('3650', 0.0),
                'dose_36500d': material['dose_rates'].get('36500', 0.0),
            }
            
            # Add composition data
            for element, fraction in material['composition'].items():
                row[f'{element}_fraction'] = fraction
                
            materials_data.append(row)
    
    return pd.DataFrame(materials_data)


def plot_score_evolution(df: pd.DataFrame, save_path: Optional[str] = None) -> None:
    """Plot the evolution of scores across iterations.
    
    Parameters
    ----------
    df : pd.DataFrame
        Materials DataFrame.
    save_path : Optional[str], optional
        Path to save the plot, by default None.
    """
    plt.figure(figsize=(12, 8))
    
    # Plot individual scores
    plt.subplot(2, 2, 1)
    for iteration in df['iteration'].unique():
        iter_data = df[df['iteration'] == iteration]
        plt.scatter([iteration] * len(iter_data), iter_data['score'], 
                   alpha=0.6, label=f'Iteration {iteration}')
    
    plt.xlabel('Iteration')
    plt.ylabel('Score')
    plt.title('Score Evolution')
    plt.grid(True, alpha=0.3)
    
    # Plot best score per iteration
    plt.subplot(2, 2, 2)
    best_scores = df.groupby('iteration')['score'].max()
    plt.plot(best_scores.index, best_scores.values, 'o-', linewidth=2, markersize=8)
    plt.xlabel('Iteration')
    plt.ylabel('Best Score')
    plt.title('Best Score per Iteration')
    plt.grid(True, alpha=0.3)
    
    # Plot score distribution
    plt.subplot(2, 2, 3)
    plt.hist(df['score'], bins=20, alpha=0.7, edgecolor='black')
    plt.xlabel('Score')
    plt.ylabel('Frequency')
    plt.title('Score Distribution')
    plt.grid(True, alpha=0.3)
    
    # Plot feasibility
    plt.subplot(2, 2, 4)
    feasible = (df['satisfy_dose'] & df['satisfy_gas']).sum()
    infeasible = len(df) - feasible
    plt.pie([feasible, infeasible], labels=['Feasible', 'Infeasible'], 
            autopct='%1.1f%%', startangle=90)
    plt.title('Feasibility Distribution')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_composition_analysis(df: pd.DataFrame, elements: List[str], 
                            save_path: Optional[str] = None) -> None:
    """Plot composition analysis and correlations with scores.
    
    Parameters
    ----------
    df : pd.DataFrame
        Materials DataFrame.
    elements : List[str]
        List of element symbols.
    save_path : Optional[str], optional
        Path to save the plot, by default None.
    """
    plt.figure(figsize=(15, 10))
    
    # Composition vs Score scatter plots
    n_elements = len(elements)
    n_plots = min(5, n_elements)  # Show up to 5 element plots
    
    for i, element in enumerate(elements[:n_plots]):
        plt.subplot(2, 3, i + 1)
        plt.scatter(df[f'{element}_fraction'], df['score'], alpha=0.6)
        plt.xlabel(f'{element} Fraction')
        plt.ylabel('Score')
        plt.title(f'{element} vs Score')
        plt.grid(True, alpha=0.3)
    
    # UMAP + K-means clustering visualization (if available)
    if UMAP_AVAILABLE and len(elements) >= 3:
        plt.subplot(2, 3, 6)
        
        # Prepare composition data
        composition_data = np.array([df[f'{element}_fraction'].values for element in elements]).T
        
        # Standardize the data
        scaler = StandardScaler()
        composition_scaled = scaler.fit_transform(composition_data)
        
        # Apply UMAP for dimensionality reduction
        reducer = umap.UMAP(n_components=2, random_state=42)
        embedding = reducer.fit_transform(composition_scaled)
        
        # Apply K-means clustering (using 5 clusters)
        kmeans = KMeans(n_clusters=5, random_state=42)
        clusters = kmeans.fit_predict(composition_scaled)
        
        # Create scatter plot colored by score with cluster boundaries
        scatter = plt.scatter(embedding[:, 0], embedding[:, 1], 
                            c=df['score'], s=60, alpha=0.7, cmap='viridis',
                            edgecolors='black', linewidth=0.5)
        plt.colorbar(scatter, label='Score')
        
        # Add cluster centers
        centers_2d = reducer.transform(scaler.transform(kmeans.cluster_centers_))
        plt.scatter(centers_2d[:, 0], centers_2d[:, 1], 
                   c='red', marker='x', s=200, linewidth=3, label='Cluster Centers')
        
        # Find and annotate highest scoring region
        best_idx = df['score'].idxmax()
        best_point = embedding[best_idx]
        best_material = df.iloc[best_idx]
        
        plt.annotate(f'Best: {best_material["score"]:.3f}\n' + 
                    '\n'.join([f'{el}: {best_material[f"{el}_fraction"]:.2f}' 
                              for el in elements[:3]]),
                    xy=(best_point[0], best_point[1]),
                    xytext=(10, 10), textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.3', fc='yellow', alpha=0.7),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
        
        plt.xlabel('UMAP Dimension 1')
        plt.ylabel('UMAP Dimension 2') 
        plt.title('Composition Space (UMAP + K-means)')
        plt.legend()
        plt.grid(True, alpha=0.3)
    else:
        # Fallback: simple composition correlation plot
        plt.subplot(2, 3, 6)
        if len(elements) >= 2:
            plt.scatter(df[f'{elements[0]}_fraction'], df[f'{elements[1]}_fraction'], 
                       c=df['score'], alpha=0.7, cmap='viridis')
            plt.colorbar(label='Score')
            plt.xlabel(f'{elements[0]} Fraction')
            plt.ylabel(f'{elements[1]} Fraction')
            plt.title(f'{elements[0]} vs {elements[1]} (Score)')
            plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_gas_production_analysis(df: pd.DataFrame, save_path: Optional[str] = None) -> None:
    """Plot gas production analysis.
    
    Parameters
    ----------
    df : pd.DataFrame
        Materials DataFrame.
    save_path : Optional[str], optional
        Path to save the plot, by default None.
    """
    plt.figure(figsize=(15, 10))
    
    # Gas production vs Score
    plt.subplot(2, 3, 1)
    plt.scatter(df['He_appm'], df['score'], alpha=0.6)
    plt.xlabel('He Production (appm)')
    plt.ylabel('Score')
    plt.title('He Production vs Score')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 3, 2)
    plt.scatter(df['H_appm'], df['score'], alpha=0.6)
    plt.xlabel('H Production (appm)')
    plt.ylabel('Score')
    plt.title('H Production vs Score')
    plt.grid(True, alpha=0.3)
    
    # Gas production distribution
    plt.subplot(2, 3, 3)
    plt.hist(df['He_appm'], bins=20, alpha=0.7, label='He', edgecolor='black')
    plt.hist(df['H_appm'], bins=20, alpha=0.7, label='H', edgecolor='black')
    plt.xlabel('Gas Production (appm)')
    plt.ylabel('Frequency')
    plt.title('Gas Production Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Dose rates vs Score
    plt.subplot(2, 3, 4)
    
    # Filter out zero or very small dose rates for log scale
    dose_cols = ['dose_14d', 'dose_365d', 'dose_3650d', 'dose_36500d']
    dose_labels = ['14 days', '1 year', '10 years', '100 years']
    colors = ['red', 'blue', 'green', 'orange']
    
    for dose_col, label, color in zip(dose_cols, dose_labels, colors):
        # Filter out zero or very small values
        valid_mask = df[dose_col] > 1e-10
        if valid_mask.sum() > 0:
            plt.scatter(df.loc[valid_mask, dose_col], df.loc[valid_mask, 'score'], 
                       alpha=0.6, label=label, color=color, s=30)
    
    plt.xlabel('Dose Rate (Sv/h/kg)')
    plt.ylabel('Score')
    plt.title('Dose Rates vs Score')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Set log scale only if we have valid data
    if any(df[col].max() > 1e-10 for col in dose_cols):
        plt.xscale('log')
        # Set reasonable x-axis limits
        min_dose = min(df[col][df[col] > 1e-10].min() for col in dose_cols if (df[col] > 1e-10).any())
        max_dose = max(df[col].max() for col in dose_cols)
        plt.xlim(min_dose * 0.1, max_dose * 10)
    
    # Feasibility breakdown
    plt.subplot(2, 3, 5)
    dose_feasible = df['satisfy_dose'].sum()
    gas_feasible = df['satisfy_gas'].sum()
    both_feasible = (df['satisfy_dose'] & df['satisfy_gas']).sum()
    total = len(df)
    
    labels = ['Dose Only', 'Gas Only', 'Both', 'Neither']
    sizes = [dose_feasible - both_feasible, gas_feasible - both_feasible, 
             both_feasible, total - dose_feasible - gas_feasible + both_feasible]
    
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
    plt.title('Feasibility Breakdown')
    
    # Score correlation matrix
    plt.subplot(2, 3, 6)
    correlation_cols = ['score', 'He_appm', 'H_appm', 'dose_14d', 'dose_365d', 'dose_3650d', 'dose_36500d']
    
    # Only include columns that have valid variation (not all zeros or constant)
    valid_cols = []
    for col in correlation_cols:
        if col in df.columns:
            col_data = df[col]
            # Check if column has valid variation
            if col_data.std() > 1e-12 and not col_data.isna().all():
                valid_cols.append(col)
    
    if len(valid_cols) >= 2:
        corr_matrix = df[valid_cols].corr()
        
        im = plt.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
        plt.colorbar(im)
        plt.xticks(range(len(valid_cols)), valid_cols, rotation=45)
        plt.yticks(range(len(valid_cols)), valid_cols)
        plt.title('Correlation Matrix')
        
        # Add correlation values as text
        for i in range(len(valid_cols)):
            for j in range(len(valid_cols)):
                corr_val = corr_matrix.iloc[i, j]
                if not np.isnan(corr_val):
                    plt.text(j, i, f'{corr_val:.2f}', 
                            ha='center', va='center', fontsize=8)
    else:
        plt.text(0.5, 0.5, 'Insufficient valid data\nfor correlation matrix', 
                ha='center', va='center', transform=plt.gca().transAxes,
                fontsize=12)
        plt.title('Correlation Matrix')
        plt.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def print_summary_statistics(df: pd.DataFrame, results: Dict[str, Any]) -> None:
    """Print summary statistics of the optimization results.
    
    Parameters
    ----------
    df : pd.DataFrame
        Materials DataFrame.
    results : Dict[str, Any]
        Original optimization results.
    """
    print("=" * 60)
    print("BAYESIAN OPTIMIZATION RESULTS SUMMARY")
    print("=" * 60)
    
    # Metadata
    metadata = results['metadata']
    print(f"Timestamp: {metadata['timestamp']}")
    print(f"Elements: {metadata['elements']}")
    print(f"Iterations: {metadata['n_iterations']}")
    print(f"Batch size: {metadata['batch_size']}")
    print(f"Total materials evaluated: {len(df)}")
    
    print(f"\nCritical Limits:")
    for gas, limit in metadata['critical_limits'].items():
        print(f"  {gas}: {limit} appm")
    
    print(f"\nDose Limits:")
    for days, limit in metadata['dose_limits'].items():
        print(f"  {days} days: {limit:.2e} Sv/h/kg")
    
    # Score statistics
    print(f"\nScore Statistics:")
    print(f"  Mean score: {df['score'].mean():.4f}")
    print(f"  Std score: {df['score'].std():.4f}")
    print(f"  Min score: {df['score'].min():.4f}")
    print(f"  Max score: {df['score'].max():.4f}")
    
    # Feasibility statistics
    feasible = (df['satisfy_dose'] & df['satisfy_gas']).sum()
    print(f"\nFeasibility:")
    print(f"  Feasible materials: {feasible} ({feasible/len(df)*100:.1f}%)")
    print(f"  Dose feasible: {df['satisfy_dose'].sum()} ({df['satisfy_dose'].mean()*100:.1f}%)")
    print(f"  Gas feasible: {df['satisfy_gas'].sum()} ({df['satisfy_gas'].mean()*100:.1f}%)")
    
    # Best material
    best_material = df.loc[df['score'].idxmax()]
    print(f"\nBest Material:")
    print(f"  Name: {best_material['material_name']}")
    print(f"  Score: {best_material['score']:.4f}")
    print(f"  Composition: {dict(zip(metadata['elements'], [best_material[f'{el}_fraction'] for el in metadata['elements']]))}")
    print(f"  He production: {best_material['He_appm']:.2f} appm")
    print(f"  H production: {best_material['H_appm']:.2f} appm")
    print(f"  Satisfies dose: {best_material['satisfy_dose']}")
    print(f"  Satisfies gas: {best_material['satisfy_gas']}")
    
    # Top 5 materials
    print(f"\nTop 5 Materials:")
    top_5 = df.nlargest(5, 'score')
    for i, (_, material) in enumerate(top_5.iterrows(), 1):
        print(f"  {i}. {material['material_name']}: {material['score']:.4f}")
    
    print("=" * 60)


def main():
    """Main function to run the analysis."""
    parser = argparse.ArgumentParser(description='Analyze Bayesian optimization results')
    parser.add_argument('results_file', type=str, help='Path to the JSON results file')
    parser.add_argument('--output-dir', type=str, default='analysis_output', 
                       help='Directory to save analysis plots')
    parser.add_argument('--no-plots', action='store_true', 
                       help='Skip generating plots')
    
    args = parser.parse_args()
    
    # Load results
    print(f"Loading results from: {args.results_file}")
    results = load_optimization_results(args.results_file)
    
    # Create DataFrame
    df = create_materials_dataframe(results)
    
    # Print summary
    print_summary_statistics(df, results)
    
    if not args.no_plots:
        # Create output directory
        output_dir = Path(args.output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # Generate plots
        print("\nGenerating plots...")
        
        plot_score_evolution(df, save_path=output_dir / 'score_evolution.png')
        plot_composition_analysis(df, results['metadata']['elements'], 
                                save_path=output_dir / 'composition_analysis.png')
        plot_gas_production_analysis(df, save_path=output_dir / 'gas_production_analysis.png')
        
        print(f"Plots saved to: {output_dir.absolute()}")
    
    # Save DataFrame to CSV for further analysis
    csv_path = Path(args.output_dir) / 'materials_data.csv'
    df.to_csv(csv_path, index=False)
    print(f"Materials data saved to: {csv_path.absolute()}")


if __name__ == "__main__":
    main() 