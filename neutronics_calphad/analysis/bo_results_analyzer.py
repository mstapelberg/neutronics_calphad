"""Analysis utilities for Bayesian optimization results from the sequential pipeline."""

import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import seaborn as sns


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
        score, satisfy_dose, satisfy_gas, dose_rates, gas_production.
    """
    records = []
    
    for iteration_data in neutronics_results['iterations']:
        iteration = iteration_data['iteration']
        
        for material in iteration_data['materials']:
            record = {
                'iteration': iteration,
                'material_name': material['material_name'],
                'score': material['score'],
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
            
            records.append(record)
    
    return pd.DataFrame(records)


def analyze_convergence(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyze optimization convergence.
    
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
    
    # Best score so far
    df['best_score_so_far'] = df['score'].cummax()
    
    # Success rate by iteration
    success_rate_by_iter = df.groupby('iteration').agg({
        'satisfy_dose': 'mean',
        'satisfy_gas': 'mean',
        'score': ['mean', 'max', 'std']
    }).round(3)
    
    # Overall statistics
    total_materials = len(df)
    dose_success_rate = df['satisfy_dose'].mean()
    gas_success_rate = df['satisfy_gas'].mean()
    combined_success_rate = (df['satisfy_dose'] & df['satisfy_gas']).mean()
    
    # Find best compositions
    best_overall = df.loc[df['score'].idxmax()]
    best_dose = df[df['satisfy_dose']].loc[df[df['satisfy_dose']]['score'].idxmax()] if df['satisfy_dose'].any() else None
    best_gas = df[df['satisfy_gas']].loc[df[df['satisfy_gas']]['score'].idxmax()] if df['satisfy_gas'].any() else None
    
    return {
        'total_materials_evaluated': total_materials,
        'dose_success_rate': dose_success_rate,
        'gas_success_rate': gas_success_rate,
        'combined_success_rate': combined_success_rate,
        'final_best_score': df['score'].max(),
        'success_rate_by_iteration': success_rate_by_iter.to_dict(),
        'best_compositions': {
            'overall': best_overall.to_dict() if best_overall is not None else None,
            'best_dose': best_dose.to_dict() if best_dose is not None else None,
            'best_gas': best_gas.to_dict() if best_gas is not None else None
        }
    }


def plot_optimization_progress(df: pd.DataFrame, save_path: Optional[str] = None) -> None:
    """
    Plot optimization progress over iterations.
    
    Parameters
    ----------
    df : pd.DataFrame
        Optimization history DataFrame.
    save_path : Path, optional
        Path to save the plot.
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Score evolution
    iteration_stats = df.groupby('iteration').agg({
        'score': ['mean', 'max', 'min'],
        'satisfy_dose': 'mean',
        'satisfy_gas': 'mean'
    })
    
    ax1 = axes[0, 0]
    iterations = iteration_stats.index
    ax1.plot(iterations, iteration_stats[('score', 'max')], 'b-', label='Best score', linewidth=2)
    ax1.plot(iterations, iteration_stats[('score', 'mean')], 'g--', label='Mean score', linewidth=1)
    ax1.fill_between(iterations, 
                     iteration_stats[('score', 'min')], 
                     iteration_stats[('score', 'max')], 
                     alpha=0.3, color='blue')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Score')
    ax1.set_title('Score Evolution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Success rates
    ax2 = axes[0, 1]
    ax2.plot(iterations, iteration_stats[('satisfy_dose', 'mean')], 'r-o', label='Dose success rate', markersize=4)
    ax2.plot(iterations, iteration_stats[('satisfy_gas', 'mean')], 'b-s', label='Gas success rate', markersize=4)
    combined_success = df.groupby('iteration').apply(lambda x: (x['satisfy_dose'] & x['satisfy_gas']).mean())
    ax2.plot(iterations, combined_success, 'g-^', label='Combined success rate', markersize=4)
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Success Rate')
    ax2.set_title('Constraint Satisfaction Rates')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1.1)
    
    # Plot 3: Composition space exploration (V vs Cr)
    ax3 = axes[1, 0]
    scatter = ax3.scatter(df['comp_V'], df['comp_Cr'], 
                         c=df['score'], cmap='viridis', 
                         alpha=0.7, s=30)
    ax3.set_xlabel('V fraction')
    ax3.set_ylabel('Cr fraction')
    ax3.set_title('Composition Space Exploration (V vs Cr)')
    plt.colorbar(scatter, ax=ax3, label='Score')
    
    # Plot 4: Dose vs Gas constraints
    ax4 = axes[1, 1]
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
    
    ax4.scatter(df[dose_col], df[gas_col], c=colors, alpha=0.7, s=30)
    ax4.set_xlabel(dose_col.replace('_', ' ').title())
    ax4.set_ylabel(gas_col.replace('_', ' ').title())
    ax4.set_title('Dose vs Gas Production')
    ax4.set_yscale('log')
    ax4.set_xscale('log')
    
    # Add legend for colors
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green', label='Both satisfied'),
        Patch(facecolor='orange', label='Dose only'),
        Patch(facecolor='blue', label='Gas only'),
        Patch(facecolor='red', label='Neither satisfied')
    ]
    ax4.legend(handles=legend_elements, loc='best')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
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
    save_path : Path, optional
        Path to save the plot.
    """
    # Select relevant columns
    comp_cols = [col for col in df.columns if col.startswith('comp_')]
    performance_cols = ['score', 'satisfy_dose', 'satisfy_gas']
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
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Heatmap saved to {save_path}")
    
    plt.show()


def generate_analysis_report(results_dir: str, output_file: Optional[str] = None) -> Dict[str, Any]:
    """
    Generate a comprehensive analysis report.
    
    Parameters
    ----------
    results_dir : Path
        Directory containing pipeline results.
    output_file : Path, optional
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
            'elements_analyzed': composition_analysis.get('elements', [])
        }
    }
    
    # Add CALPHAD analysis if available
    if 'calphad' in results:
        calphad_data = results['calphad']
        n_calphad_compositions = len(calphad_data.get('compositions', []))
        n_phase_passing = sum(1 for analysis in calphad_data.get('phase_analysis', [])
                             if analysis.get('satisfies_phase_limits', False))
        
        report['calphad_analysis'] = {
            'compositions_analyzed': n_calphad_compositions,
            'phase_passing_compositions': n_phase_passing,
            'phase_success_rate': n_phase_passing / n_calphad_compositions if n_calphad_compositions > 0 else 0,
            'overall_pipeline_success_rate': n_phase_passing / len(df) if len(df) > 0 else 0
        }
    
    # Save report
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        print(f"Analysis report saved to {output_file}")
    
    return report


# Example usage functions
def quick_analysis(results_dir: str) -> None:
    """
    Perform a quick analysis of pipeline results.
    
    Parameters
    ----------
    results_dir : str
        Path to results directory.
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
    print(f"  Best score achieved: {conv['final_best_score']:.3f}")
    
    print(f"\nComposition Space:")
    print(f"  Elements: {comp['elements']}")
    print(f"  Successful compositions: {comp['n_successful']}")
    print(f"  Unsuccessful compositions: {comp['n_unsuccessful']}")
    
    if 'calphad_analysis' in report:
        calphad = report['calphad_analysis']
        print(f"\nCALPHAD Analysis:")
        print(f"  Compositions analyzed: {calphad['compositions_analyzed']}")
        print(f"  Phase-stable compositions: {calphad['phase_passing_compositions']}")
        print(f"  Overall pipeline success: {calphad['overall_pipeline_success_rate']:.1%}")
    
    # Create plots
    results = load_pipeline_results(results_path)
    df = extract_optimization_history(results['neutronics'])
    
    print(f"\nGenerating plots...")
    plot_optimization_progress(df, save_path=os.path.join(results_path, "optimization_progress.png"))
    create_composition_heatmap(df, save_path=os.path.join(results_path, "composition_heatmap.png"))
    
    print(f"Analysis complete!")


if __name__ == "__main__":
    # Example usage
    import sys
    if len(sys.argv) > 1:
        quick_analysis(sys.argv[1])
    else:
        print("Usage: python bo_results_analyzer.py <results_directory>")
        print("Example: python bo_results_analyzer.py sequential_materials_pipeline") 