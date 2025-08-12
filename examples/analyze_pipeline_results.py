"""Example script for analyzing sequential pipeline results with UMAP visualization.

This script demonstrates how to use the improved bo_results_analyzer to analyze
neutronics and CALPHAD results from the sequential materials design pipeline,
including UMAP dimensionality reduction for composition space exploration.

Usage:
    python analyze_pipeline_results.py <results_directory> [--interactive] [--no-umap]
"""

import sys
import os
from pathlib import Path
from typing import Optional

# Add the parent directory to the path to import neutronics_calphad
sys.path.insert(0, str(Path(__file__).parent.parent))

from neutronics_calphad.analysis.bo_results_analyzer import (
    load_pipeline_results,
    extract_optimization_history,
    extract_good_compositions,
    create_umap_visualization,
    analyze_convergence,
    analyze_composition_space,
    analyze_calphad_results,
    generate_analysis_report,
    quick_analysis,
    plot_composition_space_exploration,
    plot_dose_rate_analysis
)


def analyze_specific_aspects(results_dir: str, interactive: bool = False) -> None:
    """
    Perform detailed analysis of specific aspects of the pipeline results.
    
    Parameters
    ----------
    results_dir : str
        Path to the results directory.
    interactive : bool
        Whether to create interactive plots.
    """
    print(f"=== Detailed Analysis: {results_dir} ===")
    
    # Load results
    results = load_pipeline_results(results_dir)
    
    if 'neutronics' not in results:
        print("No neutronics results found!")
        return
    
    # Extract data
    df = extract_optimization_history(results['neutronics'])
    good_compositions = extract_good_compositions(results['neutronics'])
    
    print(f"Loaded {len(df)} evaluated compositions")
    print(f"Found {len(good_compositions)} predicted good compositions")
    
    # Analyze convergence
    print("\n--- Convergence Analysis ---")
    convergence = analyze_convergence(df)
    print(f"Total materials evaluated: {convergence['total_materials_evaluated']}")
    print(f"Dose success rate: {convergence['dose_success_rate']:.1%}")
    print(f"Gas success rate: {convergence['gas_success_rate']:.1%}")
    print(f"Combined success rate: {convergence['combined_success_rate']:.1%}")
    
    # Detailed dose rate analysis
    if 'dose_analysis' in convergence:
        print("\n--- Dose Rate Analysis ---")
        for dose_key, dose_data in convergence['dose_analysis'].items():
            print(f"{dose_key}:")
            print(f"  Mean: {dose_data['mean']:.2e}")
            print(f"  Min: {dose_data['min']:.2e}")
            print(f"  Success rate: {dose_data['success_rate']:.1%}")
            print(f"  Limit: {dose_data['limit']:.0e}")
    
    # Gas production analysis
    if 'gas_analysis' in convergence:
        print("\n--- Gas Production Analysis ---")
        for gas_key, gas_data in convergence['gas_analysis'].items():
            print(f"{gas_key}:")
            print(f"  Mean: {gas_data['mean']:.1f} appm")
            print(f"  Min: {gas_data['min']:.1f} appm")
            print(f"  Success rate: {gas_data['success_rate']:.1%}")
            print(f"  Limit: {gas_data['limit']:.1f} appm")
    
    # Analyze composition space
    print("\n--- Composition Space Analysis ---")
    comp_analysis = analyze_composition_space(df)
    print(f"Elements analyzed: {comp_analysis['elements']}")
    print(f"Successful compositions: {comp_analysis['n_successful']}")
    print(f"Unsuccessful compositions: {comp_analysis['n_unsuccessful']}")
    
    # Composition statistics
    print("\nComposition ranges:")
    for element, stats in comp_analysis['overall_composition_stats'].items():
        print(f"  {element}: {stats['min']:.3f} - {stats['max']:.3f} (mean: {stats['mean']:.3f})")
    
    # Analyze CALPHAD results if available
    if 'calphad' in results:
        print("\n--- CALPHAD Analysis ---")
        calphad_analysis = analyze_calphad_results(results['calphad'])
        
        if 'error' not in calphad_analysis:
            print(f"Compositions analyzed: {calphad_analysis['n_compositions_analyzed']}")
            print(f"Phase-stable compositions: {calphad_analysis['n_phase_passing']}")
            print(f"Phase success rate: {calphad_analysis['phase_success_rate']:.1%}")
            print(f"Single phase compositions: {calphad_analysis['single_phase_count']}")
            
            # Dominant phases
            print("\nDominant phases:")
            for phase, count in calphad_analysis['dominant_phase_distribution'].items():
                print(f"  {phase}: {count}")
        else:
            print(f"CALPHAD analysis error: {calphad_analysis['error']}")
    
    # Create UMAP visualization
    print("\n--- Creating UMAP Visualization ---")
    elements = comp_analysis['elements']
    
    # Create static UMAP plot
    create_umap_visualization(
        df, good_compositions, elements,
        save_path=os.path.join(results_dir, "umap_detailed.png"),
        interactive=False
    )
    
    # Create comprehensive composition space exploration
    print("Creating composition space exploration plots...")
    plot_composition_space_exploration(df, save_path=os.path.join(results_dir, "composition_space_exploration_detailed.png"))
    
    # Create dose rate analysis
    print("Creating dose rate analysis plots...")
    plot_dose_rate_analysis(df, save_path=os.path.join(results_dir, "dose_rate_analysis_detailed.png"))
    
    # Create interactive UMAP plot if requested
    if interactive:
        print("Creating interactive UMAP plot...")
        umap_fig = create_umap_visualization(
            df, good_compositions, elements,
            save_path=None,
            interactive=True
        )
        
        if umap_fig is not None:
            interactive_path = os.path.join(results_dir, "umap_interactive_detailed.html")
            umap_fig.write_html(interactive_path)
            print(f"Interactive UMAP plot saved to: {interactive_path}")


def compare_pipeline_stages(results_dir: str) -> None:
    """
    Compare results across different pipeline stages.
    
    Parameters
    ----------
    results_dir : str
        Path to the results directory.
    """
    print(f"\n=== Pipeline Stage Comparison: {results_dir} ===")
    
    # Load results
    results = load_pipeline_results(results_dir)
    
    if 'neutronics' not in results:
        print("No neutronics results found!")
        return
    
    # Extract data
    df = extract_optimization_history(results['neutronics'])
    good_compositions = extract_good_compositions(results['neutronics'])
    
    # Stage 1: Neutronics
    total_evaluated = len(df)
    neutronics_passing = sum(1 for _, row in df.iterrows() 
                           if row['satisfy_dose'] and row['satisfy_gas'])
    neutronics_success_rate = neutronics_passing / total_evaluated if total_evaluated > 0 else 0
    
    print(f"Stage 1 (Neutronics):")
    print(f"  Total evaluated: {total_evaluated}")
    print(f"  Passing compositions: {neutronics_passing}")
    print(f"  Success rate: {neutronics_success_rate:.1%}")
    print(f"  Predicted good compositions: {len(good_compositions)}")
    
    # Stage 2: CALPHAD
    if 'calphad' in results:
        calphad_analysis = analyze_calphad_results(results['calphad'])
        
        if 'error' not in calphad_analysis:
            calphad_passing = calphad_analysis['n_phase_passing']
            calphad_success_rate = calphad_analysis['phase_success_rate']
            overall_success_rate = calphad_passing / total_evaluated if total_evaluated > 0 else 0
            
            print(f"\nStage 2 (CALPHAD):")
            print(f"  Compositions analyzed: {calphad_analysis['n_compositions_analyzed']}")
            print(f"  Phase-stable compositions: {calphad_passing}")
            print(f"  Phase success rate: {calphad_success_rate:.1%}")
            print(f"  Overall pipeline success: {overall_success_rate:.1%}")
            
            # Pipeline efficiency
            efficiency = (calphad_passing / neutronics_passing) if neutronics_passing > 0 else 0
            print(f"  Pipeline efficiency (CALPHAD/Neutronics): {efficiency:.1%}")
    
    # Pipeline state
    if 'pipeline_state' in results:
        state = results['pipeline_state']
        print(f"\nPipeline State:")
        print(f"  Current stage: {state.get('current_stage', 'unknown')}")
        print(f"  Neutronics completed: {state.get('neutronics_completed', False)}")
        print(f"  CALPHAD completed: {state.get('calphad_completed', False)}")
        print(f"  Ductility completed: {state.get('ductility_completed', False)}")


def main():
    """Main function to run the analysis."""
    if len(sys.argv) < 2:
        print("Usage: python analyze_pipeline_results.py <results_directory> [--interactive] [--detailed]")
        print("Example: python analyze_pipeline_results.py sequential_materials_pipeline_run_1 --interactive --detailed")
        return
    
    results_dir = sys.argv[1]
    interactive = '--interactive' in sys.argv
    detailed = '--detailed' in sys.argv
    
    if not os.path.exists(results_dir):
        print(f"Results directory not found: {results_dir}")
        return
    
    print("=== Pipeline Results Analysis ===")
    print(f"Results directory: {results_dir}")
    print(f"Interactive plots: {interactive}")
    print(f"Detailed analysis: {detailed}")
    
    # Quick analysis
    print("\n" + "="*50)
    quick_analysis(results_dir, create_umap=True, interactive=interactive)
    
    # Detailed analysis if requested
    if detailed:
        print("\n" + "="*50)
        analyze_specific_aspects(results_dir, interactive=interactive)
        compare_pipeline_stages(results_dir)
    
    print("\n" + "="*50)
    print("Analysis complete!")
    print(f"Check the results directory for generated plots and reports.")


if __name__ == "__main__":
    main() 