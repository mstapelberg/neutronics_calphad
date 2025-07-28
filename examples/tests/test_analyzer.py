"""Test script to verify the analyzer works with existing pipeline results."""

import sys
import os
from pathlib import Path

# Add the parent directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_analyzer():
    """Test the analyzer with existing results."""
    
    # Check if results directory exists
    results_dir = "sequential_materials_pipeline_run_1"
    
    if not os.path.exists(results_dir):
        print(f"Results directory not found: {results_dir}")
        print("Please run the pipeline first or update the path.")
        return False
    
    print(f"Testing analyzer with results from: {results_dir}")
    
    try:
        # Import the analyzer
        from neutronics_calphad.analysis.bo_results_analyzer import (
            load_pipeline_results,
            extract_optimization_history,
            extract_good_compositions
        )
        
        # Load results
        print("Loading pipeline results...")
        results = load_pipeline_results(results_dir)
        
        if 'neutronics' not in results:
            print("No neutronics results found!")
            return False
        
        # Extract data
        print("Extracting optimization history...")
        df = extract_optimization_history(results['neutronics'])
        
        print("Extracting good compositions...")
        good_compositions = extract_good_compositions(results['neutronics'])
        
        print(f"✓ Successfully loaded {len(df)} evaluated compositions")
        print(f"✓ Found {len(good_compositions)} predicted good compositions")
        
        # Test basic statistics
        if len(df) > 0:
            print(f"✓ Iterations: {df['iteration'].min()} to {df['iteration'].max()}")
            print(f"✓ Elements: {[col.replace('comp_', '') for col in df.columns if col.startswith('comp_')]}")
            
            # Test success rates
            dose_success = df['satisfy_dose'].mean()
            gas_success = df['satisfy_gas'].mean()
            combined_success = (df['satisfy_dose'] & df['satisfy_gas']).mean()
            
            print(f"✓ Dose success rate: {dose_success:.1%}")
            print(f"✓ Gas success rate: {gas_success:.1%}")
            print(f"✓ Combined success rate: {combined_success:.1%}")
        
        # Test CALPHAD results if available
        if 'calphad' in results:
            print("✓ CALPHAD results found")
            calphad_data = results['calphad']
            if 'phase_analysis' in calphad_data:
                n_phase_passing = sum(1 for analysis in calphad_data['phase_analysis'] 
                                    if analysis.get('satisfies_phase_limits', False))
                print(f"✓ Phase-stable compositions: {n_phase_passing}")
        
        # Test UMAP availability
        try:
            import umap
            import sklearn
            print("✓ UMAP and sklearn available for visualization")
        except ImportError:
            print("⚠ UMAP or sklearn not available - visualization will be limited")
        
        # Test plotly availability
        try:
            import plotly
            print("✓ Plotly available for interactive plots")
        except ImportError:
            print("⚠ Plotly not available - interactive plots will be skipped")
        
        print("\n✓ All tests passed! The analyzer is working correctly.")
        return True
        
    except Exception as e:
        print(f"✗ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_quick_analysis():
    """Test the quick analysis function."""
    
    results_dir = "sequential_materials_pipeline_run_1"
    
    if not os.path.exists(results_dir):
        print(f"Results directory not found: {results_dir}")
        return False
    
    try:
        from neutronics_calphad.analysis.bo_results_analyzer import quick_analysis
        
        print("\nTesting quick analysis...")
        quick_analysis(results_dir, create_umap=True, interactive=False)
        
        print("✓ Quick analysis completed successfully!")
        return True
        
    except Exception as e:
        print(f"✗ Error during quick analysis: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("=== Testing Pipeline Results Analyzer ===\n")
    
    # Test basic functionality
    basic_test = test_analyzer()
    
    if basic_test:
        # Test quick analysis
        quick_test = test_quick_analysis()
        
        if quick_test:
            print("\n🎉 All tests passed! You can now use the analyzer.")
            print("\nNext steps:")
            print("1. Run: python examples/analyze_pipeline_results.py sequential_materials_pipeline_run_1")
            print("2. For interactive plots: python examples/analyze_pipeline_results.py sequential_materials_pipeline_run_1 --interactive")
            print("3. For detailed analysis: python examples/analyze_pipeline_results.py sequential_materials_pipeline_run_1 --detailed")
        else:
            print("\n❌ Quick analysis test failed.")
    else:
        print("\n❌ Basic functionality test failed.") 