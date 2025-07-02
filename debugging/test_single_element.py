#!/usr/bin/env python3
"""
Quick test of the full R2S workflow for a single element.
"""

from pathlib import Path
import sys
sys.path.append('.')

from neutronics_calphad.library import run_element
import os

def test_single_element():
    """Test the complete workflow for Vanadium."""
    
    print("🧪 Testing Single Element R2S Workflow")
    print("="*40)
    
    # Test with Vanadium
    element = 'V'
    outdir = 'test_single_element'
    
    print(f"Testing element: {element}")
    print(f"Output directory: {outdir}")
    
    # Clean up any previous test
    if os.path.exists(outdir):
        import shutil
        shutil.rmtree(outdir)
    
    # Run the full R2S workflow
    try:
        run_element(
            element=element,
            outdir=outdir,
            full_chain_file=Path.home() / 'nuclear_data' / 'chain-endf-b8.0.xml',
            cross_sections=Path.home() / 'nuclear_data' / 'cross_sections.xml'
        )
        
        # Check if output was created
        h5_file = os.path.join(outdir, element, f"{element}.h5")
        if os.path.exists(h5_file):
            import h5py
            with h5py.File(h5_file, 'r') as f:
                print(f"\n✅ Output file created successfully!")
                print(f"   Contents: {list(f.keys())}")
                
                # Check dose data
                if 'dose' in f:
                    dose_data = f['dose'][...]
                    print(f"   Dose data shape: {dose_data.shape}")
                    print(f"   Dose range: {dose_data.min():.2e} - {dose_data.max():.2e}")
                
                # Check gas data
                for gas in ['He3', 'He4', 'H1', 'H2', 'H3']:
                    if f'gas/{gas}' in f:
                        gas_data = f[f'gas/{gas}'][...]
                        print(f"   {gas} production: {gas_data:.2e} atoms")
        else:
            print(f"❌ Output file not created: {h5_file}")
            
    except Exception as e:
        print(f"❌ Error during test: {e}")
        import traceback
        traceback.print_exc()
        
    print("\n💡 Next Steps:")
    print("  If test passed: Run 'neutronics-calphad build-library'")
    print("  If test failed: Debug the specific error before full run")

if __name__ == "__main__":
    test_single_element() 