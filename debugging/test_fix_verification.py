#!/usr/bin/env python3
"""
Test script to verify that the time indexing fix resolves the 10-year dose rate issue.
"""

import sys
sys.path.append('.')

from neutronics_calphad.library import run_element
from pathlib import Path
import os

def test_fix(use_reduced_chain=True, mpi_args=None):
    """Test the fix with a simple single-element run."""
    
    chain_type = "reduced" if use_reduced_chain else "full"
    mpi_info = f" with MPI {mpi_args}" if mpi_args else " (serial)"
    print(f"🧪 Testing Time Indexing Fix ({chain_type} chain{mpi_info})")
    print("="*60)
    
    element = 'T'
    outdir = 'test_fix_verification'
    
    # Clean up any previous test
    if os.path.exists(outdir):
        import shutil
        shutil.rmtree(outdir)
    
    print(f"Testing element: {element}")
    print(f"Output directory: {outdir}")
    print(f"Chain type: {chain_type}")
    
    try:
        # Run with just a few time steps to test quickly
        run_element(
            element=element,
            outdir=outdir,
            full_chain_file=Path.home() / 'nuclear_data' / 'chain-endf-b8.0.xml',
            cross_sections=Path.home() / 'nuclear_data' / 'cross_sections.xml',
            use_reduced_chain=use_reduced_chain,
            mpi_args=mpi_args
        )
        
        # Check if the 10-year issue is resolved
        import h5py
        h5_file = os.path.join(outdir, element, f"{element}.h5")
        
        if os.path.exists(h5_file):
            with h5py.File(h5_file, 'r') as f:
                times = f['dose_times'][:]
                doses = f['dose'][:]
                
                print(f"\n✅ Results Summary:")
                print(f"{'Time (years)':<15} {'Dose (µSv/h)':<15} {'Status':<10}")
                print("-" * 45)
                
                # Focus on the critical time range
                for i, (t, d) in enumerate(zip(times, doses)):
                    time_years = t / (365.25 * 24 * 3600)
                    if 4 < time_years < 26:  # Focus on 5-25 year range
                        status = "❌ ZERO!" if d == 0 else "✅ OK"
                        print(f"{time_years:<15.1f} {d:<15.2e} {status:<10}")
                        
                        # Special check for 10 years
                        if 9.5 < time_years < 10.5:
                            if d > 0:
                                print(f"🎉 SUCCESS: 10-year dose rate is non-zero: {d:.2e} µSv/h")
                            else:
                                print(f"❌ FAILED: 10-year dose rate is still zero")
        else:
            print(f"❌ Output file not created: {h5_file}")
            
    except Exception as e:
        print(f"❌ Error during test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    import sys
    
    # Parse command line arguments
    use_reduced = True
    mpi_args = None
    
    for i, arg in enumerate(sys.argv[1:], 1):
        if arg.lower() in ['full', 'f']:
            use_reduced = False
        elif arg.startswith('-n') or arg.startswith('--np'):
            # Extract number of processes: -n8, -n 8, --np=8, etc.
            if '=' in arg:
                n_procs = arg.split('=')[1]
            elif len(arg) > 2 and arg[2:].isdigit():
                n_procs = arg[2:]
            elif i < len(sys.argv) - 1:
                n_procs = sys.argv[i + 1]
            else:
                n_procs = '4'  # default
            
            mpi_args = ['mpiexec', '-n', str(n_procs)]
    
    test_fix(use_reduced_chain=use_reduced, mpi_args=mpi_args)
    
    print(f"\n💡 Usage examples:")
    print(f"   python {sys.argv[0]}              # reduced chain, serial")
    print(f"   python {sys.argv[0]} full         # full chain, serial") 
    print(f"   python {sys.argv[0]} -n8          # reduced chain, 8 MPI processes")
    print(f"   python {sys.argv[0]} full -n4     # full chain, 4 MPI processes") 