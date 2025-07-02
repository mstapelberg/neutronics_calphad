#!/usr/bin/env python3
"""
Pre-build checklist to ensure everything is ready for neutronics-calphad build-library.
"""

import os
from pathlib import Path

def run_checklist():
    """Run through all pre-build checks."""
    
    print("📋 Pre-Build Checklist for neutronics-calphad")
    print("="*60)
    
    checks = []
    
    # 1. Check nuclear data files
    print("\n1️⃣ Checking Nuclear Data Files...")
    cross_sections = Path.home() / 'nuclear_data' / 'cross_sections.xml'
    chain_file = Path.home() / 'nuclear_data' / 'chain-endf-b8.0.xml'
    
    if cross_sections.exists():
        print(f"  ✅ Cross sections found: {cross_sections}")
        checks.append(True)
    else:
        print(f"  ❌ Cross sections NOT FOUND: {cross_sections}")
        checks.append(False)
    
    if chain_file.exists():
        print(f"  ✅ Chain file found: {chain_file}")
        checks.append(True)
    else:
        print(f"  ❌ Chain file NOT FOUND: {chain_file}")
        checks.append(False)
    
    # 2. Check OpenMC environment
    print("\n2️⃣ Checking OpenMC Environment...")
    try:
        import openmc
        print(f"  ✅ OpenMC version: {openmc.__version__}")
        checks.append(True)
        
        # Check for parallel capability
        try:
            mpi_procs = int(os.environ.get('OMP_NUM_THREADS', 1))
            print(f"  ℹ️  OMP_NUM_THREADS: {mpi_procs}")
        except:
            print(f"  ℹ️  OMP_NUM_THREADS not set (will use single thread)")
            
    except ImportError:
        print(f"  ❌ OpenMC not found!")
        checks.append(False)
    
    # 3. Check geometry fix
    print("\n3️⃣ Checking Geometry Fix...")
    try:
        from neutronics_calphad.geometry_maker import create_model
        model = create_model('V')
        
        # Check boundary conditions
        surfaces = model.geometry.get_all_surfaces()
        vacuum_count = 0
        reflective_count = 0
        
        for surf in surfaces.values():
            if hasattr(surf, 'boundary_type'):
                if surf.boundary_type == 'vacuum':
                    vacuum_count += 1
                elif surf.boundary_type == 'reflective':
                    reflective_count += 1
        
        print(f"  ✅ Geometry has {vacuum_count} vacuum boundaries")
        print(f"  ✅ Geometry has {reflective_count} reflective boundaries")
        
        if vacuum_count >= 4 and reflective_count == 2:
            print(f"  ✅ Boundary conditions look correct")
            checks.append(True)
        else:
            print(f"  ⚠️  Unexpected boundary configuration")
            checks.append(False)
            
    except Exception as e:
        print(f"  ❌ Error checking geometry: {e}")
        checks.append(False)
    
    # 4. Simulation parameters
    print("\n4️⃣ Current Simulation Parameters:")
    print(f"  - Neutron batches: 1 (for depletion)")
    print(f"  - Neutrons per batch: 10,000")
    print(f"  - Photon batches: 10")  
    print(f"  - Photons per batch: 10,000")
    print(f"  - Irradiation time: 365 days")
    print(f"  - Source rate: 1e20 n/s")
    print(f"  - Cooling times: 8 steps (1s to 100y)")
    
    # 5. Time estimate
    print("\n5️⃣ Time Estimate:")
    print(f"  - Elements to run: 5 (V, Cr, Ti, W, Zr)")
    print(f"  - Estimated time per element: ~15-30 minutes")
    print(f"  - Total estimated time: 1.5-2.5 hours")
    print(f"  💡 Consider running overnight or using tmux/screen")
    
    # 6. Disk space
    print("\n6️⃣ Disk Space Check:")
    import shutil
    stat = shutil.disk_usage('.')
    free_gb = stat.free / (1024**3)
    print(f"  - Free disk space: {free_gb:.1f} GB")
    if free_gb > 10:
        print(f"  ✅ Sufficient disk space")
        checks.append(True)
    else:
        print(f"  ⚠️  Low disk space - need ~5-10 GB")
        checks.append(False)
    
    # Summary
    print("\n" + "="*60)
    print("📊 SUMMARY:")
    
    if all(checks):
        print("✅ ALL CHECKS PASSED - Ready to run build-library!")
        print("\n🚀 Next steps:")
        print("  1. (Optional) Set OMP_NUM_THREADS for parallel:")
        print("     export OMP_NUM_THREADS=8")
        print("  2. Run the build command:")
        print("     neutronics-calphad build-library")
        print("  3. Or run in background:")
        print("     nohup neutronics-calphad build-library > build.log 2>&1 &")
    else:
        print("❌ Some checks failed - please fix before running")
        print("\n🔧 Common fixes:")
        print("  - Download nuclear data if missing")
        print("  - Install/update OpenMC")
        print("  - Free up disk space")
        
    # Optional recommendations
    print("\n💡 Optional Optimizations:")
    print("  - Increase particles for better statistics (edit library.py)")
    print("  - Reduce cooling times if not all needed")
    print("  - Run single element test first:")
    print("    python test_single_element.py")

if __name__ == "__main__":
    run_checklist() 