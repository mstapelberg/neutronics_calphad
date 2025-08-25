"""
Detailed flux normalization debugging tests.

This module contains comprehensive tests to debug the flux normalization issues
that are causing unrealistically high dose rates. It examines raw OpenMC tallies,
source normalization, and compares different calculation approaches.
"""
import pytest
import numpy as np
import openmc
import openmc.deplete
from pathlib import Path
import tempfile
import os
import h5py
from typing import Dict, Any, Tuple, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from _pytest.capture import CaptureFixture
    from _pytest.fixtures import FixtureRequest
    from _pytest.logging import LogCaptureFixture
    from _pytest.monkeypatch import MonkeyPatch
    from pytest_mock.plugin import MockerFixture

from neutronics_calphad.geometry_maker import create_model
from neutronics_calphad.config import SPHERICAL
from neutronics_calphad.library import E_PER_FUSION_eV, UNITS_EV_TO_J


class TestDetailedFluxDebugging:
    """Comprehensive flux normalization debugging tests."""
    
    @pytest.fixture
    def minimal_config(self) -> Dict[str, Any]:
        """Create the most minimal possible test configuration."""
        config = SPHERICAL.copy()
        
        # Use minimal particle counts for speed
        config['settings']['particles'] = 1000
        config['settings']['batches'] = 5
        config['settings']['inactive'] = 0
        
        return config
    
    @pytest.fixture
    def temp_chain_file(self) -> str:
        """Get chain file or skip test if not available."""
        chain_file = os.environ.get('OPENMC_CHAIN_FILE')
        if not chain_file:
            pytest.skip("OPENMC_CHAIN_FILE not set")
        return chain_file
    
    @pytest.mark.slow
    def test_raw_openmc_flux_tally(self, minimal_config: Dict[str, Any]) -> None:
        """Test raw OpenMC flux tally without any depletion module processing."""
        print("\n" + "="*80)
        print("DEBUGGING: Raw OpenMC Flux Tally")
        print("="*80)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            outdir = Path(tmpdir)
            
            # Create model
            model = create_model(minimal_config)
            model.settings.particles = 1000
            model.settings.batches = 5
            model.settings.run_mode = 'fixed source'
            
            # Find the vacuum vessel material and cell
            vv_material = None
            vv_cell = None
            for material in model.materials:
                if material.name == 'vcrti':
                    vv_material = material
                    break
            
            for cell in model.geometry.get_all_cells().values():
                if cell.fill == vv_material:
                    vv_cell = cell
                    break
            
            assert vv_material is not None, "vcrti material not found"
            assert vv_cell is not None, "vcrti cell not found"
            
            # Create a simple flux tally
            cell_filter = openmc.CellFilter(vv_cell)
            energy_filter = openmc.EnergyFilter.from_group_structure('CCFE-709')
            
            flux_tally = openmc.Tally(name='flux_tally')
            flux_tally.filters = [cell_filter, energy_filter]
            flux_tally.scores = ['flux']
            
            model.tallies = openmc.Tallies([flux_tally])
            
            # Run simulation
            print(f"Running OpenMC with {model.settings.particles} particles, {model.settings.batches} batches")
            statepoint_file = model.run(cwd=outdir)
            
            # Analyze results
            with openmc.StatePoint(statepoint_file) as sp:
                # Get basic simulation info
                print(f"\nSimulation Info:")
                print(f"  - Total source particles: {sp.n_particles}")
                print(f"  - Number of batches: {sp.n_batches}")
                print(f"  - Run mode: {sp.run_mode}")
                print(f"  - Source strength: {sp.source_present}")
                
                # Get the flux tally
                tally = sp.get_tally(name='flux_tally')
                flux_mean = tally.mean.flatten()
                flux_std = tally.std_dev.flatten()
                
                print(f"\nRaw Tally Analysis:")
                print(f"  - Tally shape: {tally.mean.shape}")
                print(f"  - Number of energy groups: {len(flux_mean)}")
                print(f"  - Total flux (all groups): {np.sum(flux_mean):.6e}")
                print(f"  - Max flux in any group: {np.max(flux_mean):.6e}")
                print(f"  - Min flux in any group: {np.min(flux_mean[flux_mean > 0]):.6e}")
                print(f"  - Flux per source particle: {np.sum(flux_mean) / sp.n_particles:.6e}")
                print(f"  - Average relative error: {np.mean(flux_std[flux_mean > 0] / flux_mean[flux_mean > 0]) * 100:.2f}%")
                
                # Material volume analysis
                material_volume = getattr(vv_material, 'volume', None)
                print(f"\nMaterial Volume Analysis:")
                print(f"  - Material volume: {material_volume:.2e} cm³" if material_volume else "  - Material volume: Not set")
                
                if material_volume:
                    flux_per_cm3 = flux_mean / material_volume
                    print(f"  - Total flux per cm³: {np.sum(flux_per_cm3):.6e}")
                    print(f"  - Flux per cm³ per source particle: {np.sum(flux_per_cm3) / sp.n_particles:.6e}")
                
                # Energy group analysis
                print(f"\nEnergy Group Analysis (top 10 groups by flux):")
                sorted_indices = np.argsort(flux_mean)[::-1]
                for i in range(min(10, len(flux_mean))):
                    idx = sorted_indices[i]
                    if flux_mean[idx] > 0:
                        print(f"  Group {idx:3d}: {flux_mean[idx]:.4e} ± {flux_std[idx]:.4e}")
                
                # Physics check: flux per source neutron should be << 1
                flux_per_neutron = np.sum(flux_mean) / sp.n_particles
                print(f"\nPhysics Validation:")
                print(f"  - Flux per source neutron: {flux_per_neutron:.6e}")
                if flux_per_neutron > 1.0:
                    print(f"  ❌ ISSUE: Flux per source neutron > 1 (impossible!)")
                    print(f"     This indicates a normalization or unit problem")
                elif flux_per_neutron > 0.1:
                    print(f"  ⚠️  WARNING: Flux per source neutron > 0.1 (suspicious)")
                else:
                    print(f"  ✅ OK: Flux per source neutron < 0.1 (reasonable)")
    
    @pytest.mark.slow
    def test_compare_depletion_vs_direct_tally(self, minimal_config: Dict[str, Any], temp_chain_file: str) -> None:
        """Compare flux from openmc.deplete.get_microxs_and_flux vs direct tally."""
        print("\n" + "="*80)
        print("DEBUGGING: Depletion Module vs Direct Tally Comparison")
        print("="*80)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            outdir = Path(tmpdir)
            
            model = create_model(minimal_config)
            model.settings.particles = 1000
            model.settings.batches = 5
            
            # Method 1: Direct tally (from previous test)
            vv_material = None
            vv_cell = None
            for material in model.materials:
                if material.name == 'vcrti':
                    vv_material = material
                    break
            
            for cell in model.geometry.get_all_cells().values():
                if cell.fill == vv_material:
                    vv_cell = cell
                    break
            
            # Direct tally approach
            cell_filter = openmc.CellFilter(vv_cell)
            energy_filter = openmc.EnergyFilter.from_group_structure('UKAEA-1102')
            
            flux_tally = openmc.Tally(name='direct_flux')
            flux_tally.filters = [cell_filter, energy_filter]
            flux_tally.scores = ['flux']
            
            model.tallies = openmc.Tallies([flux_tally])
            
            print(f"Running direct tally simulation...")
            statepoint_file = model.run(cwd=outdir)
            
            # Get direct tally results
            with openmc.StatePoint(statepoint_file) as sp:
                direct_tally = sp.get_tally(name='direct_flux')
                direct_flux = direct_tally.mean.flatten()
                direct_particles = sp.n_particles
            
            print(f"Direct tally results:")
            print(f"  - Total flux: {np.sum(direct_flux):.6e}")
            print(f"  - Particles: {direct_particles}")
            print(f"  - Flux per particle: {np.sum(direct_flux) / direct_particles:.6e}")
            
            # Method 2: Depletion module approach
            print(f"\nRunning depletion module approach...")
            
            # Clear tallies for clean run
            model.tallies = openmc.Tallies()
            
            # Make material depletable for depletion module
            vv_material.depletable = True
            depletable_mats = [vv_material]
            
            try:
                flux_list, microxs_list = openmc.deplete.get_microxs_and_flux(
                    model,
                    depletable_mats,
                    energies="UKAEA-1102",
                    chain_file=temp_chain_file,
                    run_kwargs={'cwd': str(outdir)}
                )
                
                depletion_flux = flux_list[0]  # First (and only) material
                
                print(f"Depletion module results:")
                print(f"  - Total flux: {np.sum(depletion_flux):.6e}")
                print(f"  - Flux shape: {depletion_flux.shape}")
                print(f"  - Flux type: {type(depletion_flux)}")
                
                # Compare the two methods
                print(f"\nComparison:")
                print(f"  - Direct tally total: {np.sum(direct_flux):.6e}")
                print(f"  - Depletion total: {np.sum(depletion_flux):.6e}")
                
                if len(direct_flux) == len(depletion_flux):
                    ratio = np.sum(depletion_flux) / np.sum(direct_flux)
                    print(f"  - Ratio (depletion/direct): {ratio:.6e}")
                    
                    if abs(ratio - 1.0) > 0.1:
                        print(f"  ❌ ISSUE: Large difference between methods!")
                        print(f"     This suggests different normalization approaches")
                    else:
                        print(f"  ✅ OK: Methods agree within 10%")
                else:
                    print(f"  ❌ ISSUE: Different energy group counts!")
                    print(f"     Direct: {len(direct_flux)}, Depletion: {len(depletion_flux)}")
                
            except Exception as e:
                print(f"❌ Depletion module failed: {e}")
                import traceback
                traceback.print_exc()
    
    def test_theoretical_flux_calculation(self, minimal_config: Dict[str, Any]) -> None:
        """Calculate theoretical flux expectations for validation."""
        print("\n" + "="*80)
        print("DEBUGGING: Theoretical Flux Calculation")
        print("="*80)
        
        # Get geometry parameters
        minor_radius = minimal_config['geometry']['layers'][0]['thickness']  # VV inner radius
        
        # Vacuum vessel geometry (spherical shell)
        inner_radius = minor_radius  # 113 cm
        outer_radius = inner_radius + 2  # thickness = 2 cm
        
        volume = (4/3) * np.pi * (outer_radius**3 - inner_radius**3)  # cm³
        surface_area = 4 * np.pi * inner_radius**2  # cm²
        
        print(f"Geometry Analysis:")
        print(f"  - Inner radius: {inner_radius:.1f} cm")
        print(f"  - Outer radius: {outer_radius:.1f} cm")
        print(f"  - VV volume: {volume:.2e} cm³")
        print(f"  - Inner surface area: {surface_area:.2e} cm²")
        
        # Theoretical flux calculation
        # For a point source at center, isotropic emission
        source_strength = 1.0  # neutrons/s per source particle
        
        # Mean chord length for sphere ≈ 4V/S
        mean_chord_length = 4 * volume / surface_area
        print(f"  - Mean chord length: {mean_chord_length:.2f} cm")
        
        # For thin shell, approximate flux
        # Flux ≈ source_strength / (4π * r²) for shell at radius r
        theoretical_flux_at_surface = source_strength / surface_area
        
        # Volume-averaged flux (rough estimate)
        # For thin shell: flux ≈ theoretical_flux_at_surface * mean_chord_length / volume
        volume_averaged_flux = theoretical_flux_at_surface * mean_chord_length
        
        print(f"\nTheoretical Expectations (per source neutron):")
        print(f"  - Flux at inner surface: {theoretical_flux_at_surface:.6e} neutrons/cm²/s")
        print(f"  - Volume-averaged flux: {volume_averaged_flux:.6e} neutrons/cm²/s")
        print(f"  - Total flux in volume: {volume_averaged_flux * volume:.6e} neutrons/s")
        
        # For realistic fusion reactor
        power = 500e6  # W
        neutron_rate = power / (E_PER_FUSION_eV * UNITS_EV_TO_J)
        actual_flux = volume_averaged_flux * neutron_rate
        
        print(f"\nScaled to 500 MW Fusion Power:")
        print(f"  - Neutron production rate: {neutron_rate:.2e} neutrons/s")
        print(f"  - Expected flux in VV: {actual_flux:.2e} neutrons/cm²/s")
        print(f"  - Typical fusion flux range: 1e14 - 1e16 neutrons/cm²/s")
        
        if 1e14 < actual_flux < 1e16:
            print(f"  ✅ Theoretical flux in expected range")
        else:
            print(f"  ⚠️  Theoretical flux outside typical range")
    
    @pytest.mark.slow
    def test_openmc_source_normalization(self, minimal_config: Dict[str, Any]) -> None:
        """Investigate how OpenMC normalizes sources and tallies."""
        print("\n" + "="*80)
        print("DEBUGGING: OpenMC Source Normalization")
        print("="*80)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            outdir = Path(tmpdir)
            
            model = create_model(minimal_config)
            model.settings.particles = 1000
            model.settings.batches = 5
            
            # Add multiple tallies to understand normalization
            # 1. Cell flux tally
            vv_material = None
            vv_cell = None
            for material in model.materials:
                if material.name == 'vcrti':
                    vv_material = material
                    break
            
            for cell in model.geometry.get_all_cells().values():
                if cell.fill == vv_material:
                    vv_cell = cell
                    break
            
            cell_filter = openmc.CellFilter(vv_cell)
            
            # Simple single-group tally
            flux_tally = openmc.Tally(name='cell_flux')
            flux_tally.filters = [cell_filter]
            flux_tally.scores = ['flux']
            
            # Current tally (to understand source strength)
            current_tally = openmc.Tally(name='current')
            current_tally.filters = [cell_filter]
            current_tally.scores = ['current']
            
            # Absorption rate
            absorption_tally = openmc.Tally(name='absorption')
            absorption_tally.filters = [cell_filter]
            absorption_tally.scores = ['absorption']
            
            model.tallies = openmc.Tallies([flux_tally, current_tally, absorption_tally])
            
            print(f"Running normalization test...")
            statepoint_file = model.run(cwd=outdir)
            
            with openmc.StatePoint(statepoint_file) as sp:
                print(f"\nSource Information:")
                print(f"  - Source particles simulated: {sp.n_particles}")
                print(f"  - Batches: {sp.n_batches}")
                print(f"  - Particles per batch: {sp.n_particles // sp.n_batches}")
                
                # Get tally results
                flux_tally = sp.get_tally(name='cell_flux')
                current_tally = sp.get_tally(name='current')
                absorption_tally = sp.get_tally(name='absorption')
                
                flux_mean = flux_tally.mean[0, 0, 0]  # Single value
                current_mean = current_tally.mean[0, 0, 0]
                absorption_mean = absorption_tally.mean[0, 0, 0]
                
                print(f"\nTally Results (single group):")
                print(f"  - Flux: {flux_mean:.6e}")
                print(f"  - Current: {current_mean:.6e}")
                print(f"  - Absorption: {absorption_mean:.6e}")
                
                # Analysis
                material_volume = getattr(vv_material, 'volume', None)
                if material_volume:
                    print(f"\nNormalization Analysis:")
                    print(f"  - Material volume: {material_volume:.2e} cm³")
                    print(f"  - Flux per source particle: {flux_mean / sp.n_particles:.6e}")
                    print(f"  - Flux per cm³ per source particle: {flux_mean / material_volume / sp.n_particles:.6e}")
                    
                    # Check if flux makes physical sense
                    flux_per_neutron = flux_mean / sp.n_particles
                    print(f"  - Flux/(source neutron): {flux_per_neutron:.6e}")
                    
                    if flux_per_neutron > 1:
                        print(f"  ❌ CRITICAL: Flux per source neutron > 1!")
                        print(f"     This is physically impossible and indicates a normalization issue")
                        
                        # Try to diagnose the issue
                        print(f"\nDiagnostic Information:")
                        print(f"  - Raw flux tally value: {flux_mean:.6e}")
                        print(f"  - Number of source particles: {sp.n_particles}")
                        print(f"  - Expected flux per neutron: << 1")
                        print(f"  - Possible issues:")
                        print(f"    1. Flux tally not normalized per source particle")
                        print(f"    2. Flux units different than expected")
                        print(f"    3. Volume normalization issue")
                        print(f"    4. OpenMC version-specific behavior")
                    else:
                        print(f"  ✅ Flux per source neutron is reasonable")
    
    @pytest.mark.slow
    def test_step_by_step_depletion_flux(self, minimal_config: Dict[str, Any], temp_chain_file: str) -> None:
        """Step through the depletion flux calculation to find the issue."""
        print("\n" + "="*80)
        print("DEBUGGING: Step-by-step Depletion Flux Calculation")
        print("="*80)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            outdir = Path(tmpdir)
            
            model = create_model(minimal_config)
            model.settings.particles = 1000
            model.settings.batches = 5
            
            # Make material depletable
            vv_material = None
            for material in model.materials:
                if material.name == 'vcrti':
                    vv_material = material
                    material.depletable = True
                    break
            
            print(f"Material setup:")
            print(f"  - Material: {vv_material.name}")
            print(f"  - Volume: {getattr(vv_material, 'volume', 'Not set')}")
            print(f"  - Density: {vv_material.density} g/cm³")
            print(f"  - Depletable: {vv_material.depletable}")
            
            # Step 1: Call get_microxs_and_flux but intercept the process
            print(f"\nStep 1: Calling openmc.deplete.get_microxs_and_flux...")
            
            try:
                # Use monkey patching or debugging to understand what happens inside
                flux_list, microxs_list = openmc.deplete.get_microxs_and_flux(
                    model,
                    [vv_material],
                    energies="UKAEA-1102",
                    chain_file=temp_chain_file,
                    run_kwargs={'cwd': str(outdir)}
                )
                
                flux = flux_list[0]
                microxs = microxs_list[0]
                
                print(f"\nStep 2: Analyzing returned flux:")
                print(f"  - Flux array shape: {flux.shape}")
                print(f"  - Flux array type: {type(flux)}")
                print(f"  - Total flux: {np.sum(flux):.6e}")
                print(f"  - Max flux: {np.max(flux):.6e}")
                print(f"  - Min flux: {np.min(flux[flux > 0]):.6e}")
                print(f"  - Number of non-zero groups: {np.sum(flux > 0)}")
                
                # Check if this looks like a per-source-neutron flux
                print(f"\nStep 3: Flux normalization check:")
                total_flux = np.sum(flux)
                print(f"  - Total flux: {total_flux:.6e}")
                
                if total_flux > 1:
                    print(f"  ❌ PROBLEM: Total flux > 1 per source neutron")
                    print(f"     This suggests flux is NOT normalized per source neutron")
                    
                    # Try to reverse-engineer the normalization
                    # Maybe it's per unit volume?
                    material_volume = getattr(vv_material, 'volume', None)
                    if material_volume:
                        flux_per_cm3 = total_flux / material_volume
                        print(f"  - If flux is per cm³: {flux_per_cm3:.6e}")
                        
                        # Try different scaling factors
                        test_particles = [1000, 5000, 10000]  # Possible particle counts
                        for n_particles in test_particles:
                            flux_per_neutron = total_flux / n_particles
                            print(f"  - If {n_particles} source particles: {flux_per_neutron:.6e} per neutron")
                
                # Step 4: Look at the MicroXS object
                print(f"\nStep 4: Analyzing MicroXS object:")
                print(f"  - Number of nuclides: {len(microxs.nuclides)}")
                print(f"  - Number of reactions: {len(microxs.reactions)}")
                print(f"  - Energy groups: {microxs.energy_groups}")
                print(f"  - Data shape: {microxs.data.shape}")
                
                # Step 5: Check what files were created
                print(f"\nStep 5: Examining output files:")
                for file in outdir.iterdir():
                    if file.is_file():
                        print(f"  - {file.name}: {file.stat().st_size} bytes")
                        
                        # Try to read statepoint if it exists
                        if file.name.endswith('.h5') and 'statepoint' in file.name:
                            try:
                                with openmc.StatePoint(file) as sp:
                                    print(f"    Statepoint info:")
                                    print(f"    - Particles: {sp.n_particles}")
                                    print(f"    - Batches: {sp.n_batches}")
                                    print(f"    - Source present: {sp.source_present}")
                            except Exception as e:
                                print(f"    Could not read statepoint: {e}")
                
            except Exception as e:
                print(f"❌ get_microxs_and_flux failed: {e}")
                import traceback
                traceback.print_exc()
    
    def test_openmc_version_and_documentation(self) -> None:
        """Check OpenMC version and document API behavior."""
        print("\n" + "="*80)
        print("DEBUGGING: OpenMC Version and API Documentation")
        print("="*80)
        
        # Check OpenMC version
        print(f"OpenMC Version Information:")
        print(f"  - Version: {openmc.__version__}")
        
        # Check the documentation/source of get_microxs_and_flux
        import inspect
        try:
            func = openmc.deplete.get_microxs_and_flux
            sig = inspect.signature(func)
            print(f"\nget_microxs_and_flux signature:")
            print(f"  {sig}")
            
            doc = inspect.getdoc(func)
            if doc:
                print(f"\nDocumentation excerpt:")
                lines = doc.split('\n')[:20]  # First 20 lines
                for line in lines:
                    print(f"  {line}")
            else:
                print(f"  No documentation found")
                
        except Exception as e:
            print(f"Could not inspect function: {e}")
        
        # Check what energy groups are available
        print(f"\nAvailable energy group structures:")
        available_groups = ['CCFE-709', 'UKAEA-1102', 'SCALE-252']
        for group in available_groups:
            try:
                boundaries = openmc.mgxs.EnergyGroups(group)
                print(f"  - {group}: {len(boundaries.boundaries)-1} groups")
            except Exception as e:
                print(f"  - {group}: Not available ({e})")
        
        # Recommend next debugging steps
        print(f"\nRecommended Next Steps:")
        print(f"1. Check OpenMC release notes for flux tally normalization changes")
        print(f"2. Test with different energy group structures")
        print(f"3. Compare with manual flux tally (not through depletion module)")
        print(f"4. Check if issue is specific to UKAEA-1102 group structure")
        print(f"5. Verify material volume calculation is correct")


@pytest.mark.integration
class TestFluxNormalizationSolutions:
    """Test potential solutions to the flux normalization problem."""
    
    def test_manual_flux_correction(self) -> None:
        """Test manual correction of flux normalization."""
        print("\n" + "="*80)
        print("TESTING: Manual Flux Correction Approaches")
        print("="*80)
        
        # Simulate the problematic flux we're seeing
        problematic_flux = np.array([5.0, 3.2, 7.1, 0.8])  # Total = 16.1 (too high)
        
        print(f"Problematic flux (per source neutron):")
        print(f"  - Individual groups: {problematic_flux}")
        print(f"  - Total: {np.sum(problematic_flux):.2f}")
        print(f"  - Issue: Total >> 1 (impossible)")
        
        # Potential correction approaches:
        
        # 1. Scale by total (assumes flux should sum to some reasonable value)
        target_total = 0.01  # Reasonable flux per source neutron
        scaling_factor_1 = target_total / np.sum(problematic_flux)
        corrected_flux_1 = problematic_flux * scaling_factor_1
        
        print(f"\nCorrection Approach 1: Scale to reasonable total")
        print(f"  - Scaling factor: {scaling_factor_1:.6f}")
        print(f"  - Corrected total: {np.sum(corrected_flux_1):.6f}")
        
        # 2. Check if it's a volume issue
        material_volume = 2.13e5  # cm³ (typical VV volume)
        flux_per_cm3 = problematic_flux / material_volume
        
        print(f"\nCorrection Approach 2: Volume normalization")
        print(f"  - Material volume: {material_volume:.2e} cm³")
        print(f"  - Flux per cm³: {np.sum(flux_per_cm3):.6e}")
        print(f"  - This could be the correct normalization")
        
        # 3. Check if it's a particle count issue
        typical_particles = [1000, 5000, 10000]
        for n_particles in typical_particles:
            flux_per_neutron = np.sum(problematic_flux) / n_particles
            print(f"\nCorrection Approach 3: Particle normalization ({n_particles} particles)")
            print(f"  - Flux per source neutron: {flux_per_neutron:.6e}")
            if flux_per_neutron < 0.1:
                print(f"  ✅ This gives reasonable results")
            else:
                print(f"  ❌ Still too high")
        
        # Test the corrected flux in dose calculation
        print(f"\nTesting corrected flux in dose calculation:")
        power = 500e6  # W
        neutron_rate = power / (17.6e6 * 1.60218e-19)
        
        # Original (problematic) flux
        original_total_flux = np.sum(problematic_flux) * neutron_rate
        print(f"  - Original scaled flux: {original_total_flux:.2e} neutrons/cm²/s")
        
        # Corrected flux
        corrected_total_flux = np.sum(corrected_flux_1) * neutron_rate
        print(f"  - Corrected scaled flux: {corrected_total_flux:.2e} neutrons/cm²/s")
        print(f"  - Typical fusion reactor flux: 1e14 - 1e16 neutrons/cm²/s")
        
        if 1e14 < corrected_total_flux < 1e16:
            print(f"  ✅ Corrected flux is in reasonable range")
        else:
            print(f"  ❌ Corrected flux still outside reasonable range")
    
    def test_alternative_flux_calculation_methods(self) -> None:
        """Test alternative methods for flux calculation."""
        print("\n" + "="*80)
        print("TESTING: Alternative Flux Calculation Methods")
        print("="*80)
        
        # Method 1: Direct tally-based approach (bypassing depletion module)
        print(f"Alternative Method 1: Direct OpenMC Tallies")
        print(f"  - Use openmc.Tally directly")
        print(f"  - Bypass openmc.deplete.get_microxs_and_flux")
        print(f"  - Full control over normalization")
        print(f"  ✅ Recommended for debugging")
        
        # Method 2: FISPACT-style flux input
        print(f"\nAlternative Method 2: Pre-calculated Flux")
        print(f"  - Use literature/FISPACT flux spectra")
        print(f"  - Scale to appropriate neutron wall loading")
        print(f"  - Bypass OpenMC flux calculation entirely")
        
        # Method 3: Simplified 1-group flux
        print(f"\nAlternative Method 3: Simplified 1-group Calculation")
        print(f"  - Calculate total neutron flux analytically")
        print(f"  - Use average cross sections")
        print(f"  - Good for validation and debugging")
        
        # Analytical 1-group example
        neutron_wall_loading = 2.0  # MW/m²
        neutron_energy = 14.1e6  # eV
        energy_per_neutron = neutron_energy * 1.60218e-19  # J
        
        flux_1group = (neutron_wall_loading * 1e6) / energy_per_neutron  # neutrons/m²/s
        flux_1group_cm2 = flux_1group / 1e4  # neutrons/cm²/s
        
        print(f"\nAnalytical 1-group calculation:")
        print(f"  - Neutron wall loading: {neutron_wall_loading} MW/m²")
        print(f"  - 14.1 MeV neutron flux: {flux_1group_cm2:.2e} neutrons/cm²/s")
        print(f"  - This is a good reference value")
        
        # Compare with typical OpenMC results
        print(f"\nComparison with problematic OpenMC results:")
        power = 500e6  # W
        neutron_rate = power / (17.6e6 * 1.60218e-19)
        problematic_flux_per_neutron = 15.1  # What we've been seeing
        problematic_total_flux = problematic_flux_per_neutron * neutron_rate
        
        ratio = problematic_total_flux / flux_1group_cm2
        print(f"  - Problematic OpenMC flux: {problematic_total_flux:.2e} neutrons/cm²/s")
        print(f"  - Analytical reference: {flux_1group_cm2:.2e} neutrons/cm²/s")
        print(f"  - Ratio (OpenMC/analytical): {ratio:.1f}x")
        
        if ratio > 10:
            print(f"  ❌ OpenMC result is {ratio:.0f}x too high")
            print(f"     Strong indication of normalization error")
        else:
            print(f"  ✅ Results are within reasonable agreement") 