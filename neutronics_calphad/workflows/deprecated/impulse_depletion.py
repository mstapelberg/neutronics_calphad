"""Linear impulse response library for O(1) depletion calculations.

This module implements a linear superposition approach to neutronics depletion,
enabling microsecond-scale predictions instead of seconds-scale ODE integration.
The approach:
1. Pre-compute impulse responses for single elements at 1 at% in V base
2. For new compositions, linearly combine impulse responses
3. Fall back to full depletion only for borderline cases

The linearity assumption is valid for:
- Most dose rates (gamma emission is additive)
- Gas production at moderate concentrations
- First-order neutron activation

It may break for:
- High concentrations where spectrum hardening matters
- Strong resonance overlap between elements
- Significant (n,2n) threshold effects
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple

import h5py  # type: ignore
import numpy as np  # type: ignore
import openmc  # type: ignore
import openmc.deplete  # type: ignore

from neutronics_calphad.neutronics.dose import contact_dose
from neutronics_calphad.optimizer.parsers import parse_openmc_results
from neutronics_calphad.utils.visualization import _load_energy_boundaries  # type: ignore


@dataclass
class ImpulseResponse:
    """Single-element impulse response data.
    
    Attributes:
        element: Element symbol (e.g., "Cr", "Ti")
        concentration: Reference concentration (typically 0.01 for 1 at%)
        times: Time points in seconds
        inventories: Dict mapping nuclide names to activity arrays [Bq]
        dose_rates: Dose rates at each time point [Sv/h]
        gas_production: Final gas production after irradiation [appm]
        metadata: Additional metadata (flux, power, schedule, etc.)
    """
    element: str
    concentration: float
    times: np.ndarray
    inventories: Dict[str, np.ndarray]
    dose_rates: np.ndarray
    gas_production: Dict[str, float]
    metadata: Dict[str, Union[str, float, int]] = field(default_factory=dict)
    
    def save(self, filepath: Union[str, Path]) -> None:
        """Save impulse response to HDF5 file."""
        with h5py.File(filepath, 'w') as f:
            f.attrs['element'] = self.element
            f.attrs['concentration'] = self.concentration
            f.attrs['metadata'] = json.dumps(self.metadata)
            
            f.create_dataset('times', data=self.times)
            f.create_dataset('dose_rates', data=self.dose_rates)
            
            # Store inventories
            inv_group = f.create_group('inventories')
            for nuclide, activities in self.inventories.items():
                inv_group.create_dataset(nuclide, data=activities)
            
            # Store gas production
            gas_group = f.create_group('gas_production')
            for gas, value in self.gas_production.items():
                gas_group.attrs[gas] = value
    
    @classmethod
    def load(cls, filepath: Union[str, Path]) -> ImpulseResponse:
        """Load impulse response from HDF5 file."""
        with h5py.File(filepath, 'r') as f:
            element = f.attrs['element']
            concentration = f.attrs['concentration']
            metadata = json.loads(f.attrs.get('metadata', '{}'))
            
            times = f['times'][:]
            dose_rates = f['dose_rates'][:]
            
            # Load inventories
            inventories = {}
            if 'inventories' in f:
                for nuclide in f['inventories']:
                    inventories[nuclide] = f['inventories'][nuclide][:]
            
            # Load gas production
            gas_production = {}
            if 'gas_production' in f:
                for gas in f['gas_production'].attrs:
                    gas_production[gas] = f['gas_production'].attrs[gas]
            
            return cls(
                element=element,
                concentration=concentration,
                times=times,
                inventories=inventories,
                dose_rates=dose_rates,
                gas_production=gas_production,
                metadata=metadata
            )


class ImpulseLibrary:
    """Library of pre-computed impulse responses for linear depletion synthesis.
    
    This class manages a collection of single-element impulse responses and
    provides methods for linear combination to predict depletion results for
    arbitrary compositions.
    """
    
    def __init__(self, library_dir: Union[str, Path]):
        """Initialize library from directory of impulse response files.
        
        Args:
            library_dir: Directory containing impulse response HDF5 files
        """
        self.library_dir = Path(library_dir)
        # Per-element impulses keyed by concentration (atomic fraction)
        # Example: self.responses['W'][0.01] -> ImpulseResponse at 1 at%
        self.responses: Dict[str, Dict[float, ImpulseResponse]] = {}
        # Baseline (pure V) response used as 0th-order term
        self.baseline: Optional[ImpulseResponse] = None
        # Global metadata for the library
        self.metadata: Dict[str, Union[str, float, int]] = {}
        # Optional pairwise corrections loaded from disk
        self.pairwise_corrections: Dict[str, Dict[str, float]] = {}
        
        # Load all impulse responses
        if self.library_dir.exists():
            self._load_library()
    
    def _load_library(self) -> None:
        """Load all impulse response files from library directory."""
        for filepath in self.library_dir.glob("impulse_*.h5"):
            try:
                response = ImpulseResponse.load(filepath)
                # Identify baseline
                is_baseline = bool(response.metadata.get('baseline', False))
                if is_baseline or (response.element == 'V' and float(response.concentration) >= 0.999):
                    self.baseline = response
                    continue
                # Register per-element per-concentration
                self.responses.setdefault(response.element, {})[float(response.concentration)] = response
            except Exception as e:
                print(f"Warning: Failed to load {filepath}: {e}")
        
        # Load library metadata if it exists
        metadata_file = self.library_dir / "library_metadata.json"
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                self.metadata = json.load(f)
        # Load pairwise corrections if present
        pair_file = self.library_dir / "pairwise_corrections.json"
        if pair_file.exists():
            try:
                with open(pair_file, 'r') as f:
                    self.pairwise_corrections = json.load(f)
            except Exception as e:
                print(f"Warning: Failed to load pairwise corrections: {e}")
    
    def add_response(self, response: ImpulseResponse) -> None:
        """Add a new impulse response to the library.
        
        Args:
            response: ImpulseResponse to add
        """
        # Baseline stored separately
        is_baseline = bool(response.metadata.get('baseline', False))
        if is_baseline or (response.element == 'V' and float(response.concentration) >= 0.999):
            self.baseline = response
        else:
            self.responses.setdefault(response.element, {})[float(response.concentration)] = response
        
        # Save to disk
        # Use element and concentration in filename to support piecewise
        tag = 'baseline' if is_baseline else f"{response.element}_{response.concentration:.5f}"
        filepath = self.library_dir / f"impulse_{tag}.h5"
        response.save(filepath)
    
    def synthesize(
        self,
        composition: Dict[str, float],
        cooling_days: Optional[List[int]] = None,
        disable_pairwise: bool = False
    ) -> Dict[str, Union[np.ndarray, Dict[str, float]]]:
        """Synthesize depletion results using linear combination of impulses.
        
        Args:
            composition: Dict mapping element symbols to atomic fractions
            cooling_days: List of cooling days for dose rate extraction
            
        Returns:
            Dict containing:
                - times: Time array in seconds
                - dose_rates: Synthesized dose rates [Sv/h]
                - gas_production: Synthesized gas production [appm]
                - dose_at_cooling_times: Dict mapping days to dose rates
                - method: "impulse"
        """
        if not self.responses and self.baseline is None:
            raise ValueError("No impulse responses loaded (including baseline)")
        
        # Validate composition
        total = sum(composition.values())
        if not np.isclose(total, 1.0):
            raise ValueError(f"Composition must sum to 1.0, got {total}")
        
        # Get reference times from baseline if available else first element
        if self.baseline is not None:
            times = self.baseline.times.copy()
        else:
            first_elem = next(iter(self.responses.values()))
            ref_resp = next(iter(first_elem.values()))
            times = ref_resp.times.copy()
        
        # Initialize outputs
        if self.baseline is not None:
            # Start with baseline scaled by V fraction
            x_v = float(composition.get('V', 0.0))
            dose_rates = x_v * self.baseline.dose_rates.copy()
            gas_production = {gas: x_v * val for gas, val in self.baseline.gas_production.items()}
        else:
            # Fallback if no baseline
            first_elem = next(iter(self.responses.values()))
            ref_resp = next(iter(first_elem.values()))
            dose_rates = np.zeros_like(ref_resp.dose_rates)
            gas_production = {gas: 0.0 for gas in ref_resp.gas_production}
        
        # Helper: interpolate per-fraction response for an element using piecewise impulses
        def _interp_per_fraction(element: str, target_frac: float) -> Tuple[np.ndarray, Dict[str, float]]:
            """Return per-fraction dose_rates array and gas dict at target fraction.

            Uses available impulses at multiple concentrations and baseline to
            compute per-fraction contributions: (mix - (1-c0)*baseline)/c0.
            Interpolates in concentration space; for dose rates, interpolation
            occurs in log-space to preserve order-of-magnitude structure.
            """
            conc_map = self.responses.get(element, {})
            if not conc_map:
                return np.zeros_like(dose_rates), {k: 0.0 for k in gas_production.keys()}
            # Sort available concentrations
            concs = sorted(conc_map.keys())
            # Compute per-fraction arrays at available nodes
            per_frac_dose_nodes: Dict[float, np.ndarray] = {}
            per_frac_gas_nodes: Dict[float, Dict[str, float]] = {}
            base_dose = self.baseline.dose_rates if self.baseline is not None else np.zeros_like(dose_rates)
            base_gas = self.baseline.gas_production if self.baseline is not None else {k: 0.0 for k in gas_production.keys()}
            for c0 in concs:
                resp = conc_map[c0]
                # Per-fraction dose: (mix - (1-c0)*baseline)/c0
                mix_dose = resp.dose_rates
                per_frac_dose = (mix_dose - (1.0 - c0) * base_dose) / max(c0, 1e-12)
                # Gas per-fraction (scalars)
                per_frac_gas: Dict[str, float] = {}
                for gk in gas_production.keys():
                    mix_val = float(resp.gas_production.get(gk, 0.0))
                    base_val = float(base_gas.get(gk, 0.0))
                    per_frac_gas[gk] = (mix_val - (1.0 - c0) * base_val) / max(c0, 1e-12)
                per_frac_dose_nodes[c0] = per_frac_dose
                per_frac_gas_nodes[c0] = per_frac_gas
            # If only one node, return scaled
            if len(concs) == 1:
                return per_frac_dose_nodes[concs[0]], per_frac_gas_nodes[concs[0]]
            # Find bracketing nodes
            f = float(target_frac)
            if f <= concs[0]:
                lo, hi = concs[0], concs[1]
            elif f >= concs[-1]:
                lo, hi = concs[-2], concs[-1]
            else:
                lo, hi = concs[0], concs[-1]
                for i in range(len(concs) - 1):
                    if concs[i] <= f <= concs[i + 1]:
                        lo, hi = concs[i], concs[i + 1]
                        break
            # Linear interpolation weight
            t = 0.0 if hi == lo else (f - lo) / (hi - lo)
            # Dose interpolate in log-space with epsilon
            eps = 1e-30
            lo_d = per_frac_dose_nodes[lo]
            hi_d = per_frac_dose_nodes[hi]
            dose_interp = np.exp((1 - t) * np.log(np.maximum(lo_d, eps)) + t * np.log(np.maximum(hi_d, eps)))
            # Gas interpolate linearly (scalars)
            gas_interp: Dict[str, float] = {}
            for gk in gas_production.keys():
                lo_g = per_frac_gas_nodes[lo][gk]
                hi_g = per_frac_gas_nodes[hi][gk]
                gas_interp[gk] = (1 - t) * lo_g + t * hi_g
            return dose_interp, gas_interp

        # Combine contributions for non-V elements
        for element, fraction in composition.items():
            if element == 'V' or fraction <= 0.0:
                continue
            per_frac_dose, per_frac_gas = _interp_per_fraction(element, float(fraction))
            dose_rates += fraction * per_frac_dose
            for gas, per_g in per_frac_gas.items():
                gas_production[gas] += fraction * per_g
        
        # Extract dose at specific cooling times
        dose_at_cooling_times = {}
        if cooling_days is not None:
            # Find irradiation end (last non-zero source rate time)
            # This requires metadata about the schedule
            irradiation_end = self.metadata.get('irradiation_time', 2 * 365.25 * 24 * 3600)
            
            for days in cooling_days:
                cooling_time = days * 24 * 3600
                absolute_time = irradiation_end + cooling_time
                
                # Find closest time point
                idx = np.argmin(np.abs(times - absolute_time))
                if idx < len(dose_rates):
                    dose_at_cooling_times[days] = float(dose_rates[idx])

        # Apply pairwise corrections on scalar metrics (dose_at_cooling_times, gas)
        if (not disable_pairwise) and self.pairwise_corrections and cooling_days is not None:
            elems = [e for e, f in composition.items() if e != 'V' and f > 0]
            for i in range(len(elems)):
                for j in range(i + 1, len(elems)):
                    e1, e2 = elems[i], elems[j]
                    key = "+".join(sorted([e1, e2]))
                    corr = self.pairwise_corrections.get(key)
                    if not corr:
                        continue
                    xi = float(composition.get(e1, 0.0))
                    xj = float(composition.get(e2, 0.0))
                    # Δy = c*xi*xj + ai*xi^2 + aj*xj^2
                    c = float(corr.get('c', 0.0))
                    ai = float(corr.get('ai', 0.0))
                    aj = float(corr.get('aj', 0.0))
                    # Dose corrections per cooling day if provided
                    for days in cooling_days:
                        coef_key = f"dose_{int(days)}d"
                        c_k = float(corr.get(coef_key + '_c', c))
                        ai_k = float(corr.get(coef_key + '_ai', ai))
                        aj_k = float(corr.get(coef_key + '_aj', aj))
                        delta = c_k * xi * xj + ai_k * xi * xi + aj_k * xj * xj
                        if days in dose_at_cooling_times:
                            dose_at_cooling_times[days] = max(0.0, dose_at_cooling_times[days] + delta)
                    # Gas corrections: support fractional or additive models
                    for gk in list(gas_production.keys()):
                        # Fractional form keys (preferred): <gas>_frac_*; fallback to additive if absent
                        c_frac = corr.get(gk + '_frac_c')
                        ai_frac = corr.get(gk + '_frac_ai')
                        aj_frac = corr.get(gk + '_frac_aj')
                        if c_frac is not None and ai_frac is not None and aj_frac is not None:
                            r = float(c_frac) * xi * xj + float(ai_frac) * xi * xi + float(aj_frac) * xj * xj
                            # Apply multiplicatively; clamp to avoid negative
                            gas_production[gk] = max(0.0, gas_production[gk] * (1.0 + r))
                        else:
                            c_g = float(corr.get(gk + '_c', 0.0))
                            ai_g = float(corr.get(gk + '_ai', 0.0))
                            aj_g = float(corr.get(gk + '_aj', 0.0))
                            delta_g = c_g * xi * xj + ai_g * xi * xi + aj_g * xj * xj
                            gas_production[gk] = max(0.0, gas_production[gk] + delta_g)
        
        return {
            'times': times,
            'dose_rates': dose_rates,
            'gas_production': gas_production,
            'dose_at_cooling_times': dose_at_cooling_times,
            'method': 'impulse'
        }
    
    def validate_linearity(
        self,
        test_composition: Dict[str, float],
        full_depletion_result: Dict[str, Union[np.ndarray, Dict]],
        tolerance: float = 0.1
    ) -> Dict[str, float]:
        """Validate linear approximation against full depletion result.
        
        Args:
            test_composition: Composition that was tested
            full_depletion_result: Result from full depletion calculation
            tolerance: Relative error tolerance for validation
            
        Returns:
            Dict with relative errors for each metric
        """
        # Synthesize using impulse library
        synth = self.synthesize(test_composition)
        
        errors = {}
        
        # Compare dose rates (max relative error over time)
        if 'dose_rates' in full_depletion_result:
            full_dose = full_depletion_result['dose_rates']
            synth_dose = synth['dose_rates']
            
            # Align arrays if needed
            min_len = min(len(full_dose), len(synth_dose))
            full_dose = full_dose[:min_len]
            synth_dose = synth_dose[:min_len]
            
            # Compute relative error where dose is significant
            mask = full_dose > 1e-10  # Avoid division by tiny numbers
            if np.any(mask):
                rel_errors = np.abs(synth_dose[mask] - full_dose[mask]) / full_dose[mask]
                errors['dose_max_error'] = float(np.max(rel_errors))
                errors['dose_mean_error'] = float(np.mean(rel_errors))
        
        # Compare gas production
        if 'gas_production' in full_depletion_result:
            for gas in ['He_appm', 'H_appm']:
                if gas in full_depletion_result['gas_production'] and gas in synth['gas_production']:
                    full_val = full_depletion_result['gas_production'][gas]
                    synth_val = synth['gas_production'][gas]
                    if full_val > 0:
                        errors[f'{gas}_error'] = abs(synth_val - full_val) / full_val
        
        # Flag if any error exceeds tolerance
        errors['linear_valid'] = all(v <= tolerance for k, v in errors.items() if k.endswith('_error'))
        
        return errors


def build_impulse_library(
    elements: List[str],
    base_element: str = "V",
    reference_concentration: float = 0.01,
    library_dir: Union[str, Path] = "impulse_library",
    chain_file: Optional[str] = None,
    abs_file: Optional[str] = None,
    cooling_days: Optional[List[int]] = None,
    force_rebuild: bool = False,
    source_rate: Optional[float] = None,
    element_concentrations: Optional[Dict[str, List[float]]] = None
) -> ImpulseLibrary:
    """Build library of impulse responses for linear depletion synthesis.
    
    This function runs full depletion calculations for each element at a
    reference concentration in the base element, storing the results as
    impulse responses.
    
    Args:
        elements: List of elements to compute impulses for
        base_element: Base element (default: "V")
        reference_concentration: Atomic fraction for impulse (default: 0.01)
        library_dir: Directory to store impulse library
        chain_file: OpenMC depletion chain file
        abs_file: FISPACT absorption file for dose calculations
        cooling_days: Cooling days for dose extraction
        force_rebuild: Force rebuild even if responses exist
        source_rate: Neutron source rate in n/s (computed from power if None)
        element_concentrations: Optional mapping element->list of concentrations
            to build piecewise impulses; defaults to [reference_concentration]
        
    Returns:
        ImpulseLibrary instance
    """
    from neutronics_calphad.neutronics.config import SPHERICAL
    from neutronics_calphad.neutronics.geometry_maker import create_model
    from neutronics_calphad.neutronics.flux import get_flux_and_microxs
    from neutronics_calphad.neutronics.depletion import run_independent_depletion
    from neutronics_calphad.neutronics.time_scheduler import TimeScheduler
    from neutronics_calphad.utils.io import create_material
    
    library_dir = Path(library_dir)
    library_dir.mkdir(exist_ok=True)
    
    # Default chain and abs files
    if chain_file is None:
        chain_file = openmc.config.get('chain_file')
    if abs_file is None:
        abs_file = "/home/myless/Packages/fispact/nuclear_data/decay/abs_2012"
    if cooling_days is None:
        cooling_days = [30, 365, 5*365, 100*365]
    if source_rate is None:
        POWER_MW = 500
        TORUS_TO_SPHERE_VOLUME_RATIO = 1/4.03 # from notebooks/compare_volume_spherical_toroidal.ipynb
        FUSION_POWER_MEV = 17.6
        MEV_TO_J = 1.602176634e-13
        source_rate = POWER_MW * 1e6 / (FUSION_POWER_MEV * MEV_TO_J)  * TORUS_TO_SPHERE_VOLUME_RATIO
    
    # Initialize library
    library = ImpulseLibrary(library_dir)
    
    # Check existing responses
    if not force_rebuild:
        existing = set(library.responses.keys())
        elements = [e for e in elements if e not in existing]
        if not elements:
            print("All impulse responses already exist")
            return library
    
    print(f"Building impulse responses for: {elements}")
    
    # Create base model and flux
    print("Setting up neutronics model...")
    model = create_model(config=SPHERICAL)
    model.settings.particles = 10000
    
    # Calculate flux and microxs once
    flux_dir = library_dir / "flux_microxs"
    flux_dir.mkdir(exist_ok=True)
    
    flux_file = flux_dir / "flux_spectrum_1102.txt"
    microxs_file = flux_dir / "microxs_1102.csv"
    
    if not (flux_file.exists() and microxs_file.exists()):
        print("Calculating flux and microscopic cross sections...")
        get_flux_and_microxs(
            model,
            chain_file=chain_file,
            group_structure='UKAEA-1102',
            outdir=flux_dir
        )
    
    # Load flux and microxs
    flux = [np.loadtxt(flux_file, comments='#', usecols=1)]
    microxs = openmc.deplete.MicroXS.from_csv(microxs_file)
    
    # Set up time schedule
    scheduler = TimeScheduler(
        irradiation_time='2 years',
        cooling_times=[f'{d} days' for d in cooling_days],
        source_rate=source_rate,  # Use parameter instead of hardcoded value
        irradiation_steps=24
    )
    timesteps, source_rates = scheduler.get_timesteps_and_source_rates()
    
    # Store metadata
    library.metadata = {
        'irradiation_time': 2 * 365.25 * 24 * 3600,
        'reference_concentration': reference_concentration,
        'base_element': base_element,
        'chain_file': str(chain_file),
        'abs_file': str(abs_file)
    }
    
    # Save metadata
    with open(library_dir / "library_metadata.json", 'w') as f:
        json.dump(library.metadata, f, indent=2)
    
    # Always build baseline (pure V)
    print("\nBuilding baseline (pure V)...")
    base_material_name = f"{base_element}_baseline"
    base_material = create_material({base_element: 1.0}, base_material_name)
    base_material.depletable = True
    vessel_cell = model.geometry.get_cells_by_name('vessel')[0]
    if vessel_cell.fill.volume:
        base_material.volume = vessel_cell.fill.volume
    else:
        base_material.volume = 2.13e5
    vessel_cell.fill = base_material
    base_dir = library_dir / "depletion_baseline"
    base_dir.mkdir(exist_ok=True)
    base_results = run_independent_depletion(
        model=model,
        depletable_cell='vessel',
        microxs=microxs,
        flux=flux,
        chain_file=chain_file,
        timesteps=timesteps,
        source_rates=source_rates,
        outdir=base_dir
    )
    base_parsed = parse_openmc_results(
        results=base_results,
        chain_file=chain_file,
        abs_file=abs_file,
        cooling_days=cooling_days
    )
    times_dose_base, dose_dicts_base = contact_dose(
        results=base_results,
        chain_file=chain_file,
        abs_file=abs_file
    )
    base_dose_rates = np.array([sum(d.values()) for d in dose_dicts_base])
    baseline_response = ImpulseResponse(
        element=base_element,
        concentration=1.0,
        times=times_dose_base,
        inventories={},  # not needed for synthesis
        dose_rates=base_dose_rates,
        gas_production=base_parsed['gas_production'],
        metadata={
            'material_name': base_material_name,
            'volume': base_material.volume,
            'baseline': True,
            'chain_file': str(chain_file),
            'abs_file': str(abs_file)
        }
    )
    library.add_response(baseline_response)

    # Determine concentrations per element
    conc_map: Dict[str, List[float]] = {}
    for e in elements:
        conc_map[e] = list(element_concentrations.get(e, [reference_concentration])) if element_concentrations else [reference_concentration]

    # Build impulse for each element at all requested concentrations
    for element in elements:
        print(f"\nBuilding impulse response for {element}...")
        for c0 in conc_map[element]:
            c0 = float(c0)
            # Create composition: c0 of element in base
            composition = {base_element: 1.0 - c0, element: c0}
            material_name = f"{base_element}_{element}_{c0:.5f}"
            material = create_material(composition, material_name)
            material.depletable = True
            vessel_cell = model.geometry.get_cells_by_name('vessel')[0]
            if vessel_cell.fill.volume:
                material.volume = vessel_cell.fill.volume
            else:
                material.volume = 2.13e5
            vessel_cell.fill = material
            # Run depletion
            depletion_dir = library_dir / f"depletion_{element}_{c0:.5f}"
            depletion_dir.mkdir(exist_ok=True)
            print(f"Running depletion for {element} at {c0:.5f}...")
            results = run_independent_depletion(
                model=model,
                depletable_cell='vessel',
                microxs=microxs,
                flux=flux,
                chain_file=chain_file,
                timesteps=timesteps,
                source_rates=source_rates,
                outdir=depletion_dir
            )
            parsed = parse_openmc_results(
                results=results,
                chain_file=chain_file,
                abs_file=abs_file,
                cooling_days=cooling_days
            )
            times_dose, dose_dicts = contact_dose(
                results=results,
                chain_file=chain_file,
                abs_file=abs_file
            )
            dose_rates = np.array([sum(d.values()) for d in dose_dicts])
            # Build minimal inventories if needed later
            inventories = {}
            response = ImpulseResponse(
                element=element,
                concentration=c0,
                times=times_dose,
                inventories=inventories,
                dose_rates=dose_rates,
                gas_production=parsed['gas_production'],
                metadata={
                    'material_name': material_name,
                    'volume': material.volume,
                    'chain_file': str(chain_file),
                    'abs_file': str(abs_file)
                }
            )
            library.add_response(response)
            print(f"Impulse response for {element}@{c0:.5f} complete")

    # --- Spectrum features (fast/epithermal/thermal fractions) ---
    try:
        # Load energy boundaries (eV) for UKAEA-1102
        bounds = _load_energy_boundaries(1102)
        # Convert to midpoints per group if possible
        phi = flux[0]
        if len(bounds) == len(phi) + 1:
            e_hi = bounds[:-1]
            e_lo = bounds[1:]
            e_mid = np.sqrt(np.maximum(e_hi, 1e-30) * np.maximum(e_lo, 1e-30))
        else:
            # Fallback: use provided values as representative energies
            e_mid = np.array(bounds[: len(phi)], dtype=float)
        total = float(np.sum(phi)) if np.sum(phi) > 0 else 1.0
        # Thresholds in eV
        fast_thr = 1.0e5  # 0.1 MeV
        thermal_thr = 0.5  # 0.5 eV
        fast_mask = e_mid >= fast_thr
        thermal_mask = e_mid < thermal_thr
        fast_frac = float(np.sum(phi[fast_mask]) / total)
        thermal_frac = float(np.sum(phi[thermal_mask]) / total)
        epi_frac = max(0.0, 1.0 - fast_frac - thermal_frac)
        library.metadata['spectrum_features'] = {
            'fast_frac': fast_frac,
            'epi_frac': epi_frac,
            'thermal_frac': thermal_frac,
        }
        with open(library_dir / "library_metadata.json", 'w') as f:
            json.dump(library.metadata, f, indent=2)
    except Exception as e:
        print(f"Warning: Failed to compute spectrum features: {e}")
    
    print("\nImpulse library build complete!")
    return library


def composition_hash(
    composition: Dict[str, float],
    precision: int = 4
) -> str:
    """Generate hash for composition with specified precision.
    
    Args:
        composition: Dict mapping elements to atomic fractions
        precision: Decimal places for rounding
        
    Returns:
        Hex string hash
    """
    # Sort by element and round
    rounded = {}
    for elem in sorted(composition.keys()):
        val = round(composition[elem], precision)
        if val > 0:  # Only include non-zero
            rounded[elem] = val
    
    # Create string representation
    comp_str = json.dumps(rounded, sort_keys=True)
    
    # Return hash
    return hashlib.md5(comp_str.encode()).hexdigest()[:16]
