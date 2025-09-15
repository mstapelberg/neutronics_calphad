import numpy as np
import pandas as pd
from typing import List, Dict, Optional
import json
import logging
import warnings
import time

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    warnings.warn("TQDM not available. Progress bars will not be shown.")

try:
    from tc_python import TCPython, ThermodynamicQuantity
    TC_AVAILABLE = True
except ImportError:
    TC_AVAILABLE = False
    warnings.warn("Thermo-Calc Python API not available. CALPHAD calculations will use stub implementation.")
except Exception:
    # Any other exception during import (like missing license, database issues, etc.)
    TC_AVAILABLE = False
    warnings.warn("Thermo-Calc Python API import failed. CALPHAD calculations will use stub implementation.")

logger = logging.getLogger(__name__)

class CALPHADBatchCalculator:
    """Batch equilibrium calculator using Thermo-Calc or stub implementation.

    Attributes:
        database: Thermo-Calc database name.
        temperature: Calculation temperature in K.
        fixed_impurities: Fixed impurity concentrations (atomic fraction).
        phase_threshold: Dominant phase fraction threshold (0-1) for `single_phase`.
    """
    
    def __init__(self, 
                 database: str = "TCHEA8",
                 temperature: float = 873.15,
                 fixed_impurities: Optional[Dict[str, float]] = None,
                 phase_threshold: float = 0.995):
        """Initialize batch calculator.
        
        Args:
            database: Thermo-Calc database name.
            temperature: Calculation temperature in K.
            fixed_impurities: Fixed impurity concentrations (atomic fraction).
            phase_threshold: Dominant phase minimum fraction (0-1) to mark as
                single_phase. Default 0.995 corresponds to 99.5 vol%.
        """
        self.database = database
        self.temperature = temperature
        # fixed impurities based on NIFS-HEAT 2
        # https://doi.org/10.1016/j.nme.2020.100782
        if fixed_impurities is not None:
            self.fixed_impurities = fixed_impurities
        else:
            # Default impurity values based on NIFS-HEAT 2
            # https://doi.org/10.1016/j.nme.2020.100782
            if self.database.upper() == "TCHEA8":
                self.fixed_impurities = {
                    'C': 290e-6,  # 290 appm
                    'N': 440e-6,  # 440 appm
                    'O': 470e-6   # 470 appm 
                }
            elif self.database.upper() == "TCHEA7":
                self.fixed_impurities = {
                    'C': 290e-6,  # 290 appm
                    'N': 440e-6   # 440 appm
                    # O is omitted for TCHEA7
                }
            else:
                self.fixed_impurities = {
                    'C': 290e-6,  # 290 appm
                    'N': 440e-6,  # 440 appm
                    'O': 470e-6   # 470 appm 
                }
        self.phase_threshold = phase_threshold
        
        if not TC_AVAILABLE:
            logger.warning("Using stub CALPHAD implementation")
            
    def calculate_batch(self, 
                       compositions: np.ndarray,
                       elements: List[str]) -> pd.DataFrame:
        """Calculate equilibrium for batch of compositions.
        
        Args:
            compositions: Array of compositions, shape (n_samples, n_elements)
            elements: List of element symbols (excluding impurities)
            
        Returns:
            DataFrame with columns: x_V, x_Cr, ..., phase_count, dominant_phase, single_phase
        """
        # Handle empty input by returning an empty DataFrame with the expected schema
        try:
            n_rows = int(compositions.shape[0])  # type: ignore[attr-defined]
        except Exception:
            n_rows = 0
        if n_rows == 0:
            cols: Dict[str, pd.Series] = {}
            for el in elements:
                cols[f'x_{el}'] = pd.Series(dtype=float)
            cols.update({
                'phase_count': pd.Series(dtype=int),
                'dominant_phase': pd.Series(dtype=str),
                'single_phase': pd.Series(dtype=bool),
                'phases': pd.Series(dtype=str),
            })
            return pd.DataFrame(cols)

        if TC_AVAILABLE:
            try:
                return self._calculate_batch_tc(compositions, elements)
            except Exception as e:
                logger.warning(f"Thermo-Calc calculation failed: {e}. Falling back to stub implementation.")
                return self._calculate_batch_stub(compositions, elements)
        else:
            return self._calculate_batch_stub(compositions, elements)
            
    def _calculate_batch_tc(self, 
                           compositions: np.ndarray,
                           elements: List[str]) -> pd.DataFrame:
        """Thermo-Calc implementation of batch calculation.

        The method determines `single_phase` based on whether the dominant phase
        fraction meets or exceeds `self.phase_threshold` (fraction units 0-1).
        """
        results = []
        
        # Add impurity elements
        all_elements = elements + list(self.fixed_impurities.keys())
        
        with TCPython() as session:
            # Setup calculation
            calc_setup = (
                session
                .set_cache_folder(f"{self.database}_cache")
                .select_database_and_elements(self.database, all_elements)
                .get_system()
                .with_single_equilibrium_calculation()
                .set_condition("T", self.temperature)
            )
            
            # Set fixed impurity conditions
            for impurity, fraction in self.fixed_impurities.items():
                calc_setup.set_condition(f"X({impurity})", fraction)
                
            # Calculate for each composition with progress bar
            iterator = tqdm(compositions, desc="CALPHAD calculations", unit="comp") if TQDM_AVAILABLE else compositions
            for comp in iterator:
                # Normalize main elements to account for impurities
                impurity_sum = sum(self.fixed_impurities.values())
                main_sum = 1.0 - impurity_sum
                normalized_comp = comp * main_sum
                
                # Set conditions for main elements (skip the balance element)
                for i in range(1, len(elements)):
                    calc_setup.set_condition(f"X({elements[i]})", normalized_comp[i])
                    
                try:
                    # Calculate equilibrium
                    result = calc_setup.calculate()
                    stable_phases = result.get_stable_phases()
                    
                    # Get phase fractions
                    phase_fractions: Dict[str, float] = {}
                    for phase in stable_phases:
                        fraction = result.get_value_of(
                            ThermodynamicQuantity.mole_fraction_of_a_phase(phase)
                        )
                        phase_fractions[phase] = float(fraction)
                        
                    # Determine dominant phase and single_phase by threshold
                    dominant_phase = max(phase_fractions, key=phase_fractions.get) if phase_fractions else "NONE"
                    phase_count = len(stable_phases)
                    dominant_fraction = phase_fractions.get(dominant_phase, 0.0)
                    single_phase = bool(dominant_fraction >= self.phase_threshold)
                    
                except Exception as e:
                    logger.warning(f"Calculation failed for composition: {comp}, error: {e}")
                    dominant_phase = "FAILED"
                    phase_count = -1
                    single_phase = False
                    phase_fractions = {}
                    
                # Build result row
                row = {f'x_{el}': comp[i] for i, el in enumerate(elements)}
                row.update({
                    'phase_count': phase_count,
                    'dominant_phase': dominant_phase,
                    'single_phase': single_phase,
                    'phases': json.dumps(phase_fractions)
                })
                results.append(row)
                
        return pd.DataFrame(results)
        
    def _calculate_batch_stub(self, 
                             compositions: np.ndarray,
                             elements: List[str]) -> pd.DataFrame:
        """Stub implementation for testing without Thermo-Calc.

        Uses a synthetic phase distribution (fraction units 0-1) and the configured
        `phase_threshold` to determine the `single_phase` flag, emulating the new logic.
        """
        results = []
        
        for comp in compositions:
            # Synthetic phase distribution around a dominant BCC_A2 phase
            max_fraction = comp.max()
            dominant_phase = "BCC_B2"
            # Construct fractions in fraction units consistent with threshold 0.995
            if max_fraction > 0.9:
                phase_fractions = {dominant_phase: 0.997, "LAVES_C15": 0.001, "B2": 0.002}
            else:
                phase_fractions = {dominant_phase: 0.970, "B2": 0.025, "FCC_L12": 0.005}
            phase_count = len(phase_fractions)
            dominant_fraction = float(phase_fractions.get(dominant_phase, 0.0))
            single_phase = bool(dominant_fraction >= self.phase_threshold)
            
            row = {f'x_{el}': comp[i] for i, el in enumerate(elements)}
            row.update({
                'phase_count': phase_count,
                'dominant_phase': dominant_phase,
                'single_phase': single_phase,
                'phases': json.dumps(phase_fractions)
            })
            results.append(row)
            
        return pd.DataFrame(results)