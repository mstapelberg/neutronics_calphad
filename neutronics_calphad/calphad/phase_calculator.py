import numpy as np
import pandas as pd
from typing import List, Dict, Optional
import json
import logging
import warnings

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

logger = logging.getLogger(__name__)

class CALPHADBatchCalculator:
    """Batch equilibrium calculator using Thermo-Calc or stub implementation."""
    
    def __init__(self, 
                 database: str = "TCHEA7",
                 temperature: float = 823.5,
                 fixed_impurities: Optional[Dict[str, float]] = None):
        """Initialize batch calculator.
        
        Args:
            database: Thermo-Calc database name
            temperature: Calculation temperature in K
            fixed_impurities: Fixed impurity concentrations (atomic fraction)
        """
        self.database = database
        self.temperature = temperature
        # fixed impurities based on NIFS-HEAT 2
        # https://doi.org/10.1016/j.nme.2020.100782
        self.fixed_impurities = fixed_impurities or {
            'C': 290e-6,  # 290 appm
            'N': 440e-6,  # 440 appm
            'O': 470e-6   # 470 appm
        }
        
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
        """Thermo-Calc implementation of batch calculation."""
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
                
            # Calculate for each composition
            for comp in compositions:
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
                    phase_fractions = {}
                    for phase in stable_phases:
                        fraction = result.get_value_of(
                            ThermodynamicQuantity.mole_fraction_of_a_phase(phase)
                        )
                        phase_fractions[phase] = fraction
                        
                    # Determine dominant phase
                    dominant_phase = max(phase_fractions, key=phase_fractions.get)
                    phase_count = len(stable_phases)
                    single_phase = phase_count == 1
                    
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
        """Stub implementation for testing without Thermo-Calc."""
        results = []
        
        for comp in compositions:
            # Simple heuristic: single phase if largest component > 0.8
            max_fraction = comp.max()
            single_phase = max_fraction > 0.8
            
            if single_phase:
                phase_count = 1
                dominant_phase = "BCC_A2"
            else:
                phase_count = 2
                dominant_phase = "BCC_A2"
                
            row = {f'x_{el}': comp[i] for i, el in enumerate(elements)}
            row.update({
                'phase_count': phase_count,
                'dominant_phase': dominant_phase,
                'single_phase': single_phase,
                'phases': json.dumps({dominant_phase: 0.7, "LAVES": 0.3} if not single_phase else {dominant_phase: 1.0})
            })
            results.append(row)
            
        return pd.DataFrame(results)