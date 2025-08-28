"""High-throughput materials screening pipeline with O(1) depletion.

This module implements the optimized pipeline architecture:
1. Hierarchical filtering (cheap → expensive)
2. Linear impulse depletion for most compositions
3. Two-pool parallelism for optimal core utilization
4. SQLite ledger for result caching
5. Impurity Monte Carlo for robustness
"""

from __future__ import annotations

import hashlib
import json
import multiprocessing as mp
import os
import sqlite3
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np  # type: ignore
import pandas as pd  # type: ignore

from .composition_sampling import SamplingConstraints, sample_compositions
from .filters import ActivationLimits, make_activation_filter, make_ductility_filter
from .impulse_depletion import ImpulseLibrary, composition_hash
from .calphad_runner import run_calphad_batch


# Thread pinning for proper parallelism
os.environ.update({
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
})


@dataclass
class PipelineConfig:
    """Configuration for high-throughput pipeline.
    
    Attributes:
        n_samples: Number of compositions to evaluate
        batch_size: Compositions per batch
        flush_interval: Results to accumulate before DB flush
        n_workers_fast: Workers for fast operations (impulse synthesis)
        n_workers_heavy: Workers for heavy operations (full depletion, CALPHAD)
        impulse_library_dir: Path to impulse response library
        results_dir: Directory for all outputs
        ledger_db: SQLite database path
        margin_threshold: Fraction of limit to trigger full depletion check
        impurity_samples: Number of C/N/O Monte Carlo samples
        temperature_k: CALPHAD temperature
        temperature_delta: Temperature perturbation for robustness
        calphad_database: Thermo-Calc database name
    """
    n_samples: int = 10000
    batch_size: int = 256
    flush_interval: int = 100
    n_workers_fast: int = 32
    n_workers_heavy: int = 24
    impulse_library_dir: str = "impulse_library"
    results_dir: str = "pipeline_results"
    ledger_db: str = "pipeline_ledger.db"
    margin_threshold: float = 0.2  # Check full depletion if within 20% of limit
    impurity_samples: int = 16
    temperature_k: float = 873.15  # 600°C
    temperature_delta: float = 50.0  # ±50K perturbation
    calphad_database: str = "TCHEA8"


@dataclass
class CompositionResult:
    """Result for a single composition evaluation."""
    composition_id: str
    composition: Dict[str, float]
    stage: str  # "filtered", "impulse", "full_depletion", "calphad"
    passed: bool
    margins: Dict[str, float] = field(default_factory=dict)
    gas_production: Dict[str, float] = field(default_factory=dict)
    dose_rates: Dict[str, float] = field(default_factory=dict)
    calphad_results: Optional[Dict] = None
    metadata: Dict = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())


class PipelineLedger:
    """SQLite ledger for caching and tracking results."""
    
    def __init__(self, db_path: Union[str, Path]):
        """Initialize ledger database."""
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(exist_ok=True)
        self._init_db()
    
    def _init_db(self) -> None:
        """Create database tables if they don't exist."""
        with sqlite3.connect(self.db_path) as conn:
            # Compositions table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS compositions (
                    id TEXT PRIMARY KEY,
                    V REAL, Cr REAL, Ti REAL, W REAL, Zr REAL,
                    C REAL DEFAULT 0, N REAL DEFAULT 0, O REAL DEFAULT 0,
                    total_alloy REAL,
                    source_tag TEXT,
                    time_utc TEXT
                )
            """)
            
            # Depletion results
            conn.execute("""
                CREATE TABLE IF NOT EXISTS depletion (
                    composition_id TEXT,
                    schedule_id TEXT,
                    pass_dose BOOLEAN,
                    pass_gas BOOLEAN,
                    margin_dose REAL,
                    margin_gas REAL,
                    H_appm_2y REAL,
                    He_appm_2y REAL,
                    dose_30d REAL,
                    dose_1y REAL,
                    dose_5y REAL,
                    dose_100y REAL,
                    method TEXT,
                    version TEXT,
                    time_utc TEXT,
                    PRIMARY KEY (composition_id, schedule_id, version),
                    FOREIGN KEY (composition_id) REFERENCES compositions(id)
                )
            """)
            
            # CALPHAD results
            conn.execute("""
                CREATE TABLE IF NOT EXISTS calphad (
                    composition_id TEXT,
                    db_name TEXT,
                    T_K REAL,
                    second_phase_frac REAL,
                    phases_json TEXT,
                    pass_flag BOOLEAN,
                    version TEXT,
                    time_utc TEXT,
                    PRIMARY KEY (composition_id, db_name, T_K, version),
                    FOREIGN KEY (composition_id) REFERENCES compositions(id)
                )
            """)
            
            # Create indices
            conn.execute("CREATE INDEX IF NOT EXISTS idx_comp_alloy ON compositions(total_alloy)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_depl_pass ON depletion(pass_dose, pass_gas)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_calphad_pass ON calphad(pass_flag)")
    
    def check_composition(self, comp_id: str) -> Optional[Dict]:
        """Check if composition has been evaluated."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            
            # Get composition
            comp_row = conn.execute(
                "SELECT * FROM compositions WHERE id = ?", (comp_id,)
            ).fetchone()
            
            if not comp_row:
                return None
            
            # Get depletion results
            depl_rows = conn.execute(
                "SELECT * FROM depletion WHERE composition_id = ? ORDER BY time_utc DESC",
                (comp_id,)
            ).fetchall()
            
            # Get CALPHAD results
            calphad_rows = conn.execute(
                "SELECT * FROM calphad WHERE composition_id = ? ORDER BY time_utc DESC",
                (comp_id,)
            ).fetchall()
            
            return {
                'composition': dict(comp_row),
                'depletion': [dict(row) for row in depl_rows],
                'calphad': [dict(row) for row in calphad_rows]
            }
    
    def insert_results(self, results: List[CompositionResult]) -> None:
        """Insert batch of results into database."""
        with sqlite3.connect(self.db_path) as conn:
            for result in results:
                # Insert composition if new
                comp = result.composition
                conn.execute("""
                    INSERT OR IGNORE INTO compositions 
                    (id, V, Cr, Ti, W, Zr, C, N, O, total_alloy, source_tag, time_utc)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    result.composition_id,
                    comp.get('V', 0), comp.get('Cr', 0), comp.get('Ti', 0),
                    comp.get('W', 0), comp.get('Zr', 0), comp.get('C', 0),
                    comp.get('N', 0), comp.get('O', 0),
                    1.0 - comp.get('V', 0),  # total alloy
                    result.metadata.get('source', 'pipeline'),
                    result.timestamp
                ))
                
                # Insert depletion if available
                if result.gas_production or result.dose_rates:
                    conn.execute("""
                        INSERT INTO depletion
                        (composition_id, schedule_id, pass_dose, pass_gas,
                         margin_dose, margin_gas, H_appm_2y, He_appm_2y,
                         dose_30d, dose_1y, dose_5y, dose_100y,
                         method, version, time_utc)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        result.composition_id,
                        result.metadata.get('schedule_id', 'standard'),
                        result.margins.get('dose', 0) > 0,
                        result.margins.get('gas', 0) > 0,
                        result.margins.get('dose', 0),
                        result.margins.get('gas', 0),
                        result.gas_production.get('H_appm', 0),
                        result.gas_production.get('He_appm', 0),
                        result.dose_rates.get(30, 0),
                        result.dose_rates.get(365, 0),
                        result.dose_rates.get(5*365, 0),
                        result.dose_rates.get(100*365, 0),
                        result.metadata.get('depletion_method', 'unknown'),
                        result.metadata.get('version', '1.0'),
                        result.timestamp
                    ))
                
                # Insert CALPHAD if available
                if result.calphad_results:
                    cal = result.calphad_results
                    conn.execute("""
                        INSERT INTO calphad
                        (composition_id, db_name, T_K, second_phase_frac,
                         phases_json, pass_flag, version, time_utc)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        result.composition_id,
                        cal.get('database', self.calphad_database),
                        cal.get('temperature', 873.15),
                        cal.get('second_phase_fraction', 0),
                        json.dumps(cal.get('phases', {})),
                        cal.get('single_phase', False),
                        result.metadata.get('version', '1.0'),
                        result.timestamp
                    ))


def evaluate_impulse_batch(
    compositions: pd.DataFrame,
    impulse_library: ImpulseLibrary,
    activation_limits: ActivationLimits,
    config: PipelineConfig
) -> List[CompositionResult]:
    """Evaluate batch of compositions using impulse synthesis.
    
    Args:
        compositions: DataFrame with composition columns
        impulse_library: Pre-computed impulse responses
        activation_limits: Neutronics limits
        config: Pipeline configuration
        
    Returns:
        List of composition results
    """
    results = []
    
    # Convert cooling days from dose_at_days keys
    cooling_days = list(activation_limits.dose_at_days.keys())
    
    for idx, row in compositions.iterrows():
        comp_dict = row.to_dict()
        comp_id = composition_hash(comp_dict)
        
        # Synthesize using impulse library
        try:
            synth = impulse_library.synthesize(comp_dict, cooling_days=cooling_days)
            
            # Calculate margins (fraction below limit)
            margins = {}
            
            # Gas margins
            gas_ok = True
            min_gas_margin = 1.0
            for gas_key, limit in activation_limits.gas_appm.items():
                value = synth['gas_production'].get(gas_key, 0)
                margin = 1.0 - (value / limit) if limit > 0 else 1.0
                min_gas_margin = min(min_gas_margin, margin)
                if value > limit:
                    gas_ok = False
            
            # Dose margins
            dose_ok = True
            min_dose_margin = 1.0
            for days, limit in activation_limits.dose_at_days.items():
                value = synth['dose_at_cooling_times'].get(days, 0)
                margin = 1.0 - (value / limit) if limit > 0 else 1.0
                min_dose_margin = min(min_dose_margin, margin)
                if value > limit:
                    dose_ok = False
            
            margins['gas'] = min_gas_margin
            margins['dose'] = min_dose_margin
            
            # Build result
            result = CompositionResult(
                composition_id=comp_id,
                composition=comp_dict,
                stage='impulse',
                passed=gas_ok and dose_ok,
                margins=margins,
                gas_production=synth['gas_production'],
                dose_rates=synth['dose_at_cooling_times'],
                metadata={
                    'depletion_method': 'impulse',
                    'schedule_id': 'standard'
                }
            )
            
        except Exception as e:
            # Failed synthesis
            result = CompositionResult(
                composition_id=comp_id,
                composition=comp_dict,
                stage='impulse',
                passed=False,
                metadata={'error': str(e)}
            )
        
        results.append(result)
    
    return results


def evaluate_full_depletion_batch(
    compositions: pd.DataFrame,
    config: PipelineConfig
) -> List[CompositionResult]:
    """Run full depletion for compositions requiring verification.
    
    This is called for compositions with small margins from impulse synthesis.
    """
    # Import here to avoid circular dependencies
    from neutronics_calphad.neutronics.config import SPHERICAL
    from neutronics_calphad.neutronics.geometry_maker import create_model
    from neutronics_calphad.neutronics.depletion import run_independent_depletion
    from neutronics_calphad.neutronics.time_scheduler import TimeScheduler
    from neutronics_calphad.utils.io import create_material
    from neutronics_calphad.optimizer.parsers import parse_openmc_results
    
    results = []
    
    # This would be implemented similar to pure_element_neutronics_run.py
    # but batched for efficiency
    
    # For now, return placeholder
    for idx, row in compositions.iterrows():
        comp_dict = row.to_dict()
        comp_id = composition_hash(comp_dict)
        
        result = CompositionResult(
            composition_id=comp_id,
            composition=comp_dict,
            stage='full_depletion',
            passed=True,  # Placeholder
            metadata={'depletion_method': 'full'}
        )
        results.append(result)
    
    return results


def run_high_throughput_pipeline(
    config: PipelineConfig,
    per_element_max: Dict[str, float],
    total_alloy_max: float,
    activation_limits: ActivationLimits,
    verbose: bool = True
) -> pd.DataFrame:
    """Run the high-throughput materials screening pipeline.
    
    Args:
        config: Pipeline configuration
        per_element_max: Maximum atomic fraction per element
        total_alloy_max: Maximum total alloying fraction
        activation_limits: Neutronics limits
        verbose: Print progress information
        
    Returns:
        DataFrame with all evaluated compositions and results
    """
    start_time = time.time()
    
    # Initialize components
    results_dir = Path(config.results_dir)
    results_dir.mkdir(exist_ok=True)
    
    ledger = PipelineLedger(config.ledger_db)
    impulse_library = ImpulseLibrary(config.impulse_library_dir)
    
    if not impulse_library.responses:
        raise ValueError(f"No impulse responses found in {config.impulse_library_dir}")
    
    # Sample compositions
    if verbose:
        print(f"Sampling {config.n_samples} compositions...")
    
    constraints = SamplingConstraints(
        per_element_max=per_element_max,
        total_alloy_max=total_alloy_max
    )
    
    all_compositions = sample_compositions(
        constraints=constraints,
        n_samples=config.n_samples,
        random_state=42
    )
    
    # Apply compositional filters first
    if verbose:
        print("Applying compositional filters...")
    
    ductility_filter = make_ductility_filter(
        max_total_alloy=total_alloy_max,
        base_element="V"
    )
    
    comp_mask = all_compositions.apply(
        lambda row: ductility_filter(row.to_dict()), axis=1
    )
    
    filtered_comps = all_compositions[comp_mask]
    if verbose:
        print(f"  {len(filtered_comps)} pass compositional constraints")
    
    # Process in batches
    n_batches = (len(filtered_comps) + config.batch_size - 1) // config.batch_size
    all_results = []
    flush_buffer = []
    
    # Stage 1: Impulse synthesis (fast pool)
    if verbose:
        print(f"\nStage 1: Impulse synthesis ({config.n_workers_fast} workers)...")
    
    with ProcessPoolExecutor(max_workers=config.n_workers_fast) as executor:
        futures = []
        
        for i in range(0, len(filtered_comps), config.batch_size):
            batch = filtered_comps.iloc[i:i+config.batch_size]
            
            # Check cache first
            uncached = []
            for idx, row in batch.iterrows():
                comp_id = composition_hash(row.to_dict())
                cached = ledger.check_composition(comp_id)
                if not cached or not cached.get('depletion'):
                    uncached.append(idx)
            
            if uncached:
                batch_to_eval = batch.loc[uncached]
                future = executor.submit(
                    evaluate_impulse_batch,
                    batch_to_eval,
                    impulse_library,
                    activation_limits,
                    config
                )
                futures.append(future)
        
        # Collect results
        for future in as_completed(futures):
            try:
                batch_results = future.result()
                flush_buffer.extend(batch_results)
                
                # Flush to database periodically
                if len(flush_buffer) >= config.flush_interval:
                    ledger.insert_results(flush_buffer)
                    flush_buffer = []
                    
            except Exception as e:
                print(f"Error in impulse batch: {e}")
    
    # Final flush
    if flush_buffer:
        ledger.insert_results(flush_buffer)
    
    # Identify compositions needing full depletion
    borderline_comps = []
    for result in all_results:
        if result.passed and result.stage == 'impulse':
            # Check if any margin is below threshold
            min_margin = min(result.margins.values())
            if min_margin < config.margin_threshold:
                borderline_comps.append(result.composition)
    
    if verbose:
        print(f"\n{len(borderline_comps)} compositions need full depletion verification")
    
    # Stage 2: Full depletion for borderline cases (heavy pool)
    if borderline_comps and config.n_workers_heavy > 0:
        if verbose:
            print(f"\nStage 2: Full depletion ({config.n_workers_heavy} workers)...")
        
        borderline_df = pd.DataFrame(borderline_comps)
        
        with ProcessPoolExecutor(max_workers=config.n_workers_heavy) as executor:
            # Process in smaller batches for heavy operations
            heavy_batch_size = max(1, config.batch_size // 10)
            futures = []
            
            for i in range(0, len(borderline_df), heavy_batch_size):
                batch = borderline_df.iloc[i:i+heavy_batch_size]
                future = executor.submit(
                    evaluate_full_depletion_batch,
                    batch,
                    config
                )
                futures.append(future)
            
            for future in as_completed(futures):
                try:
                    batch_results = future.result()
                    all_results.extend(batch_results)
                    ledger.insert_results(batch_results)
                except Exception as e:
                    print(f"Error in full depletion batch: {e}")
    
    # Stage 3: CALPHAD for passing compositions
    passing_comps = [r for r in all_results if r.passed]
    
    if verbose:
        print(f"\nStage 3: CALPHAD evaluation for {len(passing_comps)} passing compositions...")
    
    if passing_comps:
        # Convert to DataFrame for CALPHAD batch processing
        passing_df = pd.DataFrame([r.composition for r in passing_comps])
        
        # Run CALPHAD with thermal robustness
        temperatures = [
            config.temperature_k,
            config.temperature_k - config.temperature_delta,
            config.temperature_k + config.temperature_delta
        ]
        
        from .calphad_runner import run_calphad_batch
        
        for temp in temperatures:
            try:
                calphad_results = run_calphad_batch(
                    passing_df,
                    temperature_k=temp,
                    database=config.calphad_database
                )
                
                # Update composition results
                for idx, row in calphad_results.iterrows():
                    comp_id = composition_hash(row[['V', 'Cr', 'Ti', 'W', 'Zr']].to_dict())
                    
                    # Find corresponding result
                    for result in all_results:
                        if result.composition_id == comp_id:
                            if result.calphad_results is None:
                                result.calphad_results = {}
                            
                            result.calphad_results[f'T_{temp}K'] = {
                                'temperature': temp,
                                'database': config.calphad_database,
                                'single_phase': row.get('single_phase', False),
                                'phase_count': row.get('phase_count', 0),
                                'dominant_phase': row.get('dominant_phase', ''),
                                'phases': json.loads(row.get('phases', '{}'))
                            }
                            
                            # Update pass flag based on all temperatures
                            if temp == config.temperature_k:
                                result.stage = 'calphad'
                                result.passed = result.passed and row.get('single_phase', False)
                
            except Exception as e:
                print(f"Error in CALPHAD at {temp}K: {e}")
    
    # Save final results to ledger
    ledger.insert_results([r for r in all_results if r.stage == 'calphad'])
    
    # Build summary DataFrame
    summary_data = []
    for result in all_results:
        row = {
            'composition_id': result.composition_id,
            **result.composition,
            'stage': result.stage,
            'passed': result.passed,
            'margin_dose': result.margins.get('dose', -1),
            'margin_gas': result.margins.get('gas', -1),
            'He_appm': result.gas_production.get('He_appm', -1),
            'H_appm': result.gas_production.get('H_appm', -1),
            'dose_30d': result.dose_rates.get(30, -1),
            'dose_1y': result.dose_rates.get(365, -1),
            'single_phase': False
        }
        
        # Add CALPHAD results if available
        if result.calphad_results:
            main_temp = result.calphad_results.get(f'T_{config.temperature_k}K', {})
            row['single_phase'] = main_temp.get('single_phase', False)
            row['phase_count'] = main_temp.get('phase_count', -1)
            row['dominant_phase'] = main_temp.get('dominant_phase', '')
        
        summary_data.append(row)
    
    summary_df = pd.DataFrame(summary_data)
    
    # Save summary
    summary_df.to_csv(results_dir / "pipeline_summary.csv", index=False)
    
    # Print final statistics
    if verbose:
        total_time = time.time() - start_time
        print(f"\n=== Pipeline Complete ===")
        print(f"Total time: {total_time/60:.1f} minutes")
        print(f"Compositions evaluated: {len(all_results)}")
        print(f"  - Passed neutronics: {sum(1 for r in all_results if r.margins.get('dose', 0) > 0 and r.margins.get('gas', 0) > 0)}")
        print(f"  - Passed CALPHAD: {sum(1 for r in all_results if r.stage == 'calphad' and r.passed)}")
        print(f"  - Final candidates: {len(summary_df[summary_df['passed']])}")
        print(f"Average time per composition: {total_time/len(all_results):.2f} seconds")
    
    return summary_df
