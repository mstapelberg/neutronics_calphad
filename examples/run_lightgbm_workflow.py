#!/usr/bin/env python3
"""Example: Run LightGBM-based active learning workflow.

This demonstrates the new approach using:
- ILR transform for compositional features
- LightGBM quantile regression
- BoTorch acquisition for active learning
- Joint probability of feasibility
"""

import os
import sys
from pathlib import Path
from typing import Any, Tuple, Optional, Dict, List
import json
import warnings
import numpy as np
import pandas as pd

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure OpenMC
import openmc
openmc.config['chain_file'] = os.environ.get(
    'OPENMC_CHAIN_FILE',
    '/home/myless/nuclear_data/tendl21-fispact20-gefy61-chain.xml'
)
openmc.config['cross_sections'] = os.environ.get(
    'OPENMC_CROSS_SECTIONS',
    '/home/myless/nuclear_data/tendl-2021-hdf5/cross_sections.xml'
)

# Thread control for parallelism
os.environ.update({
    "OMP_NUM_THREADS": "1",  # [[memory:7338192]]
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
})

from neutronics_calphad.workflows.lightgbm_workflow import (
    active_loop, lgbm_joint_feasibility_score,
    DOSE_TIMES_H, TARGET_NAMES, LIMITS, CNO_FRAC,
    Composition, raw_to_ilr, sample_feasible_raw,
    suggest_candidates_joint_pof, build_feature_matrix_ilr,
    train_lightgbm_quantiles, fit_gp_for_acquisition,
    inv_log10_transform,
)
from neutronics_calphad.workflows.lightgbm_simulator import create_simulator_from_config
from neutronics_calphad.utils.utils import filter_openmc_warnings
from neutronics_calphad.workflows.calphad_runner import run_calphad_batch
from neutronics_calphad.workflows.depletion_results_loader import load_existing_depletion_results
from neutronics_calphad.workflows.batch_depletion_subprocess import composition_hash
import torch  # type: ignore

# Apply warning filters
filter_openmc_warnings()


def main():
    """Run the LightGBM active learning workflow."""
    print("=== LightGBM Active Learning Workflow ===")
    
    # Configuration
    RESULTS_DIR = Path("analysis_results/lightgbm_workflow")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Set actual problem parameters
    # Fixed impurity fraction (C+N+O)
    import neutronics_calphad.workflows.lightgbm_workflow as lgbm_wf
    lgbm_wf.CNO_FRAC = 0.005  # 0.5 at% total impurities
    
    # Dose times in days (preferred), with hours derived for the simulator
    lgbm_wf.DOSE_TIMES_D = [30.0, 365.0, 5*365.0, 100*365.0]  # 30d, 1y, 5y, 100y
    lgbm_wf.DOSE_TIMES_H = [24.0 * d for d in lgbm_wf.DOSE_TIMES_D]
    
    # Update target names (days-based)
    lgbm_wf.TARGET_NAMES = [f"dose_d{int(d)}" for d in lgbm_wf.DOSE_TIMES_D] + ["He_2y", "H_2y"]
    lgbm_wf.M = len(lgbm_wf.TARGET_NAMES)
    
    # Set actual limits (in NATURAL scale), keyed by days
    lgbm_wf.LIMITS = {
        "dose_d30": 1e3,       # 30 days
        "dose_d365": 1.0,      # 1 year
        "dose_d1825": 0.01,    # 5 years
        "dose_d36500": 0.0001, # 100 years
        "He_2y": 586.1,
        "H_2y": 1200.0,
    }
    # Sync module-level constants used below with updated config
    global TARGET_NAMES, LIMITS, DOSE_TIMES_H, CNO_FRAC
    TARGET_NAMES = lgbm_wf.TARGET_NAMES
    LIMITS = lgbm_wf.LIMITS
    DOSE_TIMES_H = lgbm_wf.DOSE_TIMES_H
    CNO_FRAC = lgbm_wf.CNO_FRAC
    
    # Configure simulator
    flux_microxs_dir = Path("analysis_results/impulse_library/library/flux_microxs")
    if not flux_microxs_dir.exists():
        print(f"ERROR: Flux/MicroXS directory not found: {flux_microxs_dir}")
        print("Please run neutronics calculations first to generate flux and cross sections.")
        return
    
    # Create configured simulator
    simulator = create_simulator_from_config(
        flux_microxs_dir=flux_microxs_dir,
        chain_file=openmc.config['chain_file'],
        abs_file="/home/myless/Packages/fispact/nuclear_data/decay/abs_2012",
        output_dir=RESULTS_DIR / "depletion_runs",
        dose_times_h=lgbm_wf.DOSE_TIMES_H,
        max_parallel_jobs=32,
        threads_per_process=1  # [[memory:7338192]]
    )
    
    # Replace the module's simulator with our configured one
    lgbm_wf.run_simulator = simulator

    n_init = 32
    batch_q = 32
    n_iters = 5
    rng_seed = 42
    
    # Silence LightGBM feature-name warnings
    warnings.filterwarnings("ignore", message=r"X does not have valid feature names.*")

    # Env option: use existing depletion results only
    USE_EXISTING_ONLY = os.environ.get("LGBM_USE_EXISTING", "0") == "1"
    USE_EXISTING_ONLY = False

    # Run active learning loop
    print(f"\nStarting active learning with:")
    print(f"  Initial samples: {n_init}")
    print(f"  Batch size: {batch_q}")
    print(f"  Iterations: {n_iters}")
    print(f"  CNO fraction: {lgbm_wf.CNO_FRAC}")
    print(f"  Dose times (days): {lgbm_wf.DOSE_TIMES_D}")
    print(f"  Limits: {lgbm_wf.LIMITS}")
    
    X_all = Y_all = lgbm_model = gp_model = None  # type: ignore

    # Iteration logger (ground truth only). Predictions will be added after training.
    def _log_iteration(iter_idx: int, X_batch: np.ndarray, Y_batch: np.ndarray) -> None:
        """Append iteration record with composition IDs; skip duplicates by iter index.

        The composition IDs correspond to comp_<hash> directories produced by the
        batch depletion runner, so downstream users can link iterations to outputs.
        """
        out = RESULTS_DIR / "depletion_runs" / "iterations.jsonl"
        out.parent.mkdir(parents=True, exist_ok=True)

        # Build composition IDs using the same hashing scheme as the runner
        comp_ids: List[str] = []
        for row in X_batch:
            cr, ti, w, zr = float(row[0]), float(row[1]), float(row[2]), float(row[3])
            # Match lightgbm_simulator.run_simulator: V = 1 - (Cr + Ti + W + Zr)
            v = 1.0 - (cr + ti + w + zr)
            if v < 0.0:
                v = 0.0
            comp = {"V": v, "Cr": cr, "Ti": ti, "W": w, "Zr": zr}
            s = sum(comp.values())
            if s > 0:
                comp = {k: (val / s) for k, val in comp.items()}
            comp_ids.append(composition_hash(comp))

        rec = {
            "iter": int(iter_idx),
            "X_raw": X_batch.tolist(),  # [Cr, Ti, W, Zr]
            "Y_nat": Y_batch.tolist(),  # in natural scale, TARGET_NAMES order
            "comp_ids": comp_ids,
        }

        # Deduplicate by iteration index (idempotent restarts)
        existing_iters: set[int] = set()
        if out.exists():
            try:
                with open(out, "r") as fin:
                    for line in fin:
                        try:
                            d = json.loads(line)
                            if isinstance(d, dict) and "iter" in d:
                                existing_iters.add(int(d["iter"]))
                        except Exception:
                            continue
            except Exception:
                pass
        if int(iter_idx) in existing_iters:
            print(f"Skipping duplicate iteration record for iter={int(iter_idx)}")
            return

        with open(out, "a") as f:
            f.write(json.dumps(rec) + "\n")
    if USE_EXISTING_ONLY:
        loaded = load_existing_depletion_results(RESULTS_DIR / "depletion_runs", DOSE_TIMES_H)
        if loaded is not None:
            X_all, Y_all = loaded
            X_ilr = build_feature_matrix_ilr(X_all)
            lgbm_model = train_lightgbm_quantiles(X_ilr, Y_all)
            gp_model = fit_gp_for_acquisition(X_all, Y_all)
            print(f"Loaded {X_all.shape[0]} existing evaluations.")
        else:
            print("No existing results found; running simulator.")
    if X_all is None:
        try:
            # Warm start from raw iterations log if present and continue remaining iters
            iter_log = RESULTS_DIR / "depletion_runs" / "iterations.jsonl"
            X0 = Y0 = None
            start_iter = 0
            if iter_log.exists():
                try:
                    X0, Y0, iter_ids = _load_iterations_log(iter_log)
                    if len(iter_ids) > 0:
                        start_iter = max(int(i) for i in iter_ids)
                    print(f"Warm start from iteration log with {X0.shape[0]} rows (last iter={start_iter}).")
                except Exception as e:
                    print(f"Warning: failed to load iteration log: {e}")
                    X0 = Y0 = None
                    start_iter = 0

            remaining_iters = max(0, n_iters - start_iter)

            def _log_iteration_with_offset(local_iter_idx: int, X_batch: np.ndarray, Y_batch: np.ndarray) -> None:
                # active_loop already adds iter_offset to the provided index
                _log_iteration(int(local_iter_idx), X_batch, Y_batch)

            X_all, Y_all, lgbm_model, gp_model = active_loop(
                n_init=n_init,
                batch_q=batch_q,
                n_iters=remaining_iters,
                rng_seed=rng_seed,
                on_iteration=_log_iteration_with_offset,
                X0=X0,
                Y0=Y0,
                iter_offset=start_iter,
            )
        except Exception as e:
            print(f"Error in active loop: {e}")
            import traceback
            traceback.print_exc()
            return
    
    print(f"\nActive learning complete!")
    print(f"Total evaluations: {X_all.shape[0]}")
    
    # Analyze results
    analyze_results(X_all, Y_all, lgbm_model, RESULTS_DIR)
    evaluate_surrogate(X_all, Y_all, lgbm_model, RESULTS_DIR)
    
    # Save results
    save_results(X_all, Y_all, RESULTS_DIR)

    # Augment iteration log with predictions from the fitted LightGBM
    try:
        augment_iterations_with_predictions(RESULTS_DIR / "depletion_runs" / "iterations.jsonl",
                                            RESULTS_DIR / "depletion_runs" / "iterations_with_pred.jsonl",
                                            lgbm_model)
        print("Wrote iterations_with_pred.jsonl with predicted feasibility per batch.")
    except Exception as e:
        print(f"Warning: failed to augment iteration log with predictions: {e}")

    # Persist the GP model and evaluate CALPHAD on GP-proposed candidates
    gp_path = save_gp_model(gp_model, RESULTS_DIR)
    print(f"Saved GP model to: {gp_path}")

    cand_df, calphad_df = propose_and_run_calphad(gp_model, RESULTS_DIR, n_candidates=1024)
    calphad_csv = RESULTS_DIR / "calphad_candidates.csv"
    calphad_df.to_csv(calphad_csv, index=False)
    print(f"CALPHAD candidate results saved to: {calphad_csv}")

    # Optional: train/val/test split for hold-out evaluation on existing dataset
    split_env = os.environ.get("LGBM_SPLIT", "0.8,0.1,0.1")
    try:
        s_tr, s_va, s_te = [float(x) for x in split_env.split(',')]
    except Exception:
        s_tr, s_va, s_te = 0.8, 0.1, 0.1
    if abs(s_tr + s_va + s_te - 1.0) > 1e-6:
        s_tr, s_va, s_te = 0.8, 0.1, 0.1
    try:
        splits = split_dataset(X_all, Y_all, (s_tr, s_va, s_te), rng_seed)
        X_tr, Y_tr = splits['train']
        X_va, Y_va = splits['val']
        X_te, Y_te = splits['test']
        # Fit a fresh LightGBM on train only
        lgbm_hold = train_lightgbm_quantiles(build_feature_matrix_ilr(X_tr), Y_tr)
        evaluate_surrogate(X_va, Y_va, lgbm_hold, RESULTS_DIR / Path('val'))
        evaluate_surrogate(X_te, Y_te, lgbm_hold, RESULTS_DIR / Path('test'))
        print("Saved hold-out validation/test metrics and predictions.")
    except Exception as e:
        print(f"Warning: failed hold-out evaluation: {e}")


def analyze_results(X_raw, Y_nat, lgbm_model, output_dir):
    """Analyze and visualize results from active learning."""
    import matplotlib.pyplot as plt
    
    # Convert to compositions for analysis
    compositions = []
    for i in range(X_raw.shape[0]):
        cr, ti, w, zr = X_raw[i]
        v = 1.0 - (cr + ti + w + zr + CNO_FRAC)
        compositions.append({
            'V': max(0, v),
            'Cr': cr,
            'Ti': ti,
            'W': w,
            'Zr': zr
        })
    
    comp_df = pd.DataFrame(compositions)
    
    # Check feasibility using LightGBM predictions
    from neutronics_calphad.workflows.lightgbm_workflow import build_feature_matrix_ilr
    X_ilr = build_feature_matrix_ilr(X_raw)
    
    # Get feasibility scores
    scores = lgbm_joint_feasibility_score(lgbm_model, X_raw, method="quantile_rule")
    feasible_mask = scores > 0.5
    
    print(f"\nFeasibility Analysis:")
    print(f"  Feasible: {feasible_mask.sum()} / {len(feasible_mask)} ({100*feasible_mask.mean():.1f}%)")
    
    # Choose dose index (1y if present)
    try:
        dose_1y_idx = TARGET_NAMES.index("dose_d365")
    except ValueError:
        try:
            dose_1y_idx = TARGET_NAMES.index("dose_t8760h")
        except ValueError:
            dose_1y_idx = 0

    # Find best compositions
    if feasible_mask.sum() > 0:
        feasible_idx = np.where(feasible_mask)[0]
        dose_1y_feasible = Y_nat[feasible_idx, dose_1y_idx]
        best_idx = feasible_idx[np.argmin(dose_1y_feasible)]
        best_comp = compositions[best_idx]
        print(f"\nBest composition (lowest 1y dose among feasible):")
        print(f"  V-{best_comp['V']:.3f} Cr-{best_comp['Cr']:.3f} "
              f"Ti-{best_comp['Ti']:.3f} W-{best_comp['W']:.3f} Zr-{best_comp['Zr']:.3f}")
        print(f"  1y dose: {Y_nat[best_idx, dose_1y_idx]:.2f} Sv/h")
        print(f"  He: {Y_nat[best_idx, -2]:.1f} appm")
        print(f"  H: {Y_nat[best_idx, -1]:.1f} appm")
    
    # Visualizations
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    # Discrete colormap for feasibility (avoid yellow/white blends)
    from matplotlib.colors import ListedColormap, BoundaryNorm
    feas_int = feasible_mask.astype(int)
    feas_cmap = ListedColormap(["#d62728", "#2ca02c"])  # red=0, green=1
    feas_norm = BoundaryNorm([-0.5, 0.5, 1.5], feas_cmap.N)
    
    # 1. Composition space coverage
    ax = axes[0, 0]
    scatter = ax.scatter(
        comp_df['Cr'], comp_df['Ti'], c=feas_int,
        cmap=feas_cmap, norm=feas_norm, alpha=0.9, s=40,
        linewidths=0.2, edgecolors='k')
    ax.set_xlabel('Cr fraction')
    ax.set_ylabel('Ti fraction')
    ax.set_title('Composition Space (Cr-Ti projection)')
    plt.colorbar(scatter, ax=ax, label='Feasible')
    
    # 2. Dose vs Gas trade-off
    ax = axes[0, 1]
    dose_1y = Y_nat[:, dose_1y_idx]
    he_2y = Y_nat[:, -2]
    scatter = ax.scatter(
        dose_1y, he_2y, c=feas_int,
        cmap=feas_cmap, norm=feas_norm, alpha=0.9, s=40,
        linewidths=0.2, edgecolors='k')
    ax.set_xlabel('1y Dose (Sv/h)')
    ax.set_ylabel('He Production (appm)')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_title('Dose vs Gas Trade-off')
    
    # Add limit lines (convert label to 1y if present)
    for k in ('dose_d365', 'dose_t8760h'):
        if k in LIMITS:
            ax.axvline(LIMITS[k], color='r', linestyle='--', alpha=0.5)
            break
    ax.axhline(LIMITS['He_2y'], color='r', linestyle='--', alpha=0.5)
    
    # 3. Iteration progress
    ax = axes[1, 0]
    n_init = 32
    iters = (np.arange(len(feasible_mask)) - n_init) // 8
    iter_feasible = []
    for it in range(max(iters) + 1):
        mask = iters <= it
        if mask.sum() > 0:
            iter_feasible.append(feasible_mask[mask].mean())
    
    ax.plot(iter_feasible, 'o-')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Feasibility Rate')
    ax.set_title('Active Learning Progress')
    ax.grid(True, alpha=0.3)
    
    # 4. Element distribution for feasible
    ax = axes[1, 1]
    if feasible_mask.sum() > 0:
        feasible_comps = comp_df[feasible_mask]
        feasible_comps[['Cr', 'Ti', 'W', 'Zr']].boxplot(ax=ax)
        ax.set_ylabel('Atomic Fraction')
        ax.set_title('Element Distribution (Feasible Only)')
    else:
        ax.text(0.5, 0.5, 'No feasible compositions found', 
                ha='center', va='center', transform=ax.transAxes)
    
    plt.tight_layout()
    plt.savefig(output_dir / "lightgbm_results.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nPlots saved to: {output_dir}/lightgbm_results.png")


def save_results(X_raw, Y_nat, output_dir):
    """Save results to CSV for further analysis."""
    # Build full results DataFrame
    results = []
    
    for i in range(X_raw.shape[0]):
        cr, ti, w, zr = X_raw[i]
        v = 1.0 - (cr + ti + w + zr + CNO_FRAC)
        
        row = {
            'V': max(0, v),
            'Cr': cr,
            'Ti': ti,
            'W': w,
            'Zr': zr,
        }
        
        # Add target values
        for j, name in enumerate(TARGET_NAMES):
            row[name] = Y_nat[i, j]
        
        # Add pass/fail for each constraint
        for j, name in enumerate(TARGET_NAMES):
            row[f'{name}_pass'] = Y_nat[i, j] <= LIMITS[name]
        
        # Overall pass
        row['all_pass'] = all(Y_nat[i, j] <= LIMITS[name] 
                             for j, name in enumerate(TARGET_NAMES))
        
        results.append(row)
    
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_dir / "lightgbm_results.csv", index=False)
    
    print(f"Results saved to: {output_dir}/lightgbm_results.csv")
    
    # Print summary statistics
    print("\nSummary Statistics:")
    print(f"  Total evaluations: {len(results_df)}")
    print(f"  Passing all constraints: {results_df['all_pass'].sum()} ({100*results_df['all_pass'].mean():.1f}%)")
    
    for name in TARGET_NAMES:
        pass_rate = results_df[f'{name}_pass'].mean()
        print(f"  {name} pass rate: {100*pass_rate:.1f}%")


def evaluate_surrogate(X_raw: np.ndarray, Y_nat: np.ndarray, lgbm_model: Any, output_dir: Path) -> None:
    """Evaluate LightGBM quantile ensemble calibration and screening accuracy.

    Saves predictions and metrics to disk.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    X_ilr = build_feature_matrix_ilr(X_raw)
    preds = lgbm_model.predict_quantiles(X_ilr)

    rows: List[Dict[str, float]] = []
    metrics: Dict[str, Dict[str, float]] = {}
    quantiles = [0.1, 0.5, 0.9]

    # Per-target metrics
    for j, name in enumerate(TARGET_NAMES):
        y_true = Y_nat[:, j].astype(float)
        metrics[name] = {}
        # Predicted quantiles in natural scale
        q_pred = {q: inv_log10_transform(preds[name][q]) for q in quantiles}
        # Pinball loss and coverage
        for q in quantiles:
            yq = q_pred[q]
            diff = y_true - yq
            pinball = np.mean(np.maximum(q * diff, (q - 1) * diff))
            coverage = float(np.mean(y_true <= yq))
            metrics[name][f"q{int(q*100)}_pinball"] = float(pinball)
            metrics[name][f"q{int(q*100)}_coverage"] = coverage
        # Median MAE
        mae = float(np.mean(np.abs(y_true - q_pred[0.5])))
        metrics[name]["median_mae"] = mae

    # Screening performance vs true pass
    # Predicted pass via quantile rule (same as lgbm_joint_feasibility_score)
    pred_pass = np.ones(X_raw.shape[0], dtype=bool)
    for j, name in enumerate(TARGET_NAMES):
        ub_nat = inv_log10_transform(preds[name][0.9])
        pred_pass &= (ub_nat <= LIMITS[name])
    true_pass = np.ones(X_raw.shape[0], dtype=bool)
    for j, name in enumerate(TARGET_NAMES):
        true_pass &= (Y_nat[:, j] <= LIMITS[name])
    tp = int(np.sum(pred_pass & true_pass))
    fp = int(np.sum(pred_pass & ~true_pass))
    fn = int(np.sum(~pred_pass & true_pass))
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    metrics["screening"] = {
        "tp": tp, "fp": fp, "fn": fn,
        "precision": float(precision), "recall": float(recall)
    }

    # Save predictions table
    for i in range(X_raw.shape[0]):
        row: Dict[str, float] = {"Cr": X_raw[i, 0], "Ti": X_raw[i, 1], "W": X_raw[i, 2], "Zr": X_raw[i, 3]}
        for j, name in enumerate(TARGET_NAMES):
            row[f"{name}_true"] = float(Y_nat[i, j])
            for q in quantiles:
                row[f"{name}_q{int(q*100)}"] = float(inv_log10_transform(preds[name][q][i]))
            row[f"{name}_pass_true"] = bool(Y_nat[i, j] <= LIMITS[name])
            row[f"{name}_pass_pred"] = bool(inv_log10_transform(preds[name][0.9][i]) <= LIMITS[name])
        row["all_pass_true"] = bool(true_pass[i])
        row["all_pass_pred"] = bool(pred_pass[i])
        rows.append(row)
    pd.DataFrame(rows).to_csv(output_dir / "lightgbm_predictions.csv", index=False)

    # Save metrics JSON
    with open(output_dir / "lightgbm_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)


def split_dataset(
    X_raw: np.ndarray,
    Y_nat: np.ndarray,
    ratios: Tuple[float, float, float],
    seed: int,
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Random train/val/test split with given ratios.

    Returns dict with keys 'train', 'val', 'test' mapping to (X, Y).
    """
    assert len(ratios) == 3
    r_tr, r_va, r_te = ratios
    n = X_raw.shape[0]
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_tr = int(r_tr * n)
    n_va = int(r_va * n)
    tr = idx[:n_tr]
    va = idx[n_tr:n_tr + n_va]
    te = idx[n_tr + n_va:]
    return {
        'train': (X_raw[tr], Y_nat[tr]),
        'val': (X_raw[va], Y_nat[va]),
        'test': (X_raw[te], Y_nat[te]),
    }


def augment_iterations_with_predictions(src_jsonl: Path, dst_jsonl: Path, lgbm_model: Any) -> None:
    """Read iterations.jsonl and write a new JSONL with predicted feasibility per batch.

    Adds fields: 'pred_pass_per_target' and 'pred_pass_all'.
    """
    if not src_jsonl.exists():
        return
    import io
    buf_out = io.StringIO()
    with open(src_jsonl, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            X_batch = np.asarray(rec.get('X_raw', []), dtype=float)
            if X_batch.size == 0:
                continue
            X_ilr = build_feature_matrix_ilr(X_batch)
            preds = lgbm_model.predict_quantiles(X_ilr)
            pred_pass_per_target: Dict[str, List[bool]] = {}
            pred_pass_all: List[bool] = []
            # Conservative pass: 0.9-quantile <= limit
            for i in range(X_batch.shape[0]):
                per_target: Dict[str, bool] = {}
                all_ok = True
                for name in TARGET_NAMES:
                    ub = inv_log10_transform(preds[name][0.9][i])
                    ok = bool(ub <= LIMITS[name])
                    per_target[name] = ok
                    if not ok:
                        all_ok = False
                for name in TARGET_NAMES:
                    pred_pass_per_target.setdefault(name, []).append(bool(per_target[name]))
                pred_pass_all.append(all_ok)
            rec['pred_pass_per_target'] = pred_pass_per_target
            rec['pred_pass_all'] = pred_pass_all
            buf_out.write(json.dumps(rec) + "\n")
    dst_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with open(dst_jsonl, 'w') as f:
        f.write(buf_out.getvalue())


def save_gp_model(gp_model: Any, output_dir: Path) -> Path:
    """Save the BoTorch GP model to disk.

    Args:
        gp_model: Trained ModelListGP from the active loop.
        output_dir: Directory to save the model file.

    Returns:
        Path to the saved model artifact.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "gp_model.pth"
    try:
        torch.save(gp_model.state_dict(), path)
    except Exception:
        # Fallback: save entire object (less stable across versions)
        torch.save(gp_model, path)
    return path


def propose_and_run_calphad(gp_model: Any, output_dir: Path, n_candidates: int = 32) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Propose GP candidates via Joint PoF and run CALPHAD on them.

    Args:
        gp_model: Trained ModelListGP used for acquisition.
        output_dir: Base output directory.
        n_candidates: Number of candidates to evaluate with CALPHAD.

    Returns:
        Tuple of (candidates_df [Cr,Ti,W,Zr,V], calphad_results_df).
    """
    # Build limits vector in natural scale
    import numpy as _np
    limit_vec = _np.array([LIMITS[name] for name in TARGET_NAMES], dtype=float)

    # Suggest candidates in RAW space [Cr, Ti, W, Zr]
    X_next = suggest_candidates_joint_pof(
        gp_model, q=n_candidates, limits_nat=limit_vec,
        use_fast_pool=True,      # Enable fast pool method
        pool_size=16384,         # 2x more candidates for better quality
        # Remove gradient-based parameters since they're not used with fast_pool=True
        # num_restarts=20,       # Not used with fast_pool=True
        # raw_samples=512        # Not used with fast_pool=True
    )
    # Convert to full composition with V as balance (includes fixed CNO_FRAC implicitly as in features)
    rows = []
    for i in range(X_next.shape[0]):
        cr, ti, w, zr = [float(x) for x in X_next[i]]
        v = max(0.0, 1.0 - (cr + ti + w + zr + CNO_FRAC))
        # Renormalize across 5 parts (V+Cr+Ti+W+Zr), excluding CNO as constant
        s = v + cr + ti + w + zr
        if s > 0:
            v, cr, ti, w, zr = [x / s for x in (v, cr, ti, w, zr)]
        rows.append({"V": v, "Cr": cr, "Ti": ti, "W": w, "Zr": zr})

    cand_df = pd.DataFrame(rows)

    # Run CALPHAD batch on proposed candidates
    calphad_df = run_calphad_batch(cand_df)
    calphad_df.to_csv(output_dir / "calphad_candidates_raw.csv", index=False)
    return cand_df, calphad_df


def _load_iterations_log(log_path: Path) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """Load iterations.jsonl and concatenate X_raw/Y_nat; return iter indices too.

    Returns (X, Y, iter_ids). If the log is empty, raises ValueError.
    """
    X_rows: List[List[float]] = []
    Y_rows: List[List[float]] = []
    iter_ids: List[int] = []
    with open(log_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            xr = rec.get('X_raw')
            yr = rec.get('Y_nat')
            it = rec.get('iter')
            if isinstance(xr, list) and isinstance(yr, list):
                if len(xr) == len(yr) and len(xr) > 0:
                    X_rows.extend(xr)
                    Y_rows.extend(yr)
                    try:
                        iter_ids.append(int(it))
                    except Exception:
                        pass
    if not X_rows:
        raise ValueError("Empty iteration log")
    return np.asarray(X_rows, dtype=float), np.asarray(Y_rows, dtype=float), iter_ids


if __name__ == "__main__":
    main()
