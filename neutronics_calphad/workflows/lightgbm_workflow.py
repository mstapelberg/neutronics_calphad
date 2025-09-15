# -*- coding: utf-8 -*-
"""
Starter kit: ILR transform + LightGBM (quantile) + BoTorch acquisition for V–Cr–Ti–W–Zr.
- Compositional features use ILR (5-part: [V, Cr, Ti, W, Zr]) with a Helmert basis.
- LightGBM trains per-target quantiles in log10-space.
- BoTorch GP (per target) models log10 outputs in RAW variable space [Cr, Ti, W, Zr],
  and acquisition maximizes joint probability of feasibility under your constraints.
"""

from __future__ import annotations
import math
import os
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Callable

# --- LightGBM (quantile) ---
import lightgbm as lgb

# --- Torch / BoTorch / GPyTorch ---
import torch
from torch.distributions import Normal
from botorch.models import SingleTaskGP, ModelListGP
from botorch.fit import fit_gpytorch_mll
from botorch.optim import optimize_acqf
from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.models.transforms.outcome import Standardize
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood

# =============================================================================
# 0) Problem configuration (YOU set these)
# =============================================================================

# Fixed total impurity fraction (atomic): C+N+O.
CNO_FRAC: float = 0.00  # <-- put your fixed impurity at. frac here (e.g., 0.005). 0 if negligible.

# Dose times (days) – prefer days throughout the workflow for clarity
DOSE_TIMES_D: List[float] = [30.0, 365.0, 5 * 365.0, 100 * 365.0]
# Also expose hours for internal adapters that still expect hours
DOSE_TIMES_H: List[float] = [24.0 * d for d in DOSE_TIMES_D]

# Targets = dose at each time (days) + He_2y + H_2y
TARGET_NAMES: List[str] = [f"dose_d{int(d)}" for d in DOSE_TIMES_D] + ["He_2y", "H_2y"]
M = len(TARGET_NAMES)

# Limits keyed by TARGET_NAMES
LIMITS: Dict[str, float] = {
    "dose_d30": 1e3,
    "dose_d365": 1.0,
    "dose_d1825": 1e-2,
    "dose_d36500": 1e-4,
    "He_2y": 586.0,
    "H_2y": 1200.0,
}

# LightGBM quantiles to train
QUANTILES = [0.1, 0.5, 0.9]

# Numerical safety
EPS = 1e-12
# Optional grid step for alloying elements [Cr, Ti, W, Zr]; set via env LGBM_GRID_STEP
try:
    _GRID_STEP_ENV = os.environ.get("LGBM_GRID_STEP", "")
    GRID_STEP: Optional[float] = float(_GRID_STEP_ENV) if _GRID_STEP_ENV else 0.001
    if GRID_STEP is not None and GRID_STEP <= 0:
        GRID_STEP = None
except Exception:
    GRID_STEP = None

# =============================================================================
# 1) ILR utilities (Helmert basis) for 5-part compositions
# =============================================================================

def helmert_submatrix(D: int) -> np.ndarray:
    """Return (D-1) x D Helmert submatrix with orthonormal rows."""
    H = np.zeros((D, D))
    # First row (not used in submatrix) can be 1/sqrt(D), but we only need rows 1..D-1 below.
    for i in range(1, D):
        # Row i: first i entries = 1/sqrt(i*(i+1)), entry i+1 = -i/sqrt(i*(i+1))
        H[i, :i] = 1.0 / math.sqrt(i * (i + 1))
        H[i, i] = -i / math.sqrt(i * (i + 1))
    return H[1:, :]  # shape: (D-1, D)

# Precompute for D=5 parts: [V, Cr, Ti, W, Zr]
D = 5
_H = helmert_submatrix(D)  # (4 x 5)
_HT = _H.T

def ilr_from_closed_5(x5: np.ndarray) -> np.ndarray:
    """
    ILR of a 5-part composition (positive, sums to 1).
    Uses Helmert basis: z = H @ ln(x).
    """
    x5 = np.asarray(x5, dtype=float)
    assert x5.shape[-1] == 5
    x5 = np.clip(x5, EPS, None)
    ln = np.log(x5)
    return _H @ ln  # shape (..., 4)

def ilr_inv_to_closed_5(z: np.ndarray) -> np.ndarray:
    """
    Inverse ILR to a 5-part composition (sum=1).
    clr = H^T z; x ∝ exp(clr).
    """
    z = np.asarray(z, dtype=float)
    clr = _HT @ z  # length 5
    x = np.exp(clr)
    return x / x.sum()

# =============================================================================
# 2) Composition feature pipeline (your caps enforced here)
# =============================================================================

@dataclass
class Composition:
    Cr: float
    Ti: float
    W: float
    Zr: float

def raw_to_closed5(comp: Composition, cno_frac: float = CNO_FRAC) -> np.ndarray:
    """
    Map raw variables [Cr, Ti, W, Zr] to the 5-part composition [V, Cr, Ti, W, Zr]
    with fixed C+N+O = cno_frac. The 5 parts are then *renormalized* to sum to 1
    (C/N/O excluded since they are constant).
    """
    cr, ti, w, zr = comp.Cr, comp.Ti, comp.W, comp.Zr
    v = 1.0 - (cr + ti + w + zr + cno_frac)
    if v <= 0:
        # This should be prevented by constraints; clamp to a tiny positive if needed.
        v = EPS
    x5 = np.array([v, cr, ti, w, zr], dtype=float)
    x5 = np.clip(x5, EPS, None)
    x5 = x5 / x5.sum()  # renormalize over the 5 varying parts
    return x5  # sums to 1

def raw_to_ilr(comp: Composition, cno_frac: float = CNO_FRAC) -> np.ndarray:
    """Convenience: raw -> 5-part closure -> ILR(4-dim)."""
    x5 = raw_to_closed5(comp, cno_frac)
    return ilr_from_closed_5(x5)

# =============================================================================
# 3) Constraint-aware random sampling in your feasible region
# =============================================================================

def sample_feasible_raw(n: int, rng: np.random.Generator) -> np.ndarray:
    """
    Sample n points (Cr, Ti, W, Zr) uniformly under:
      0 ≤ Cr,Ti,W ≤ 0.20
      0 ≤ Zr ≤ 0.049
      Cr + Ti + W + Zr ≤ 0.20
    Strategy:
      - Sample S_alloy ~ U(0, 0.20)
      - Sample Zr ~ U(0, min(0.049, S_alloy))
      - Let S_CTW = S_alloy - Zr. Draw Dirichlet(1,1,1) on (Cr,Ti,W) scaled by S_CTW.
    """
    X = np.zeros((n, 4), dtype=float)
    for i in range(n):
        S = rng.uniform(0.0, 0.20)
        zr_max = min(0.049, S)
        zr = rng.uniform(0.0, zr_max)
        s_ctw = max(S - zr, 0.0)
        # Dirichlet for 3 parts (uniform on simplex)
        if s_ctw > 0:
            parts = rng.dirichlet(np.ones(3))
            cr, ti, w = s_ctw * parts
        else:
            cr = ti = w = 0.0
        # Clip within individual caps (they already are ≤ S ≤ 0.20)
        X[i] = [min(cr, 0.20), min(ti, 0.20), min(w, 0.20), zr]
    return X  # shape (n,4)

def _quantize_raw_floor(X: np.ndarray, step: Optional[float]) -> np.ndarray:
    """Quantize raw variables [Cr, Ti, W, Zr] to a floor grid of given step.

    This preserves feasibility because values only decrease:
    - Per-element bounds remain satisfied
    - Sum constraint Cr+Ti+W+Zr ≤ 0.20 remains satisfied

    Args:
        X: Array of shape (n,4) for [Cr, Ti, W, Zr].
        step: Grid step (e.g., 0.001). If None, returns X unchanged.

    Returns:
        Quantized array of shape (n,4).
    """
    if step is None:
        return X
    s = float(step)
    if not np.isfinite(s) or s <= 0:
        return X
    Xq = np.floor(np.asarray(X, dtype=float) / s) * s
    # Clamp within explicit caps for safety
    Xq[:, 0] = np.clip(Xq[:, 0], 0.0, 0.20)  # Cr
    Xq[:, 1] = np.clip(Xq[:, 1], 0.0, 0.20)  # Ti
    Xq[:, 2] = np.clip(Xq[:, 2], 0.0, 0.20)  # W
    Xq[:, 3] = np.clip(Xq[:, 3], 0.0, 0.049) # Zr
    # Sum constraint is still satisfied due to floor, but clip for numerical safety
    sums = Xq.sum(axis=1)
    over = sums > 0.20 + 1e-12
    if np.any(over):
        # Project by decrementing largest of {Cr,Ti,W} first in step units
        for i in np.where(over)[0]:
            excess = sums[i] - 0.20
            if excess <= 0:
                continue
            # Work in integer grid units to avoid drift
            units = int(np.ceil(excess / s))
            # Indices for Cr,Ti,W only for removal priority
            order = np.argsort(-Xq[i, :3])  # descending among first 3
            j = 0
            while units > 0 and (Xq[i, :3] > 0).any():
                col = int(order[j % 3])
                take = min(units, int(np.floor(Xq[i, col] / s)))
                if take <= 0:
                    j += 1
                    continue
                Xq[i, col] -= take * s
                units -= take
                j += 1
    return Xq

# =============================================================================
# 4) LightGBM quantile models (per target, log-space)
# =============================================================================

class LGBMQuantileEnsemble:
    """
    Trains one LGBMRegressor per (target, quantile). Targets are in log10-space.
    """
    def __init__(self, target_names: List[str], alphas: List[float] = QUANTILES, params: Dict = None):
        self.target_names = target_names
        self.alphas = alphas
        self.params = params or dict(
            n_estimators=700,
            learning_rate=0.05,
            num_leaves=64,
            min_child_samples=20,
            subsample=0.9,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            verbose=-1,
        )
        self.models: Dict[Tuple[str, float], lgb.LGBMRegressor] = {}

    def fit(self, X_ilr: np.ndarray, Y_log: np.ndarray):
        assert Y_log.shape[1] == len(self.target_names)
        for j, name in enumerate(self.target_names):
            yj = Y_log[:, j]
            for alpha in self.alphas:
                model = lgb.LGBMRegressor(objective="quantile", alpha=alpha, **self.params)
                model.fit(X_ilr, yj)
                self.models[(name, alpha)] = model

    def predict_quantiles(self, X_ilr: np.ndarray) -> Dict[str, Dict[float, np.ndarray]]:
        """
        Returns: {target_name: {alpha: y_pred_log10 (n,)}}.
        """
        out: Dict[str, Dict[float, np.ndarray]] = {}
        for name in self.target_names:
            out[name] = {}
            for alpha in self.alphas:
                out[name][alpha] = self.models[(name, alpha)].predict(X_ilr)
        return out

# =============================================================================
# 5) BoTorch GP + Joint Probability-of-Feasibility acquisition
# =============================================================================

def fit_gp_model_list(X_raw: np.ndarray, Y_log: np.ndarray) -> ModelListGP:
    """
    Fit a separate SingleTaskGP per output on log10-transformed targets.
    Uses float64 and CUDA if available for speed.
    """
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    torch.set_default_dtype(torch.float64)
    X = torch.as_tensor(X_raw, dtype=torch.float64, device=device)
    models = []
    for m in range(Y_log.shape[1]):
        y = torch.as_tensor(Y_log[:, [m]], dtype=torch.float64, device=device)
        gp = SingleTaskGP(X, y, outcome_transform=Standardize(m=1)).to(device)
        models.append(gp)
    model_list = ModelListGP(*models).to(device)
    mll = SumMarginalLogLikelihood(model_list.likelihood, model_list).to(device)
    fit_gpytorch_mll(mll)
    return model_list

class JointPoFAcq(AcquisitionFunction):
    """
    Joint probability that *all* constraints are satisfied:
      Y_i <= limit_i  for every target i.
    Works with ModelListGP (independent posteriors across outputs).
    For q>1, returns the SUM of joint PoF over the q points (maximize expected #feasible).
    """
    def __init__(self, model: ModelListGP, limits_log10: np.ndarray):
        super().__init__(model=model)
        # Get device from model and ensure limits tensor is on the same device
        device = next(model.parameters()).device
        self.register_buffer("limits", torch.as_tensor(limits_log10.reshape(1, -1), dtype=torch.double, device=device))
        self.std_normal = Normal(loc=torch.zeros(1, dtype=torch.double, device=device), scale=torch.ones(1, dtype=torch.double, device=device))

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        X shape: (..., q, d). Returns shape (...,).
        """
        if X.dim() == 2:  # (q, d) -> (1, q, d)
            X = X.unsqueeze(0)

        # Compute PoF per output, per candidate
        # pof has shape (*batch, q, M)
        pofs = []
        for m, gp in enumerate(self.model.models):
            post = gp.posterior(X)
            mu = post.mean.squeeze(-1)    # (*batch, q)
            var = post.variance.squeeze(-1).clamp_min(1e-12)
            sigma = var.sqrt()
            z = (self.limits[..., m].expand_as(mu) - mu) / sigma  # (*batch, q)
            p = self.std_normal.cdf(z)  # (*batch, q)
            pofs.append(p.unsqueeze(-1))
        pof_all = torch.cat(pofs, dim=-1).prod(dim=-1)  # (*batch, q)

        # Aggregate across q points by sum (maximize expected #feasible per batch)
        return pof_all.sum(dim=-1)  # (*batch,)

def suggest_candidates_joint_pof(
    model_list: ModelListGP,
    q: int,
    limits_nat: np.ndarray,
    num_restarts: int = 8,
    raw_samples: int = 128,
    use_fast_pool: bool = True,
    pool_size: int = 8192,
) -> np.ndarray:
    """
    Optimize JointPoF over RAW variables [Cr, Ti, W, Zr], with linear constraints.

    Bounds:
        Cr,Ti,W ∈ [0, 0.20], Zr ∈ [0, 0.049]
    Linear inequality:
        Cr + Ti + W + Zr ≤ 0.20
        
    Args:
        use_fast_pool: If True, use fast candidate pool approach (10-100x faster).
        pool_size: Number of candidates to sample in pool mode.
    """
    # Convert limits to log10-scale
    limits_log10 = np.log10(np.clip(limits_nat, EPS, None))
    acq = JointPoFAcq(model_list, limits_log10)

    # Match device/dtype to model
    dev = next(model_list.parameters()).device
    dtype = torch.float64
    
    if use_fast_pool:
        # Fast candidate pool approach: sample many feasible points, score all, pick top-q
        print(f"Using fast pool selection with {pool_size} candidates...")
        
        # Generate a large pool of feasible candidates on CPU
        rng = np.random.default_rng()
        pool_np = sample_feasible_raw(pool_size, rng)  # (pool_size, 4)
        
        # Move to GPU and evaluate acquisition in batches to avoid OOM
        batch_size = min(1024, pool_size)  # Process in chunks
        scores = []
        
        with torch.no_grad():  # No gradients needed for pool evaluation
            for i in range(0, pool_size, batch_size):
                batch = pool_np[i:i+batch_size]
                X_batch = torch.tensor(batch, dtype=dtype, device=dev)
                # Evaluate JointPoF for each point (not batched as q)
                score_batch = acq(X_batch.unsqueeze(1)).squeeze()  # Treat each as q=1
                scores.append(score_batch)
        
        all_scores = torch.cat(scores)
        
        # Select top-q points
        topk_indices = torch.topk(all_scores, k=min(q, pool_size)).indices
        selected = pool_np[topk_indices.cpu().numpy()]
        
        return selected  # shape (q, 4)
    
    else:
        # Original gradient-based optimization
        lb = torch.tensor([0.0, 0.0, 0.0, 0.0], dtype=dtype, device=dev)
        ub = torch.tensor([0.20, 0.20, 0.20, 0.049], dtype=dtype, device=dev)
        bounds = torch.stack([lb, ub])

        # Linear constraint: sum_i x_i ≤ 0.20
        # BoTorch requires tensors for indices/coefficients in some versions
        ineq_cons = [
            (
                torch.tensor([0, 1, 2, 3], dtype=torch.long, device=dev),
                torch.tensor([1.0, 1.0, 1.0, 1.0], dtype=dtype, device=dev),
                0.20,
            )
        ]

        cand, _ = optimize_acqf(
            acq_function=acq,
            bounds=bounds,
            q=q,
            num_restarts=num_restarts,
            raw_samples=raw_samples,
            inequality_constraints=ineq_cons,
            options={"maxiter": 100},
        )
        return cand.detach().cpu().numpy()  # shape (q,4)

# =============================================================================
# 6) Active loop skeleton tying it all together
# =============================================================================

def log10_transform(y: np.ndarray) -> np.ndarray:
    return np.log10(np.clip(y, EPS, None))

def inv_log10_transform(ylog: np.ndarray) -> np.ndarray:
    return np.power(10.0, ylog)

def build_feature_matrix_ilr(X_raw: np.ndarray) -> np.ndarray:
    """
    X_raw: (n,4) raw comps [Cr, Ti, W, Zr].
    Returns X_ilr: (n,4) ILR coords from [V, Cr, Ti, W, Zr] closure.
    """
    X_ilr = np.zeros_like(X_raw)
    for i in range(X_raw.shape[0]):
        comp = Composition(*X_raw[i])
        X_ilr[i] = raw_to_ilr(comp)
    return X_ilr

def run_simulator(X_raw: np.ndarray) -> np.ndarray:
    """
    Run neutronics depletion simulator for compositions.
    
    Given raw compositions (n,4), return Y in NATURAL scale, shape (n, M):
      columns correspond to TARGET_NAMES = [dose_t..., ..., He_2y, H_2y]
    
    This function should be replaced with a configured version from
    lightgbm_simulator.create_simulator_from_config() for production use.
    """
    # Import here to avoid circular dependencies
    from .lightgbm_simulator import run_simulator as _run_simulator
    
    # Use the imported simulator with dose times from module config
    return _run_simulator(
        X_raw=X_raw,
        dose_times_h=DOSE_TIMES_H,
        threads_per_process=os.environ.get("OMP_NUM_THREADS", 4)  # Use 4 threads per depletion based on benchmarks
    )

def train_lightgbm_quantiles(X_ilr: np.ndarray, Y_nat: np.ndarray) -> LGBMQuantileEnsemble:
    Y_log = log10_transform(Y_nat)
    ens = LGBMQuantileEnsemble(TARGET_NAMES)
    ens.fit(X_ilr, Y_log)
    return ens

def fit_gp_for_acquisition(X_raw: np.ndarray, Y_nat: np.ndarray) -> ModelListGP:
    Y_log = log10_transform(Y_nat)
    return fit_gp_model_list(X_raw, Y_log)

def active_loop(
    n_init: int = 64,
    batch_q: int = 8,
    n_iters: int = 10,
    rng_seed: int = 0,
    on_iteration: Optional[Callable[[int, np.ndarray, np.ndarray], None]] = None,
    X0: Optional[np.ndarray] = None,
    Y0: Optional[np.ndarray] = None,
    use_fast_pool: bool = True,
    pool_size: int = 8192,
    iter_offset: int = 0,
    # Exploration controls (mirrors CALPHAD loop)
    weight_mode: str = "adaptive",
    w_exploit: float = 0.75,
    w_boundary: float = 0.25,
    w_diversity: float = 0.15,
    diversity_min: float = 0.15,
    diversity_initial: float = 0.6,
    diversity_decay_rate: float = 0.2,
    boundary_early: float = 0.25,
    boundary_late: float = 0.30,
    boundary_switch_at_labels: int = 300,
) -> Tuple[np.ndarray, np.ndarray, LGBMQuantileEnsemble, ModelListGP]:
    """
    Run active learning loop with optional warm start and iteration offset.

    Args:
      n_init: Number of initial random samples if no warm-start data is provided.
      batch_q: Batch size per iteration.
      n_iters: Number of acquisition iterations to run (excluding warm-start data).
      rng_seed: Random seed for reproducibility.
      on_iteration: Optional callback called as on_iteration(iter_index, X_batch, Y_batch).
      X0: Optional warm-start inputs of shape (n0, 4) for [Cr, Ti, W, Zr].
      Y0: Optional warm-start targets of shape (n0, M) in NATURAL scale.
      use_fast_pool: Whether to use the fast candidate pool approach for acquisition.
      pool_size: Candidate pool size when use_fast_pool is True.
      iter_offset: Absolute iteration offset used for printing and callback indices
                   to support continuation from previous runs (e.g., when warm-starting).

    Returns:
      X_all_raw: (N,4) accumulated raw comps
      Y_all_nat: (N,M) accumulated targets in NATURAL scale
      lgbm: trained LightGBM quantile ensemble
      gp_model: fitted ModelListGP for acquisition
    """
    rng = np.random.default_rng(rng_seed)

    # --- 1) Initial design & eval ---
    if X0 is not None and Y0 is not None:
        # Warm start from existing data; skip new initial sampling
        X_raw = np.asarray(X0, dtype=float)
        Y_nat = np.asarray(Y0, dtype=float)
    else:
        X_raw = sample_feasible_raw(n_init, rng)        # (n_init, 4)
        # Quantize alloying elements to grid if configured
        X_raw = _quantize_raw_floor(X_raw, GRID_STEP)
        Y_nat = run_simulator(X_raw)                    # (n_init, M)
        if on_iteration is not None:
            # iteration 0 = initial design
            on_iteration(0, X_raw.copy(), Y_nat.copy())

    # --- 2) Train models ---
    X_ilr = build_feature_matrix_ilr(X_raw)
    lgbm = train_lightgbm_quantiles(X_ilr, Y_nat)
    gp_model = fit_gp_for_acquisition(X_raw, Y_nat)

    # --- 3) Iterative acquisition ---
    limit_vec = np.array([LIMITS[name] for name in TARGET_NAMES], dtype=float)

    for it in range(n_iters):
        # Propose a feasible batch using exploration-aware selection when pool mode is enabled
        import time
        t_start = time.time()
        if use_fast_pool and weight_mode in {"adaptive", "fixed"}:
            # Build pool via random feasible sampling
            pool_np = sample_feasible_raw(pool_size, rng)  # (pool_size, 4)

            # Quantize pool BEFORE scoring, enforce per-element minimums, drop duplicates
            pool_np = _quantize_raw_floor(pool_np, GRID_STEP)
            min_elem = 0.004
            mask = (
                (pool_np[:, 0] >= min_elem) &
                (pool_np[:, 1] >= min_elem) &
                (pool_np[:, 2] >= min_elem) &
                (pool_np[:, 3] >= min_elem)
            )
            filtered = pool_np[mask] if mask.size and mask.any() else pool_np
            # Drop duplicates (round to 6dp for robust keys)
            seen = set()
            uniq_rows = []
            for row in filtered:
                key = tuple(float(round(x, 6)) for x in row)
                if key in seen:
                    continue
                seen.add(key)
                uniq_rows.append(row)
            if uniq_rows and len(uniq_rows) >= batch_q:
                pool_np = np.asarray(uniq_rows, dtype=float)
            # Else keep original quantized pool_np to ensure enough candidates

            # Score exploitation via Joint PoF (per-candidate)
            limits_log10 = np.log10(np.clip(limit_vec, EPS, None))
            acq = JointPoFAcq(gp_model, limits_log10)
            dev = next(gp_model.parameters()).device
            dtype = torch.float64
            scores = []
            with torch.no_grad():
                bs = min(1024, pool_size)
                for i in range(0, pool_size, bs):
                    batch = pool_np[i:i+bs]
                    Xb = torch.tensor(batch, dtype=dtype, device=dev)
                    score_b = acq(Xb.unsqueeze(1)).squeeze()  # (bs,)
                    scores.append(score_b)
            proba = torch.cat(scores).detach().cpu().numpy()  # (pool_size,)

            # Boundary score: prefer joint PoF near 0.5
            boundary_scores = 1.0 - np.abs(proba - 0.5) * 2.0
            boundary_scores = np.clip(boundary_scores, 0.0, 1.0)

            # Diversity score: distance from history X_raw
            def _min_distances(pool_arr: np.ndarray, hist_arr: np.ndarray, chunk: int = 1024) -> np.ndarray:
                if hist_arr.shape[0] == 0:
                    # Fallback: distance from centroid of pool
                    centroid = pool_arr.mean(axis=0, keepdims=True)
                    return np.linalg.norm(pool_arr - centroid, axis=1)
                mins = np.empty(pool_arr.shape[0], dtype=float)
                for j in range(0, pool_arr.shape[0], chunk):
                    pb = pool_arr[j:j+chunk]
                    # squared distances to history
                    d2 = np.sum((pb[:, None, :] - hist_arr[None, :, :]) ** 2, axis=2)
                    mins[j:j+pb.shape[0]] = np.sqrt(np.min(d2, axis=1))
                return mins

            min_distances = _min_distances(pool_np, X_raw)
            # Rank-scale diversity to [0,1]
            ranks = np.argsort(np.argsort(min_distances))
            diversity_scores = ranks / (len(min_distances) - 1 + 1e-9)

            # Determine weights
            if weight_mode == "fixed":
                w_div_eff = float(max(0.0, min(1.0, w_diversity)))
                w_bnd_eff = float(max(0.0, min(1.0, w_boundary)))
                w_exp_eff = float(max(0.0, 1.0 - (w_div_eff + w_bnd_eff)))
            else:
                n_labels = X_raw.shape[0]
                w_div_eff = float(max(diversity_min, diversity_initial * np.exp(-diversity_decay_rate * it)))
                w_bnd_eff = float(boundary_early if n_labels < boundary_switch_at_labels else boundary_late)
                w_div_eff = float(max(0.0, min(1.0, w_div_eff)))
                w_bnd_eff = float(max(0.0, min(1.0, w_bnd_eff)))
                w_exp_eff = float(max(0.0, 1.0 - (w_div_eff + w_bnd_eff)))

            # Early-iteration diversification if labels scarce
            early_explore = (it < 2) or (X_raw.shape[0] < 50)
            if early_explore:
                select_idx = np.argsort(diversity_scores)[-batch_q:]
            else:
                blended_scores = (
                    w_exp_eff * proba +
                    w_bnd_eff * boundary_scores +
                    w_div_eff * diversity_scores
                )
                blended_scores = blended_scores + 1e-6 * np.random.random(len(blended_scores))

                # Quotas to ensure representation
                k_div = int(batch_q * w_div_eff)
                k_bnd = int(batch_q * w_bnd_eff)
                k_exp = batch_q - k_div - k_bnd

                idx_div = np.argsort(diversity_scores)[-k_div:] if k_div > 0 else np.array([], dtype=int)
                mask = np.ones(pool_np.shape[0], dtype=bool)
                mask[idx_div] = False

                rem_after_div = np.where(mask)[0]
                idx_bnd_rel = np.argsort(boundary_scores[mask])[-k_bnd:] if k_bnd > 0 else np.array([], dtype=int)
                idx_bnd = rem_after_div[idx_bnd_rel]
                mask[idx_bnd] = False

                rem_after_bnd = np.where(mask)[0]
                idx_exp_rel = np.argsort(proba[mask])[-k_exp:] if k_exp > 0 else np.array([], dtype=int)
                idx_exp = rem_after_bnd[idx_exp_rel]

                select_idx = np.unique(np.concatenate([idx_div, idx_bnd, idx_exp]))
                if len(select_idx) < batch_q:
                    need = batch_q - len(select_idx)
                    remaining = np.setdiff1d(np.arange(pool_np.shape[0]), select_idx, assume_unique=True)
                    top_off = remaining[np.argsort(blended_scores[remaining])[-need:]]
                    select_idx = np.concatenate([select_idx, top_off])

            X_next = pool_np[select_idx]
        else:
            # Original behavior: optimize Joint PoF (pool or gradient-based)
            X_next = suggest_candidates_joint_pof(
                gp_model, q=batch_q, limits_nat=limit_vec,
                num_restarts=20, raw_samples=512,
                use_fast_pool=use_fast_pool, pool_size=pool_size
            )

        t_select = time.time() - t_start
        print(f"[iter {iter_offset + it + 1}] Candidate selection took {t_select:.1f}s")

        # Evaluate simulator at proposed points
        t_start = time.time()
        # Quantize alloying elements to grid if configured
        X_next = _quantize_raw_floor(X_next, GRID_STEP)
        Y_next = run_simulator(X_next)  # (q,M)
        t_sim = time.time() - t_start
        print(f"[iter {iter_offset + it + 1}] Depletion simulation took {t_sim:.1f}s")
        
        if on_iteration is not None:
            on_iteration(iter_offset + it + 1, X_next.copy(), Y_next.copy())

        # Augment dataset
        X_raw = np.vstack([X_raw, X_next])
        Y_nat = np.vstack([Y_nat, Y_next])

        # Refit models (cheap compared to simulator)
        t_start = time.time()
        X_ilr = build_feature_matrix_ilr(X_raw)
        lgbm = train_lightgbm_quantiles(X_ilr, Y_nat)
        gp_model = fit_gp_for_acquisition(X_raw, Y_nat)
        t_fit = time.time() - t_start
        
        print(f"[iter {iter_offset + it + 1}/{iter_offset + n_iters}] dataset size: {X_raw.shape[0]}, model fitting took {t_fit:.1f}s")

    return X_raw, Y_nat, lgbm, gp_model

# =============================================================================
# 7) Screening helper: probability of feasibility using LightGBM quantiles
# =============================================================================

def lgbm_joint_feasibility_score(
    lgbm: LGBMQuantileEnsemble,
    X_raw: np.ndarray,
    method: str = "quantile_rule",
) -> np.ndarray:
    """
    Fast screening using LightGBM:
      method="quantile_rule": declare 'likely feasible' if the 0.9-quantile (upper bound)
      is <= limit for every target. Returns a boolean mask or score (here float in {0,1}).
    """
    X_ilr = build_feature_matrix_ilr(X_raw)
    preds = lgbm.predict_quantiles(X_ilr)
    scores = np.ones(X_raw.shape[0], dtype=float)
    for j, name in enumerate(TARGET_NAMES):
        ub = preds[name][0.9]           # log10(upper quantile)
        ub_nat = inv_log10_transform(ub)
        scores *= (ub_nat <= LIMITS[name]).astype(float)
    return scores  # 1.0 if passes the conservative screen, else 0.0
