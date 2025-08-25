"""End-to-end workflow: constrained sampling, filtering, embedding, CALPHAD.

Instructions:
- Ensure pure-element OpenMC results exist under RESULTS_DIR used below.
- Update CHAIN_FILE and ABS_FILE paths for your environment.
- Optionally set PER_ELEMENT_MAX from your `MAX_COMPOSITIONS` outputs.

Outputs:
- Embedding scatter with validity overlay: analysis_results/workflow/embedding.png
- CALPHAD results CSV for valid compositions: analysis_results/workflow/calphad_results.csv
- Example ternary slices saved under analysis_results/workflow/ternary_*.png
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict

import numpy as np  # type: ignore
import pandas as pd  # type: ignore
import matplotlib.pyplot as plt  # type: ignore

from neutronics_calphad.workflows.workflow import (
    sample_and_filter,
    embed_and_cluster_compositions,
    evaluate_calphad_for_valid,
)
from neutronics_calphad.workflows.filters import ActivationLimits
from neutronics_calphad.workflows.pure_results_loader import load_pure_results
from neutronics_calphad.workflows.ternary import plot_ternary_slice
from neutronics_calphad.workflows.composition_sampling import make_ternary_grid


# --- Configuration ---
RESULTS_BASE = Path("analysis_results/pure_element_neutronics_run")
WORKFLOW_OUT = Path("analysis_results/workflow")
WORKFLOW_OUT.mkdir(parents=True, exist_ok=True)

# Update these paths for your environment
CHAIN_FILE = os.environ.get("OPENMC_CHAIN_FILE", "/home/myless/nuclear_data/tendl21-fispact20-gefy61-chain.xml")
ABS_FILE = os.environ.get("FISPACT_ABS_FILE", "/home/myless/Packages/fispact/nuclear_data/decay/abs_2012")

# Per-element max atomic fractions from your prior analysis (edit as needed)
PER_ELEMENT_MAX: Dict[str, float] = {
    "Cr": 0.49,
    "Ti": 0.697,
    "W": 0.95,
    "Zr": 0.049,
}

# Ductility heuristic: total alloying (Cr+Ti+W+Zr) <= 0.20
TOTAL_MAX = 0.20

# Activation limits consistent with examples/pure_element_neutronics_run.py
CRIT_LIMITS = {"He_appm": 1172.2 / 2.0, "H_appm": 1200.0}
DOSE_LIMITS = {30: 1e3, 365: 1.0, 5 * 365: 1e-2, 36500: 1e-4}


def main() -> None:
    """Run the constrained sampling + CALPHAD workflow and save figures/CSV."""
    # Load pure-element results to drive activation filters
    pure_results = load_pure_results(
        results_dir=str(RESULTS_BASE),
        chain_file=CHAIN_FILE,
        abs_file=ABS_FILE,
        elements=("V", "Cr", "Ti", "W", "Zr"),
        cooling_days=tuple(sorted(DOSE_LIMITS.keys())),
    )
    if set(pure_results.keys()) != {"V", "Cr", "Ti", "W", "Zr"}:
        print("Warning: Missing pure results for some elements; activation filter may be permissive.")

    activation_limits = ActivationLimits(gas_appm=CRIT_LIMITS, dose_at_days=DOSE_LIMITS)

    # Sample and filter
    comps = sample_and_filter(
        per_element_max=PER_ELEMENT_MAX,
        total_alloy_max=TOTAL_MAX,
        pure_results=pure_results,
        activation_limits=activation_limits,
        n_samples=1000,
        random_state=42,
    )
    print(f"Sampled {len(comps)} compositions; valid count = {comps['is_valid'].sum()}")

    # Embed and cluster
    emb, labels = embed_and_cluster_compositions(comps, random_state=42)
    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    from neutronics_calphad.workflows.embedding import plot_embedding as _plot_emb

    _plot_emb(emb, valid_mask=comps["is_valid"].values, labels=labels, ax=ax, title="Composition manifold with filters")
    fig.tight_layout()
    fig.savefig(WORKFLOW_OUT / "embedding.png", dpi=220)
    plt.close(fig)

    # Run CALPHAD for valid compositions
    cal_df = evaluate_calphad_for_valid(comps, temperature_k=823.5, database="TCHEA7")
    cal_df.to_csv(WORKFLOW_OUT / "calphad_results.csv", index=False)

    # Example ternary slices
    # 1) Fix Zr at 0.02 and vary (V, Cr, Ti) ternary
    grid1 = make_ternary_grid(fixed_element="Zr", fixed_value=0.02, varying_elements=("V", "Cr", "Ti"), resolution=51)
    # 2) Fix V at 0.85 and vary (Cr, Ti, W)
    grid2 = make_ternary_grid(fixed_element="V", fixed_value=0.85, varying_elements=("Cr", "Ti", "W"), resolution=51)

    # For coloring, use a simple validity surrogate: pass ductility+per-element caps
    def _is_within_caps(df: pd.DataFrame) -> np.ndarray:
        # per-element caps
        ok_per = np.ones(len(df), dtype=bool)
        for el, cap in PER_ELEMENT_MAX.items():
            ok_per &= df[el].values <= cap + 1e-12
        # ductility cap
        total_alloy = 1.0 - df["V"].values
        ok_duct = total_alloy <= TOTAL_MAX + 1e-12
        return ok_per & ok_duct

    v1 = _is_within_caps(grid1)
    v2 = _is_within_caps(grid2)

    fig1, ax1 = plt.subplots(1, 1, figsize=(6, 5))
    plot_ternary_slice(grid1, ("V", "Cr", "Ti"), value=v1.astype(int), ax=ax1, title="Ternary slice: Zr=0.02 (1=within caps)")
    fig1.tight_layout()
    fig1.savefig(WORKFLOW_OUT / "ternary_Zr_0p02_VCrTi.png", dpi=220)
    plt.close(fig1)

    fig2, ax2 = plt.subplots(1, 1, figsize=(6, 5))
    plot_ternary_slice(grid2, ("Cr", "Ti", "W"), value=v2.astype(int), ax=ax2, title="Ternary slice: V=0.85 (1=within caps)")
    fig2.tight_layout()
    fig2.savefig(WORKFLOW_OUT / "ternary_V_0p85_CrTiW.png", dpi=220)
    plt.close(fig2)

    print(f"Saved outputs under: {WORKFLOW_OUT}")


if __name__ == "__main__":
    main()


