# Requirements: pandas, numpy, matplotlib, scipy
from typing import Tuple

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# --- Load your sweep CSV ---
df = pd.read_csv("./analysis_results/particle_sweep_neutronics_run_batches/final_analysis/monte_carlo_sweep_summary.csv")

# --- Helper: log–log linear fit ---
def fit_loglog(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float]:
    """Fit a log-log line and return slope, intercept, and R^2."""
    mask = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    xlog = np.log10(x[mask]); ylog = np.log10(y[mask])
    m, b, r, p, se = stats.linregress(xlog, ylog)
    return m, b, r**2

# --- Identify columns ---
hist_col = next(c for c in df.columns if "hist" in c.lower() or "total" in c.lower())

flux_cols = [c for c in df.columns if "flux" in c.lower() and ("l2" in c.lower() or "error" in c.lower())]
activity_cols = [c for c in df.columns if "activity" in c.lower() and ("l2" in c.lower() or "error" in c.lower())]
if not activity_cols:
    activity_cols = [c for c in df.columns
                     if any(tag in c.lower() for tag in [" v", "_v", "zr", " w", "_w"])
                     and ("l2" in c.lower() or "error" in c.lower())]

series = {}
if flux_cols:
    series["Flux"] = (df[hist_col].to_numpy(), df[flux_cols[0]].to_numpy())
for col in activity_cols[:3]:
    series[col.replace("_", " ")] = (df[hist_col].to_numpy(), df[col].to_numpy())

# --- Single consolidated chart (log–log) ---
plt.figure(figsize=(8, 6))

colors = ["#2A33C3", "#A35D00", "#0B7285", "#8F2D56", "#6E8B00"]
fit_lines = []
legend_handles = []
legend_labels = []

for idx, (label, (x, y)) in enumerate(series.items()):
    color = colors[idx % len(colors)]
    sc = plt.scatter(x, y, s=30, color=color)
    legend_handles.append(sc); legend_labels.append(label)
    m, b, r2 = fit_loglog(x, y)
    xs = np.linspace(np.nanmin(x[x > 0]), np.nanmax(x[x > 0]), 200)
    ys = 10**(b + m * np.log10(xs))
    plt.plot(xs, ys, linewidth=1.5, color=color)
    fit_lines.append(f"{label}: slope={m:.3f}, R^2={r2:.3f}")

plt.xscale("log"); plt.yscale("log")
plt.xlabel("Total histories (particles × batches)", fontweight="bold")
plt.ylabel("Relative L2 deviation vs reference", fontweight="bold")
plt.legend(legend_handles, legend_labels, loc="lower left")
plt.gca().text(0.98, 0.98, "\n".join(fit_lines),
               transform=plt.gca().transAxes, va="top", ha="right",
               fontsize=9, bbox=dict(boxstyle="round", alpha=0.2))
plt.tight_layout()
plt.savefig("./analysis_results/particle_sweep_neutronics_run_batches/appendix_particle_sweep.pdf", dpi=300)
