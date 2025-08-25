import pandas as pd, numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

# --- Load & select columns ---
df = pd.read_csv("DBTT_Data_V-Main_Data.csv")
y = pd.to_numeric(df["DBTT(C)"], errors="coerce")
V = pd.to_numeric(df["V"], errors="coerce")
mask = y.notna() & V.notna()
y = y[mask].to_numpy()
X = V[mask].to_numpy().reshape(-1, 1)

# --- Train/test split & fit ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
lm = LinearRegression()
lm.fit(X_train, y_train)
y_pred = lm.predict(X_test)

# --- Metrics (documented in sklearn/scipy) ---
r2 = r2_score(y_test, y_pred)              # R² definition per sklearn docs
mae = mean_absolute_error(y_test, y_pred)  # MAE per sklearn docs
r, p = pearsonr(X.ravel(), y)              # Pearson r,p over full available data
n = len(y)

# --- Publication-ish styling (Matplotlib rcParams) ---
plt.rcParams.update({
    "figure.figsize": (4.0, 3.2),   # good journal-friendly aspect
    "figure.dpi": 150,              # on-screen; savefig will control export
    "font.size": 9,                 # ~9 pt body text
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.linewidth": 0.8,
    "grid.alpha": 0.25,
})

fig, ax = plt.subplots()

# --- Scatter and fitted line ---
ax.scatter(X, y, s=16)  # small markers for dense points; default color
x_span = np.linspace(X.min(), X.max(), 200).reshape(-1,1)
ax.plot(x_span, lm.predict(x_span), linewidth=1.5)

# --- Optional: simple 95% CI band for the mean prediction (OLS closed form) ---
#   CI here is for the *mean* of y|x, not individual prediction intervals.
#   Comment out if you prefer a cleaner look.
X1 = np.c_[np.ones_like(X_train), X_train]
XtX_inv = np.linalg.inv(X1.T @ X1)
x1_span = np.c_[np.ones_like(x_span), x_span]
y_hat = lm.predict(x_span)
# s^2 (residual variance) from train split
res = y_train - lm.predict(X_train)
s2 = (res**2).sum() / (len(y_train) - 2)
se_mean = np.sqrt(np.einsum("ij,jk,ik->i", x1_span, XtX_inv, x1_span) * s2)
ax.fill_between(x_span.ravel(), y_hat - 1.96*se_mean, y_hat + 1.96*se_mean, alpha=0.15)

# --- Axes labels & grid ---
ax.set_xlabel("V (wt%)")
ax.set_ylabel("DBTT (°C)")
ax.grid(True)

# --- Stats panel (top-left, axes coords) ---
text = (
    f"n = {n}\n"
    f"slope = {lm.coef_[0]:.2f} °C/(wt% V)\n"
    f"intercept = {lm.intercept_:.1f} °C\n"
    f"$R^2$ (test) = {r2:.3f}\n"
    f"MAE (test) = {mae:.1f} °C\n"
    f"Pearson r = {r:.3f} (p = {p:.2e})"
)
ax.text(
    0.02, 0.02, text,
    ha="left", va="bottom", transform=ax.transAxes,
    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8, linewidth=0.8)
)

fig.set_size_inches(6.0, 4.5)
fig.tight_layout()

# --- Save both raster and vector; vector is best for journals ---
fig.savefig("dbtt_vs_v.png", dpi=600, bbox_inches="tight")  # high-res raster
fig.savefig("dbtt_vs_v.svg", bbox_inches="tight")           # vector (infinite zoom)

plt.close(fig)

