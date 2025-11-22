import pandas as pd
import numpy as np
from iminuit import Minuit
from scipy.stats import poisson
import matplotlib.pyplot as plt
import re

# ============================================================
# INPUT
# ============================================================
input_pkl = "bdt_distributions/bdt_binned_with_cross_sections.pkl"
df = pd.read_pickle(input_pkl)

signal_names = [
    "wzp6_ee_Henueqq_ecm125.root",
    "wzp6_ee_Htaunutauqq_ecm125.root"
]

# Classify signal vs background
df["category"] = np.where(df["process"].isin(signal_names), "signal", "background")

# Aggregate per bin *and process*
agg_proc = df.groupby(["bin_center", "process"])["reco_level_events"].sum().unstack(fill_value=0)

# Separate signal and backgrounds
signal_cols = [p for p in agg_proc.columns if p in signal_names]
bkg_cols = [p for p in agg_proc.columns if p not in signal_names]

# Compute total S and B per bin
agg_proc["S"] = agg_proc[signal_cols].sum(axis=1)
agg_proc["B_total"] = agg_proc[bkg_cols].sum(axis=1)
agg_proc.reset_index(inplace=True)

# Asimov observed counts
N_obs = agg_proc["S"] + agg_proc["B_total"]

# ============================================================
# Helper: Clean process labels
# ============================================================
def clean_label(name):
    """
    Extract substring between 'ee_' and '_ecm125'
    Example: 'wzp6_ee_Hqqmunumu_ecm125.root' → 'Hqqmunumu'
    """
    match = re.search(r"ee_(.+?)_ecm125", name)
    if match:
        return match.group(1)
    return name.replace(".root", "")

# ============================================================
# Step plots
# ============================================================
plt.figure(figsize=(8, 7), dpi=300)

# Plot each background separately
for i, bkg in enumerate(bkg_cols):
    plt.step(
        agg_proc["bin_center"],
        agg_proc[bkg],
        where='mid',
        label=clean_label(bkg),
        linewidth=1.2
    )
    plt.fill_between(
        agg_proc["bin_center"],
        0,
        agg_proc[bkg],
        step='mid',
        alpha=0.2
    )

# Plot each signal separately (with red tones)
colors = ["red", "darkorange", "firebrick", "tomato"]
for i, sig in enumerate(signal_cols):
    plt.step(
        agg_proc["bin_center"],
        agg_proc[sig],
        where='mid',
        label=f"Signal ({clean_label(sig)})",
        color=colors[i % len(colors)],
        linewidth=2
    )

plt.yscale("log")
plt.xlabel("BDT bin center")
plt.ylabel("Events (log scale)")
plt.title("BDT Input Distribution by Process")
plt.legend(fontsize=6, ncol=2)
plt.tight_layout()
plt.savefig("bdt_breakdown.png")
plt.show()

# ============================================================
# Profile-likelihood fit with per-background nuisances
# ============================================================

# Prepare arrays
S = agg_proc["S"].values
B_components = agg_proc[bkg_cols].values  # shape: (nbins, nbkg)
N_obs = S + B_components.sum(axis=1)

# Avoid division-by-zero issues
B_components = np.clip(B_components, 1e-9, None)

sigma_B_rel = 0.05  # 5% relative uncertainty per background

def nll(mu, *thetas):
    """
    Profile likelihood:
      mu: signal strength
      thetas[i]: scale factor for background i
    """
    thetas = np.array(thetas)
    B_scaled = np.sum(B_components * thetas, axis=1)
    expected = mu * S + B_scaled

    # Poisson term
    poisson_term = -np.sum(poisson.logpmf(N_obs, expected))

    # Gaussian constraints for nuisance parameters
    gauss_term = 0.5 * np.sum(((thetas - 1) / sigma_B_rel)**2)

    return poisson_term + gauss_term

# Initialize parameters
init_vals = [1.0] + [1.0]*len(bkg_cols)
param_names = ["mu"] + [f"theta_{b}" for b in bkg_cols]

m = Minuit(nll, *init_vals, name=param_names)
m.limits["mu"] = (0, None)
for b in bkg_cols:
    m.limits[f"theta_{b}"] = (0, None)

m.migrad()

# Extract results
mu_hat = m.values["mu"]
mu_err = m.errors["mu"] if np.isfinite(m.errors["mu"]) else np.nan
theta_vals = {clean_label(b): m.values[f"theta_{b}"] for b in bkg_cols}

# ============================================================
# Safe significance calculation
# ============================================================
B_sum = np.clip(B_components.sum(axis=1), 1e-9, None)
if np.isfinite(mu_err) and mu_err > 0:
    significance = mu_hat / mu_err
else:
    # Fallback: Asimov approximation
    significance = np.sqrt(2 * np.sum((S + B_sum) * np.log(1 + S / B_sum) - S))

# ============================================================
# Print results
# ============================================================
print("\n=== Profile Likelihood Fit Results (Per-Background) ===")
print(f"Best-fit signal strength: μ = {mu_hat:.3f} ± {mu_err:.3f}")
print(f"Expected significance: Z = {significance:.2f}σ\n")
print("Background scale factors:")
for b, val in theta_vals.items():
    print(f"  {b:20s}: θ = {val:.3f}")
