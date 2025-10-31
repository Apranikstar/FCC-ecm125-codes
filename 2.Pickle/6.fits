import pandas as pd
import numpy as np
from iminuit import Minuit
from scipy.stats import poisson
import matplotlib.pyplot as plt


# ------------------ INPUT ------------------
input_pkl = "off-mu-bdt_binned_with_cross_sections.pkl"
df = pd.read_pickle(input_pkl)

signal_names = [
    "wzp6_ee_Hqqmunumu_ecm125.root",
    "wzp6_ee_Hqqtaunutau_ecm125.root"
]

# Classify signal vs background
df["category"] = np.where(df["process"].isin(signal_names), "signal", "background")

# Aggregate per BDT bin
agg = df.groupby(["bin_center", "category"])["reco_level_events"].sum().unstack(fill_value=0)
agg["S"] = agg["signal"]
agg["B"] = agg["background"]
agg.reset_index(inplace=True)

# Asimov observed counts
N_obs = agg["S"] + agg["B"]


# ------------------ STAGE 1: Step plots in log scale ------------------
plt.figure(figsize=(8,7),dpi=300)

# Background as filled step
plt.step(
    agg["bin_center"],
    agg["B"],
    where='mid',
    label="Background",
    color='C0',
    linewidth=1.5
)
plt.fill_between(
    agg["bin_center"],
    0,
    agg["B"],
    step='mid',
    alpha=0.3,
    color='C0'
)

# Signal as line step
plt.step(
    agg["bin_center"],
    agg["S"],
    where='mid',
    label="Signal",
    color='C1',
    linewidth=2
)

plt.yscale("log")
plt.xlabel("BDT bin center")
plt.ylabel("Normalized events(log scale)")
plt.title("Off-Shell Muon BDT Binned Input Distribution")
plt.legend()
plt.tight_layout()
plt.savefig("off-mu.png")
plt.show()



# ============================================================
# STAGE 2: Fit signal strength μ using Poisson likelihood
# ============================================================

from iminuit import Minuit
from scipy.stats import poisson
import numpy as np

def nll(mu):
    """Negative log-likelihood for Poisson counting model"""
    muS = mu * agg["S"]
    expected = muS + agg["B"]
    return -np.sum(poisson.logpmf(N_obs, expected))

# Fit mu >= 0
m = Minuit(nll, mu=1.0)
m.limits["mu"] = (0, None)
m.migrad()

# Fit results
mu_hat = m.values["mu"]
mu_err = m.errors["mu"]

# Safe significance calculation
if mu_err > 0:
    significance = mu_hat / mu_err
else:
    # Use Asimov formula if error is zero or undefined
    S = agg["S"]
    B = agg["B"]
    significance = np.sqrt(2 * np.sum((S + B) * np.log(1 + S/B) - S))

print("\n=== Fit Results ===")
print(f"Best-fit signal strength: μ = {mu_hat:.3f} ± {mu_err:.3f}")
print(f"Expected significance: Z = {significance:.2f}σ")


# ============================================================
# STAGE 2: Profile-likelihood fit with background nuisance
# ============================================================

import numpy as np
from iminuit import Minuit
from scipy.stats import poisson, norm

# -------------------- Input --------------------
# S: expected signal per bin
# B: expected background per bin
# N_obs: observed events (Asimov dataset)
S = agg["S"].values
B = agg["B"].values
N_obs = S + B  # Asimov dataset

# Assume 5% relative background uncertainty as an example
sigma_B_rel = 0.05
sigma_B = sigma_B_rel * B

# -------------------- Negative log-likelihood --------------------
def nll(mu, theta):
    """
    Profile likelihood:
    - mu: signal strength
    - theta: multiplicative factor for background normalization
    """
    # Background with nuisance
    B_eff = B * theta
    # Poisson term
    poisson_term = -np.sum(poisson.logpmf(N_obs, mu * S + B_eff))
    # Gaussian constraint for nuisance
    gauss_term = 0.5 * np.sum(((theta - 1)/sigma_B_rel)**2)
    return poisson_term + gauss_term

# -------------------- Minimize --------------------
m = Minuit(nll, mu=1.0, theta=1.0)
m.limits["mu"] = (0, None)
m.limits["theta"] = (0, None)  # background positive
m.migrad()

# -------------------- Extract results --------------------
mu_hat = m.values["mu"]
theta_hat = m.values["theta"]
mu_err = m.errors["mu"]

# Safe significance calculation
if mu_err > 0:
    significance = mu_hat / mu_err
else:
    # Asimov formula with nuisance approximated as no uncertainty
    significance = np.sqrt(2 * np.sum((S + B) * np.log(1 + S/B) - S))

# -------------------- Print --------------------
print("\n=== Profile Likelihood Fit Results ===")
print(f"Best-fit signal strength: μ = {mu_hat:.3f} ± {mu_err:.3f}")
print(f"Best-fit background scale: θ = {theta_hat:.3f}")
print(f"Expected significance: Z = {significance:.2f}σ")
