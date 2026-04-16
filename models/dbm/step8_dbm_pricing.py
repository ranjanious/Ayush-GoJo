"""
Phase B, Step 8 -- DBM Bond Pricing and Portfolio Valuation
===========================================================
Feeds DBM short-rate paths into the shared pricing and risk
infrastructure. Computes terminal portfolio values at T = 1, 2, 5, 10
years and confirms the $90-$110 average price sanity check.

Run:
    python models/dbm/step8_dbm_pricing.py

Outputs:
    figures/step8_figure.png
    results/step8_results.json
"""

import sys, os, json, math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from shared.bond_pricing import price_bond, price_bond_flat, portfolio_value
from shared.risk_engine   import summary_stats


def _nan_to_none(v):
    """Convert float NaN/Inf to None for strict-JSON output."""
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return None
    return v

# -- Parameters -----------------------------------------------------------
np.random.seed(42)

R0    = 0.0372    # 3-month CMT, March 17 2026
MU    = 0.0       # zero drift (benchmark); replace in Phase F
SIGMA = 0.01      # annual vol placeholder; replace in Phase F
DT    = 1 / 12    # monthly grid
N_PATHS  = 10_000
N_STEPS  = 120    # 10 years

F   = 100.0
C_A = 1.84        # semiannual coupon, Bond A (3.68% par yield, 2-year)
C_B = 2.10        # semiannual coupon, Bond B (4.20% par yield, 10-year)

PAY_DATES_A = np.arange(0.5, 2.5,  0.5)   # [0.5, 1.0, 1.5, 2.0]
PAY_DATES_B = np.arange(0.5, 10.5, 0.5)   # [0.5, ..., 10.0]

CFS_A = np.full(len(PAY_DATES_A), C_A); CFS_A[-1] += F
CFS_B = np.full(len(PAY_DATES_B), C_B); CFS_B[-1] += F

HORIZONS = [1, 2, 5, 10]


# -- DBM simulator (Euler-Maruyama) ---------------------------------------
def simulate_dbm(r0, mu, sigma, dt, n_steps, n_paths):
    """
    dr_t = mu*dt + sigma*dW_t
    Discrete update: r_{k+1} = r_k + mu*dt + sigma*sqrt(dt)*Z_k
    """
    paths = np.zeros((n_paths, n_steps + 1))
    paths[:, 0] = r0
    Z = np.random.standard_normal((n_paths, n_steps))
    for k in range(n_steps):
        paths[:, k + 1] = paths[:, k] + mu * dt + sigma * np.sqrt(dt) * Z[:, k]
    return paths


# -- Run simulation -------------------------------------------------------
paths = simulate_dbm(R0, MU, SIGMA, DT, N_STEPS, N_PATHS)
print(f"DBM paths: {paths.shape}  (n_paths x n_steps+1)")

# -- Price bonds and compute portfolio values -----------------------------
records = {}

for T in HORIZONS:
    pA = price_bond(paths, CFS_A, PAY_DATES_A, T, DT)
    pB = price_bond(paths, CFS_B, PAY_DATES_B, T, DT)

    # Bond A matures at year 2; treat as par received after maturity
    if T >= 2.0:
        pA = np.full(N_PATHS, F)
    if T >= 10.0:
        pB = np.full(N_PATHS, F)

    pV = portfolio_value(pA, pB)

    # Constant-rate benchmark
    pA_const = price_bond_flat(R0, CFS_A, PAY_DATES_A, T) if T < 2.0  else F
    pB_const = price_bond_flat(R0, CFS_B, PAY_DATES_B, T) if T < 10.0 else F
    pV_const = portfolio_value(pA_const, pB_const)

    sA = summary_stats(pA, f"Bond A @ T={T}", V0=F)
    sB = summary_stats(pB, f"Bond B @ T={T}", V0=F)
    sV = summary_stats(pV, f"Portfolio @ T={T}", V0=100.0)

    records[T] = dict(pA=pA, pB=pB, pV=pV,
                      pA_const=pA_const, pB_const=pB_const, pV_const=pV_const,
                      sA=sA, sB=sB, sV=sV)

    sanity = "PASS" if 90 <= sA["Mean"] <= 110 and 90 <= sB["Mean"] <= 110 else "FAIL"
    print(f"\nT={T}yr | BondA mean=${sA['Mean']:.2f} std=${sA['Std']:.2f} "
          f"| BondB mean=${sB['Mean']:.2f} std=${sB['Std']:.2f} "
          f"| Portfolio VaR99=${sV['VaR_99']:.2f} P(loss)={sV['P_loss']*100:.1f}% "
          f"| Sanity: {sanity}")


# -- Save results JSON ----------------------------------------------------
os.makedirs("results", exist_ok=True)
serializable = {}
for T in HORIZONS:
    r = records[T]
    serializable[str(T)] = {
        "const_portf" : _nan_to_none(float(r["pV_const"])),
        "Bond_A"      : {k: _nan_to_none(float(v)) for k, v in r["sA"].items() if k != "Label"},
        "Bond_B"      : {k: _nan_to_none(float(v)) for k, v in r["sB"].items() if k != "Label"},
        "Portfolio"   : {k: _nan_to_none(float(v)) for k, v in r["sV"].items() if k != "Label"},
    }
with open("results/step8_results.json", "w") as f:
    json.dump(serializable, f, indent=2, allow_nan=False)
print("\nSaved: results/step8_results.json")


# -- Figure ---------------------------------------------------------------
os.makedirs("figures", exist_ok=True)
t_grid = np.arange(0, N_STEPS + 1) * DT

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Phase B * Step 8 -- DBM Diagnostics", fontsize=12, fontweight='bold')

# Panel 1: rate paths fan chart
ax = axes[0]
for i in range(200):
    ax.plot(t_grid, paths[i] * 100, color="#4472C4", alpha=0.08, linewidth=0.6)
pcts = np.percentile(paths, [5, 25, 50, 75, 95], axis=0)
ax.plot(t_grid, pcts[2] * 100, color="#C00000", linewidth=1.5, label="Median")
ax.fill_between(t_grid, pcts[0]*100, pcts[4]*100, color="#4472C4", alpha=0.15, label="5-95th pct")
ax.fill_between(t_grid, pcts[1]*100, pcts[3]*100, color="#4472C4", alpha=0.25, label="25-75th pct")
ax.axhline(R0 * 100, color="black", linestyle="--", linewidth=1, label=f"r0 = {R0*100:.2f}%")
ax.set_xlabel("Time (years)"); ax.set_ylabel("Short rate (%)")
ax.set_title("DBM Short-Rate Paths (200 shown)")
ax.legend(fontsize=8); ax.grid(alpha=0.3)

# Panel 2: Bond B price distributions
ax2 = axes[1]
for T, col in {1: "#4472C4", 2: "#ED7D31", 5: "#70AD47"}.items():
    ax2.hist(records[T]["pB"], bins=60, alpha=0.45, color=col, density=True, label=f"T={T}yr")
ax2.axvline(100.0, color="black", linestyle="--", linewidth=1.2, label="Par ($100)")
ax2.set_xlabel("Bond B Mark-to-Market Price ($)")
ax2.set_title("Bond B Terminal Price Distribution")
ax2.legend(fontsize=8); ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("figures/step8_figure.png", dpi=180, bbox_inches="tight")
plt.close()
print("Saved: figures/step8_figure.png")
