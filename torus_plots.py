"""Level, power and runtime of the CKSD test on the torus vs. an MMD two-sample baseline (same design as the rankings).

Every model is a cosine model centred at `centre`, with concentrations (k1, k2, k3) = c(x) * kappa0: the analogue of
the Mallows spread nu(x) in permutations_plot.py. P_{Y|x} = Q1 has c(x) = |x|, Q2 has c(x) = c_h1 |x|, and Q3 equals Q2
on a random `fraction` of the observed x values and Q1 elsewhere.

Writes figures/torus_{Q1,Q2,Q3,runtime}.pdf.
"""
import time
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from torus import TorusGaussianKernel, TorusSteinKernel, cosine_score, sample_cosine
from utils import GaussianKernel1D, mmd_test, plot_rejection_rate, plot_runtime, rejection_rates, set_plot_style, \
    wild_bootstrap_pvalue

#%% experiment

rng = np.random.default_rng(43)
# ns = np.geomspace(100, 1000, 10).astype(int)
ns = [10, 20, 50, 100, 200, 500]
ns = np.array([n if n % 2 == 0 else n+1 for n in ns])  # having them even is simpler
nrep, B = 100, 1000
c_h1 = 5
kappa0 = np.array([3.0, 2.0, 1.0])     # (k1, k2, k3) at c = 1; unimodal for every c since k3 < k1 k2 / (k1 + k2)
centre = np.array([0.0, 0.0])          # (mu, nu)

KQ = TorusSteinKernel(GaussianKernel1D(gamma=1.0), TorusGaussianKernel(gamma=1.0))


def cosine_params(x, c):
    """(k1, k2, k3, mu, nu) at the covariates x of the cosine model with concentration c(x)"""
    s = c(x)
    return s * kappa0[0], s * kappa0[1], s * kappa0[2], np.full_like(x, centre[0]), np.full_like(x, centre[1])


def c3(x, S):
    """concentration of Q3: c_h1|x| on the chosen x values S, |x| elsewhere"""
    return np.where(np.isin(x, S), c_h1 * np.abs(x), np.abs(x))


fraction = 0.2
out = []
for n in tqdm(list(ns), desc="n"):
    for rep in range(nrep):
        xs = rng.uniform(-1, 1, n)
        S = rng.choice(xs, size=int(n * fraction), replace=False)  # the fraction% of x values where Q3 = Q2

        ts = sample_cosine(rng, *cosine_params(xs, np.abs))       # P_{Y|x}, angles (n, 2)

        xs1, xs2 = xs[:n // 2], xs[n // 2:]
        ts1 = ts[:n // 2]                                         # observed data, half 1; ts2 is replaced by ts2p

        # each model = its concentration c(x); the same c is used for CKSD (score) and MMD (sampling)
        models = {
            "Q1": lambda x: np.abs(x),
            "Q2": lambda x: c_h1 * np.abs(x),
            "Q3": lambda x: c3(x, S),
        }
        for name, c in models.items():
            # CKSD: all n points, only needs the score d_theta log q
            t0 = time.time()
            K = KQ.gram(lambda x, t: cosine_score(t, *cosine_params(x, c)), xs, ts)
            p_cksd = wild_bootstrap_pvalue(K, rng, B)
            t1 = time.time()

            # MMD: half 1 observed vs. half 2 with y resampled from the model
            ts2p = sample_cosine(rng, *cosine_params(xs2, c))
            mmd, p_mmd = mmd_test(xs1, ts1, xs2, ts2p, KQ.k, KQ.l, rng, B)
            t2 = time.time()

            out.append({
                "n": n, "rep": rep, "model": name,
                "cksd": K.mean(), "p_cksd": p_cksd, "time_cksd": t1 - t0,
                "mmd": mmd, "p_mmd": p_mmd, "time_mmd": t2 - t1,
            })

df = pd.DataFrame(out)

#%% plot style

figures_dir = Path("figures")
figures_dir.mkdir(parents=True, exist_ok=True)

# mpl.use("TkAgg")
set_plot_style(fontsize=10)

#%% figures

alpha = 0.05
figsize = (6.5 / 4, 1.6)
palette = "cubehelix"
hue_order = ["CKSD", "MMD"]
legend_width = 0.9

# axis labels
xlabel = "$n$"
ylabel_level = "Level"
ylabel_power = "Power"
ylabel_runtime = "Runtime [s]"

# long format: one row per (n, rep, model, test)
cols = ["n", "rep", "model"]
dflong = pd.concat([
    df[cols].assign(test="CKSD", p_value=df["p_cksd"], time=df["time_cksd"]),
    df[cols].assign(test="MMD", p_value=df["p_mmd"], time=df["time_mmd"]),
], ignore_index=True)

rates, times, ylim_power = rejection_rates(dflong, alpha)
style = dict(figsize=figsize, palette=palette)

# Q1: level
plot_rejection_rate(rates, "Q1", figures_dir / "torus_Q1.pdf", hue_order,
                    xlabel=xlabel, ylabel=ylabel_level, alpha=alpha, **style)

# Q2, Q3: power, on a common y-range
plot_rejection_rate(rates, "Q2", figures_dir / "torus_Q2.pdf", hue_order,
                    xlabel=xlabel, ylabel=ylabel_power, ylim=ylim_power, **style)
plot_rejection_rate(rates, "Q3", figures_dir / "torus_Q3.pdf", hue_order,
                    xlabel=xlabel, ylabel=ylabel_power, ylim=ylim_power, **style)

# runtime, with the legend of all four figures
plot_runtime(times, figures_dir / "torus_runtime.pdf", hue_order,
             xlabel=xlabel, ylabel=ylabel_runtime, legend_width=legend_width, legend_loc="center", **style)
