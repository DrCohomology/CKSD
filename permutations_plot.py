"""Level, power and runtime of the CKSD test on rankings (Sec. 4.2) vs. an MMD two-sample baseline.

Covariates x ~ U(-1, 1); the rankings of na = 5 items follow a Mallows model centred at the identity ranking, with
spread nu(x): q_x(y) ∝ exp(-nu(x) d_K(y, centre)). P_{Y|x} = Q1 has nu(x) = |x|, Q2 has nu(x) = nu_h1 |x|, and Q3
equals Q2 on a random `fraction` of the observed x values and Q1 elsewhere: Q1 gives the level, Q2 and Q3 the power.

Competitors: the CKSD with three orderings of S_n (SJT / lexicographic / random shuffle), which fix the cyclic shift
of the Stein operator, and the MMD test of the observed first half of the sample against the second half with y
resampled from the model.

Writes figures/permutations_{Q1,Q2,Q3,runtime}.pdf.
"""
import itertools
import time
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.random import default_rng
from tqdm.auto import tqdm

from permutations import MallowsKernel, YangSteinKernel1D, gray_space, kendall
from utils import GaussianKernel1D, mmd_test, plot_rejection_rate, plot_runtime, rejection_rates, sample_rows, \
    set_plot_style, wild_bootstrap_pvalue

#%% experiment

rng = np.random.default_rng(43)
# ns = np.geomspace(100, 1000, 10).astype(int)
ns = [10, 20, 50, 100, 200, 500]
ns = np.array([n if n % 2 == 0 else n+1 for n in ns])  # having them even is simpler
na, nrep, B = 5, 100, 1000
nu_h1 = 5

# reference list of S_n: lexicographic (01234, 01243, ..., 43210); rankings are sampled as indices into it.
# Every ordering is a permutation `order` of this list, i.e. its space is space_lex[order].
space_lex = np.array(list(itertools.permutations(range(na))))
M = len(space_lex)
lex_index = {tuple(p): i for i, p in enumerate(space_lex)}
center1 = np.arange(na)
center2 = np.arange(na, 0, -1)                       # currently unused
d_c1 = kendall(space_lex, center1[None])[:, 0]       # Kendall distance of every ranking to the centre (lex. index)
d_c2 = kendall(space_lex, center2[None])[:, 0]

k = GaussianKernel1D(gamma=1.0)
l = MallowsKernel(na)
order_sjt = np.array([lex_index[tuple(p)] for p in gray_space(na)])
order_lex = np.arange(M)
KQ_sjt = YangSteinKernel1D(k, l, space_lex[order_sjt])
KQ_lex = YangSteinKernel1D(k, l, space_lex[order_lex])
L_lex = KQ_lex.L                                     # l on space_lex, used by the MMD test


def l_lex(ja, jb):
    """Mallows kernel between rankings given by their index in space_lex: a lookup in L_lex."""
    return L_lex[np.ix_(ja, jb)]


def nu3(x, S):
    """spread of Q3: nu_h1|x| on the chosen x values S, |x| elsewhere"""
    return np.where(np.isin(x, S), nu_h1 * np.abs(x), np.abs(x))


fraction = 0.2
out = []
for n in tqdm(list(ns), desc="n"):
    for rep in range(nrep):
        xs = rng.uniform(-1, 1, n)
        S = rng.choice(xs, size=int(n * fraction), replace=False)  # the fraction% of x values where Q3 = Q2

        P = np.exp(-np.abs(xs)[:, None] * d_c1[None, :])  # P_{Y|x}
        P /= P.sum(axis=1, keepdims=True)
        js = sample_rows(rng, P)                          # indices of space_lex

        xs1, xs2 = xs[:n // 2], xs[n // 2:]
        js1 = js[:n // 2]                                 # observed data, half 1; js2 is replaced by js2p

        # CKSD competitors: same data, different ordering of S_n (a fresh random shuffle in every repetition)
        order_random = rng.permutation(M)
        cksd_tests = {
            "CKSD (jst)": (KQ_sjt, order_sjt),
            "CKSD (lex)": (KQ_lex, order_lex),
            "CKSD (random)": (YangSteinKernel1D(k, l, space_lex[order_random]), order_random),
        }
        boot_seed = rng.integers(2 ** 32)                 # same bootstrap multipliers for all orderings

        # each model = its spread nu(x); the same nu is used for CKSD (log q) and MMD (sampling)
        models = {
            "Q1": lambda x: np.abs(x),
            "Q2": lambda x: nu_h1 * np.abs(x),
            "Q3": lambda x: nu3(x, S),
        }
        for name, nu in models.items():
            # CKSD: all n points, only needs log q (up to a constant)
            for test, (KQ, order) in cksd_tests.items():
                pos = np.empty(M, dtype=int)
                pos[order] = np.arange(M)
                js_o, d_o = pos[js], d_c1[order]          # the same rankings and distances, indexed in this ordering
                t0 = time.time()
                K = KQ.gram(lambda x, j: -nu(x) * d_o[j], xs, js_o)
                p_cksd = wild_bootstrap_pvalue(K, default_rng(boot_seed), B)
                t1 = time.time()
                out.append({"n": n, "rep": rep, "model": name, "test": test,
                            "stat": K.mean(), "p_value": p_cksd, "time": t1 - t0})

            # MMD: half 1 observed vs. half 2 with y resampled from the model
            t0 = time.time()
            Q = np.exp(-nu(xs2)[:, None] * d_c1[None, :])
            Q /= Q.sum(axis=1, keepdims=True)
            js2p = sample_rows(rng, Q)
            mmd, p_mmd = mmd_test(xs1, js1, xs2, js2p, k, l_lex, rng, B)
            t1 = time.time()
            out.append({"n": n, "rep": rep, "model": name, "test": "MMD",
                        "stat": mmd, "p_value": p_mmd, "time": t1 - t0})

df = pd.DataFrame(out)                                    # long format: one row per (n, rep, model, test)

#%% plot style

figures_dir = Path("figures")
figures_dir.mkdir(parents=True, exist_ok=True)

# mpl.use("TkAgg")
set_plot_style(fontsize=10)

#%% figures

alpha = 0.05
figsize = (6.5 / 4, 1.6)
palette = "cubehelix"
hue_order = ["CKSD (jst)", "CKSD (lex)", "CKSD (random)", "MMD"]
legend_width = 1.7

# axis labels
xlabel = "$n$"
ylabel_level = "Level"
ylabel_power = "Power"
ylabel_runtime = "Runtime [s]"

rates, times, ylim_power = rejection_rates(df, alpha)
style = dict(figsize=figsize, palette=palette)

# Q1: level
plot_rejection_rate(rates, "Q1", figures_dir / "permutations_Q1.pdf", hue_order,
                    xlabel=xlabel, ylabel=ylabel_level, alpha=alpha, **style)

# Q2, Q3: power, on a common y-range
plot_rejection_rate(rates, "Q2", figures_dir / "permutations_Q2.pdf", hue_order,
                    xlabel=xlabel, ylabel=ylabel_power, ylim=ylim_power, **style)
plot_rejection_rate(rates, "Q3", figures_dir / "permutations_Q3.pdf", hue_order,
                    xlabel=xlabel, ylabel=ylabel_power, ylim=ylim_power, **style)

# runtime, with the legend of all four figures
plot_runtime(times, figures_dir / "permutations_runtime.pdf", hue_order,
             xlabel=xlabel, ylabel=ylabel_runtime, legend_width=legend_width, legend_loc="center left", **style)
