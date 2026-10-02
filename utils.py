"""Code shared by the CKSD experiments on rankings (permutations.py) and on the torus (torus.py).

- `GaussianKernel1D`: the kernel k on the covariate space X = R;
- `sample_rows`: vectorised sampling from a batch of discrete distributions;
- `wild_bootstrap_pvalue`, `mmd_test`: the two tests compared in the experiments;
- `set_plot_style`, `rejection_rates`, `plot_rejection_rate`, `plot_runtime`: the level / power / runtime figures.
"""
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


# ====================================================================================================== kernels


class GaussianKernel1D:
    """Gaussian kernel on the real line, k(x, x') = exp(-gamma (x - x')^2).

    Parameters
    ----------
    gamma : float, default 1.0
        Inverse squared length scale.
    """

    def __init__(self, gamma=1.0):
        self.gamma = gamma

    def __call__(self, x, xp):
        """Gram matrix k(x[i], xp[j]) of the 1-d arrays x (N,) and xp (M,) -> (N, M)."""
        return np.exp(-self.gamma * (x[:, None] - xp[None, :]) ** 2)


# ===================================================================================================== sampling


def sample_rows(rng, P):
    """One index per row of the row-stochastic matrix P (inverse-CDF, vectorised).

    Parameters
    ----------
    rng : numpy.random.Generator
    P : (N, M) array
        Row i is a probability vector over {0, ..., M - 1}.

    Returns
    -------
    (N,) int array
        Entry i is drawn from P[i], independently across rows.
    """
    u = rng.random((len(P), 1))
    return np.minimum((P.cumsum(axis=1) < u).sum(axis=1), P.shape[1] - 1)


# ======================================================================================================== tests


def wild_bootstrap_pvalue(K, rng, B=1000):
    """p-value of the CKSD test, by the Rademacher wild bootstrap of the V-statistic.

    The statistic is the V-statistic n^-2 sum_ij K_ij = K.mean(); each bootstrap replicate is n^-2 w^T K w,
    with i.i.d. Rademacher weights w in {-1, +1}^n.

    Parameters
    ----------
    K : (n, n) array
        Gram matrix of the Stein kernel K_Q on the sample, K_ij = K_Q((x_i, y_i), (x_j, y_j)).
    rng : numpy.random.Generator
        Source of the bootstrap weights.
    B : int, default 1000
        Number of bootstrap replicates.

    Returns
    -------
    float
        (1 + #{b : replicate_b >= statistic}) / (1 + B).
    """
    n = len(K)
    W = rng.choice([-1.0, 1.0], size=(B, n))
    boot = ((W @ K) * W).sum(axis=1) / n ** 2
    return (1 + np.sum(boot >= K.mean())) / (1 + B)


def mmd_test(xa, ya, xb, yb, k, l, rng, B=1000):
    """Permutation test of H0: (xa, ya) and (xb, yb) have the same joint law, with the MMD.

    Uses the product kernel k(x, x') l(y, y') on X x Y and the V-statistic estimate of MMD^2; its null distribution
    is approximated by B random relabellings of the pooled sample.

    Parameters
    ----------
    xa, xb : (na,), (nb,) arrays
        Covariates of the two samples.
    ya, yb : arrays with na and nb rows
        Responses of the two samples, in whatever representation `l` takes (angles (n, 2) on the torus,
        indices into the list of all rankings for S_n).
    k, l : callable
        Kernels on X and on Y; called on two arrays of points, they return the Gram matrix.
    rng : numpy.random.Generator
        Source of the relabellings.
    B : int, default 1000
        Number of relabellings.

    Returns
    -------
    stat : float
        MMD^2 (V-statistic).
    p_value : float
        (1 + #{b : relabelled_b >= stat}) / (1 + B).
    """
    x, y = np.concatenate([xa, xb]), np.concatenate([ya, yb])
    G = k(x, x) * l(y, y)                                      # joint Gram matrix of the pooled sample
    a = np.r_[np.full(len(xa), 1 / len(xa)), np.full(len(xb), -1 / len(xb))]
    stat = a @ G @ a                                           # MMD^2 (V-statistic)
    A = np.stack([rng.permutation(a) for _ in range(B)])       # random relabellings
    perm = ((A @ G) * A).sum(axis=1)
    return stat, (1 + np.sum(perm >= stat)) / (1 + B)


# ===================================================================================================== plotting

FIGSIZE = (6.5 / 4, 1.6)      # one panel = a quarter of the 6.5 in text width
PALETTE = "cubehelix"


def set_plot_style(fontsize=10):
    """Style of the paper figures: seaborn "ticks" theme, all text typeset by LaTeX in Times at `fontsize` pt."""
    sns.set_theme(style="ticks", context="paper", font="times new roman")
    mpl.rcParams['text.usetex'] = True
    mpl.rcParams['text.latex.preamble'] = r"""
    \usepackage{mathptmx}
    \usepackage{amsmath}
"""
    mpl.rc("font", family="Times New Roman", size=fontsize)


def rejection_rates(dflong, alpha=0.05, power_models=("Q2", "Q3")):
    """Rejection rates and mean runtimes from the results in long format.

    Parameters
    ----------
    dflong : DataFrame
        One row per (n, rep, model, test), with at least the columns n, model, test, p_value, time.
    alpha : float, default 0.05
        Nominal level: a test rejects when p_value <= alpha.
    power_models : sequence of str
        Models under the alternative; their largest rejection rate sets the common y-range of the power plots.

    Returns
    -------
    rates : DataFrame
        Columns model, test, n, reject: the fraction of repetitions in which the test rejects.
    times : DataFrame
        Columns test, n, time: the mean runtime in seconds, over models and repetitions.
    ylim_power : tuple
        (0, 1.05 * largest power), shared by the power plots so that they can be compared.
    """
    dflong = dflong.assign(reject=dflong["p_value"] <= alpha)
    rates = dflong.groupby(["model", "test", "n"], as_index=False)["reject"].mean()
    times = dflong.groupby(["test", "n"], as_index=False)["time"].mean()
    ylim_power = (0, 1.05 * rates.loc[rates["model"].isin(power_models), "reject"].max())
    return rates, times, ylim_power


def plot_rejection_rate(rates, model, path, hue_order, *, ylabel, xlabel="$n$", alpha=None, ylim=None,
                        figsize=FIGSIZE, palette=PALETTE, show=True):
    """Rejection rate against n of every test, under one model, saved to `path`.

    Under the null model this is the empirical level: pass `alpha` to draw the nominal level as a dotted line.
    Under an alternative it is the empirical power: pass the common `ylim` from `rejection_rates`.

    Parameters
    ----------
    rates : DataFrame
        Output of `rejection_rates`.
    model : str
        Which model to plot ("Q1", "Q2", ...).
    path : str or Path
        Output file; the format follows the extension.
    hue_order : list of str
        Tests to plot, in legend order; sets both colour and line style.
    ylabel, xlabel : str
        Axis labels.
    alpha : float, optional
        Nominal level, drawn as a horizontal dotted line.
    ylim : tuple, optional
        y-range of the axis.
    figsize, palette :
        Size in inches and seaborn palette.
    show : bool, default True
        Call plt.show() after saving.
    """
    fig, ax = plt.subplots(figsize=figsize)
    sns.lineplot(rates[rates["model"] == model], x="n", y="reject", hue="test", hue_order=hue_order, palette=palette,
                 style="test", style_order=hue_order, legend=False, ax=ax)
    if alpha is not None:
        ax.axhline(alpha, c="silver", ls=":")
    ax.set(xlabel=xlabel, ylabel=ylabel)
    if ylim is not None:
        ax.set(ylim=ylim)
    sns.despine()
    plt.tight_layout()
    plt.savefig(path)
    if show:
        plt.show()
    return fig, ax


def plot_runtime(times, path, hue_order, *, ylabel="Runtime [s]", xlabel="$n$", legend_width=0.9,
                 legend_loc="center", figsize=FIGSIZE, palette=PALETTE, show=True):
    """Mean runtime against n of every test, with the legend of all the figures in a panel on the right.

    Parameters
    ----------
    times : DataFrame
        Output of `rejection_rates`.
    path : str or Path
        Output file; the format follows the extension.
    hue_order : list of str
        Tests to plot, in legend order; sets both colour and line style.
    ylabel, xlabel : str
        Axis labels.
    legend_width : float
        Width in inches of the legend panel, added to the width of the plot.
    legend_loc : str
        Position of the legend inside its panel.
    figsize, palette :
        Size in inches of the plot (without the legend panel) and seaborn palette.
    show : bool, default True
        Call plt.show() after saving.
    """
    fig, (ax, lax) = plt.subplots(1, 2, figsize=(figsize[0] + legend_width, figsize[1]),
                                  width_ratios=[figsize[0], legend_width])
    sns.lineplot(times, x="n", y="time", hue="test", hue_order=hue_order, ax=ax, palette=palette,
                 style="test", style_order=hue_order)
    ax.set(xlabel=xlabel, ylabel=ylabel)
    handles, labels = ax.get_legend_handles_labels()
    ax.get_legend().remove()
    lax.legend(handles, labels, frameon=False, loc=legend_loc)
    lax.axis("off")
    sns.despine(ax=ax)
    plt.tight_layout()
    plt.savefig(path)
    if show:
        plt.show()
    return fig, (ax, lax)
