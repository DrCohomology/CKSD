# CKSD experiments: permutations and rotamers

Synthetic experiments for the conditional kernel Stein discrepancy (CKSD) goodness-of-fit test, on two response
spaces: permutations of 5 items (the symmetric group S_5, Sec. 4.2) and the torus S^1 x S^1 (Sec. 4.1).
Each experiment estimates the level, power and runtime of the CKSD test and compares them with an MMD two-sample test.

## Files

| File | Contents |
|---|---|
| `utils.py` | Shared code: Gaussian kernel on the covariates, vectorised row sampling, wild-bootstrap p-value of the CKSD, MMD permutation test, plot style and plotting helpers |
| `permutations.py` | Rankings: Steinhaus-Johnson-Trotter order, Kendall distance, Mallows kernel, discrete Stein kernel built on a cyclic shift of S_n |
| `permutations_plot.py` | Experiment on rankings and its figures |
| `torus.py` | Torus: chart and tangent vectors, Gaussian kernel and its derivatives, Stein kernel, cosine model (score, normaliser, sampler) |
| `torus_plots.py` | Experiment on the torus and its figures |

## Design

The covariate is x ~ U(-1, 1) and the data are drawn from the conditional model Q1. The tested models are

- Q1 (H0): Mallows spread (rankings) or cosine-model concentration (torus) equal to |x|; 
- Q2 (H1 global shift): spread or concentration 5|x| for every x; 
- Q3 (H1 local shift): as Q2 on a random 20 % of the observed x values, as Q1 elsewhere.

Sample sizes n = 10, 20, 50, 100, 200, 500, with 100 repetitions each, B = 1000 bootstrap replicates (resp. permutations for the MMD),
nominal level alpha = 0.05, seed 43.

- CKSD uses all n points and needs only log q (rankings) or its gradient (torus), never the normalizing constant.
  On rankings it is run with three orderings of S_n, which fix the cyclic shift of the Stein operator:
  SJT (neighbours at Kendall distance 1), lexicographic, and a random shuffle.
- MMD compares the first half of the sample with the second half, whose responses are resampled from the model.

## Running

```bash
python permutations_plot.py
python torus_plots.py
```

Each script writes four figures to `figures/`: `<name>_Q1.pdf` (level), `<name>_Q2.pdf`, `<name>_Q3.pdf` (power)
and `<name>_runtime.pdf` (runtime and the legend). The scripts are split into `#%%` cells, so in an IDE the figure
cells can be rerun without repeating the experiment. Axis labels, figure size and palette are set at the top of the
figure cell.

## Requirements

Python 3 with `numpy`, `scipy`, `pandas`, `matplotlib`, `seaborn`, `tqdm`.
The figures are typeset with LaTeX (`text.usetex`), which needs a TeX installation with `mathptmx`, `amsmath` and
`cm-super`.
