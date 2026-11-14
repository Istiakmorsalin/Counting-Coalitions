# Counting Coalitions

**The combinatorics of Shapley values and the cost of explainable machine learning**

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Istiakmorsalin/Counting-Coalitions/blob/main/shapley_values_implementation_final.ipynb)

Md Istiak Morsalin · Prianka Barman
STAT 8030: Applied, Discrete & Combinatorial Mathematics, Kennesaw State University (Fall 2026)

![Poster](docs/poster_preview.png)

## Overview

SHAP explains a model's prediction by giving each feature its **Shapley value**:

$$
\phi_i=\sum_{S\subseteq N\setminus\{i\}}\frac{|S|!\,(n-|S|-1)!}{n!}\,\big[v(S\cup\{i\})-v(S)\big]
$$

The sum runs over all $2^{n-1}$ coalitions of the other features, so exact SHAP is exponential in the number of
features $n$. In practice it is approximated by sampling, with little guidance on how much sampling is enough.

This project uses counting arguments to explain **why** the formula works, and to predict **what it costs**:

- **Counting proofs** of the three properties that make Shapley values a sound attribution.
- **A counting rule, $m(n+1) < 2^n$,** that predicts when sampling with $m$ permutations is cheaper than exact
  computation, confirmed on real data.
- **Measured sample budgets** for a target accuracy, for a linear and a tree-based model.

## Research questions

| | Question | Answer |
|---|---|---|
| **RQ1** | Why do the weights $\frac{k!(n-k-1)!}{n!}$ form a valid average? | Double counting: $\binom{n-1}{k}k!(n-1-k)!=(n-1)!$, so every coalition size gets exactly $1/n$ of the weight. |
| **RQ2** | Why does the subset formula equal an average over all $n!$ orderings? | A bijection: exactly $k!(n-k-1)!$ orderings put a given $S$ before $i$. So averaging random orderings is **unbiased**. |
| **RQ3** | Why do the values always add up to $v(N)-v(\emptyset)$? | Telescoping along each ordering (also proved by induction). |
| **RQ4** | How does the cost grow, and how many samples are enough? | Exact cost doubles per feature; see the results below. |

![RQ2: the bijection](docs/rq2_bijection.png)

## Methodology

![Methodology](docs/methodology.png)

- **Data:** [UCI Bank Marketing](https://archive.ics.uci.edu/dataset/222/bank+marketing) (`bank-full`), 16 features;
  a 12,000-row stratified sample; target = did the client subscribe to a term deposit (11.7% yes).
- **Varying $n$:** the first $n$ features of one fixed random ordering, for $n = 4, \dots, 16$, retraining each model.
- **Models:** logistic regression and gradient-boosted trees (scikit-learn). Test AUC at $n=16$: 0.865 and 0.918.
- **Game:** $v(S)=\frac{1}{|B|}\sum_{b\in B} f(x_S, b_{N\setminus S})$, the interventional value function, with
  $|B| = 50$ background rows. It is the same game as SHAP's `Independent` masker.
- **Estimators**, all implemented from scratch:
  - **exact:** all $2^n$ coalitions;
  - **permutation sampling:** $m$ random orderings;
  - **KernelSHAP:** constrained weighted least squares, given the *same* budget of $m(n+1)$ evaluations of $v$.
- **Evaluation:** $m \in \{10, 50, 100, 500, 1000\}$, 5 test instances × 3 seeds. Relative $\ell_1$ error against
  exact values, Kendall's $\tau$ on feature rankings, top-3 overlap and runtime.

## Results

| Finding | Value |
|---|---|
| Exact runtime per added feature | **≈ 2×** (matches $2^n$) |
| One exact explanation at $n = 16$ (trees) | **≈ 50 s**: 65,536 coalitions, 3.3 M predictions |
| Predicted break-even for $m = 100$, from $m(n+1) < 2^n$ | **$n = 11$** (measured: 12, on the even-$n$ grid) |
| Settings where KernelSHAP has lower mean error than permutation sampling | **70 / 70** |
| KernelSHAP's error advantage (median) | **14×** logistic regression, **2.1×** trees |
| Lowest Kendall's $\tau$ for any $m \ge 100$ | **0.92** |

![Exact cost](docs/cost.png)

![Accuracy vs budget](docs/accuracy.png)

**Rule of thumb:** the smallest $m$ with ≤ 5% error *and* Kendall's $\tau \ge 0.9$.

| Model | Method | $n=4$ | 8 | 12 | 16 |
|---|---|---:|---:|---:|---:|
| Logistic | KernelSHAP  | 10 | 10  | 10  | 10  |
| Logistic | Permutation | 10 | 10  | 50  | 50  |
| Trees    | KernelSHAP  | 10 | 50  | 100 | 500 |
| Trees    | Permutation | 50 | 100 | 500 | 500 |

Both estimators are exact when features do not interact, so the budget is driven by feature interactions. A
near-additive model needs few samples at any $n$; a tree model needs more as $n$ grows.

### One real client, explained

![One client explained](docs/client_explained.png)

Exact Shapley values for a client who subscribed and whom the tree model correctly flagged. The bars add up exactly
from the base rate (0.18) to the prediction (0.79), which is efficiency (RQ3) on real data. The top driver is
**call duration**, which is only known after the call ends. This is a known leak in this dataset, and the
explanation exposes it. It inflates AUC but not the estimator comparison, which is measured against exact values for
the same model.

## Running the notebook

Everything is in [`shapley_values_implementation_final.ipynb`](shapley_values_implementation_final.ipynb).

1. Open it in Colab with the badge above, or locally with Python 3.10 or newer and
   `pip install numpy pandas scipy scikit-learn matplotlib shap`.
2. In the configuration cell, set:
   - `QUICK = True` for a smoke test of a few minutes, or `QUICK = False` for the full study, which takes about an
     hour on a free Colab CPU.
   - `USE_DATASETS = ["Bank"]` (the default). The other options are `"Adult"`, `"Pima"`, `"California"`, `"Heart"`
     and `"Synthetic16"`.
3. Run all cells. CSVs and figures are written to `results/` and zipped as `shapley_results.zip`.

| Section | Contents |
|---|---|
| 1 | Coalitions as bitmasks; subset and permutation formulas |
| 2 to 4 | RQ1 to RQ3: exact-arithmetic checks and the $n = 3$ worked example |
| 5 to 6 | Data, models, the value function, the three estimators, and a cross-check against `shap` |
| 6.2 | One client explained (the panel above) |
| 7 to 9 | RQ4: exact cost, accuracy against budget, and the rule of thumb with a Hoeffding comparison |

Results use fixed seeds, so accuracy numbers reproduce exactly when the data comes from the same source (OpenML,
with the UCI archive as a fallback). Runtimes vary slightly from run to run and between machines.

## Limitations

- One dataset, including the leaking `duration` feature.
- Features are treated as independent when they are removed (the interventional value function).
- 50 background rows, and 5 instances × 3 seeds per setting.

## References

1. L. S. Shapley. A value for *n*-person games. *Contributions to the Theory of Games II*, 307–317, 1953.
2. S. M. Lundberg and S.-I. Lee. A unified approach to interpreting model predictions. *NeurIPS*, 2017.
3. E. Štrumbelj and I. Kononenko. Explaining prediction models and individual predictions with feature contributions.
   *Knowledge and Information Systems* 41(3), 2014.
4. J. Castro, D. Gómez and J. Tejada. Polynomial calculation of the Shapley value based on sampling.
   *Computers & Operations Research* 36(5), 2009.
5. I. Covert and S.-I. Lee. Improving KernelSHAP: Practical Shapley value estimation using linear regression.
   *AISTATS*, 2021.
6. S. Moro, P. Cortez and P. Rita. A data-driven approach to predict the success of bank telemarketing.
   *Decision Support Systems* 62, 2014.

The full poster is in [`docs/poster.pdf`](docs/poster.pdf).
