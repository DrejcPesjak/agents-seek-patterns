# Work log: is there hidden low-dimensional structure in `example_X.csv`?

**Verdict: `none`.** Every test below matched a pure-noise reference (i.i.d. Uniform(0,1) matrices of the same 4000 x 50 shape).

Scripts: `explore1.py` … `explore5.py` (tests), `make_plots.py` (figures in `plots/`).

## What the matrix looks like

- 4000 rows x 50 columns, all values in (0, 1), no duplicates, full float64 precision.
- Each column is individually uniform: mean 0.49–0.51, std 0.283–0.294 (uniform is 0.289), worst KS p-value 0.013 over 50 columns, which is unremarkable.

Since the marginals are flat, any structure would have to live in the *relationships* between columns or rows. That is what the rest of the log looks for.

## What I tried

| # | Idea | Result on data | Same test on noise | Outcome |
|---|------|----------------|--------------------|---------|
| 1 | PCA / singular spectrum | 20.3 … 16.3, smooth | same band | nothing (`plots/01`, `plots/02`, `plots/07`, `plots/08`) |
| 2 | Max linear correlation between columns | 0.055 | ~0.06 | nothing |
| 3 | Circular correlation, harmonics 1–3 (columns as angles) | 0.055 | 0.050 | nothing |
| 4 | Pairwise dependence, binned chi-square at 4, 8, 16 bins | max 41 / 120 / 323 | max 33–47 / 94–107 / 324–359 | nothing (`plots/03`, `plots/04`, `plots/09`) |
| 5 | Modular relations between pairs, `a*x_i ± b*x_j mod 1`, a,b up to 48 | 0.0613 | 0.0619 | nothing |
| 6 | Modular relations among triples, coefficients ±1, ±2 | 0.058 | 0.059 | nothing |
| 7 | Global relations over all columns (sum, alternating sum, weighted 1..50, mod 1) | 0.006–0.014 | noise floor 0.03 | nothing |
| 8 | Nearest-neighbour distances, Euclidean and on the torus | quantiles 1.65 … 2.34 | 1.66 … 2.36 | nothing (`plots/05`) |
| 9 | Intrinsic dimension (TwoNN) | 32.8 | 31.9 | nothing |
| 10 | Row order: lag autocorrelation down the rows | 0.0045 | 0.0057 | nothing |
| 11 | Column order: autocorrelation and Fourier spectrum along each row | flat | flat | nothing |
| 12 | Per-row summaries (mean, variance, max, circular resultant, sorted gaps) | KS p 0.22–0.98 vs noise | n/a | nothing |
| 13 | Number-stream artefacts: bit patterns, RANDU-style lattice in row- and column-major order | 0.006 | noise floor 0.007 | nothing |
| 14 | Gaussianised PCA and PCA of the (cos, sin) torus embedding | same spectrum | same spectrum | nothing |
| 15 | Predict each column from the other 49 (extra-trees, with cos/sin features, 4-fold CV) | best R² = -0.002 | best R² = -0.005 | nothing (`plots/06`) |
| 16 | Classifier: real rows vs fresh uniform rows | 50.6% accuracy | 50% is chance | nothing |

## What worked

Nothing found structure. What did work was running each statistic on fresh uniform matrices alongside the data, which gave an honest noise level for each one.

## What looked interesting but wasn't

- Column pair (37, 42) scored chi-square 120 at 8x8 bins, above the five noise maxima I had at the time (94–107). It did not hold up: at 4 bins and 16 bins the data maxima sit inside the noise range, the pair's scatter is a plain uniform square (`plots/04`), and it is the largest of 1225 pairs.
- The low-order-digit KS p-value was 0.045; one borderline value among dozens of tests is expected by chance.

## What these tests cannot rule out

- Relations among 4 or more specific columns with modular arithmetic, other than the three all-column ones in row 7.
- Modular relations with coefficients larger than those tried (48 for pairs, 2 for triples).
- Columns built as `frac(w · z)` from a low-dimensional `z` with large non-integer weights. With 4000 rows this is practically indistinguishable from noise.

Tests 15 and 16 are generic and would have caught most of these if the effect were learnable from 4000 rows, and they came back at chance.

## Plots

- `plots/01_pca_spectrum.png`: singular values against the range from 20 noise matrices
- `plots/02_pca_top3.png`: the top three principal components, shown as three 2D views (the system matplotlib cannot draw 3D axes; the 3D figures below were made through uv)
- `plots/03_pairwise_chi2.png`: distribution of pairwise dependence scores, data vs noise
- `plots/04_top_pairs.png`: scatter of the three highest-scoring column pairs
- `plots/05_nn_distances.png`: nearest-neighbour distance distribution, data vs noise
- `plots/06_column_predictability.png`: cross-validated R² for each column predicted from the rest
- `plots/07_pca_3d_data_vs_noise.png`: 3D scatter of the top three principal components, data next to a noise matrix
- `plots/08_pca_3d_four_views.png`: the same data projection from four viewing angles
- `plots/09_column_triples_3d.png`: 3D scatter of raw column triples built from the highest-scoring pairs

## 3D plotting note

The system Python's matplotlib fails to import `Axes3D` (mixed system and pip installs). The 3D figures come from `make_plots_3d.py`, run as `uv run --no-project --with matplotlib --with numpy python make_plots_3d.py`, which pulled matplotlib 3.11.2 into a throwaway uv environment. They show the same thing as the 2D views: no shape from any angle.
