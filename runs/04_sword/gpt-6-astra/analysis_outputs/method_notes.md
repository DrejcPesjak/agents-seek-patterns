# Analysis notes

**Final result: structure.** Two six-column modular combinations recover a noisy closed sword outline. See `sword_validation.png`, `sword_detail.png`, and `../submission.json`.

- u = mod(x3-x9+x12+2*x25+2*x27-x29,1)
- v = mod(2*x2+2*x4-x14-x15-x20-x28,1)

The closed 160-point polyline was fit on the first 5,000 rows and checked on the remaining 3,000. Held-out median distance is 0.002711; 98.27% lie within 0.01. Independent uniform noise has median distance 0.210471, with 3.318% within 0.01. Both noise coordinates have standard deviation 1/sqrt(12), so normalization scales these distances by sqrt(12).

Only `X.csv` was read as source data. No directory listing, unrelated workspace files, dataset lookup, or internet access was used. All other files read by the analysis were outputs or scripts created during this run. Independent uniform samples and synthetic positive controls were generated locally.

The actual input has 8,000 rows and 30 columns, all finite and in [0,1]. No duplicate rows were found.

The search deliberately goes beyond ordinary PCA:

- Column uniformity tests and histograms; Pearson correlations; covariance spectra on raw and Gaussian marginal scores.
- Pairwise contingency tests at 4, 8 and 16 bins; binary digit parity tests at several resolutions.
- Row sums, radii and lag checks; nearest-neighbor distances in ordinary and circular embeddings against independently simulated uniform noise.
- Extra Trees and gradient-boosted classifiers trained to distinguish input rows from uniform noise. On separate rows their AUCs were 0.48743 and 0.49791.
- Fourier searches use the concentration abs(mean(exp(2*pi*i*(k dot x)))). Independent uniform variables give a population value of zero for every nonzero integer k. Curves hidden by modulo sums can give large concentrations even when every original column, and every original column pair, is uninformative.
- Exhaustive signed pair, triple, quadruple and six-column Fourier combinations. Expanded searches include pair coefficients through four, a third coefficient through sixteen, and four-column combinations with coefficients through two.
- A larger six-column search uses all triple features whose coefficients have magnitudes one or two, comparing sums and differences of features. An eight-column extension began comparing all four-column signed sums; it was stopped after the decisive six-column result and is not claimed complete. Overlaps also test certain combinations with repeated coefficients. Screening uses fewer rows for speed; hits are checked on 3,000 separate rows, not accepted based on the search score.
- Nonlinear pair feature searches include products, squared sums/differences, ratios, radii and centered angles at two harmonics, and test dependencies between disjoint column pairs.
- Binary digit matrices have full GF(2) rank at scales 1 through 128: no exact dense XOR constraint was detected. Sparse parity searches extend to six columns.
- Lattice reduction searches for dense small-integer exact or narrow modulo constraints. A separate synthetic control with an exact 30-column modular constraint was recovered with concentration 1.0. No comparable relation was found in the input.
- Gaussian-score independent component analysis explores non-Gaussian linear combinations. The most promising initial components had p-values 0.00718, 0.000412 and 0.000973, but separate checks gave 0.321, 0.598 and 0.711. They were rejected as non-replicating search hits.

The early negative results did not establish independence. The expanded coefficient search was decisive: the two Fourier concentrations replicated at 0.89545 and 0.62564 on 3,000 separate rows. The resulting 2D scatter directly revealed the sword outline. The final drawing follows this outline with segments only; no interior area is filled. It stays inside the modulo borders, so no splitting is needed. The held-out geometric validation establishes that the drawing captures source structure rather than selected visible-row quirks.

`log.txt` contains numerical results and progress. The Python scripts in the workspace reproduce the individual analyses. `overview.png`, `marginals.png`, `neighbors_False.png`, `neighbors_True.png`, and `ica_check.png` contain visual diagnostics.
