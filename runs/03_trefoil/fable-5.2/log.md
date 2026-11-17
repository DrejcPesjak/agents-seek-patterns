# Analysis log

## Step 0 — setup
- uv venv in ./.venv with numpy/scipy/pandas/matplotlib/scikit-learn. Only X.csv is used.

## Step 1 — basic checks (s1_basic.py)
- File actually has 8000 rows x 30 cols (task text said 5000). No NaNs, all values in [0,1].
- Every column passes KS test vs U(0,1); max |pairwise corr| = 0.036 (noise level); covariance eigenvalues flat; no row-order autocorrelation.

## Step 2 — sparse Fourier scan (s2_fourier.py)
- |mean exp(2πi a·x)| for integer a on pairs (|coef|<=4) and triples (|coef|<=2). Max statistics 10.1 / 13.5 vs expected-max ~10.1 / 11.8 → nothing.

## Step 3 — more null scans (s3_hist.py, s4_fourier_wide.py, s5_rows.py, s6_triples.py)
- 2D/3D histogram chi-square for all pairs/triples, full-dim kNN distances, single-column spectra up to freq 2000: null.
- Characters: pairs |coef|<=40, triples |coef|<=8, quads |coef|<=2, 5- and 6-column ±1 sums: all at expected-max noise level.
- Row-level statistics (sum, radius, order stats, gaps, ...) vs simulated uniform: null. Columns are not rank-transformed; no row-order effects.

## Step 4 — lattice reduction for dense mod-1 relations (s7_lll.py, s8_relation.py)
- Idea: if some integer vector a has a·x ≈ const (mod 1), then (a, W*(a·Δx - m)) is a short vector in a lattice built from row differences. LLL+BKZ finds it; planted controls (12 columns, σ up to 0.02) recovered.
- **HIT**: a = x3 - x6 + 2*x8 - 2*x10 - x29 (mod 1) is concentrated: character statistic 6941 (null max ≈ 17), values in a band 0.526 ± 0.115.
- (s9_explore.py, plot_relation.png) Residual is bounded and multimodal → looks like a projection of a curve; no other characters on those 5 columns; residual not explained by sparse characters of ≤4 columns.

## Step 5 — find the remaining coordinates (s10_resid.py, s11_gpu.py)
- GPU (cupy) scan of all 5-column characters with coefficients in {-2..2} (~73M tests, all 8000 rows). Exactly three hits, on disjoint column sets:
  - u1 = mod(x3 - x6 + 2*x8 - 2*x10 - x29, 1)      stat 6941
  - u2 = mod(x4 - 2*x13 - x19 + 2*x22 + x30, 1)    stat 4590
  - u3 = mod(x2 + 2*x12 + x16 + x24 + 2*x27, 1)    stat 4219
  (noise maximum ≈ 18). A 6-column coef<=2 scan was attempted but ran out of GPU memory and was not rerun.

## Step 6 — shape (s12_latent.py, s13_fit.py; plot_latent_raw.png, plot_knot_3d.png, plot_fit_quality.png)
- (u1,u2,u3) trace a trefoil knot: a closed curve centred near (0.526,0.528,0.527), not crossing the mod-1 border.
- Fitted a closed Fourier curve (6 harmonics) on 5000 rows, checked on the other 3000: mean distance 0.0063 (train 0.0063), max 0.021; uniform-noise rows: mean 0.32, 0.2% within the data's 99th-percentile distance. Scatter ≈ 0.004 std per axis. Whole loop populated.

## Step 7 — submission (s14_submit.py)
- submission.json: status "structure", the 3 maps above, 200-point closed polyline (200 segments, no triangles). Re-evaluated from the written file: data mean distance 0.0063, noise mean 0.32.
