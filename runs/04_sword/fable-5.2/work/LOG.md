# Analysis log (X.csv: actually 8000 rows x 30 cols)

1. a1.py  – basic stats: all columns uniform[0,1], max |corr| 0.03, flat PCA spectrum. Nothing.
2. a2.py  – torus Fourier scan |mean exp(2πi k·x)|, 1–3 columns, |k|≤3–4. Nothing.
3. a3.py  – chi-square of 2/3/4-column histograms. Nothing.
4. a4.py  – row statistics (sum, min, max, gaps…), nearest-neighbour distances (full, pairs, triples), row autocorrelation. Nothing.
5. a5.py  – Fourier scan for 4 cols (|k|≤2), 5 and 6 cols (±1). Nothing.
6. a6.py  – gradient-boosting classifier data vs uniform noise: AUC 0.50. Nothing.
7. a7.py  – Gaussianise + FastICA projection pursuit: same as noise.
8. a8.py  – high-frequency Fourier (single ≤3000, pairs ≤48, triples ≤6): nothing, BUT low-bit check of the doubles flagged x3.
9. a9/a10 – all columns are multiples of 2^-53 except x3 (229 rows) and x14 (355 rows); trailing-zero stats differ for exactly these two → they are computed columns.
10. a11/a15 – regress trailing-zero count of x3 / x14 on other columns → x3 driven by x9,x12,x25,x27,x29; x14 by x2,x4,x15,x20,x28.
11. a12/a16 – integer-coefficient search:
    u = (x3 − x9 + x12 + 2x25 + 2x27 − x29) mod 1   (|char. mean| 0.90)
    v = (x14 − 2x2 − 2x4 + x15 + x20 + x28) mod 1    (|char. mean| 0.63)
12. a17 – scatter (u,v): closed dagger-shaped curve (p_uv.png); noise through same map is uniform.
13. a18 – fitted 180-vertex closed polyline (p_fit.png): rms distance 0.0044, max 0.019, length 1.63.
14. a19 – wrote ../submission.json.

Plots: p_uv.png, p_fit.png, p_x14.png, p_x3_x14.png
Not found / not checked exhaustively: a third hidden coordinate (no other column shows the float fingerprint).
