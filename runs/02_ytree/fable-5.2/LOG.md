# Log

1. `scripts/01_basic.py` — marginals uniform on [0,1] (KS ok), max |corr| 0.044 (null level), PCA flat,
   circular means null, no row-to-row autocorrelation. Nothing.
2. `scripts/02_pairs_triples.py` — pair 2D chi2, pair/triple Fourier characters exp(2*pi*i*(a xi + b xj + c xk)),
   small integer coefs. All at null level. Nothing.
3. `scripts/03_subsets_hist_knn.py` — chi2 histograms and torus nearest-neighbour distance over all
   2-, 3-, 4-subsets of columns. Pairs/triples null. Two 4-subsets stand out massively:
   (x10,x23,x25,x30): chi2 z=43, kNN z=-57;  (x1,x6,x11,x24): chi2 z=41, kNN z=-57. Everything else |z|<5.
4. `scripts/04_quads.py` — Fourier-character search (coefs -4..4) inside each quad:
   |E exp(2πi(2x10+2x23+x25+x30))| = 0.97 and |E exp(2πi(2x1-2x6-x11-x24))| = 0.96; nothing else.
5. `scripts/05_uv.py`, `06_inner.py` — u=(2x10+2x23+x25+x30)%1 and v=(2x1-2x6-x11-x24)%1 are each confined to a
   narrow band (~0.43-0.62, ~0.39-0.59), with non-Gaussian shape. Scatter u vs v = a three-armed "Y" (plots/05_uv.png).
   u, v do not depend on any individual column (plots/06_u_vs_cols.png).
6. `scripts/07_fitY.py` — fitted the Y: junction (0.5174, 0.5108); arms at 18.9°, 142.1°, -97.9°; lengths ≈0.10, 0.11, 0.12;
   arms straight (no curvature in perpendicular residuals), thickness sd ≈ 0.005; points uniform along the arms (plots/07_Y_zoom.png).
7. `scripts/08_third.py` — looked for a third coordinate: position on the Y shows no dependence on single/pair column
   characters; torus-kNN scan of all 142,454 other 5-column subsets is null (min z=-4.5 vs -57 for the real quads).
   Relations needing 6+ columns were not scanned. Conclusion: 2-D map, Y-shaped 1-D structure; 22 of 30 columns are noise.
8. `scripts/09_submit.py` — writes submission.json, checks it on a 4000/1000 row split and against uniform noise (plots/09_final.png).
