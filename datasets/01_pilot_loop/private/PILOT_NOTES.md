# Torus planted-geometry pilot

This is research prototype code, not a polished benchmark. It samples two disjoint, signed integer rows of support size s, generates an independently randomized smooth closed Fourier curve, and draws uniformly on torus fibers. The seeds are fixed for reproducibility; they must be replaced with private random seeds for actual agent evaluation. The CSV contains only X. The NPZ contains the planted directions and coordinates and must be kept private from evaluated agents.

## Reproduce

Python dependencies: numpy, scipy, scikit-learn, sympy, matplotlib. Run from this directory. Timings use `time.process_time()` (CPU seconds), including feature construction but excluding generator time. The pilot solvers receive the support size and use a known rank-two stopping criterion; they do not receive the planted rows or the shape.

- `python pilot.py`: generate all 18 cases at N=4000, calibrate initial diagnostics against 19 null draws, save NPZ data.
- `python joint_baseline.py`: extended joint diagnostic calibration for D=20 and 50 (can take time for D=100 due to random-init FastICA; the D=100 run in this pilot was stopped).
- `python joint100.py`: extended D=100 diagnostics with coordinate-aligned symmetric-whitening ICA initialization. Randomized-init ICA did not converge in the initial D=100 run.
- `python recover.py`: split-frequency matrix multiplication solver, capped at 180 million candidate pairs per partition by default. `extra_recovery.py` adds a larger-cap D=100 s=5 test, a retry for D=50 s=3, and three N=1000 runs.
- `python lsh.py` and `python lsh_fast.py`: approximate correlation hashing pilots. The latter was stopped after the first two reported runs. They are experimental, not production implementations.
- `g++ -O3 -march=native -shared -fPIC popscan.cpp -o popscan.so`; `python pop_attack.py`: compressed binary correlation screening. The C++ scanner uses AVX-512 VPOPCNTDQ, so an appropriate CPU is required or the kernel must be adapted. The pilot cap was approximately 70 CPU seconds per case. `resume.py` continues the prescribed partition RNG sequence for D=100 s=6 after the initial short run.
- `python lll_test.py` and `python lll_exact.py`: illustrative SymPy LLL attacks; only D=20 and 50, s=4 completed in the reported runs. The wider experiments are not evidence of a general LLL impossibility.
- `python leak_test.py`: compare planted, perturbed and random real projection characteristic functions to the exact uniform null.
- `python shape_demo.py`: reconstruct a 64-vertex polygon in the recovered basis of D=50, s=6 using visible data, score against fresh latent samples, test exact rational rank scoring and save the plot. The geometry score uses sampled toroidal nearest distances as a numerical approximation; it is not a general exact Hausdorff or topology evaluator.

## Limitations

One generated instance per D and s; N=4000 for the complete grid, 19 matched null calibrations per D; no agent experiments; no full one-hour sweeps; larger capped configurations are unresolved rather than demonstrated impossible. The shape family is only randomized smooth loops (not general graphs or knots). The recovery solvers are informed of the test condition, so the experiment establishes instance recoverability rather than family-blind agent performance. No result should be interpreted as a universal guarantee against PCA/ICA/density methods, or as a computational hardness proof.

All JSON files are measured pilot outputs; incomplete logs/runs are intentionally not presented as successes. The archived example is D=50, s=6, N=4000. Never give its NPZ or the result JSON files to an agent evaluating that example.
