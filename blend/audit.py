#!/usr/bin/env python3
"""Low-order concealment audit: does the released data look like its matched null?

Run:
    python -m blend.audit out/sword/agent/X.csv --nulls 39
    python -m blend.audit out/sword_full/agent/X.csv --key out/sword_full/private/answer_key.json

Basic data is compared with uniform draws on the generator's prime grid.
Full data needs --key: its null is uniform data pushed through the same private
rotation and noise, which only the answer key knows.

Five diagnostics: max |correlation|, two PCA-spectrum statistics, the largest
pair chi-square (8x8 bins) and the largest triple chi-square (4x4x4 bins).
Pair/triple scans cost O(D^3 N). This is a diagnostic gate, not a proof of
hardness and not a test of recoverability.
"""
from __future__ import annotations
import argparse
import itertools
import json
from pathlib import Path
import numpy as np

from .generate import PRIME, disguise

NAMES = ['max_abs_corr', 'PCA_max_min_ratio', 'PCA_spectrum_RMS',
         'max_pair_chi2_8x8', 'max_triple_chi2_4x4x4']


def _bins(x: np.ndarray, n: int, unit_interval: bool) -> np.ndarray:
    """Per-column bin index. Equal-width on [0,1), else equal-count (rank) bins."""
    if unit_interval:
        return np.floor(x*n).astype(np.int32).T
    ranks = np.argsort(np.argsort(x, axis=0), axis=0)
    return (ranks*n // len(x)).astype(np.int32).T


def diagnostics(x: np.ndarray, unit_interval: bool = True) -> np.ndarray:
    N, D = x.shape
    corr = np.corrcoef(x, rowvar=False)
    max_corr = float(abs(corr[np.triu_indices(D, 1)]).max()) if D > 1 else 0.
    eig = np.linalg.eigvalsh(np.cov(x, rowvar=False))
    # Expected variance per column: 1/12 for uniform; the spectrum mean otherwise.
    unit = 1/12 if unit_interval else float(eig.mean())
    p8 = _bins(x, 8, unit_interval)
    p4 = _bins(x, 4, unit_interval)
    max_pair, max_triple = 0., 0.
    expected = N/64
    for i, j in itertools.combinations(range(D), 2):
        count = np.bincount(8*p8[i]+p8[j], minlength=64)
        max_pair = max(max_pair, float(((count-expected)**2).sum()/expected))
    for i, j, k in itertools.combinations(range(D), 3):
        count = np.bincount(16*p4[i]+4*p4[j]+p4[k], minlength=64)
        max_triple = max(max_triple, float(((count-expected)**2).sum()/expected))
    return np.array([max_corr, float(eig[-1]/eig[0]),
                     float(np.sqrt(np.mean((eig/unit-1)**2))),
                     max_pair, max_triple])


def audit(x: np.ndarray, n_null: int = 39, seed: int = 2027, key: dict | None = None) -> dict:
    """Rank the data's diagnostics among n_null matched null datasets."""
    if not (19 <= n_null <= 10000):
        raise ValueError('At least 19 null datasets are required')
    if x.ndim != 2 or not np.isfinite(x).all():
        raise ValueError('Expected a finite N x M matrix')
    full = bool(key) and key.get('encoding') == 'full'
    N, M = x.shape
    rng = np.random.default_rng(seed)
    if full:
        R = np.asarray(key['disguise']['R'])
        sigma = float(key['disguise']['noise_sigma'])
        if R.shape != (M, M):
            raise ValueError('answer key rotation does not match the data width')
        def null():
            return disguise(rng.integers(0, PRIME, size=(N, M//2), dtype=np.int64), R, sigma, rng)
    else:
        if not np.all((0 <= x) & (x < 1)):
            raise ValueError('Values outside [0,1): for full-BLEND data pass --key answer_key.json')
        def null():
            return rng.integers(0, PRIME, size=(N, M), dtype=np.int64)/PRIME
    obs = diagnostics(x, not full)
    nulls = np.array([diagnostics(null(), not full) for _ in range(n_null)])
    vals = np.vstack([obs, nulls])
    # Exchangeable rank, inclusive of the observed sample and ties.
    p_all = np.mean(vals[None, :, :] >= vals[:, None, :], axis=1)
    p_obs = p_all[0]
    min_p = p_all.min(axis=1)
    omnibus = float(np.mean(min_p <= min_p[0]))
    result = {
        'encoding': 'full' if full else 'basic',
        'null_datasets': n_null,
        'diagnostics': NAMES,
        'observed': obs.tolist(),
        'null_median': np.median(nulls, axis=0).tolist(),
        'empirical_p_values': p_obs.tolist(),
        'omnibus_p': omnibus,
        'pass_at_0_05': bool(omnibus > .05 and np.all(p_obs > .05)),
        'note': ('This audit measures only these five diagnostics. It does not test recoverability, '
                 'exact lattice correctness, topology, or every possible artifact.'),
    }
    if not full:
        # Numeric-format check: a column computed differently from the others
        # (the legacy floating-point leak) shows up as an outlier here.
        grid_error = np.abs(x*PRIME-np.rint(x*PRIME))
        result['prime_grid_max_roundtrip_error'] = float(grid_error.max())
        result['prime_grid_max_error_by_column'] = grid_error.max(axis=0).tolist()
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('csv', type=Path)
    ap.add_argument('--nulls', type=int, default=39)
    ap.add_argument('--seed', type=int, default=2027)
    ap.add_argument('--key', type=Path, help='private/answer_key.json (required for full-BLEND data)')
    ap.add_argument('--output', type=Path)
    args = ap.parse_args()
    x = np.loadtxt(args.csv, delimiter=',', skiprows=1)
    key = json.loads(args.key.read_text()) if args.key else None
    report = json.dumps(audit(x, args.nulls, args.seed, key), indent=2)+'\n'
    if args.output:
        args.output.write_text(report)
    print(report)


if __name__ == '__main__':
    main()
