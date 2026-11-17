#!/usr/bin/env python3
"""BLEND generator: hide a low-dimensional shape in high-dimensional data.

Two encodings, selected with "encoding" in the config:

  basic   X in [0,1)^(N x D).      B x mod 1 = u, sampled exactly on a shared
          prime grid so every column (pivot or not) has the same numeric format.
  full    Y in R^(N x 2D).         y = R phi(x) + eta, where phi maps every
          column to (cos 2 pi x, sin 2 pi x), R is a random rotation and eta is
          Gaussian noise. The basic X is never released.

Scope: 1 <= k <= 3 latent dimensions; D and N arbitrary. Only numpy required.

Usage:
    python -m blend.generate --config blend/configs/sword.json --out out/sword

Outputs:
    out/sword/agent/X.csv                        the only file an agent may see
    out/sword/private/private_planted.csv        held-out rows, same source
    out/sword/private/private_noise.csv          matched null rows
    out/sword/private/answer_key.json
    out/sword/private/reference_submission.json  ORACLE witness, not a blind solver
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

# Prime field: every nonzero coefficient in {-2,-1,1,2} is invertible.
# All output columns, including pivots, live on precisely the same grid.
PRIME = 1_000_000_007


@dataclass
class Geometry:
    vertices: np.ndarray  # (V, k) UNWRAPPED coordinate chart
    edges: np.ndarray     # (E, 2) indexes into vertices
    kind: str


def _rot2(theta: float) -> np.ndarray:
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s], [s, c]])


def geometry_from_config(cfg: dict[str, Any], rng: np.random.Generator, k: int) -> Geometry:
    name = cfg.get('kind', 'polyline')
    if name == 'random':
        options = cfg.get('choices', ['sword', 'y_tree', 'x_cross', 'two_loops', 'figure8', 'random_tree'] if k == 2 else ['trefoil'])
        name = str(rng.choice(options))
    if name == 'polyline':
        V = np.asarray(cfg['vertices'], dtype=float)
        if 'edges' in cfg:
            E = np.asarray(cfg['edges'], dtype=np.int64)
        else:
            E = np.array([(i, i + 1) for i in range(len(V) - 1)] +
                         ([(len(V) - 1, 0)] if cfg.get('closed', False) else []), dtype=np.int64)
    elif name in ('sword', 'y_tree', 'x_cross', 'two_loops', 'figure8', 'random_tree'):
        if k != 2:
            raise ValueError(f'{name} is a 2-D shape')
        if name == 'sword':
            # Unwrapped stylized sword outline. Vertices arranged in boundary order.
            xy = [
                (0.00, 0.36), (-.043, .14), (-.045, .09),
                (-.10, .09), (-.13, .125), (-.18, .13), (-.195, .085),
                (-.185, .040), (-.145, .035), (-.110, .070),
                (-.047, .065), (-.046, .015), (-.043, -.11),
                (-.058, -.13), (-.061, -.185), (-.04, -.215),
                (0, -.225), (.04, -.215), (.061, -.185), (.058, -.13),
                (.043, -.11), (.046, .015), (.047, .065),
                (.110, .070), (.145, .035), (.185, .040),
                (.195, .085), (.18, .13), (.13, .125), (.10, .09),
                (.045, .09), (.043, .14)
            ]
            V = np.array(xy, float)
            E = np.array([(i, (i + 1) % len(V)) for i in range(len(V))], np.int64)
        elif name == 'y_tree':
            angles = np.arange(3) * 2 * np.pi / 3 + rng.uniform(-.18, .18, 3)
            V = np.vstack((np.zeros((1, 2)), np.column_stack((np.cos(angles), np.sin(angles))) *
                           rng.uniform(.15, .22, (3, 1))))
            E = np.array([[0, 1], [0, 2], [0, 3]], np.int64)
        elif name == 'x_cross':
            V = np.array([[0, 0], [-.17, -.17], [.17, .17], [-.17, .17], [.17, -.17]], float)
            E = np.array([[0, 1], [0, 2], [0, 3], [0, 4]], np.int64)
        elif name == 'random_tree':
            count = int(rng.integers(4, 7))
            V_list = [np.zeros(2)]
            E_list = []
            angles = np.arange(count - 1) * (2 * np.pi / (count - 1)) + rng.uniform(-.2, .2, count - 1)
            for i in range(1, count):
                parent = 0 if i < 4 else int(rng.integers(1, i))
                point = V_list[parent] + rng.uniform(.10, .18)*np.array([np.cos(angles[i-1]), np.sin(angles[i-1])])
                V_list.append(point)
                E_list.append([parent, i])
            V = np.asarray(V_list, dtype=float)
            E = np.asarray(E_list, dtype=np.int64)
        elif name == 'two_loops':
            t = np.arange(64) * 2 * np.pi / 64
            V = np.vstack((np.column_stack((-.12 + .07 * np.cos(t), .09 * np.sin(t))),
                           np.column_stack((+.12 + .07 * np.cos(t), .09 * np.sin(t)))))
            E = np.array([(offset + i, offset + (i + 1) % 64)
                          for offset in (0, 64) for i in range(64)], np.int64)
        else:
            t = np.arange(160) * 2 * np.pi / 160
            V = np.column_stack((.19 * np.sin(t), .14 * np.sin(2 * t)))
            E = np.array([(i, (i + 1) % len(V)) for i in range(len(V))], np.int64)
    elif name == 'trefoil':
        if k != 3:
            raise ValueError('trefoil requires k=3')
        count = int(cfg.get('vertices_count', 160))
        t = np.arange(count) * 2 * np.pi / count
        raw = np.column_stack(((2 + np.cos(3*t))*np.cos(2*t),
                               (2 + np.cos(3*t))*np.sin(2*t),
                               np.sin(3*t)))
        V = raw * float(cfg.get('scale', .07))
        E = np.array([(i, (i + 1) % count) for i in range(count)], np.int64)
    else:
        raise ValueError(f'Unknown shape kind {name!r}. Use polyline for custom drawings.')

    if V.ndim != 2 or V.shape[1] != k or not np.isfinite(V).all():
        raise ValueError(f'Geometry vertices must be finite with {k} coordinates')
    if E.ndim != 2 or E.shape[1] != 2 or len(E) == 0:
        raise ValueError('Geometry needs at least one edge')
    if np.any(E < 0) or np.any(E >= len(V)) or np.any(E[:, 0] == E[:, 1]):
        raise ValueError('Invalid geometry edge')

    # Random rigid orientation and placement, unless user supplied an unwrapped
    # custom polyline which is left exactly as specified by default.
    if name != 'polyline' or cfg.get('random_transform', False):
        if k == 2:
            V = V @ _rot2(rng.uniform(-np.pi, np.pi)).T
        elif k == 3:
            Q, R = np.linalg.qr(rng.normal(size=(3, 3)))
            Q *= np.sign(np.diag(R))
            V = V @ Q
        center = rng.uniform(.46, .54, k)
        V = V + center
    return Geometry(V, E, name)


def _det(M: np.ndarray) -> int:
    """Exact determinant for 1x1 through 3x3 int matrices."""
    n = len(M)
    if n == 1:
        return int(M[0, 0])
    if n == 2:
        return int(M[0, 0])*int(M[1, 1]) - int(M[0, 1])*int(M[1, 0])
    if n == 3:
        a, b, c = [int(z) for z in M[0]]
        d, e, f = [int(z) for z in M[1]]
        g, h, i = [int(z) for z in M[2]]
        return a*(e*i-f*h)-b*(d*i-f*g)+c*(d*h-e*g)
    raise ValueError('Exact matrix verification supports k<=3')


def max_minor_gcd(B: np.ndarray) -> int:
    """Index of row lattice in its saturation; 1 iff saturated."""
    k, D = B.shape
    g = 0
    for cols in itertools.combinations(range(D), k):
        g = math.gcd(g, _det(B[:, cols]))
        if g == 1:
            return 1
    return g


def delta_exact(B: np.ndarray) -> int:
    """Minimum support of a nonzero rational rowspace vector for k<=3.

    Any largest set of zero columns lies in an (k-1)-dimensional hyperplane;
    enumerate normals of hyperplanes spanned by k-1 columns.
    """
    k, D = B.shape
    if k == 1:
        return int(np.count_nonzero(B[0]))
    if k == 2:
        normals = [np.array([-int(v[1]), int(v[0])], dtype=np.int64)
                   for v in B.T if np.any(v)]
    else:
        normals = []
        for i, j in itertools.combinations(range(D), 2):
            normal = np.cross(B[:, i], B[:, j])
            if np.any(normal):
                normals.append(normal)
    if not normals:
        return 0
    return D - max(int(np.count_nonzero((n @ B) == 0)) for n in normals)


def delta_mod_prime(B: np.ndarray, p: int) -> int:
    """Minimum rowspace support over F_p, which controls discrete-grid secrecy."""
    k, D = B.shape
    if k == 1:
        return int(np.count_nonzero(B[0] % p))
    if k == 2:
        normals = [np.array([-int(v[1]), int(v[0])], dtype=np.int64)
                   for v in B.T if np.any(v % p)]
    else:
        normals = []
        for i, j in itertools.combinations(range(D), 2):
            normal = np.cross(B[:, i], B[:, j])
            if np.any(normal % p):
                normals.append(normal)
    if not normals:
        return 0
    return D - max(int(np.count_nonzero((n @ B) % p == 0)) for n in normals)


def verify_B(B: np.ndarray, min_delta: int) -> dict[str, int]:
    if B.ndim != 2 or not (1 <= B.shape[0] <= 3):
        raise ValueError('B must have 1, 2, or 3 rows')
    if max_minor_gcd(B) != 1:
        raise ValueError('B is not saturated (gcd of maximal minors != 1)')
    delta = delta_exact(B)
    if delta < min_delta:
        raise ValueError(f'delta(B)={delta} is below required {min_delta}')
    finite_delta = delta_mod_prime(B, PRIME)
    if finite_delta < min_delta:
        raise ValueError(f'finite-field delta={finite_delta} is below required {min_delta}')
    return {'delta': delta, 'finite_field_delta': finite_delta, 'saturation_index': 1}


def choose_B(cfg: dict[str, Any], rng: np.random.Generator, D: int, k: int) -> np.ndarray:
    if 'B' in cfg:
        B = np.array(cfg['B'], dtype=np.int64)
        if B.shape != (k, D):
            raise ValueError(f'Explicit B must have shape {(k, D)}')
        return B
    support = cfg.get('support_per_direction', 4)
    supports = [int(support)]*k if isinstance(support, int) else [int(z) for z in support]
    if len(supports) != k or min(supports) < 1:
        raise ValueError('support_per_direction must provide k positive integers')
    disjoint = bool(cfg.get('disjoint_supports', True))
    if disjoint and sum(supports) > D:
        raise ValueError('Disjoint supports exceed D')
    coefs = np.array(cfg.get('coefficients', [-2, -1, 1, 2]), dtype=np.int64)
    if coefs.ndim != 1 or len(coefs) == 0 or np.any(coefs == 0) or np.max(abs(coefs)) > 1000:
        raise ValueError('coefficients must contain small nonzero integers')
    for _ in range(1000):
        B = np.zeros((k, D), dtype=np.int64)
        p = rng.permutation(D)
        at = 0
        for j, s in enumerate(supports):
            cols = p[at:at+s] if disjoint else rng.choice(D, size=s, replace=False)
            at += s if disjoint else 0
            B[j, cols] = rng.choice(coefs, size=s)
        if max_minor_gcd(B) == 1 and delta_exact(B) >= int(cfg.get('min_delta', min(supports))):
            return B
    raise RuntimeError('Could not sample saturated B with desired delta in 1000 attempts')


def inverse_mod_prime(M: np.ndarray, p: int) -> np.ndarray:
    """Gauss-Jordan over F_p. Uses Python integers to avoid overflow."""
    k = len(M)
    A = [[int(M[i, j]) % p for j in range(k)] + [int(i == j) for j in range(k)] for i in range(k)]
    for col in range(k):
        pivot = next((r for r in range(col, k) if A[r][col] != 0), None)
        if pivot is None:
            raise ValueError('Pivot minor singular modulo the prime')
        A[col], A[pivot] = A[pivot], A[col]
        inv = pow(A[col][col], -1, p)
        A[col] = [(v * inv) % p for v in A[col]]
        for r in range(k):
            if r == col:
                continue
            t = A[r][col]
            A[r] = [(a-t*b) % p for a, b in zip(A[r], A[col])]
    return np.array([row[k:] for row in A], np.int64)


def choose_pivots(B: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    k, D = B.shape
    for cols in itertools.combinations(range(D), k):
        A = B[:, cols]
        if _det(A) % PRIME:
            return np.asarray(cols, dtype=np.int64), inverse_mod_prime(A, PRIME)
    raise ValueError('No invertible pivot over F_p')


def sample_latents(rng: np.random.Generator, geom: Geometry, N: int, sigma: float) -> np.ndarray:
    V, E = geom.vertices, geom.edges
    spans = V[E[:, 1]] - V[E[:, 0]]
    lengths = np.linalg.norm(spans, axis=1)
    if not np.all(lengths > 0):
        raise ValueError('Zero-length geometry edge')
    ids = rng.choice(len(E), size=N, p=lengths/lengths.sum())
    t = rng.random((N, 1))
    latent = V[E[ids, 0]] + t*spans[ids] + rng.normal(0, sigma, (N, V.shape[1]))
    return np.rint((latent % 1)*PRIME).astype(np.int64) % PRIME


def embed(rng: np.random.Generator, B: np.ndarray, U: np.ndarray,
          pivots: np.ndarray, inverse: np.ndarray) -> np.ndarray:
    """Exact prime-grid sampling uniformly along the fiber B*x = U (mod p)."""
    N, k = U.shape
    D = B.shape[1]
    x = rng.integers(0, PRIME, (N, D), dtype=np.int64)
    x[:, pivots] = 0
    # Coefficient magnitudes are bounded; if supplying huge explicit B, reject it.
    if int(np.max(np.sum(np.abs(B), axis=1))) > 1_000_000:
        raise ValueError('B coefficients too large for safe int64 vectorization')
    rhs = (U - x @ B.T) % PRIME
    solution = np.zeros((N, k), dtype=np.int64)
    for i in range(k):
        for j in range(k):
            # Reduce after every multiply: no k*PRIME^2 accumulated overflow.
            solution[:, i] = (solution[:, i] + (rhs[:, j]*inverse[i, j]) % PRIME) % PRIME
    x[:, pivots] = solution
    if not np.array_equal((x @ B.T) % PRIME, U):
        raise AssertionError('Prime-grid embedding failed exact modular constraint')
    return x


def split_wrap_geometry(geom: Geometry) -> dict[str, Any]:
    """Convert unwrapped torus edges into Euclidean pieces across box borders."""
    V, E = geom.vertices, geom.edges
    out: list[list[float]] = []
    edges: list[list[int]] = []
    seen: dict[tuple[float, ...], int] = {}

    def idx(p: np.ndarray) -> int:
        key = tuple(round(float(z), 13) for z in p)
        if key not in seen:
            seen[key] = len(out)
            out.append([float(z) for z in key])
        return seen[key]

    for a, b in E:
        x, y = V[a], V[b]
        ts = [0., 1.]
        for axis in range(V.shape[1]):
            lo, hi = sorted([x[axis], y[axis]])
            if abs(float(y[axis] - x[axis])) < 1e-14:
                continue
            for boundary in range(math.floor(float(lo)) + 1, math.ceil(float(hi))):
                t = (boundary - x[axis]) / (y[axis]-x[axis])
                if 1e-12 < t < 1 - 1e-12:
                    ts.append(float(t))
        ts = sorted(set(ts))
        for t0, t1 in zip(ts, ts[1:]):
            offset = np.floor(x+(t0+t1)/2*(y-x))
            aa = x+t0*(y-x)-offset
            bb = x+t1*(y-x)-offset
            aa = np.clip(aa, 0, 1)
            bb = np.clip(bb, 0, 1)
            ia, ib = idx(aa), idx(bb)
            if ia != ib:
                edges.append([ia, ib])
    return {'points': out, 'segments': edges, 'triangles': []}


def observable_characters(U: np.ndarray, k: int, min_rho: float, harmonic_max: int) -> dict[str, Any]:
    angles = 2 * np.pi * (U.astype(np.float64) / PRIME)
    good = []
    for m in itertools.product(range(-harmonic_max, harmonic_max+1), repeat=k):
        if not any(m):
            continue
        # Sign pair equivalent; keep unique representative.
        if next(z for z in m if z) < 0:
            continue
        rho = float(abs(np.exp(1j*(angles @ np.asarray(m, float))).mean()))
        if rho >= min_rho:
            good.append({'m': list(m), 'rho': rho})
    if not good:
        return {'pass': False, 'characters': [], 'observable_index': 0}
    G = np.array([z['m'] for z in good], dtype=np.int64).T
    index = max_minor_gcd(G)
    return {'pass': bool(index == 1), 'characters': good, 'observable_index': index}


def circle_features(Z: np.ndarray) -> np.ndarray:
    """phi: prime-grid integers (N, D) -> (cos, sin) pairs (N, 2D)."""
    angle = 2 * np.pi * (Z.astype(np.float64) / PRIME)
    phi = np.empty((Z.shape[0], 2 * Z.shape[1]))
    phi[:, 0::2] = np.cos(angle)
    phi[:, 1::2] = np.sin(angle)
    return phi


def random_rotation(rng: np.random.Generator, n: int) -> np.ndarray:
    """Haar-distributed orthogonal matrix."""
    Q, R = np.linalg.qr(rng.normal(size=(n, n)))
    return Q * np.sign(np.diag(R))


def disguise(Z: np.ndarray, R: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    """Full BLEND observation y = R phi(x) + eta."""
    Y = circle_features(Z) @ R.T
    if sigma > 0:
        Y = Y + rng.normal(0, sigma, Y.shape)
    return Y


def save_matrix(p: Path, Y: np.ndarray) -> None:
    """Released full-BLEND data. Every column gets the same fixed-point format."""
    header = ','.join('x'+str(i+1) for i in range(Y.shape[1]))
    np.savetxt(p, Y, delimiter=',', header=header, comments='', fmt='%.10f')


def linear_terms(row: np.ndarray, name) -> str:
    terms = []
    for i, v in enumerate(row):
        if v:
            terms.append(('+' if v > 0 else '-') +
                         (str(abs(int(v)))+'*' if abs(int(v)) != 1 else '') + name(i))
    return ''.join(terms).lstrip('+')


def save_csv(p: Path, X: np.ndarray) -> None:
    header = ','.join('x'+str(i+1) for i in range(X.shape[1]))
    # Every column is derived from the same prime-grid integer representation.
    np.savetxt(p, X.astype(float)/PRIME, delimiter=',', header=header, comments='', fmt='%.17g')


def run(config: dict[str, Any], out: Path) -> dict[str, Any]:
    seed = int(config['seed'])
    D, N, k = (int(config[z]) for z in ('D', 'N', 'k'))
    if not (1 <= k <= 3 and D >= k and N >= 100):
        raise ValueError('Requires 1<=k<=3, D>=k, N>=100')
    sigma = float(config.get('noise_sigma', .005))
    if not (0 <= sigma <= .5):
        raise ValueError('noise_sigma must be in [0,.5]')
    encoding = str(config.get('encoding', 'basic'))
    if encoding not in ('basic', 'full'):
        raise ValueError('encoding must be "basic" or "full"')
    dis_cfg = config.get('disguise', {})
    obs_sigma = float(dis_cfg.get('noise_sigma', .01))
    if not (0 <= obs_sigma <= .5):
        raise ValueError('disguise.noise_sigma must be in [0,.5]')
    generators = [np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(7)]
    r_shape, r_weights, r_train, r_private, r_null, r_disguise = generators[:6]
    geom = geometry_from_config(config['shape'], r_shape, k)
    B = choose_B(config.get('lattice', {}), r_weights, D, k)
    min_delta = int(config.get('lattice', {}).get('min_delta', 1))
    structure = verify_B(B, min_delta)
    pivots, inverse = choose_pivots(B)
    U = sample_latents(r_train, geom, N, sigma)
    X = embed(r_train, B, U, pivots, inverse)
    Uhold = sample_latents(r_private, geom, N, sigma)
    Xhold = embed(r_private, B, Uhold, pivots, inverse)
    Xnull = r_null.integers(0, PRIME, (N, D), dtype=np.int64)
    rho_floor = float(config.get('fourier_min_abs', .3))
    visible = observable_characters(U, k, rho_floor, int(config.get('harmonic_max', 2)))
    if not visible['pass']:
        raise ValueError('Observable Fourier characters do not span primitive latent lattice at requested threshold')
    drawing = split_wrap_geometry(geom)
    if len(drawing['points']) > 200:
        raise ValueError(f'Oracle drawing has {len(drawing["points"])} vertices (>200). Reduce geometry vertices_count')

    agent = out/'agent'; private = out/'private'
    agent.mkdir(parents=True, exist_ok=True)
    private.mkdir(parents=True, exist_ok=True)
    if encoding == 'basic':
        save_csv(agent/'X.csv', X)
        save_csv(private/'private_planted.csv', Xhold)
        save_csv(private/'private_noise.csv', Xnull)
        disguise_key = None
    else:
        # The matched null goes through the same rotation and noise, so the only
        # difference between planted and null rows is the hidden constraint.
        R = (random_rotation(r_disguise, 2*D) if dis_cfg.get('rotate', True) else np.eye(2*D))
        save_matrix(agent/'X.csv', disguise(X, R, obs_sigma, r_disguise))
        save_matrix(private/'private_planted.csv', disguise(Xhold, R, obs_sigma, r_disguise))
        save_matrix(private/'private_noise.csv', disguise(Xnull, R, obs_sigma, r_disguise))
        disguise_key = {'columns_released': 2*D, 'noise_sigma': obs_sigma,
                        'feature_order': 'phi[2j]=cos(2 pi x_j), phi[2j+1]=sin(2 pi x_j); y = R phi + eta',
                        'R': R.tolist()}
    key = {
        'seed': seed, 'encoding': encoding, 'prime_grid_modulus': PRIME, 'D': D, 'N': N, 'k': k,
        'B': B.tolist(), 'shape_kind': geom.kind,
        'shape_vertices_unwrapped': geom.vertices.tolist(), 'shape_edges': geom.edges.tolist(),
        'noise_sigma': sigma, 'grader_epsilon': float(config.get('grader_epsilon', .055)),
        'pivot_columns_one_based': (pivots+1).tolist(),
        'delta': structure['delta'], 'finite_field_delta': structure['finite_field_delta'], 'saturated': True,
        'fourier_observability': visible,
        'disguise': disguise_key,
        'limitations': ('The reference submission is an oracle witness derived from ground truth. '
                        'This generator does NOT test whether a blind reference solver can recover the structure, '
                        'nor does it run concealment/null diagnostics. Run those gates separately.')
    }
    (private/'answer_key.json').write_text(json.dumps(key, indent=2)+'\n')
    if encoding == 'basic':
        reference = {'status': 'structure',
                     'map': ['mod(' + linear_terms(b, lambda i: f'x{i+1}') + ',1)' for b in B],
                     'drawing': drawing}
    else:
        # Undo the rotation only for the columns the hidden directions use:
        # z(2t+1), z(2t+2) are the cos and sin of the t-th used column.
        used = np.flatnonzero(np.any(B != 0, axis=0))
        slot = {int(c): t for t, c in enumerate(used)}
        unmix = np.vstack([R[:, [2*c, 2*c+1]].T for c in used])
        reference = {'status': 'structure',
                     'unmix': unmix.tolist(),
                     'map': ['mod((' + linear_terms(b, lambda i: f'atan2(z{2*slot[i]+2},z{2*slot[i]+1})') + ')/(2*pi),1)'
                             for b in B],
                     'drawing': drawing}
    reference['report'] = 'ORACLE geometry witness. Not a recovered result; do not give to a blind agent.'
    if any(len(s) > 200 for s in reference['map']):
        raise ValueError('Oracle expression too long for submission contract')
    (private/'reference_submission.json').write_text(json.dumps(reference, indent=2)+'\n')
    summary = {'encoding': encoding, 'D': D, 'N': N, 'k': k, 'columns_released': D if encoding == 'basic' else 2*D, 'shape_kind': geom.kind,
               'delta': structure['delta'], 'finite_field_delta': structure['finite_field_delta'], 'saturated': True,
               'observable_character_count': len(visible['characters']),
               'oracle_drawing_vertices': len(drawing['points']),
               'paths': {'public': str(agent/'X.csv'), 'private': str(private)},
               'grade_with': f'python -m seek.grade SUBMISSION.json --private {private}'}
    return summary


def main() -> None:
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--config', type=Path, required=True)
    cli.add_argument('--out', type=Path, required=True)
    args = cli.parse_args()
    result = run(json.loads(args.config.read_text()), args.out)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
