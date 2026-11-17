#!/usr/bin/env python3
"""Regenerate the README figures in docs/figures/ from datasets/ and runs/.

Run from the repo root:  python results/make_figures.py
Needs numpy, scipy, matplotlib. Takes about a minute.
"""
import json
import sys
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from blend.generate import PRIME, run as blend_generate  # noqa: E402
from results.grade_runs import lattice_recovered, planted_B, submitted_A  # noqa: E402
from seek import grade as G  # noqa: E402

OUT = ROOT/'docs'/'figures'
SURFACE, INK, INK2, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#e4e3df'
DATA, AGENT, TRUTH = '#2a78d6', '#eb6834', '#1baf7a'   # validated categorical slots 1-3
plt.rcParams.update({
    'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE, 'savefig.facecolor': SURFACE,
    'font.size': 9, 'axes.titlesize': 10, 'axes.titleweight': 'bold', 'axes.titlelocation': 'left',
    'axes.edgecolor': GRID, 'axes.labelcolor': INK2, 'text.color': INK,
    'xtick.color': INK2, 'ytick.color': INK2, 'xtick.labelsize': 8, 'ytick.labelsize': 8,
    'axes.spines.top': False, 'axes.spines.right': False, 'legend.frameon': False,
    'legend.fontsize': 8, 'figure.dpi': 160,
})


def load(path):
    return np.loadtxt(path, delimiter=',', skiprows=1)


def save(fig, name):
    fig.savefig(OUT/name, bbox_inches='tight')
    plt.close(fig)
    print('wrote', (OUT/name).relative_to(ROOT))


def title(ax, head, sub):
    """Bold takeaway line with a quieter second line under it."""
    ax.set_title(sub, fontsize=8, fontweight='normal', color=INK2, pad=5)
    ax.annotate(head, xy=(0, 1), xycoords='axes fraction', xytext=(0, 19), textcoords='offset points',
                fontsize=9.5, fontweight='bold', color=INK, annotation_clip=False)


def frame(ax, P, pad=1.25):
    """Square view centred on the drawing, no ticks or spines."""
    c = (P[:, :2].max(axis=0)+P[:, :2].min(axis=0))/2
    r = np.ptp(P[:, :2], axis=0).max()/2*pad
    ax.set_xlim(c[0]-r, c[0]+r); ax.set_ylim(c[1]-r, c[1]+r); ax.set_aspect('equal')
    ax.set_xticks([]); ax.set_yticks([])
    for side in ax.spines.values():
        side.set_visible(False)


def square(ax, lo=0., hi=1.):
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect('equal')
    ax.set_xticks([lo, hi]); ax.set_yticks([lo, hi])


def apply_map(sub, X):
    """Evaluate a submission's map on rows X with the grader's own parser."""
    G.VARS = {f'x{i+1}': i for i in range(X.shape[1])}
    trees, P, E, T = G.validate(sub)
    return G.mapped(trees, X), P, E


def draw_segments(ax, P, E, dims=(0, 1), **kw):
    for n, (i, j) in enumerate(E):
        a, b = P[i][list(dims)], P[j][list(dims)]
        if np.abs(a-b).max() < .5:          # skip edges that only exist through the wrap
            ax.plot([a[0], b[0]], [a[1], b[1]], label=kw.pop('label', None) if n == 0 else None, **kw)


def truth_in_agent_frame(sub, key, holdout, A_out):
    """Planted vertices expressed in the agent's coordinates: a = U (B x) + c (mod 1)."""
    B = np.array(key['B'])
    A = submitted_A(sub['map'], B.shape[1])
    U = np.rint(A @ np.linalg.pinv(B))
    latent = (holdout @ B.T) % 1
    resid = A_out - latent @ U.T
    shift = np.angle(np.exp(2j*np.pi*resid).mean(axis=0))/(2*np.pi)
    lo = np.floor(A_out.min(axis=0)*2)/2          # agent range is [0,1) or [-.5,.5)
    V = np.array(key['true_vertices']) @ U.T + shift
    return (V-lo) % 1 + lo, np.array(key['true_edges'])


# --------------------------------------------------------------------------- hero
def fig_hero():
    """Trefoil knot: raw columns, the hidden 3-D curve seen along its symmetry axis, the agent's answer."""
    from matplotlib.colors import LinearSegmentedColormap
    d = ROOT/'datasets'/'03_trefoil'
    X = load(d/'X.csv.gz'); hold = load(d/'private'/'private_planted.csv.gz')
    key = json.loads((d/'private'/'answer_key.json').read_text())
    sub = json.loads((ROOT/'runs'/'03_trefoil'/'gpt-6-astra'/'submission.json').read_text())
    a, P, E = apply_map(sub, hold)
    V, VE = truth_in_agent_frame(sub, key, hold, a)
    # Viewing frame: the two widest principal axes of the planted knot; the third is depth.
    centre = V.mean(axis=0)
    axes3 = np.linalg.svd(V-centre, full_matrices=False)[2]
    view = lambda Q: (np.asarray(Q)-centre) @ axes3.T
    a_v, P_v, V_v = view(a), view(P), view(V)
    seen = view(apply_map(sub, X)[0])
    depth = LinearSegmentedColormap.from_list('depth', ['#a9cdf5', DATA, '#0c2f5e'])
    lim = np.abs(V_v[:, :2]).max()*1.18

    fig, ax = plt.subplots(1, 3, figsize=(10.5, 3.9))
    ax[0].scatter(X[:, 1], X[:, 11], s=1.5, color=DATA, alpha=.35, linewidths=0, rasterized=True)
    square(ax[0]); ax[0].set_xlabel('x2'); ax[0].set_ylabel('x12')
    title(ax[0], '1. What the agent gets', '8,000 rows x 30 columns that look like noise')

    order = np.argsort(seen[:, 2])                      # far points first, near points on top
    ax[1].scatter(seen[order, 0], seen[order, 1], c=seen[order, 2], cmap=depth, s=3, linewidths=0, rasterized=True)
    title(ax[1], '2. What is hidden', 'a knot in three integer combinations of 5 columns each')
    ax[1].annotate('darker = closer to the viewer', xy=(.5, -.02), xycoords='axes fraction', ha='center',
                   va='top', fontsize=8, color=INK2)

    ax[2].scatter(a_v[:, 0], a_v[:, 1], s=2, color=DATA, alpha=.25, linewidths=0, rasterized=True, label='held-out rows')
    draw_segments(ax[2], V_v, VE, color=TRUTH, lw=3.4, alpha=.9, label='planted knot', solid_capstyle='round')
    draw_segments(ax[2], P_v, E, color=AGENT, lw=1.1, label="agent's drawing")
    title(ax[2], '3. What the agent hands back', 'its own map and drawing, scored on new rows')
    for axis in ax[1:]:
        axis.set_xlim(-lim, lim); axis.set_ylim(-lim, lim); axis.set_aspect('equal')
        axis.set_xticks([]); axis.set_yticks([])
        for side in axis.spines.values():
            side.set_visible(False)
    handles, names = ax[2].get_legend_handles_labels()
    ax[2].legend(handles, names, loc='upper center', bbox_to_anchor=(.5, .02), ncol=3, markerscale=4,
                 columnspacing=1.2, handlelength=1.6)
    fig.tight_layout()
    save(fig, 'hero.png')


# ------------------------------------------------------------------- results grid
def fig_results():
    panels = [('01_pilot_loop', 'gpt-6-astra'), ('02_ytree', 'gpt-6-astra'),
              ('02_ytree', 'gpt-6-astra_minimal-prompt'), ('02_ytree', 'fable-5.2'),
              ('03_trefoil', 'gpt-6-astra'), ('03_trefoil', 'fable-5.2'),
              ('04_sword', 'gpt-6-astra'), ('04_sword', 'fable-5.2')]
    label = {'gpt-6-astra': 'GPT-6 Astra', 'gpt-6-astra_minimal-prompt': 'GPT-6 Astra, minimal prompt',
             'fable-5.2': 'Fable 5.2'}
    fig, axes = plt.subplots(2, 4, figsize=(11, 5.9))
    for ax, (ds, agent) in zip(axes.flat, panels):
        d = ROOT/'datasets'/ds
        sub = json.loads((ROOT/'runs'/ds/agent/'submission.json').read_text())
        pilot = ds.startswith('01')
        rows = np.loadtxt(d/'X.csv.gz', delimiter=',') if pilot else load(d/'private'/'private_planted.csv.gz')
        a, P, E = apply_map(sub, rows)
        ax.scatter(a[:, 0], a[:, 1], s=2, color=DATA, alpha=.3, linewidths=0, rasterized=True,
                   label='visible rows (pilot)' if pilot else 'held-out rows')
        if not pilot:
            key = json.loads((d/'private'/'answer_key.json').read_text())
            V, VE = truth_in_agent_frame(sub, key, rows, a)
            if a.shape[1] == 3:        # look along the knot's symmetry axis
                centre = V.mean(axis=0); axes3 = np.linalg.svd(V-centre, full_matrices=False)[2]
                a, P, V = ((np.asarray(Q)-centre) @ axes3.T for Q in (a, P, V))
            draw_segments(ax, V, VE, color=TRUTH, lw=3.2, alpha=.9, label='planted shape', solid_capstyle='round')
        draw_segments(ax, P, E, color=AGENT, lw=1.1, label="agent's drawing")
        frame(ax, P)
        note = ', viewed along its axis' if a.shape[1] == 3 else ''
        title(ax, {'01': 'Pilot loop', '02': 'Y-tree', '03': 'Trefoil knot', '04': 'Sword'}[ds[:2]], label[agent]+note)
    handles, names = axes[0, 1].get_legend_handles_labels()
    fig.legend(handles, names, loc='lower center', ncol=3, markerscale=4, bbox_to_anchor=(.5, -.01))
    fig.tight_layout(rect=(0, .04, 1, 1), h_pad=2.5)
    save(fig, 'results_grid.png')


# -------------------------------------------------------------------- concealment
def fig_concealment():
    d = ROOT/'datasets'/'04_sword'
    X = load(d/'X.csv.gz'); N, D = X.shape
    B = np.array(json.loads((d/'private'/'answer_key.json').read_text())['B'])
    rng = np.random.default_rng(7)
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 3.6))
    eig = np.sort(np.linalg.eigvalsh(np.cov(X, rowvar=False)))[::-1]*12
    null = np.array([np.sort(np.linalg.eigvalsh(np.cov(rng.random((N, D)), rowvar=False)))[::-1]*12 for _ in range(200)])
    k = np.arange(1, D+1)
    ax[0].fill_between(k, null.min(axis=0), null.max(axis=0), color=INK2, alpha=.18, linewidth=0,
                       label='range over 200 pure-noise matrices')
    ax[0].plot(k, eig, color=DATA, lw=2, marker='o', ms=3.5, label='sword dataset')
    ax[0].set_ylim(.8, 1.2); ax[0].set_xlabel('principal component'); ax[0].set_ylabel('variance / (1/12)')
    ax[0].set_title('PCA sees a pure-noise spectrum'); ax[0].legend(loc='upper right')
    ax[0].grid(axis='y', color=GRID, lw=.6)
    # Fourier characters: random sparse integer directions vs the two planted ones
    E1 = np.exp(2j*np.pi*X); mags = np.empty(20000)
    for t in range(len(mags)):
        cols = rng.choice(D, 6, replace=False); co = rng.choice([-2, -1, 1, 2], 6)
        mags[t] = abs(np.exp(2j*np.pi*(X[:, cols] @ co)).mean())
    planted = np.abs(np.exp(2j*np.pi*(X @ B.T)).mean(axis=0))
    ax[1].hist(mags, bins=40, color=INK2, alpha=.45, label='20,000 random 6-column directions')
    for i, m in enumerate(planted):
        ax[1].axvline(m, color=DATA, lw=2, label='the 2 planted directions' if i == 0 else None)
        ax[1].annotate(f'b{i+1}: {m:.2f}', (m, 3), xytext=(-6, 0),
                       textcoords='offset points', ha='right', color=INK, fontsize=8)
    ax[1].set_xlim(0, max(planted)*1.12); ax[1].set_yscale('log')
    ax[1].set_xlabel(r'character size  $|\frac{1}{N}\sum_n e^{2\pi i\, a\cdot x_n}|$'); ax[1].set_ylabel('directions')
    ax[1].set_title('Only the exact integer directions light up'); ax[1].legend(loc='upper left', bbox_to_anchor=(.08, 1.0))
    fig.tight_layout()
    save(fig, 'concealment.png')


# ------------------------------------------------------------- floating-point leak
def fig_fp_leak(tmp):
    legacy = load(ROOT/'datasets'/'04_sword'/'X.csv.gz')
    cfg = json.loads((ROOT/'blend'/'configs'/'sword.json').read_text())
    blend_generate(cfg, tmp/'sword')
    fixed = load(tmp/'sword'/'agent'/'X.csv')
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 3.2))
    for a, x, title in ((ax[0], legacy, 'Legacy generator: the two solved columns stand out'),
                        (ax[1], fixed, 'Prime-grid generator: every column looks the same')):
        off = 100*((x*2.0**53) % 1 != 0).mean(axis=0)
        a.bar(np.arange(1, len(off)+1), off, width=.72, color=DATA)
        a.set_xlabel('column'); a.set_ylabel('% of values off the 2$^{-53}$ grid')
        a.set_title(title); a.set_xticks([1, 5, 10, 15, 20, 25, 30]); a.grid(axis='y', color=GRID, lw=.6)
        a.set_axisbelow(True)
    for col in np.flatnonzero(((legacy*2.0**53) % 1 != 0).any(axis=0)):
        v = 100*((legacy[:, col]*2.0**53) % 1 != 0).mean()
        ax[0].annotate(f'x{col+1}', (col+1, v), xytext=(0, 3), textcoords='offset points', ha='center', fontsize=8)
    ax[0].set_ylim(0, ax[0].get_ylim()[1]*1.12); ax[1].set_ylim(0, 100)
    fig.tight_layout()
    save(fig, 'fp_leak.png')


# --------------------------------------------------------------------- full BLEND
def fig_full(tmp):
    cfg = json.loads((ROOT/'blend'/'configs'/'sword_full.json').read_text())
    blend_generate(cfg, tmp/'full')
    Y = load(tmp/'full'/'agent'/'X.csv'); hold = load(tmp/'full'/'private'/'private_planted.csv')
    key = json.loads((tmp/'full'/'private'/'answer_key.json').read_text())
    ref = json.loads((tmp/'full'/'private'/'reference_submission.json').read_text())
    W = np.array(ref['unmix'])
    Z = Y @ W.T
    G.VARS = {f'x{i+1}': i for i in range(Y.shape[1])} | {f'z{i+1}': Y.shape[1]+i for i in range(len(W))}
    trees, P, E, _ = G.validate(ref)
    a = G.mapped(trees, np.hstack([hold, hold @ W.T]))
    fig, ax = plt.subplots(1, 3, figsize=(10.5, 3.7))
    ax[0].scatter(Y[:, 0], Y[:, 1], s=1.5, color=DATA, alpha=.35, linewidths=0, rasterized=True)
    ax[0].set_aspect('equal'); ax[0].set_xlabel('y1'); ax[0].set_ylabel('y2')
    title(ax[0], '1. Released: 60 mixed columns', 'no column is a coordinate any more')
    ax[1].scatter(Z[:, 0], Z[:, 1], s=1.5, color=DATA, alpha=.35, linewidths=0, rasterized=True)
    ax[1].set_aspect('equal'); ax[1].set_xlabel('z1 = r1 . y'); ax[1].set_ylabel('z2 = r2 . y')
    title(ax[1], '2. Undo the rotation', 'each original column is a noisy circle')
    ax[2].scatter(a[:, 0], a[:, 1], s=2, color=DATA, alpha=.3, linewidths=0, rasterized=True, label='held-out rows')
    draw_segments(ax[2], P, E, color=TRUTH, lw=1.6, label='planted outline')
    frame(ax[2], np.array(P), 1.1)
    title(ax[2], '3. Angles, then integer directions', 'the same kind of shape, slightly blurred')
    ax[2].legend(loc='lower right', markerscale=4)
    fig.tight_layout()
    save(fig, 'full_blend.png')


if __name__ == '__main__':
    OUT.mkdir(parents=True, exist_ok=True)
    fig_hero()
    fig_results()
    fig_concealment()
    with tempfile.TemporaryDirectory() as t:
        fig_fp_leak(Path(t))
        fig_full(Path(t))
