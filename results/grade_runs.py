#!/usr/bin/env python3
"""Re-grade every run in runs/ with seek/grade.py and write results/results.csv.

Run from the repo root:  python results/grade_runs.py

For each runs/<dataset>/<agent>/submission.json this records the grader score
on the dataset's held-out rows, and whether the integer directions in the
submitted map span exactly the planted lattice. The pilot dataset has no
held-out rows, so its score is taken from the archived chat evaluation.
"""
import csv
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from seek.grade import grade_file  # noqa: E402

AGENTS = {
    'gpt-6-astra': ('GPT-6 Astra (medium)', 'prompt.txt'),
    'gpt-6-astra_minimal-prompt': ('GPT-6 Astra (medium)', 'prompt_minimal.txt'),
    'fable-5.2': ('Claude Fable 5.2', 'prompt.txt'),
}
NOTES = {
    ('01_pilot_loop', 'fable-5.2'): 'Concluded "none" in LOG.md; no submission.json written. File was named example_X.csv; exact prompt not archived.',
    ('01_pilot_loop', 'gpt-6-astra'): 'No held-out rows for the pilot; score is from the chat evaluation on freshly generated rows.',
    ('04_sword', 'fable-5.2'): 'Found the pivot columns through the floating-point side channel of the legacy generator.',
}
TERM = re.compile(r'([+-]?)\s*(?:(\d+)\s*\*\s*)?x(\d+)')


def planted_B(dataset: Path) -> np.ndarray:
    key = dataset/'private'/'answer_key.json'
    if key.exists():
        return np.array(json.loads(key.read_text())['B'])
    return np.load(dataset/'private'/'answer_key.npz')['B']


def submitted_A(maps: list[str], D: int) -> np.ndarray:
    """Integer coefficients of x1..xD in each map expression (sparse linear forms)."""
    A = np.zeros((len(maps), D), dtype=int)
    for row, text in enumerate(maps):
        for sign, coef, col in TERM.findall(text):
            A[row, int(col)-1] += (-1 if sign == '-' else 1)*int(coef or 1)
    return A


def lattice_recovered(A: np.ndarray, B: np.ndarray) -> bool:
    """Same rational row space and A = U B with U an integer matrix of determinant +-1."""
    rank = np.linalg.matrix_rank
    if not (rank(A) == rank(B) == rank(np.vstack([A, B])) == len(B) == len(A)):
        return False
    U = A @ np.linalg.pinv(B)
    return bool(np.allclose(U, np.rint(U), atol=1e-9) and abs(round(np.linalg.det(np.rint(U)))) == 1)


def chat_eval_score(run: Path):
    path = run/'eval.md'
    if not path.exists():
        return ''
    found = re.search(r'\*\*(\d+\.\d+)%\*\*', path.read_text())
    return round(float(found.group(1))/100, 4) if found else ''


def main():
    rows = []
    for run in sorted(p for p in (ROOT/'runs').glob('*/*') if p.is_dir()):
        dataset, agent = run.parent.name, run.name
        model, prompt = AGENTS[agent]
        row = {'dataset': dataset, 'agent': agent, 'model': model, 'prompt': prompt,
               'status': '', 'score': '', 'hit_planted': '', 'hit_noise': '',
               'drawing_precision': '', 'lattice_recovered': '', 'drawing_points': '',
               'score_source': '', 'chat_eval_score': chat_eval_score(run),
               'notes': NOTES.get((dataset, agent), '')}
        sub = run/'submission.json'
        private = ROOT/'datasets'/dataset/'private'
        if not sub.exists():
            row.update(status='none', score=0.0, lattice_recovered=False, score_source='no submission')
        else:
            obj = json.loads(sub.read_text())
            B = planted_B(ROOT/'datasets'/dataset)
            row.update(status=obj['status'], drawing_points=len(obj['drawing']['points']),
                       lattice_recovered=lattice_recovered(submitted_A(obj['map'], B.shape[1]), B))
            if (private/'private_planted.csv.gz').exists():
                g = grade_file(sub, private)
                row.update(score=round(g['score'], 4), hit_planted=round(g['hit_planted'], 4),
                           hit_noise=round(g['hit_noise'], 4),
                           drawing_precision=round(g['drawing_precision'], 4),
                           score_source='seek/grade.py')
            else:
                row.update(score=row['chat_eval_score'], score_source='chat evaluation')
        rows.append(row)
        print(f"{dataset:15s} {agent:28s} {row['status']:9s} score={row['score']} lattice={row['lattice_recovered']}")
    with open(ROOT/'results'/'results.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == '__main__':
    main()
