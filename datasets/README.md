# Datasets

The four instances the agents were tested on. All of them come from the
**legacy** generators (`blend/legacy/`), not from `blend/generate.py`.

| Folder | Rows x columns | Hidden shape | Latent dims | Columns per direction | Latent noise |
|---|---|---|---|---|---|
| `01_pilot_loop` | 4,000 x 50 | smooth closed loop | 2 | 6 | none |
| `02_ytree` | 5,000 x 30 | Y-shaped tree | 2 | 4 | 0.005 |
| `03_trefoil` | 8,000 x 30 | trefoil knot | 3 | 5 | 0.005 |
| `04_sword` | 8,000 x 30 | sword outline | 2 | 6 | 0.004 |

Each folder has:

- `X.csv.gz`: exactly what the agent was given (as `X.csv`).
- `private/`: what the agent must never see.
  - `answer_key.json`: seed, planted integer matrix `B`, true shape, generator diagnostics.
  - `private_planted.csv.gz`, `private_noise.csv.gz`: 5,000 held-out rows each, used by the grader.
  - `reference_submission.json`: the legacy generator's own blind reference solution.

Grade any submission with:

    python -m seek.grade runs/04_sword/fable-5.2/submission.json --private datasets/04_sword/private

## Things to know

- **These instances are burned.** Their answers are public in this repo, so
  they are examples, not a test set. Generate fresh instances for new runs.
- **The pilot is different.** Its CSV has no header row, its `private/` holds
  only `answer_key.npz` (`B` and the latent coordinates `U` of the visible
  rows), and it has no held-out rows, so `seek.grade` cannot score it.
  `PILOT_NOTES.md` is the README of the original prototype.
- **All four have the floating-point leak** in their computed (pivot) columns:
  `x4` in the pilot, `x11`/`x25` in the Y-tree, `x3`/`x4` in the trefoil, `x3`/`x14`
  in the sword. See `blend/legacy/README.md`.
