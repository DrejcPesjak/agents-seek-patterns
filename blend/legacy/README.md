# Legacy generators (archived, not maintained)

These are the original per-level scripts that produced the datasets the agents
were actually tested on. They are kept unchanged for the record.

| File | Produced | Shape |
|---|---|---|
| `generator_level2_ytree.py` | `datasets/02_ytree` | random 4-6 node tree (came out as a Y) |
| `generator_level3_trefoil.py` | `datasets/03_trefoil` | torus knot in 3 latent dimensions |
| `generator_level4_sword.py` | `datasets/04_sword` | sword outline |
| `grade.py` | the grader they import (fixed to 30 columns) | |

The 50-column pilot (`datasets/01_pilot_loop`) came from earlier prototype
code that was not archived; only its data and answer key survive.

## Why they were replaced

They embed in floating point: free columns come straight from the random
number generator, and the `k` pivot columns are then *computed* as
`((u - x B^T) B_P^-1) mod 1`. Generator output is always a multiple of
2^-53; a computed value below 0.5 often is not. So the pivot columns can be
read off from the binary representation alone, without any statistics. One
agent found this on the sword dataset (columns `x3` and `x14`). See
`docs/figures/fp_leak.png`.

`blend/generate.py` samples everything on one prime grid instead, so pivot
and free columns have the same numeric format.

## Running them

Each script expects `grade.py` and (for `--reproduce`) the instance's
`answer_key.json` next to it, writes `../agent/X.csv`, and needs
scikit-learn. They also contain two things the current generator does not
have: a blind reference solver (`solve_lattice`, a split Fourier search) and
acceptance gates that reject an instance unless that solver recovers it.
