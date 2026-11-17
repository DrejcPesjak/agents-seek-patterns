# **Level 3 — Evaluation**

**Verdict: Fully correct**

&#x20;Passed

Official Level-3 grader score

# **99.18%**

Hit planted

**0.9946**

Hit noise

**0.0028**

Drawing precision

**1.0000**

Ran the original `private/grade.py` on `submission(4).json`, using ε = 0.055.

The new submission scores slightly higher than the reference solver's **99.10%**.

## **2. Exact lattice verification**

The agent recovered all three hidden directions exactly.

The submitted matrix \\(A\\) is related to the planted matrix \\(B\\) by

\\[ A=\begin{pmatrix} 0&1&0\\\ 0&0&1\\\ 1&0&0 \end{pmatrix}B \\]

| Verification               | Result   |
| -------------------------- | -------- |
| Planted rank               | 3        |
| Agent rank                 | 3        |
| Stacked rank               | 3        |
| Transformation determinant | +1       |
| **Lattice recovery**       | **100%** |

## **3. Geometry compared with ground truth**

The planted shape was a **(2,3) trefoil knot**, exactly as the agent identified.

Level-3 submitted drawing versus private ground truth

Blue: held-out private observations. Green: actual planted curve. Orange: agent's submitted drawing.

| Measurement                      | Result   |
| -------------------------------- | -------- |
| Mean agent → true curve distance | 0.000451 |
| Mean true → agent curve distance | 0.000441 |
| Maximum sampled deviation        | 0.000899 |
| Planted noise σ                  | 0.005    |
| Submitted vertices               | 200      |
| Submitted segments               | 200      |

The reconstruction is extremely close to the underlying noiseless geometry.

The graph structure also matches: one connected component, one closed cycle, no endpoints and no branch points. I did not independently compute a knot invariant, but the recovered curve closely matches the known trefoil.

## **4. Verdict**

**Fully correct — Level 3 passed.**

The agent recovered all three integer directions exactly, correctly identified the trefoil knot and reconstructed the curve with sub-0.001 geometric error.

Nothing significant was missed. This submission scores **99.18%**, compared with **99.20%** for the previous Level-3 submission and **99.10%** for the reference solver.
