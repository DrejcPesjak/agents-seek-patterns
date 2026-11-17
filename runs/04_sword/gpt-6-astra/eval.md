# **Level 4 — Sword evaluation**

Fully correct

The agent recovered the hidden sword outline and both integer directions.

Official grader score

# **94.74%**

I ran the original Level-4 private grader on `submission(5).json`, without modifications.

| Metric            | Agent      | Reference solver |
| ----------------- | ---------- | ---------------- |
| Hit planted       | 1.0000     | 1.0000           |
| Hit noise         | 0.0526     | 0.0548           |
| Drawing precision | 1.0000     | 1.0000           |
| **Final score**   | **0.9474** | **0.9452**       |

The agent slightly outperformed the reference solver.

## **Exact lattice verification**

The agent's two expressions are **identical to the planted directions**, including coefficient signs and ordering.

\\[ B\_{\text{agent}}=B\_{\text{true}} \\]

| Check                | Result   |
| -------------------- | -------- |
| Planted rank         | 2        |
| Agent rank           | 2        |
| Stacked rank         | 2        |
| Basis transformation | Identity |
| **Lattice recovery** | **100%** |

## **Geometry compared with ground truth**

Level-4 sword drawing compared to ground truth

Blue: 5,000 held-out observations. Green dashed: true sword. Orange: agent's 160-vertex drawing. Right: crossguard detail.

| Measurement                        | Result   |
| ---------------------------------- | -------- |
| Mean true outline → agent drawing  | 0.000787 |
| Mean agent drawing → true outline  | 0.000699 |
| 95th-percentile true → agent error | 0.001996 |
| Maximum true → agent deviation     | 0.009117 |
| Planted noise σ                    | 0.004    |
| Connected components               | 1        |
| Closed cycles                      | 1        |

The reconstruction preserves the sword's blade, point, curved crossguard, grip and pommel. The biggest differences are around a few sharp corners, but the outline is extremely close.

## **Comparison with the previous sword submission**

| Metric           | Previous | New         |
| ---------------- | -------- | ----------- |
| Score            | 94.74%   | **94.74%**  |
| Planted hit rate | 99.88%   | **100.00%** |
| Noise hit rate   | 5.14%    | 5.26%       |
| Drawing vertices | 180      | **160**     |
| Lattice recovery | 100%     | 100%        |

The interesting difference is methodological: the previous agent exploited the floating-point fingerprint. This new agent **reports using Fourier-character search**, without mentioning that shortcut. That is encouraging, although the report alone cannot verify every step of its process.

**Final verdict: Fully correct — Level 4 passed again.** The integer directions are exact, the geometry is highly accurate, and nothing structurally important was missed.
