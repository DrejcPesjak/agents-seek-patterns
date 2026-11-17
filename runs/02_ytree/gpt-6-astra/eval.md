# **Level 2 — Evaluation results**

&#x20;Fully correct — exact lattice recovery and correct Y-tree geometry.

I ran the original `private/grade.py` on your new submission against the **second dataset (Level 2, 5,000 × 30)**.

Final discovery score

# **98.82%**

Hit planted

**0.9980**

Hit noise

**0.0098**

Drawing precision

**1.0000**

Grader tolerance ε = 0.055

## **Ground-truth comparison**

The agent recovered both planted integer directions exactly, in reversed order:

\\[ A=\begin{pmatrix}0&1\\\1&0\end{pmatrix}B \\]

The planted matrix, submitted matrix and stacked matrix all have rank 2. **Lattice recovery = 100%.**

The hidden shape was a four-node Y-shaped tree. The agent correctly submitted one junction, three endpoints and three connecting segments.

| Vertex   | Coordinate error |
| -------- | ---------------- |
| Junction | 0.000662         |
| Arm 1    | 0.000329         |
| Arm 2    | 0.000857         |
| Arm 3    | 0.000561         |

All four vertices are within 0.001 of the ground truth, compared with the planted noise standard deviation of 0.005.

## **Visual verification**

Agent's Y-tree compared with the private ground truth

Blue points are the 5,000 unseen private observations. Orange is the agent's drawing; dashed green is the planted geometry.

## **Comparison with the previous Level-2 submission**

| Metric           | Previous | New        |
| ---------------- | -------- | ---------- |
| Final score      | 98.82%   | **98.82%** |
| Lattice recovery | 100%     | 100%       |
| Drawing vertices | 61       | **4**      |
| Drawing segments | 60       | **3**      |
| Topology         | Correct  | Correct    |

**Verdict: Fully correct.** The new submission achieves the same score with a much simpler, more interpretable representation. It recovers the exact underlying tree topology without needing 61 sampled points.

The report isn't graded automatically, but its description is also consistent with the private ground truth.
