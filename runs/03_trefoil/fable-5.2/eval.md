# **Level 3 — Evaluation results**

**Fully correct. The agent discovered all three hidden directions and correctly reconstructed the trefoil knot.** It again slightly outperformed the reference solver.

## **1. Actual grader results**

Final discovery score

## **99.20%**

Hit planted

# **99.46%**

Hit noise

# **0.26%**

Drawing precision

# **100%**

I ran the original, unmodified grader on the submitted JSON.

| Metric            | Agent      | Reference solver |
| ----------------- | ---------- | ---------------- |
| Hit planted       | 0.9946     | 0.9938           |
| Hit noise         | 0.0026     | 0.0028           |
| Drawing precision | 1.0000     | 1.0000           |
| **Final score**   | **0.9920** | **0.9910**       |

## **2. Visual comparison**

Recovered trefoil knot

The orange line is the agent's reconstruction, the green line is the ground truth, and the blue dots are 5,000 unseen private observations.

The two curves are almost perfectly overlapping.

## **3. Exact lattice recovery**

The planted directions were:

\\[ \begin{aligned} b_1 &= x_2+2x\_{12}+x\_{16}+x\_{24}+2x\_{27}\\\ b_2 &= x_3-x_6+2x_8-2x\_{10}-x\_{29}\\\ b_3 &= x_4-2x\_{13}-x\_{19}+2x\_{22}+x\_{30} \end{aligned} \\]

The agent recovered all three exactly, just in a different order.

The basis transformation is:

\\[ A= \begin{pmatrix} 0&1&0\\\ 0&0&1\\\ 1&0&0 \end{pmatrix}B \\]

| Check                      | Result   |
| -------------------------- | -------- |
| Planted matrix rank        | 3        |
| Recovered matrix rank      | 3        |
| Stacked matrix rank        | 3        |
| Transformation determinant | 1        |
| **Lattice recovery**       | **100%** |

## **4. Shape recovery**

The actual hidden shape was a **(2,3)-torus knot**, also known as a trefoil knot.

The agent identified it correctly and reconstructed it using a closed Fourier series with six harmonics.&#x20;

submission.json



| Geometric metric                                | Result   |
| ----------------------------------------------- | -------- |
| Mean distance from reconstruction to true curve | 0.000316 |
| Mean distance from true curve to reconstruction | 0.000313 |
| Maximum sampled reconstruction error            | 0.000867 |
| Planted noise σ                                 | 0.005    |
| Submitted vertices                              | 200      |
| Connected components                            | 1        |
| Closed loops                                    | 1        |

### **Reconstruction error vs. planted noise**

Mean reconstruction error

**0.000316**

Maximum sampled error

**0.000867**

Planted noise standard deviation

**0.005000**

The mean reconstruction error is approximately **16 times smaller than the noise standard deviation**.

I verified geometric proximity and graph connectivity, although I did not independently compute a formal knot invariant.

## **5. Benchmark progress**

**Level 2 — Y-shaped tree**

**98.82%**

**Level 3 — Trefoil knot**

**99.20%**

## **6. Verdict**

**Fully correct — Level 3 passed.**

The agent recovered the entire lattice, correctly identified the knot, and reconstructed its geometry with negligible error.

Interestingly, the agent reports independently attempting lattice reduction before moving to GPU-accelerated Fourier search. It is selecting appropriate mathematical tools without being told the generator family.

For Level 4, I would introduce a fundamentally different concealment mechanism rather than merely increasing sparsity or geometric complexity.
