# **First dataset — Evaluation results**

&#x20;Fully correct

The agent recovered both hidden directions exactly and reconstructed the underlying curve with extremely high precision.

I evaluated it against the **original torus pilot (4,000 rows × 50 columns)**, which matches the dimensions described in the submission.

## **1. Grading results**

Final discovery score

## **95.34%**

Hit planted

# **1.0000**

Hit noise

# **0.0466**

Drawing precision

# **1.0000**

Used 5,000 newly generated planted observations, 5,000 uniform noise observations and ε = 0.055.

The original pilot had no official `grade.py`, so I applied our later Level-2 grading formula to independently generated holdout data.

## **2. Exact lattice recovery**

The original hidden directions were:

\\[ \begin{aligned} b_1&=x\_{19}-x\_{29}+x\_{31}+x\_{36}+x\_{38}+x\_{46}\\\ b_2&=-x_4+x_9-x\_{20}-x\_{26}+x\_{30}+x\_{50} \end{aligned} \\]

The agent recovered:

\\[ a_1=-b_2,\qquad a_2=b_1. \\]

The basis transformation is

\\[ A= \begin{pmatrix} 0&-1\\\ 1&0 \end{pmatrix}B \\]

| Verification   | Result     |
| -------------- | ---------- |
| Planted rank   | 2          |
| Recovered rank | 2          |
| Stacked rank   | 2          |
| Lattice score  | **1.0000** |

The agent's phase offsets and coordinate centering are also correct. Its mapped coordinates match the known latent coordinates to within \\(1.33\times10^{-15}\\).

## **3. Visual recovery**

Blue: observations. Green: actual generated curve. Orange: agent's submitted 200-vertex reconstruction. The right panel uses 5,000 newly generated observations.

| Geometry metric                 | Error     |
| ------------------------------- | --------- |
| Mean agent drawing → true curve | 0.0000214 |
| Mean true curve → agent drawing | 0.0001230 |
| Maximum true curve → drawing    | 0.0008080 |

The agent correctly identified a single smooth, closed, slightly asymmetric loop. It also recovered the original Fourier-generated geometry rather than merely drawing an approximate enclosing shape.

## **4. Verdict**

**Fully correct — 95.34% geometric discovery score and 100% lattice recovery.**

The agent recovered both hidden directions, the correct curve and its geometry, with negligible error.

One important distinction: this submission is for the original **50-column pilot**, not the first Level-2 blind test (30 columns). The latter would reject expressions involving `x50`.
