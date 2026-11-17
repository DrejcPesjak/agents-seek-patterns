# **Level 4 — Evaluation results**

**Fully correct. The agent recovered the sword, both hidden directions, and discovered a previously unnoticed weakness in our dataset generator.**

## **1. Actual grader results**

Final discovery score

## **94.74%**

Hit planted

# **99.88%**

Hit noise

# **5.14%**

Drawing precision

# **100%**

I ran the original, unmodified grader.

| Metric            | Agent      | Reference solver |
| ----------------- | ---------- | ---------------- |
| Hit planted       | 0.9988     | 1.0000           |
| Hit noise         | 0.0514     | 0.0548           |
| Drawing precision | 1.0000     | 1.0000           |
| **Final score**   | **0.9474** | **0.9452**       |

Once again, the agent slightly outperformed the reference solver.

## **2. Visual recovery**

Sword reconstruction

- Blue: 5,000 unseen private observations.
- Orange: agent's reconstruction.
- Dashed green: actual planted sword.

The agent correctly identified the blade, crossguard, grip and pommel.

## **3. Exact lattice recovery**

The agent recovered both planted directions exactly, with the second direction negated.

The basis transformation is:

\\[ A=\begin{pmatrix} 1&0\\\ 0&-1 \end{pmatrix}B \\]

| Property                  | Result   |
| ------------------------- | -------- |
| Planted matrix rank       | 2        |
| Recovered matrix rank     | 2        |
| Stacked matrix rank       | 2        |
| Lattice recovery          | **100%** |
| Connected components      | 1/1      |
| Closed loops              | 1/1      |
| Mean reconstruction error | 0.000817 |
| Planted noise σ           | 0.004000 |

The reconstruction slightly rounds off some sharp corners, but otherwise accurately captures the entire geometry.

## **4. More importantly: the agent found a flaw in our generator**

This is the most interesting part of its submission.

Instead of completing the expensive Fourier search, the agent discovered a **floating-point side channel**.

> The break came from floating-point fingerprints: 28 columns are exact multiples of 2^-53 with the usual trailing-zero pattern, but x3 and x14 are not, so they were computed.

I independently verified this claim using the actual datasets.

### **Values outside the PRNG's numerical grid**

Actual measurements across all 30 columns of the public dataset.

| Dataset              | x3  | x14 | All other columns |
| -------------------- | --- | --- | ----------------- |
| Public data          | 229 | 355 | 0                 |
| Private planted data | 123 | 192 | 0                 |
| Private noise data   | 0   | 0   | 0                 |

These two columns are indeed the generator's pivot columns.

The agent reports subsequently using trailing-zero statistics to identify the variables involved in each constraint. I verified the numerical side channel, although I haven't independently reproduced that second stage of its attack.

### **How we should fix this**

For the next dataset, I would apply identical fine-grained random dithering and quantization to every column in both planted and null datasets.

For example:

```
X = (X + rng.uniform(-1e-9, 1e-9, X.shape)) % 1X = np.round(X, decimals=8) % 1
```

This adds negligible noise relative to the planted noise, while removing the specific numerical-grid fingerprint. We should experimentally check that the attack no longer works.

## **5. Benchmark progress**

**Level 2 — Y-tree**

**98.82%**

**Level 3 — Trefoil knot**

**99.20%**

**Level 4 — Sword**

**94.74%**

## **Verdict**

**Fully correct — Level 4 passed.**

The agent not only discovered the intended hidden structure but also uncovered an unintended computational fingerprint that our concealment tests completely missed.

In some ways, this is the most interesting result of all three experiments: the agent discovered something about the generating mechanism that we hadn't anticipated ourselves.
