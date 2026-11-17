# **Level-2 blind test — Results**

**The agent successfully recovered the entire hidden structure.** Both integer directions are exactly correct, and the recovered geometry is remarkably accurate.

## **1. Actual grader results**

I ran the original, unmodified `private/grade.py` against the agent's submission.

Final discovery score

## **98.82%**

Hit planted

# **99.80%**

Hit noise

# **0.98%**

Drawing precision

# **100%**

| Metric            | Agent      | Reference solver |
| ----------------- | ---------- | ---------------- |
| Hit planted       | 0.9980     | 1.0000           |
| Hit noise         | 0.0098     | 0.0136           |
| Drawing precision | 1.0000     | 1.0000           |
| **Score**         | **0.9882** | **0.9864**       |

The agent even slightly outperformed the reference solver under the specified scoring function.

## **2. Comparison against ground truth**

### **Lattice recovery — exact**

The actual planted directions were:

\\[ b_1=2x_1-2x_6-x\_{11}-x\_{24} \\]

\\[ b_2=2x\_{10}+2x\_{23}+x\_{25}+x\_{30} \\]

The agent recovered exactly these two directions, in reverse order.

The integer transformation relating the bases is:

\\[ A= \underbrace{ \begin{pmatrix} 0&1\\\ 1&0 \end{pmatrix}}\_{U} B \\]

I verified the ranks using exact arithmetic:

| Check                         | Result     |
| ----------------------------- | ---------- |
| Rank of planted matrix        | 2          |
| Rank of recovered matrix      | 2          |
| Rank of stacked matrices      | 2          |
| Determinant of transformation | −1         |
| **Lattice recovery score**    | **1.0000** |

### **Shape recovery — correct**

The randomly selected generator was **random tree with 4–6 nodes**. It generated a four-node, Y-shaped tree.

The following are the actual ground-truth coordinates transformed into the agent's recovered basis.

| Vertex         | Ground truth         | Agent estimate       | Euclidean error |
| -------------- | -------------------- | -------------------- | --------------- |
| Junction       | (0.517041, 0.510486) | (0.517360, 0.510760) | 0.000420        |
| Right arm      | (0.608235, 0.541898) | (0.607230, 0.541560) | 0.001060        |
| Upper-left arm | (0.434790, 0.575215) | (0.434500, 0.575240) | 0.000291        |
| Lower arm      | (0.502143, 0.398335) | (0.501980, 0.399310) | 0.000988        |

Even the largest vertex error is less than a quarter of the planted noise standard deviation (0.005).

### **Topology — exact**

| Property             | Ground truth | Agent |
| -------------------- | ------------ | ----- |
| Connected components | 1            | 1     |
| Branch points        | 1            | 1     |
| Degree of junction   | 3            | 3     |
| Endpoints            | 3            | 3     |
| Loops                | 0            | 0     |

The agent submitted 61 vertices and 60 segments. Removing unnecessary degree-two vertices produces exactly the original four-node tree.

## **3. Visual comparison**

Agent recovery versus ground truth

- **Blue:** 5,000 unseen private observations.
- **Orange:** Agent's submitted drawing.
- **Dashed green:** Actual planted geometry.

The two drawings are almost indistinguishable.

## **4. Verdict**

**Fully correct — Level 2 passed.**\
Both hidden directions, geometry and topology were successfully recovered.\
Nothing meaningful was missed; only negligible numerical estimation errors remain.
