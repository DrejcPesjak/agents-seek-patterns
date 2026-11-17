# Mathematical Foundations of Hidden Geometry in Nearly Independent Distributions

> Long-form companion to the [README](../README.md). The construction in sections 5 and 13 is
> implemented in `blend/generate.py`, section 11 is its `"encoding": "full"` mode, and section 12
> is `seek/grade.py`.

The method we developed is a way to **embed a low-dimensional, geometrically meaningful structure inside high-dimensional numerical data while making every sufficiently small subset of observed variables statistically independent**.

It combines ideas from harmonic analysis, integer lattices, modular arithmetic, Haar probability measures, geometry, and statistical hypothesis testing.

The central idea is:

> Construct a dataset whose individual coordinates look like independent uniform noise, but whose carefully chosen integer combinations reveal a recognizable geometric object.

Our previous experiments encoded a Y-shaped tree, a trefoil knot, and a sword outline. All three used essentially the same mathematical construction.

Below is a formal treatment, including the guarantees, proofs, recovery methods, and limitations.

## 1. The mathematical problem

Let

```math
X\in[0,1)^{N\times D}
```

be an observed dataset containing $`N`$ observations and $`D`$ features.

An investigator receives only $`X`$. There are no labels, target variables, or descriptions of what might be hidden.

We want to construct $`X`$ such that:

1. Its low-dimensional marginals look exactly like independent uniform random variables.
2. There exists a hidden transformation $`f`$ under which the observations reveal a recognizable geometric structure.
3. Discovering $`f`$ requires examining interactions among several variables.
4. The structure can be recovered and evaluated without access to the generator.

The transformation in our current family is

```math
\boxed{f_B(x)=Bx\bmod 1}
```

where

```math
B\in\mathbb Z^{k\times D}, \qquad k\ll D
```

is a hidden integer matrix of rank $`k`$.

For example, with $`D=30`$ and $`k=2`$,

```math
x\in[0,1)^{30} \quad\longmapsto\quad u=Bx\bmod1\in[0,1)^2.
```

The $`30`$-dimensional observation looks unstructured, while the two-dimensional output might trace a sword.

**Hidden geometry **$`M`$

A tree, knot, silhouette, graph, or other structure

Sample latent coordinates $`u`$

**Modular embedding **$`Bx=u\pmod1`$

Randomly choose one high-dimensional preimage $`x`$

Release only $`X`$

**Apparently independent high-dimensional data**

Agent must recover both $`B`$ and the geometry

The crucial difference from ordinary dimensionality reduction is that we **generate the high-dimensional observations from the hidden structure**, rather than simply projecting existing observations onto a lower-dimensional space.

## 2. Why modular arithmetic?

### 2.1 The torus

Define the one-dimensional torus as

```math
\mathbb T=\mathbb R/\mathbb Z.
```

It can be represented by the interval $`[0,1)`$, with endpoints identified.

Addition takes place modulo 1:

```math
a\oplus b=(a+b)\bmod1.
```

The $`D`$-dimensional torus is

```math
\mathbb T^D=(\mathbb R/\mathbb Z)^D.
```

Although $`[0,1)^D`$ looks like a hypercube, it has periodic geometry: crossing the right boundary of one coordinate brings us back to its left boundary.

This is important because integer matrices act naturally on tori.

Given

```math
B\in\mathbb Z^{k\times D},
```

define

```math
F_B:\mathbb T^D\rightarrow\mathbb T^k, \qquad F_B(x)=Bx\bmod1.
```

Because $`B`$ contains integers,

```math
F_B(x+z)=F_B(x)
```

for every $`z\in\mathbb Z^D`$.

Thus the map is well-defined on equivalence classes modulo integers.

If $`B`$ has rank $`k`$, this torus homomorphism is surjective: every latent point $`u\in\mathbb T^k`$ has at least one preimage.

### 2.2 The fiber of a latent point

For a particular $`u`$, define its fiber:

```math
\boxed{ F_B^{-1}(u) = \{x\in\mathbb T^D:Bx=u\pmod1\}. }
```

This is the set of all high-dimensional vectors encoding the same latent coordinates.

When $`B`$ is saturated (defined below), its kernel is a connected torus of dimension $`D-k`$, and each fiber is a translated copy of that torus.

For $`D=30`$ and $`k=2`$, the fiber has dimension 28.

Therefore, **one point in the hidden two-dimensional sword corresponds to an entire 28-dimensional family of observations**.

This large amount of freedom is what allows the observed coordinates to behave like independent random variables.

## 3. Constructing the hidden geometry

Let

```math
M\subseteq\mathbb T^k
```

denote the hidden geometric object.

We define a probability distribution $`\mu_M`$ concentrated on or near $`M`$.

### 3.1 A graph or silhouette

A shape such as a sword can be represented by a graph

```math
G=(V,E),
```

where

```math
V=\{v_1,\ldots,v_m\}\subset\mathbb T^2
```

contains geometric vertices and

```math
E\subseteq\{1,\ldots,m\}^2
```

contains connected edges.

A point on an edge $`(i,j)`$ can be sampled as

```math
u(t)=(1-t)v_i+tv_j, \qquad t\sim U(0,1).
```

To sample uniformly by arc length across the whole graph, choose edge $`e`$ with probability

```math
P(e)=\frac{L_e}{\sum_{e'\in E}L_{e'}},
```

where $`L_e`$ is its length.

For boundary-crossing torus edges, this interpolation must use an unwrapped integer lift before reducing modulo 1.

### 3.2 Parametric curves and knots

Alternatively, define

```math
g:[0,1)\rightarrow\mathbb T^k
```

and sample

```math
u=g(t),\qquad t\sim\nu.
```

For example, the standard trefoil knot can be parameterized in $`\mathbb R^3`$ as

```math
q(t)= \begin{pmatrix} (2+\cos3t)\cos2t\\ (2+\cos3t)\sin2t\\ \sin3t \end{pmatrix}, \quad t\in[0,2\pi).
```

After applying an affine transformation

```math
g(t)=c+Aq(t),
```

we can place the knot inside the interior of $`[0,1)^3`$.

The sampling distribution $`\nu`$ matters: uniform sampling of $`t`$ is not generally uniform sampling by arc length. For arc-length sampling, its density should be proportional to $`\|g'(t)\|`$.

### 3.3 Latent noise

To avoid an unrealistically perfect manifold, add wrapped Gaussian noise:

```math
U=(Z+\varepsilon)\bmod1,
```

where

```math
Z\sim\mu_M,\qquad \varepsilon\sim\mathcal N(0,\sigma^2I_k).
```

In our experiments, we typically used

```math
\sigma\approx0.004\text{–}0.005.
```

This produces a thin noisy neighborhood around the shape, rather than points lying exactly on a curve.

The complete latent distribution is the convolution

```math
\mu=\mu_M*\mathcal N_{\mathbb T^k}(0,\sigma^2I_k).
```

Once $`\mu`$ is defined, the concealment mechanism does not need to know whether it represents a sword, a tree, or a knot.

That is why the generator can be made shape-independent.

## 4. The hidden integer matrix

The next ingredient is a rank-$`k`$ matrix

```math
B= \begin{pmatrix} b_1^\top\\ \vdots\\ b_k^\top \end{pmatrix} \in\mathbb Z^{k\times D}.
```

Each row represents a hidden integer direction.

For instance,

```math
u_1=(x_1-x_2+2x_3-2x_4)\bmod1
```

and

```math
u_2=(x_5+2x_6-x_7-2x_8)\bmod1.
```

The associated matrix, restricting attention to eight coordinates, is

```math
B= \begin{pmatrix} 1&-1&2&-2&0&0&0&0\\ 0&0&0&0&1&2&-1&-2 \end{pmatrix}.
```

The remaining coordinates in a larger dataset can be placed anywhere and can participate in more complicated relationships.

Two properties of $`B`$ are especially important.

### 4.1 Sparsity and minimum support

For a vector $`v`$, define its support:

```math
\operatorname{supp}(v)=\{j:v_j\ne0\}.
```

Its support size is

```math
\|v\|_0=|\operatorname{supp}(v)|.
```

A naive complexity measure would be the support size of each individual row $`b_i`$.

But that is insufficient.

For example, two rows could each contain six nonzero coefficients, yet their difference might involve only two coordinates.

We therefore define

```math
\boxed{ \delta(B)= \min_{\substack{v\in\operatorname{row}_{\mathbb Q}(B)\\v\ne0}} \|v\|_0. }
```

This measures the sparsest nonzero vector in the **entire rational row space**, rather than in the particular basis used by the generator.

An equivalent characterization is

```math
\boxed{ \delta(B)= \min_{\substack{I\subseteq\{1,\ldots,D\}\\ \operatorname{rank}_{\mathbb Q}(B_{I^c})<k}} |I|. }
```

Here $`B_{I^c}`$ is the matrix restricted to columns outside $`I`$.

For our simple example, the two directions have disjoint supports of size four. No nonzero combination can cancel coordinates shared between them, because there are none.

Consequently,

```math
\delta(B)=4.
```

This number is central to the concealment guarantee.

### 4.2 Saturation

Let

```math
\Lambda_B= \{m^\top B:m\in\mathbb Z^k\} \subseteq\mathbb Z^D
```

be the integer row lattice generated by $`B`$.

The lattice is saturated if

```math
\boxed{ \Lambda_B= \operatorname{row}_{\mathbb Q}(B)\cap\mathbb Z^D. }
```

Informally, there are no missing integer vectors in the rational subspace spanned by the rows of $`B`$.

A rank-$`k`$ integer matrix is saturated precisely when the greatest common divisor of its nonzero $`k\times k`$ minors is 1.

Equivalently, all $`k`$ nonzero Smith-normal-form invariant factors equal 1.

Saturation gives us a clean integer lattice representation and a connected kernel.

**Important distinction:** Saturation does not imply that some particular $`k\times k`$ submatrix has determinant $`\pm1`$. For example, the one-row matrix $`(2,3)`$ is saturated even though neither entry is $`\pm1`$.

For a universal generator, saturation should be verified using exact integer arithmetic rather than assumed.

## 5. Generating high-dimensional observations

We now have:

- A hidden matrix $`B`$.
- A latent distribution $`\mu`$.
- A desired number of observations $`N`$.

The goal is to sample $`X`$ such that

```math
BX\bmod1\sim\mu
```

while making the observed coordinates maximally random.

The correct sampling rule is:

```math
\boxed{ U\sim\mu,\qquad X\mid U=u\sim\operatorname{Haar}(F_B^{-1}(u)). }
```

In words:

1. Sample a latent point from the hidden shape.
2. Sample uniformly from all high-dimensional vectors that map to that point.

The second step is critical.

Merely solving $`Bx=u`$ with some deterministic choice of $`x`$ would leak structure through the individual coordinates.

### 5.1 Practical implementation using pivot columns

Suppose $`B`$ contains a $`k\times k`$ unimodular submatrix $`B_P`$, satisfying

```math
\det B_P=\pm1.
```

Partition the columns into pivot columns $`P`$ and free columns $`F`$:

```math
B=(B_P\mid B_F).
```

The constraint becomes

```math
B_Px_P+B_Fx_F=u\pmod1.
```

Sample the free coordinates independently:

```math
x_F\sim U([0,1)^{D-k}).
```

Then solve

```math
\boxed{ x_P=B_P^{-1}(u-B_Fx_F)\bmod1. }
```

Because $`B_P`$ is unimodular, this gives exactly one pivot solution on the torus for each choice of free coordinates.

The resulting $`X`$ is Haar-uniform on the fiber.

### 5.2 Worked example

Using our eight-dimensional example,

```math
\begin{aligned} u_1 &=x_1-x_2+2x_3-2x_4\pmod1,\\ u_2 &=x_5+2x_6-x_7-2x_8\pmod1. \end{aligned}
```

Choose $`x_1,x_5`$ as pivots.

Sample

```math
x_2,x_3,x_4,x_6,x_7,x_8 \overset{\mathrm{iid}}{\sim}U(0,1).
```

Then compute

```math
\boxed{ \begin{aligned} x_1&=(u_1+x_2-2x_3+2x_4)\bmod1,\\ x_5&=(u_2-2x_6+x_7+2x_8)\bmod1. \end{aligned} }
```

This guarantees that the two intended modular expressions recover the original latent point.

For a general saturated matrix without a unimodular pivot submatrix, Smith normal form provides a change of integer coordinates that reduces the map to a coordinate projection. We can sample the remaining coordinates uniformly and transform back.

For nonsaturated matrices, the same construction requires additional care to sample uniformly across all disconnected components of a fiber.

## 6. The main concealment theorem

This is arguably the most important mathematical result underlying our benchmark.

**Theorem — Exact low-order marginal independence**

Let $`B\in\mathbb Z^{k\times D}`$ have rank $`k`$. Let $`U`$ have any probability distribution on $`\mathbb T^k`$, and sample $`X`$ Haar-uniformly on the fiber $`Bx=U\pmod1`$.

Then, for every coordinate subset $`I`$ satisfying

```math
|I|<\delta(B),
```

we have

```math
\boxed{ X_I\sim U([0,1)^{|I|}). }
```

In other words, those observed coordinates are **exactly mutually independent uniform variables**, regardless of the hidden shape.

### 6.1 Proof

Choose a coordinate subset $`I`$ with $`|I|<\delta(B)`$. Let $`F=I^c`$.

By the definition of $`\delta(B)`$,

```math
\operatorname{rank}_{\mathbb Q}(B_F)=k.
```

Therefore, the map

```math
x_F\longmapsto B_Fx_F\bmod1
```

is surjective onto $`\mathbb T^k`$.

For every prescribed value of $`x_I`$ and every latent point $`u`$, there exists a solution $`x_F`$ satisfying

```math
B_Ix_I+B_Fx_F=u\pmod1.
```

Thus the projection of the fiber onto the coordinates $`I`$ is surjective.

The projection of Haar measure on that fiber is Haar measure on $`\mathbb T^{|I|}`$.

Haar measure on the coordinate torus is precisely the product-uniform distribution.

Consequently,

```math
X_I\sim U([0,1)^{|I|}).
```

```math
\qquad\blacksquare
```

### 6.2 What the theorem guarantees

Suppose

```math
\delta(B)=6.
```

Then every subset of at most five columns is jointly independent uniform.

In particular:

| Statistic                            | Population-level behavior |
| ------------------------------------ | ------------------------- |
| Individual histograms                | Uniform                   |
| Means and variances                  | Same as iid uniform       |
| Pearson correlations                 | Zero                      |
| Pairwise mutual information          | Zero                      |
| Any triple dependency test           | Null                      |
| Every five-column joint distribution | Product-uniform           |
| Covariance matrix                    | $`\frac1{12}I_D`$       |
| Population PCA spectrum              | All eigenvalues equal     |

This isn't merely a weak statistical signal or an approximation.

**These properties hold exactly at the population level.**

Finite datasets will still exhibit random fluctuations, just as genuinely independent uniform data do.

The theorem does not guarantee that the entire distribution is indistinguishable from iid noise. It says only that all coordinate subsets smaller than $`\delta(B)`$ have the same distributions.

A sufficiently high-order method can recover the hidden structure.

### 6.3 Why this resembles RANDU

RANDU's famous failure was invisible to some ordinary one-dimensional tests but appeared when points were examined jointly in three dimensions, revealing parallel planes.

Our construction follows a related principle:

```math
\text{simple-looking individual coordinates} \quad\not\Rightarrow\quad \text{joint independence}.
```

However, the mechanism is different.

RANDU uses arithmetic dependencies caused by a linear congruential generator. Our construction deliberately plants modular Fourier dependencies and embeds an arbitrary latent geometry in them.

## 7. Fourier analysis: where the hidden structure lives

We have established that low-order coordinate distributions cannot reveal the planted structure.

So what can?

The answer is the **Fourier transform of the joint distribution on the torus**.

### 7.1 Fourier characters

For an integer frequency vector

```math
a\in\mathbb Z^D,
```

define the Fourier character

```math
\chi_a(x)=e^{2\pi i a^\top x}.
```

The Fourier coefficient of the observed distribution $`P_X`$ at frequency $`a`$ is

```math
\widehat P_X(a)= \mathbb E_{X\sim P_X}[e^{2\pi i a^\top X}].
```

For independent uniform coordinates,

```math
\widehat P_{\mathrm{null}}(a)=0
```

for every nonzero $`a\in\mathbb Z^D`$.

This follows from

```math
\int_0^1e^{2\pi iax}\,dx=0 \qquad(a\in\mathbb Z\setminus\{0\}).
```

Now consider our hidden distribution.

### 7.2 The Fourier-support theorem

**Theorem — Fourier coefficients of the fiber construction**

For the generative model defined above,

```math
\boxed{ \widehat P_X(a)= \begin{cases} \widehat\mu(m),&a=B^\top m,\quad m\in\mathbb Z^k,\\ 0,&a\notin B^\top\mathbb Z^k. \end{cases} }
```

Here

```math
\widehat\mu(m)= \mathbb E_{U\sim\mu}[e^{2\pi i m^\top U}].
```

Thus, every nonzero Fourier coefficient of the high-dimensional observations lies in the integer lattice generated by the rows of $`B`$.

For saturated $`B`$, this lattice is exactly

```math
\operatorname{row}_{\mathbb Q}(B)\cap\mathbb Z^D.
```

### 7.3 Proof

Condition on the latent variable $`U=u`$.

The conditional distribution of $`X`$ is Haar-uniform on $`F_B^{-1}(u)`$.

A Fourier character is constant on that fiber precisely when it is trivial on the kernel of $`F_B`$.

By the character theory of compact abelian groups, the characters trivial on this kernel are exactly those with frequencies

```math
a=B^\top m,\qquad m\in\mathbb Z^k.
```

If $`a`$ does not have this form, integrating the character over the fiber gives zero:

```math
\mathbb E[\chi_a(X)\mid U=u]=0.
```

If $`a=B^\top m`$, then

```math
\begin{aligned} \chi_a(X) &=e^{2\pi i(B^\top m)^\top X}\\ &=e^{2\pi i m^\top BX}\\ &=e^{2\pi i m^\top U}. \end{aligned}
```

Therefore,

```math
\mathbb E[\chi_a(X)\mid U=u] =e^{2\pi i m^\top u}.
```

Taking the expectation over $`U`$ yields the stated formula.

```math
\qquad\blacksquare
```

This result is the algebraic core of the method.

### 7.4 Why it works

Consider one planted direction

```math
b=(0,0,1,-1,0,2,0,-2,\ldots).
```

If the latent distribution has

```math
|\widehat\mu(e_1)|=0.8,
```

then the corresponding high-dimensional frequency satisfies

```math
|\widehat P_X(b)|=0.8.
```

But frequencies involving fewer than $`\delta(B)`$ columns have zero population Fourier coefficients.

An agent could examine every one-, two-, three-, four- and five-column Fourier interaction and discover nothing when $`\delta(B)=6`$.

Yet the correct six-column frequency could have a very strong signal.

That creates the characteristic contrast:

```math
\boxed{ \underbrace{\widehat P_X(a)=0 \text{ for all nonzero }a\text{ with }\|a\|_0<\delta(B)}_{\text{Low-order concealment}} }
```

```math
\boxed{ \underbrace{|\widehat P_X(B^\top m)|>0 \text{ for some }m}_{\text{Recoverable higher-order structure}} }
```

### 7.5 Effect of latent Gaussian noise

Wrapped Gaussian noise has an especially convenient Fourier transform.

If

```math
U=(Z+\varepsilon)\bmod1,\qquad \varepsilon\sim\mathcal N(0,\sigma^2I_k),
```

then

```math
\boxed{ \widehat\mu(m)= \widehat\mu_M(m) e^{-2\pi^2\sigma^2\|m\|_2^2}. }
```

Thus noise attenuates high frequencies, but does not introduce low-order dependence into the observed variables.

This provides an independent difficulty parameter.

Increasing $`\sigma`$ reduces the strength of detectable Fourier characters. Increasing $`\delta(B)`$ increases the number of coordinates that must participate before a signal can appear.

These are different forms of difficulty.

## 8. Observability and identifiability

The existence of a hidden matrix $`B`$ does not guarantee that it is recoverable from the observed distribution.

This distinction is important for a rigorous benchmark.

### 8.1 Observable frequencies

Define the nonzero observable Fourier frequencies:

```math
\mathcal S= \{a\in\mathbb Z^D\setminus\{0\}: \widehat P_X(a)\ne0\}.
```

We can then define the observable rational subspace:

```math
V_{\mathrm{obs}} = \operatorname{span}_{\mathbb Q}(\mathcal S).
```

By the Fourier-support theorem,

```math
V_{\mathrm{obs}} \subseteq \operatorname{row}_{\mathbb Q}(B).
```

If

```math
\boxed{ \dim V_{\mathrm{obs}}=k, }
```

then the observable Fourier frequencies span the planted rational row space.

If $`B`$ is saturated, the integer row lattice is identifiable at the population level as

```math
\Lambda_B=V_{\mathrm{obs}}\cap\mathbb Z^D.
```

The condition matters because some latent distributions may be uniform along one direction.

For instance, if $`k=2`$ but

```math
\mu=\nu\otimes U(\mathbb T),
```

then all nonzero Fourier coefficients involving the second latent coordinate vanish.

The observed distribution contains no information about that hidden coordinate direction.

It would be incorrect to require an agent to recover something that is not statistically identifiable.

### 8.2 Strong observable frequencies

Population identifiability alone is insufficient for finite datasets.

A Fourier coefficient could be nonzero but have magnitude $`10^{-8}`$, effectively impossible to distinguish from sampling noise using 5,000 observations.

Define

```math
\mathcal S_\rho= \{a\in\mathbb Z^D: |\widehat P_X(a)|\ge\rho\}.
```

A useful generator requirement is

```math
\boxed{ \operatorname{rank}_{\mathbb Q} \bigl(\operatorname{span}_{\mathbb Q}\mathcal S_\rho\bigr)=k. }
```

For our earlier tests, we used strong Fourier signals, commonly with the threshold

```math
\rho=0.3.
```

This ensures that at least $`k`$ independent latent directions are visible through sufficiently strong frequencies.

It does not by itself ensure an efficient search algorithm can find them.

### 8.3 The true effective concealment order

There is another useful quantity:

```math
\delta_{\mathrm{obs}}= \min_{a\in\mathcal S}\|a\|_0.
```

We have

```math
\delta_{\mathrm{obs}}\ge\delta(B).
```

The inequality may be strict because some Fourier coefficients of the latent geometry vanish.

Therefore, $`\delta(B)`$ gives a guaranteed concealment level, whereas $`\delta_{\mathrm{obs}}`$ describes the first support size at which the actual planted distribution has a detectable Fourier character.

For finite samples, we can further define a strong-signal version using $`\mathcal S_\rho`$.

## 9. How an agent can recover the hidden structure

The discovery task is effectively an inverse problem:

```math
X\longrightarrow \widehat B\longrightarrow\widehat M.
```

It contains two conceptually different challenges.

First, recover the hidden integer directions. Second, reconstruct the geometry in the resulting coordinates.

### 9.1 Estimating Fourier coefficients

Given samples $`x_1,\ldots,x_N`$, estimate

```math
\boxed{ \widehat\rho(a)= \left| \frac1N \sum_{n=1}^N e^{2\pi i a^\top x_n} \right|. }
```

Under the iid uniform null,

```math
\mathbb E[\widehat\rho(a)^2]=\frac1N \qquad(a\ne0).
```

The typical null magnitude is therefore of order

```math
O(N^{-1/2}).
```

For $`N=8{,}000`$,

```math
N^{-1/2}\approx0.0112.
```

An observed magnitude of 0.6 or 0.9 is enormous relative to this single-frequency noise scale.

However, when testing millions or billions of candidate frequencies, multiple testing matters.

If $`M`$ candidates are examined, the largest null fluctuation is generally on the order of

```math
\sqrt{\frac{\log M}{N}},
```

up to constants and dependencies between tests.

Thus a recovery system should use independent validation data or account for the number of searches conducted.

### 9.2 Sparse integer frequency search

Our previous datasets used integer coefficients such as

```math
a_j\in\{-2,-1,0,1,2\}.
```

The goal is to find vectors $`a`$ satisfying

```math
\|a\|_0=s
```

and maximizing $`\widehat\rho(a)`$.

An exhaustive search over exactly $`s`$ nonzero coordinates and $`2C`$ allowed nonzero coefficients contains

```math
\boxed{ M=\binom Ds(2C)^s }
```

candidate vectors before accounting for sign equivalence.

For example, with

```math
D=30,\quad s=6,\quad C=2,
```

this is approximately

```math
2.43\times10^9
```

frequencies.

So direct exhaustive evaluation quickly becomes expensive.

### 9.3 Meet-in-the-middle Fourier search

The key computational improvement used by our reference solvers is to divide each candidate into two parts:

```math
a=a_L+a_R.
```

Then

```math
e^{2\pi ia^\top x} = e^{2\pi ia_L^\top x} e^{2\pi ia_R^\top x}.
```

Define feature matrices whose columns contain the complex exponential features for candidate half-frequencies:

```math
L_{n,p}=e^{2\pi ia_{L,p}^\top x_n},
```

```math
R_{n,q}=e^{2\pi ia_{R,q}^\top x_n}.
```

Their product gives

```math
\boxed{ C=\frac1N L^\top R. }
```

Notice the ordinary transpose, rather than the conjugate transpose, because we want the sum $`a_L+a_R`$.

The entry $`C_{pq}`$ is the empirical Fourier coefficient associated with that combined frequency.

This allows us to evaluate very large groups of candidate interactions using optimized matrix multiplication rather than evaluating every full frequency independently.

The search can be performed over randomized column partitions, small screening subsets, and verification subsets.

This is a computational strategy, not a proof of polynomial-time recoverability.

Its cost still grows combinatorially with sparsity.

### 9.4 Recovering the lattice

Once significant frequencies are found, collect independent integer vectors

```math
a_1,\ldots,a_r.
```

If the observable frequencies span the hidden rational row space, they determine

```math
\operatorname{row}_{\mathbb Q}(B).
```

For saturated $`B`$, taking the integer saturation of the span recovers the primitive lattice.

However, **rank equality alone is not enough** to establish equality of integer lattices.

For example,

```math
\begin{pmatrix} 1&0\\ 0&1 \end{pmatrix}, \qquad \begin{pmatrix} 2&0\\ 0&1 \end{pmatrix}
```

span the same rational space but generate different integer lattices.

A proper exact verification therefore needs integer-lattice arithmetic, such as Hermite or Smith normal form.

### 9.5 Recovering the geometry

After finding a candidate basis $`\widehat B`$, transform every observation:

```math
\widehat u_n=\widehat Bx_n\bmod1.
```

The resulting point cloud may reveal:

- A branching graph.
- A smooth closed curve.
- Multiple disconnected components.
- A planar silhouette.
- A knot in three dimensions.

Geometry reconstruction can then use graph estimation, density ridges, principal curves, parametric fitting, or other methods appropriate to the observed structure.

This is deliberately separate from lattice recovery.

A solver that recovers the correct $`B`$ but fails to recognize a trefoil has solved the algebraic part but not the complete task.

## 10. Equivalent solutions and coordinate transformations

The planted matrix is not uniquely determined as a particular list of rows.

Suppose

```math
A\in GL_k(\mathbb Z),
```

meaning $`A`$ is an integer matrix with

```math
\det A=\pm1.
```

Define

```math
B'=AB.
```

Then

```math
B'x\bmod1=A(Bx)\bmod1.
```

Therefore,

```math
\boxed{ u'=Au\bmod1. }
```

Because $`A`$ has an integer inverse, this transformation is invertible on the torus.

So $`B`$ and $`B'`$ represent the same integer lattice in different coordinates.

An agent should receive full credit for either representation, provided it transforms the geometry consistently.

More generally, translations

```math
u'=Au+t\bmod1
```

also produce valid coordinate descriptions of the same latent structure.

### Why this matters for grading

If the planted geometry is a Y-shaped graph centered at

```math
(0.5,0.5),
```

but an agent negates one hidden direction, the geometry may appear reflected.

That is not a discovery failure.

The underlying shape and lattice are equivalent.

We should grade the equivalence class

```math
\boxed{ [(B,\mu)] = \{(AB,\,(u\mapsto Au+t)_*\mu): A\in GL_k(\mathbb Z),\ t\in\mathbb T^k\}, }
```

with the understanding that the translation $`t`$ is implemented by adding constants in the submitted map.

The true answer is an equivalence class of representations, not one particular matrix and coordinate orientation.

## 11. An optional stronger disguise: hiding the torus coordinates themselves

So far, the observations are explicitly represented as coordinates in $`[0,1)^D`$.

This means an agent familiar with modular Fourier searches has a strong methodological advantage.

We previously discussed an additional transformation that hides this representation.

### 11.1 Circle embedding

Replace each coordinate $`x_j`$ with its unit-circle representation:

```math
\phi_j(x_j)= \begin{pmatrix} \cos(2\pi x_j)\\ \sin(2\pi x_j) \end{pmatrix}.
```

Concatenate all coordinates:

```math
\phi(x)= \begin{pmatrix} \cos(2\pi x_1)\\ \sin(2\pi x_1)\\ \vdots\\ \cos(2\pi x_D)\\ \sin(2\pi x_D) \end{pmatrix} \in\mathbb R^{2D}.
```

Now apply a random orthogonal matrix

```math
R\in O(2D)
```

and optionally add measurement noise:

```math
\boxed{ Y=R\phi(X)+\eta. }
```

The released observations are now in $`\mathbb R^{2D}`$, not in the original torus-coordinate representation.

An investigator no longer sees the individual $`x_j`$ directly.

### 11.2 Why low-degree statistics remain concealed

Every sine and cosine term can be expressed using Fourier characters:

```math
\cos(2\pi x_j)= \frac{e^{2\pi ix_j}+e^{-2\pi ix_j}}2
```

and

```math
\sin(2\pi x_j)= \frac{e^{2\pi ix_j}-e^{-2\pi ix_j}}{2i}.
```

A polynomial of total degree $`r`$ in the circle features expands into a linear combination of Fourier characters whose frequency vectors have at most $`r`$ nonzero coordinates.

Therefore, if

```math
r<\delta(B),
```

every nonconstant character appearing in such an expansion has expectation zero, exactly as under an iid-uniform torus null.

Thus,

```math
\boxed{ \mathbb E_{\mathrm{planted}}[p(Y)] = \mathbb E_{\mathrm{null}}[p(Y)] }
```

for every polynomial $`p`$ of total degree less than $`\delta(B)`$, provided the planted and null observations undergo the same circle embedding, rotation, and independent measurement-noise distribution.

Orthogonal rotation does not change this statement because linear combinations of polynomial features preserve polynomial degree.

A sharper threshold is

```math
d_F= \min_{\substack{a\ne0\\\widehat P_X(a)\ne0}} \|a\|_1.
```

Polynomial moments of degree less than $`d_F`$ match the null.

For example, if $`d_F\ge7`$, all moments through degree six match exactly.

### 11.3 How could an agent recover the hidden coordinates?

Before rotation, the circle features satisfy

```math
\cos^2(2\pi x_j)+\sin^2(2\pi x_j)=1.
```

After rotation, these constraints become

```math
\boxed{ y^\top P_jy=1, }
```

where $`P_j`$ is a rank-two orthogonal projector onto the rotated plane associated with coordinate $`j`$.

In the noiseless case, these are exact quadratic relations. An investigator could try recovering the family of circle planes using quadratic vanishing constraints and simultaneous matrix diagonalization.

With measurement noise, they become approximate relations.

After recovering the planes, the investigator can reconstruct angular coordinates, up to shifts, reflections and permutations:

```math
\widehat x_j=\frac{\operatorname{atan2}(s_j,c_j)}{2\pi}\bmod1.
```

Then the original modular Fourier recovery problem becomes accessible again.

This gives a two-stage discovery task:

```math
\boxed{ Y \xrightarrow{\text{recover circle coordinates}} \widehat X \xrightarrow{\text{discover integer lattice}} \widehat B \xrightarrow{\text{reconstruct}} \widehat M. }
```

This is a promising more difficult benchmark variant.

However, the existence of the quadratic relations does not guarantee efficient or unique recovery with finite noisy data.

Also, the circle constraints occur in null datasets as well, so finding them is only recovery of the coordinate system, not detection of the planted geometry.

## 12. The mathematical foundation of grading

Our present grader measures how effectively a submitted transformation and drawing distinguish the planted distribution from independent noise.

Let the agent submit a map

```math
f:\mathbb T^D\rightarrow\mathbb R^r
```

and a geometric drawing

```math
\widehat M\subset\mathbb R^r.
```

The drawing is represented by points, line segments and triangles.

### 12.1 Normalize output coordinates

Let

```math
Q=f(X_{\mathrm{null}})
```

be the map evaluated on private null observations.

For each output dimension $`j`$, compute

```math
s_j=\operatorname{Std}(Q_j).
```

Define

```math
\widetilde f_j(x)=\frac{f_j(x)}{s_j}
```

and scale the drawing coordinates by the same factors.

The grader rejects maps with near-zero $`s_j`$.

This normalization makes the distance threshold less sensitive to arbitrary coordinate scaling.

### 12.2 Distance to a drawing

For a mapped observation $`z`$, define

```math
d(z,\widehat M) = \inf_{p\in\widehat M}\|z-p\|_2.
```

For a segment from $`a`$ to $`b`$, the projection parameter is

```math
t^*= \operatorname{clip}_{[0,1]} \left( \frac{(z-a)^\top(b-a)}{\|b-a\|_2^2} \right),
```

giving

```math
d(z,[a,b]) = \|z-a-t^*(b-a)\|_2.
```

For triangles, the distance is to the closest point on the filled triangle, including its boundary.

The final distance is the minimum over all drawing primitives.

### 12.3 Planted and null hit rates

For tolerance $`\epsilon`$, define

```math
h_P= \Pr_{X\sim P_X} \left[ d(\widetilde f(X),\widetilde M)\le\epsilon \right]
```

and

```math
h_0= \Pr_{X\sim P_0} \left[ d(\widetilde f(X),\widetilde M)\le\epsilon \right].
```

The detection contrast is

```math
\boxed{ S_{\mathrm{contrast}}=\max(0,h_P-h_0). }
```

A drawing that covers the entire output space will generally produce both $`h_P\approx1`$ and $`h_0\approx1`$, so it receives little or no credit.

### 12.4 Drawing precision

A detector could achieve high planted recall by submitting a huge or excessively complicated drawing.

To discourage this, measure the fraction of the drawing close to actual planted observations.

For samples $`q`$ distributed along the drawing, define

```math
p_M= \Pr_{q\sim\nu_{\widehat M}} \left[ \min_{z\in\widetilde f(X_{\mathrm{private}})} \|q-z\|_2\le\epsilon \right].
```

Our current score is

```math
\boxed{ S_{\mathrm{geometry}} = \max(0,h_P-h_0)\,p_M. }
```

This was the scoring function used for the Y-tree, trefoil and sword datasets.

It is an effective **geometric detection score**, but not a complete interpretation metric.

### 12.5 Separate lattice and topology metrics

For a research benchmark, I would introduce independent scores.

**Exact lattice recovery**

If the submitted map contains integer modular directions collected into $`\widehat B`$, evaluate whether

```math
\Lambda_{\widehat B}=\Lambda_B.
```

For two saturated rank-$`k`$ bases, this is equivalent to the existence of

```math
A\in GL_k(\mathbb Z)
```

such that

```math
\widehat B=AB.
```

For partial credit, one can measure rational subspace overlap:

```math
r_\cap= \operatorname{rank}(\widehat B)+ \operatorname{rank}(B)- \operatorname{rank} \begin{pmatrix} \widehat B\\B \end{pmatrix},
```

```math
r_\cup= \operatorname{rank} \begin{pmatrix} \widehat B\\B \end{pmatrix},
```

```math
S_{\mathrm{subspace}}=\frac{r_\cap}{r_\cup}.
```

But this must be distinguished from exact integer-lattice equality.

**Topology recovery**

For a finite embedded graph, useful invariants include:

```math
\beta_0=\text{number of connected components}
```

and

```math
\beta_1=|E|-|V|+\beta_0,
```

the number of independent graph cycles.

The degree sequence can also reveal branch points and endpoints.

For knots, a closed cycle alone is insufficient: both an unknot and a trefoil have one connected component and one graph cycle. Knot type requires additional geometric or topological analysis, potentially using knot invariants.

Even then, a noisy geometric sample may not uniquely determine topology. Coincident crossings or strands that pass extremely close to one another can make different structures observationally indistinguishable.

So the full benchmark should report something like

```math
\boxed{ (S_{\mathrm{lattice}}, S_{\mathrm{geometry}}, S_{\mathrm{topology}}, S_{\mathrm{null}}) }
```

rather than relying exclusively on one scalar.

## 13. Finite precision: why the mathematical guarantee can fail in code

The continuous construction assumes exact arithmetic on $`\mathbb T^D`$.

A computer produces finite-precision numbers.

In our sword experiment, the agent discovered that some columns were generated through floating-point modular calculations while others came directly from a random-number generator.

That exposed the hidden pivot columns.

The underlying mathematical construction was sound, but its numerical implementation leaked the answer.

### 13.1 Exact finite-grid construction

A more robust implementation uses a prime modulus $`p`$.

Generate integer coordinates

```math
z_j\in\{0,\ldots,p-1\}
```

and represent observations as

```math
x_j=\frac{z_j}{p}.
```

The hidden constraint becomes

```math
\boxed{ Bz=u\pmod p. }
```

All arithmetic is performed exactly in the finite field

```math
\mathbb F_p.
```

If the chosen pivot matrix $`B_P`$ is invertible modulo $`p`$,

```math
z_P= B_P^{-1}(u-B_Fz_F)\pmod p.
```

Only after completing the modular computation do we convert the integers to floating-point coordinates for storage.

Null datasets use the same finite grid.

This removes the specific distinction between coordinates produced by the random-number generator and coordinates produced by floating-point arithmetic.

### 13.2 What happens to the independence theorem?

There is a finite-field analogue of the concealment theorem.

For any coordinate subset $`I`$, if

```math
\operatorname{rank}_{\mathbb F_p}(B_{I^c})=k,
```

then the marginal distribution of $`Z_I`$, conditional on any latent $`u`$, is uniform on

```math
\mathbb F_p^{|I|}.
```

Therefore, if this condition holds for every $`|I|<\delta`$, those low-order marginals are exactly independent discrete uniforms.

One important subtlety is that

```math
\operatorname{rank}_{\mathbb Q}(B_{I^c})=k
```

does not always imply

```math
\operatorname{rank}_{\mathbb F_p}(B_{I^c})=k.
```

A minor can be nonzero over the rationals but vanish modulo $`p`$.

Therefore, the finite-grid generator should explicitly verify the required ranks over $`\mathbb F_p`$.

Choosing a large prime helps, but is not a substitute for validation.

The latent geometry must also be quantized consistently onto the same grid.

This addresses our known floating-point leak, not every possible numerical or procedural side channel.

## 14. What actually determines difficulty?

The important difficulty parameters are not simply $`N`$, $`D`$, and $`k`$.

I would distinguish the following:

| Parameter           | Mathematical role             | Effect                                              |
| ------------------- | ----------------------------- | --------------------------------------------------- |
| $`D`$             | Ambient dimension             | More possible hidden column combinations            |
| $`k`$             | Latent dimension              | More directions to discover                         |
| $`\delta(B)`$     | Minimum row-space support     | Higher guaranteed concealment order                 |
| $`\rho`$          | Fourier signal magnitude      | Smaller signals need more observations              |
| $`N`$             | Sample size                   | Controls statistical estimation error               |
| $`C`$             | Allowed coefficient magnitude | Expands integer-frequency search space              |
| $`\sigma`$        | Latent noise                  | Attenuates Fourier signals and blurs geometry       |
| Geometry complexity | Shape complexity              | Makes reconstruction and interpretation harder      |
| Embedding family    | Representation mechanism      | Determines what kind of inverse problem is required |

A particularly useful measure is the smallest support size needed to obtain a full-rank set of *strong* observable frequencies:

```math
\boxed{ s_\rho= \min\left\{ s: \operatorname{rank}_{\mathbb Q} \operatorname{span}_{\mathbb Q} \left\{ a: |\widehat P_X(a)|\ge\rho,\, \|a\|_0\le s \right\} =k \right\}. }
```

This is more informative about Fourier-search difficulty than $`\delta(B)`$ alone.

For example, $`\delta(B)=4`$ guarantees that all three-variable interactions are null, but it does not mean a spanning set of useful Fourier frequencies can necessarily be found using four-variable searches.

Also, a difficult-looking shape does not necessarily produce a difficult Fourier-recovery problem.

A highly detailed sword silhouette might produce stronger low-frequency Fourier signals than a simpler, more symmetrically distributed shape.

### Mathematical concealment versus computational hardness

These are distinct guarantees.

We can prove:

```math
P_{X_I}=U([0,1)^{|I|}) \qquad \text{for }|I|<\delta(B).
```

But this does **not** prove that recovering $`B`$ requires exponential time, or that no polynomial-time method exists.

A clever algorithm might exploit lattice reduction, numerical artifacts, high-order moments, or a previously unknown algebraic shortcut.

The benchmark demonstrates controlled concealment and experimentally measured recovery difficulty, not a general cryptographic hardness theorem.

There is also a fundamental upper limit to concealment: if the latent distribution $`\mu`$ is uniform on all of $`\mathbb T^k`$, then the resulting $`X`$ is exactly uniform on $`\mathbb T^D`$.

In that case, there is no observable planted structure at all, and $`B`$ is unidentifiable.

So the method balances two competing objectives:

```math
\text{Conceal low-order interactions}
```

while

```math
\text{Preserve enough higher-order information for recovery}.
```

## 15. A compact mathematical specification

The entire family can be summarized as follows.

Let

```math
\theta=(D,N,k,B,M,\mu_M,\sigma).
```

Assume $`B\in\mathbb Z^{k\times D}`$ has full row rank and is saturated.

Define the generative process

```math
\boxed{ \begin{aligned} Z_n&\sim\mu_M,\\ \varepsilon_n&\sim\mathcal N(0,\sigma^2I_k),\\ U_n&=(Z_n+\varepsilon_n)\bmod1,\\ X_n\mid U_n&\sim \operatorname{Haar} \bigl(\{x:Bx=U_n\pmod1\}\bigr). \end{aligned} }
```

Release only

```math
X=\{X_1,\ldots,X_N\}.
```

The resulting distribution satisfies three central properties.

**1. Concealment**

```math
\boxed{ X_I\sim U([0,1)^{|I|}) \quad\forall I:\ |I|<\delta(B). }
```

**2. Fourier localization**

```math
\boxed{ \widehat P_X(a)= \begin{cases} \widehat\mu(m),&a=B^\top m,\\ 0,&\text{otherwise}. \end{cases} }
```

**3. Exact latent recovery given the correct matrix**

```math
\boxed{ BX_n\bmod1=U_n. }
```

If the nonzero observable Fourier frequencies span a $`k`$-dimensional rational subspace, the saturated hidden row lattice is identifiable from the population distribution.

Given the recovered matrix, the latent geometry can be reconstructed statistically from the projected observations, subject to noise, sampling and geometric identifiability.

These are the mathematical foundations of the method.

## 16. What is proved, and what remains empirical?

| Claim                                                                      | Status                                                  |
| -------------------------------------------------------------------------- | ------------------------------------------------------- |
| Every subset smaller than $`\delta(B)`$ is independent uniform           | **Mathematically proved**                               |
| Fourier support lies in the hidden integer row lattice                     | **Mathematically proved**                               |
| Gaussian noise attenuates Fourier coefficients by a known factor           | **Mathematically proved**                               |
| Correct $`B`$ recovers the latent coordinates                            | **Exact by construction**                               |
| Full-rank observable Fourier spectrum identifies the saturated row lattice | **Mathematically established under stated assumptions** |
| A particular finite dataset passes concealment tests                       | Empirical                                               |
| A particular non-oracle solver recovers $`B`$                            | Empirical                                               |
| The agent discovers the correct geometry or topology                       | Empirical                                               |
| The computational problem is inherently hard                               | **Not proved**                                          |
| The generator has no implementation leaks                                  | Requires ongoing adversarial validation                 |

### Final perspective

The distinctive feature of our construction is that it creates a **controlled separation between marginal statistical simplicity and joint structural complexity**.

Instead of merely hiding a pattern behind many irrelevant dimensions, the distribution is constructed so that every low-order view is genuinely uniform, while a small collection of higher-order modular relationships encodes the geometry.

That makes it useful for testing whether an agent can move beyond ordinary exploratory statistics, discover non-obvious mathematical transformations, recover an interpretable latent representation, and explain the structure it reveals.

For a formal paper, I would organize the mathematical contribution around the **fiber-sampling construction, low-order concealment theorem, Fourier-support theorem, identifiability conditions, and representation-invariant evaluation**. The computational recovery algorithms and procedural shape families would form the experimental methodology.