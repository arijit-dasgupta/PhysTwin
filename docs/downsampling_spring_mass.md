# Spring-Mass Coarsening for the Warp Kernel Implementation

This note writes down the coarsening math concretely for the current Warp spring-mass code. The goal is to produce a **new spring-mass model of the same form** as the original one, but with fewer object nodes and fewer springs, so that the same kernel structure can still be used.

The two coarsening strategies covered here are:

1. **K-means coarsening**
2. **Local graph coarsening**

In both cases, the final output is a new coarse spring-mass system with:

- coarse object vertices
- coarse masses
- coarse spring endpoints
- coarse spring rest lengths
- coarse spring log-stiffness values
- the same control-point mechanism
- the same force, damping, gravity, and collision update pattern

The most important code-specific fact is that the Warp implementation already expects a combined spring graph whose endpoints can refer either to:

- object points, or
- control points appended after the object points

and that spring forces are only accumulated into object nodes, never into control nodes. So the coarse model should preserve exactly that convention.

---

## 1. Fine model being coarsened

Let:

- $N$ be the number of object points
- $C$ be the number of control points
- $E$ be the number of springs

The spring endpoint list lives on the combined index set
$$
\{1,\dots,N+C\}.
$$

The object state is:

$$
x_i(t) \in \mathbb{R}^3,\qquad v_i(t) \in \mathbb{R}^3,\qquad i=1,\dots,N
$$

and the control trajectories are:

$$
u_c(t) \in \mathbb{R}^3,\qquad \dot u_c(t) \in \mathbb{R}^3,\qquad c=1,\dots,C.
$$

For each spring $e\in\{1,\dots,E\}$, define:

- endpoints:
  $$
  (a_e,b_e)\in\{1,\dots,N+C\}^2
  $$
- rest length:
  $$
  \ell_e^0 > 0
  $$
- log-stiffness parameter:
  $$
  Y_e
  $$

The code uses the actual stiffness
$$
k_e = \mathrm{clamp}(\exp(Y_e),\,Y_{\min},\,Y_{\max})
$$
and skips the spring if $\exp(Y_e)\le Y_{\min}$.

So it is useful to define the active effective stiffness
$$
\tilde{k}_e
=
\mathbf{1}[\exp(Y_e)>Y_{\min}]
\cdot
\mathrm{clamp}(\exp(Y_e),\,Y_{\min},\,Y_{\max}).
$$

### Combined endpoint lookup

Define the combined endpoint position and velocity functions:

$$
z_p(t)=
\begin{cases}
x_p(t), & p\le N \\
u_{p-N}(t), & p>N
\end{cases}
$$

$$
w_p(t)=
\begin{cases}
v_p(t), & p\le N \\
\dot u_{p-N}(t), & p>N.
\end{cases}
$$

For spring $e=(a_e,b_e)$, define

$$
d_e = z_{b_e}-z_{a_e},\qquad
r_e = \|d_e\|,\qquad
\hat d_e = \frac{d_e}{\max(r_e,\varepsilon)}.
$$

### Spring force law

The code computes

$$
f_e^{\text{spring}}
=
\tilde{k}_e\left(\frac{r_e}{\ell_e^0}-1\right)\hat d_e.
$$

The dashpot term is

$$
f_e^{\text{dashpot}}
=
\gamma_{\text{dash}}
\big((w_{b_e}-w_{a_e})\cdot \hat d_e\big)\hat d_e.
$$

So the total spring-edge force is

$$
f_e = f_e^{\text{spring}} + f_e^{\text{dashpot}}.
$$

This force is added to the first object endpoint and subtracted from the second object endpoint. If an endpoint is a control point, no force is accumulated there.

### Per-point update

For object point $i$, with mass $m_i$, the total force is

$$
F_i^{\text{all}} = F_i^{\text{spr}} + m_i g,
$$

with gravity
$$
g = (0,0,-9.8)\cdot \text{reverse\_factor}.
$$

Then the code applies

$$
a_i = \frac{F_i^{\text{all}}}{m_i},
\qquad
v_i^{(1)} = v_i + a_i\,\Delta t,
\qquad
v_i^{(2)} = v_i^{(1)}\exp(-\Delta t\,\gamma_{\text{drag}}).
$$

Then object-collision and ground-collision logic modify the velocity and update positions.

So the fine model is fully specified by:

- object rest positions
- object masses
- control trajectories
- spring endpoints
- spring rest lengths
- spring $Y$-values
- global dashpot damping
- global drag damping
- collision parameters and masks

The coarse model must be redefined in exactly the same format.

---

## 2. Generic coarsening template

Both K-means and local graph coarsening produce a partition map

$$
\pi:\{1,\dots,N\}\to\{1,\dots,K\},
$$

where $K\ll N$ is the number of coarse object nodes.

Interpretation:

- fine object node $i$ belongs to coarse cluster $\pi(i)$

For coarse node $a$, define the cluster

$$
C_a = \{i\in\{1,\dots,N\}:\pi(i)=a\}.
$$

Control points are kept unchanged in the first version.

So the coarse system has:

- $K$ coarse object nodes
- the same $C$ control nodes
- a combined coarse index set $\{1,\dots,K+C\}$

### 2.1 Coarse masses

Mass lumping:

$$
M_a = \sum_{i\in C_a} m_i.
$$

### 2.2 Coarse rest positions

Mass-weighted centroid:

$$
X_a^0 = \frac{1}{M_a}\sum_{i\in C_a} m_i x_i^0.
$$

### 2.3 Coarse initial state

If needed, initialize coarse positions and velocities from fine initial conditions by:

$$
X_a(0)=\frac{1}{M_a}\sum_{i\in C_a} m_i x_i(0),
\qquad
V_a(0)=\frac{1}{M_a}\sum_{i\in C_a} m_i v_i(0).
$$

---

## 3. Rebuilding the coarse spring graph

Once $\pi$ is fixed, the coarse spring graph is built by grouping fine springs into coarse spring buckets.

### 3.1 Object-object fine springs

If a fine spring $e$ connects two object nodes $i,j\le N$, define

$$
\alpha=\pi(i),\qquad \beta=\pi(j).
$$

If $\alpha\neq\beta$, then this fine spring contributes to coarse object-object edge $(\alpha,\beta)$.

Define the bucket

$$
\mathcal{E}_{\alpha\beta}^{oo}
=
\{e=(i,j)\in\mathcal{E}: i,j\le N,\ \pi(i)=\alpha,\ \pi(j)=\beta,\ \alpha\neq\beta\}.
$$

### 3.2 Object-control fine springs

If a fine spring connects fine object node $i\le N$ to control point $c$, and $\pi(i)=\alpha$, then it contributes to coarse object-control edge $(\alpha,c)$.

Define

$$
\mathcal{E}_{\alpha c}^{oc}
=
\{e\in\mathcal{E}: e \text{ connects fine object } i \text{ to control } c,\ \pi(i)=\alpha\}.
$$

### 3.3 Coarse spring endpoint indexing for Warp

The Warp code expects control points to be indexed after object points.

So in the coarse model:

- coarse object nodes use indices $1,\dots,K$
- control points use indices $K+1,\dots,K+C$

If using 0-based code arrays, this becomes:

- object indices: `0, ..., K-1`
- control indices: `K, ..., K+C-1`

This is important because the same spring kernel can then be reused directly.

---

## 4. Coarse spring parameters

For every nonempty coarse spring bucket, define a coarse spring.

### 4.1 Rest length choices

#### Baseline: centroid-distance rest length

For object-object coarse spring $(\alpha,\beta)$,

$$
L_{\alpha\beta}^0 = \|X_\alpha^0 - X_\beta^0\|.
$$

For object-control coarse spring $(\alpha,c)$, using the initial control position $u_c^0$,

$$
L_{\alpha c}^0 = \|X_\alpha^0 - u_c^0\|.
$$

This is the cleanest first choice.

#### Variation A: arithmetic mean of fine rest lengths

$$
L_q^0 = \frac{1}{|\mathcal{B}_q|}\sum_{e\in\mathcal{B}_q}\ell_e^0
$$

where $\mathcal{B}_q$ is the fine spring bucket corresponding to coarse spring $q$.

#### Variation B: stiffness-weighted mean

$$
L_q^0 =
\frac{\sum_{e\in\mathcal{B}_q}\tilde{k}_e \ell_e^0}
{\sum_{e\in\mathcal{B}_q}\tilde{k}_e}.
$$

For the first implementation, centroid-distance is the best baseline.

---

### 4.2 Coarse stiffness choices

The Warp code stores spring values as `spring_Y`, but the kernel actually uses $\exp(Y)$ then clamps it. So the natural thing is:

1. define a coarse effective stiffness $K_q$
2. convert back to
   $$
   Y_q^{\text{coarse}} = \log K_q
   $$

#### Baseline: summed stiffness

$$
K_q = \sum_{e\in\mathcal{B}_q}\tilde{k}_e.
$$

Then

$$
Y_q^{\text{coarse}} = \log K_q.
$$

This is the best first choice because the coarse spring is standing in for multiple fine cross-cluster interactions.

#### Variation A: average stiffness

$$
K_q = \frac{1}{|\mathcal{B}_q|}\sum_{e\in\mathcal{B}_q}\tilde{k}_e.
$$

This often makes the coarse model too soft.

#### Variation B: scaled sum

$$
K_q = \lambda_k \sum_{e\in\mathcal{B}_q}\tilde{k}_e.
$$

This is a useful one-parameter family if you later want to tune stiffness globally.

#### Variation C: mean-count interpolation

$$
K_q = \lambda_k |\mathcal{B}_q|^\rho \cdot \mathrm{mean}_{e\in\mathcal{B}_q}\tilde{k}_e,
\qquad 0\le \rho \le 1.
$$

This interpolates between average and sum.

For the first implementation, use summed stiffness.

---

### 4.3 Coarse damping and drag

The current Warp code uses:

- one global dashpot damping scalar
- one global drag damping scalar

not per-edge damping arrays.

So the simplest coarse model is:

$$
\gamma_{\text{dash}}^{\text{coarse}} = \gamma_{\text{dash}}^{\text{fine}},
\qquad
\gamma_{\text{drag}}^{\text{coarse}} = \gamma_{\text{drag}}^{\text{fine}}.
$$

If needed later, introduce global retuning scalars:

$$
\gamma_{\text{dash}}^{\text{coarse}} = \lambda_d \gamma_{\text{dash}}^{\text{fine}},
\qquad
\gamma_{\text{drag}}^{\text{coarse}} = \lambda_{\text{drag}} \gamma_{\text{drag}}^{\text{fine}}.
$$

---

### 4.4 Coarse collision masks

If object-collision masks exist on fine nodes, define coarse mask by majority vote:

$$
\mathrm{mask}_a = \mathrm{mode}\{\mathrm{mask}_i : i\in C_a\}.
$$

A safer variant is to forbid merges that mix incompatible masks.

For a first pass, majority vote is acceptable, but protected-mask merging is more mechanically sensible.

---

## 5. K-means coarsening

Here the partition $\pi$ is produced by clustering rest-pose node positions.

Let the target coarse count be

$$
K = \left\lceil \frac{N}{r} \right\rceil
$$

for downsampling factor $r$.

### 5.1 Plain K-means

Solve

$$
\min_{\pi,\{\mu_a\}_{a=1}^K}
\sum_{i=1}^N \|x_i^0 - \mu_{\pi(i)}\|^2.
$$

### 5.2 Mass-weighted K-means

A slightly better choice for this spring-mass setting is

$$
\min_{\pi,\{\mu_a\}_{a=1}^K}
\sum_{i=1}^N m_i\,\|x_i^0 - \mu_{\pi(i)}\|^2.
$$

This makes heavier nodes matter more in defining the partition.

### 5.3 Protected-node K-means

If some nodes should remain untouched, such as control-adjacent nodes or collision-critical nodes, define a protected set $\mathcal{P}$ and keep each protected node as a singleton cluster. Then run K-means only on the remaining nodes.

### 5.4 K-means coarse model

Once $\pi_{\text{kmeans}}$ is obtained:

1. define clusters $C_a$
2. compute $M_a$ and $X_a^0$
3. build coarse spring buckets
4. define coarse rest lengths
5. define coarse $K_q$
6. convert to coarse $Y_q^{\text{coarse}}$
7. reuse the same Warp simulator structure

So K-means changes only **how the partition is obtained**. Everything after that follows the same generic reconstruction pipeline.

---

## 6. Local graph coarsening

Here the partition is produced by repeated **local graph contractions**.

Let the fine object-object spring graph be

$$
G=(V,E_{oo}),
\qquad
V=\{1,\dots,N\}
$$

where $E_{oo}$ contains only fine springs whose endpoints are both object nodes.

### 6.1 Admissible merges

The simplest rule is:

$$
(i,j)\text{ is admissible} \iff (i,j)\in E_{oo}.
$$

So only spring-adjacent object nodes may be merged.

You can also add constraints:

- do not merge protected nodes
- do not merge nodes with incompatible masks
- only merge if rest-pose distance is below a threshold

### 6.2 Merge cost choices

#### Baseline geometric cost

$$
\Delta(i,j)=\|x_i^0-x_j^0\|^2.
$$

This is the cleanest first local graph coarsener.

#### Variation A: geometry + stiffness mismatch

$$
\Delta(i,j)
=
\alpha_x \|x_i^0-x_j^0\|^2
+
\alpha_k
\left|
\sum_{e\ni i}\tilde{k}_e
-
\sum_{e\ni j}\tilde{k}_e
\right|^2.
$$

#### Variation B: geometry + benchmark-response similarity

If you later compute benchmark trajectory descriptors $\bar x_i$, then

$$
\Delta(i,j)
=
\alpha_x \|x_i^0-x_j^0\|^2
+
\alpha_m \|\bar x_i-\bar x_j\|^2.
$$

#### Variation C: geometry + role penalty

$$
\Delta(i,j)
=
\alpha_x \|x_i^0-x_j^0\|^2
+
\alpha_b \mathbf{1}[\text{role mismatch}].
$$

For the first implementation, use the geometric cost only.

### 6.3 Greedy contraction

Initialize each fine node as its own supernode.

At each stage, choose the admissible pair with smallest merge cost:

$$
(i_t,j_t)
=
\arg\min_{(i,j)\in\mathcal{A}_t}\Delta_t(i,j).
$$

Merge them.

Continue until the number of active supernodes is

$$
K = \left\lceil \frac{N}{r} \right\rceil.
$$

The final supernodes define the partition $\pi_{\text{graph}}$.

### 6.4 Batch-matching variant

Instead of one merge at a time, choose a set of disjoint low-cost admissible pairs and merge them simultaneously. This is often faster offline, but less exact than fully greedy.

### 6.5 Supernode update during contraction

If supernodes $A$ and $B$ merge into $S=A\cup B$, then

$$
M_S = M_A + M_B
$$

and

$$
X_S^0
=
\frac{M_A X_A^0 + M_B X_B^0}{M_S}.
$$

This is equivalent to the mass-weighted centroid of all fine nodes inside the new supernode.

### 6.6 Graph-coarsened spring-mass model

Once the final partition $\pi_{\text{graph}}$ is obtained, the coarse model is rebuilt in exactly the same way as before:

- bucket fine springs by coarse endpoint pair
- define coarse rest lengths
- define coarse stiffnesses
- define coarse `spring_Y`
- define coarse masses and initial states
- reuse the same Warp kernels

So, again, the difference from K-means lies only in **how the partition is generated**.

---

## 7. Exact redefinition of the coarse Warp-compatible model

Once any partition $\pi$ is fixed, the new coarse model is:

### 7.1 Coarse object rest vertices

$$
X_a^0 = \frac{1}{M_a}\sum_{i:\pi(i)=a} m_i x_i^0,
\qquad a=1,\dots,K.
$$

### 7.2 Coarse masses

$$
M_a = \sum_{i:\pi(i)=a} m_i.
$$

### 7.3 Coarse spring list

Each coarse spring $q$ corresponds to either:

- an object-object pair $(\alpha,\beta)$, or
- an object-control pair $(\alpha,c)$

with endpoints stored in the same combined indexing convention expected by the Warp kernel.

### 7.4 Coarse rest lengths

Use one of the formulas above, with centroid-distance as the baseline.

### 7.5 Coarse `spring_Y`

For each coarse spring $q$,

$$
K_q = \sum_{e\in\mathcal{B}_q}\tilde{k}_e,
\qquad
Y_q^{\text{coarse}} = \log K_q.
$$

### 7.6 Coarse drag and dashpot

Initially unchanged.

### 7.7 Coarse collisions

Use coarse masks and the same collision logic, but now on $K$ object nodes instead of $N$.

---

## 8. Coarse force law in exactly the same form

The reason this approach fits the Warp implementation well is that the coarse system still uses the same kernel-level formulas.

For each coarse spring $q=(p_q^1,p_q^2)$, define endpoint states

$$
\tilde z_p,\qquad \tilde w_p
$$

over the combined set of coarse object nodes and unchanged control nodes.

Then

$$
\tilde d_q = \tilde z_{p_q^2}-\tilde z_{p_q^1},
\qquad
\tilde r_q=\|\tilde d_q\|,
\qquad
\hat{\tilde d}_q=\frac{\tilde d_q}{\max(\tilde r_q,\varepsilon)}.
$$

The active effective stiffness is

$$
\tilde K_q
=
\mathbf{1}[\exp(Y_q^{\text{coarse}})>Y_{\min}]
\cdot
\mathrm{clamp}(\exp(Y_q^{\text{coarse}}),Y_{\min},Y_{\max}).
$$

The spring term is

$$
\tilde f_q^{\text{spring}}
=
\tilde K_q
\left(\frac{\tilde r_q}{L_q^0}-1\right)\hat{\tilde d}_q.
$$

The dashpot term is

$$
\tilde f_q^{\text{dashpot}}
=
\gamma_{\text{dash}}^{\text{coarse}}
\big((\tilde w_{p_q^2}-\tilde w_{p_q^1})\cdot \hat{\tilde d}_q\big)\hat{\tilde d}_q.
$$

And total force is

$$
\tilde f_q = \tilde f_q^{\text{spring}} + \tilde f_q^{\text{dashpot}}.
$$

So no new simulator type is introduced. Only the arrays are replaced.

---

## 9. Recommended first exact instantiations

### 9.1 K-means baseline

1. Choose
   $$
   K=\left\lceil \frac{N}{r}\right\rceil
   $$
2. Run **mass-weighted K-means** on rest positions
3. Compute $M_a$ and $X_a^0$
4. Keep control points unchanged
5. Build coarse spring buckets
6. Set
   $$
   L_q^0 = \text{centroid distance}
   $$
7. Set
   $$
   K_q = \sum_{e\in\mathcal{B}_q}\tilde{k}_e
   $$
8. Set
   $$
   Y_q^{\text{coarse}}=\log K_q
   $$
9. Keep dashpot and drag unchanged
10. Reuse the same Warp kernels

### 9.2 Local graph coarsening baseline

1. Choose
   $$
   K=\left\lceil \frac{N}{r}\right\rceil
   $$
2. Use only object-object spring adjacency as admissible merge edges
3. Use merge cost
   $$
   \Delta(i,j)=\|x_i^0-x_j^0\|^2
   $$
4. Greedily contract until $K$ supernodes remain
5. Convert the resulting supernodes into partition $\pi_{\text{graph}}$
6. Rebuild the coarse spring-mass model exactly as above

This is a good comparison because both methods use the same reconstruction formulas after the partition step.

---

## 10. Main variations worth keeping in mind

### Partition construction
- plain K-means
- mass-weighted K-means
- protected-node K-means
- greedy local graph coarsening
- batched local graph coarsening

### Coarse rest lengths
- centroid-distance
- arithmetic mean of fine rest lengths
- stiffness-weighted mean of fine rest lengths

### Coarse stiffness
- sum
- scaled sum
- average
- interpolated mean-count family

### Collision handling
- majority-vote mask
- forbid incompatible merges
- protect collision-critical nodes from merging

---

## 11. Practical code-specific notes

Because of the Warp code structure, the most natural coarse representation is still a sparse spring graph. This matters for speed.

The current implementation has:

- one thread per spring for `eval_springs`
- atomics into force buffers
- one thread per point for integration and damping
- object collision logic that is still point-based

So reducing both:

- number of object nodes
- number of springs

is what gives speedup.

That is why the coarse model should remain a spring-mass graph, not a dense reduced-order basis model.

Also, because `spring_Y` is exponentiated and clamped inside the kernel, it is cleaner to think in terms of effective stiffness first and then convert back to a coarse $Y$-value.

---

## 12. Compact summary

The full coarse pipeline is:

### Step A: choose a partition
$$
\pi =
\begin{cases}
\pi_{\text{kmeans}} & \text{for K-means} \\
\pi_{\text{graph}} & \text{for local graph coarsening}
\end{cases}
$$

### Step B: lump masses and positions
$$
M_a = \sum_{i:\pi(i)=a} m_i,
\qquad
X_a^0 = \frac{1}{M_a}\sum_{i:\pi(i)=a} m_i x_i^0
$$

### Step C: induce coarse spring buckets
Group fine springs by their coarse endpoint pair.

### Step D: define coarse spring parameters
$$
L_q^0 = \text{centroid-distance baseline}
$$
$$
K_q = \sum_{e\in\mathcal{B}_q}\tilde{k}_e
$$
$$
Y_q^{\text{coarse}} = \log K_q
$$

### Step E: run the same simulator type
Same spring force law, same drag, same gravity, same collision structure, but on the coarse arrays.

---

## 13. What to use first

If the goal is a clean first benchmark:

- start with **mass-weighted K-means**
- then try **local graph coarsening with geometric merge cost**

and keep everything else identical:

- same rest-length rule
- same stiffness aggregation rule
- same damping
- same controls
- same evaluation metric

That way, the comparison isolates the effect of **partition strategy** rather than mixing several design changes at once.