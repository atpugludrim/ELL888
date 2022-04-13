 \

: These notes have not been subjected to the usual scrutiny reserved for
formal publications. They may be distributed outside this class only
with the permission of the Course Coordinator.

Introduction
============

Support Vector Machines (SVMs) are older than the state of Haryana (the
linear variant at least), and the kernel version of SVMs were proposed
by Vapnik *et al.* in 1992. Vapnik and Chernvonenkis along with others
also developed theory key thoeretical concepts that constitute
statistical learning theory. In what follows, we will discuss the
theoretical and practical advancements since then that led to the work
titled "Kernel optimization using conformal maps for the minimal
complexity machine" by Jayadeva *et al.*

Since this is the congruence of two paths, one leading to MCMs and the
other leading to kernel optimization in a data dependent way, in the
following we first discuss MCMs, and then kernel optimization, and
finally how the two fit together.

Background
==========

The key concepts that lead to MCMs were available in statistical
learning theory long ago, but weren't applied until 2015. So let's
discuss the key concepts that naturally lead to minimal complexity
machines.

Statistical Learning Theory
---------------------------

It all starts by stating the objective of Machine Learning. The model of
learning from examples can be descibed using three components [@slt]:

1.  a generator of random vectors $x$, drawn independently from a fixed
    but unknown distribution $P(x)$;

2.  a supervisor that returns an output vector $y$ for every input
    vector $x$, according to a conditional distribution function
    $P(y\;\lvert\;x)$, also fixed but unknown;

3.  a learning machine capable of implementing a set of functions
    $f(x,\alpha), \alpha\in\Lambda$.

The problem of learning then is to choose the "right" function from the
family of functions $f(x,\alpha), \alpha\in\Lambda$. We want to choose
this function so that it predicts the supervisor's response on seen as
well as previously unseen data in the best possible way. The function
choice has to be made based on a set of pairs $\{(x_i,y_i)\}_{i=1}^M$
called the training set. These samples are drawn from the distribution
$P(x)P(y\;\lvert\;x)$ which is the joint distribution $P(x,y)$. The $M$
samples are assumed to be *independent and identically distributed
(i.i.d.)*.

Next we need to define what "predicting the response in the best
possible way means". For this we need to define how correct is
$f(x,\alpha)$ is compared to $y$. This is done by defining an error
function (also called loss function or cost function) that assigns some
real valued number to two objects (may be vectors, matrices, scalars, or
general mathematical objects). This number represents the "cost"
incurred by predicting $f(x,\alpha)$ given $y$.

The next important quantity is the *risk functional* which is the
expectation of loss. Since, $x$ is a random vector, so the output of
$f(\cdot,\alpha)$ is also random, so is the loss $\mathcal{L}$. Thus, it
makes sense to take the expectation of it.
$$R(\alpha)=\mathbb{E}_{(x,y)\sim
    P(x,y)}\biggl[\mathcal{L}\bigl(y,f(x,\alpha)\bigr)\biggr]=\int
    \mathcal{L}\bigl(y,f(x,\alpha)\bigr)dP(x,y)$$ The goal is then to
find $\alpha^*$ that minimizes the risk functional $R(\alpha)$. The
problem is that $P(x,y)$ is unknown, all that's known is the training
set $\{(x_i,y_i)\}_{i=1}^M$.

What we do have then is:
$$R_{\text{emp}}(\alpha)=\frac{1}{M}\sum_{i=1}^M
    \mathcal{L}\bigl(y,f(x,\alpha)\bigr)$$ $R_{\text{emp}}$ is called
the *empirical risk*. Given these quantities, we have the following
result from statistical learning theory: (Vapnik) If
$0\le\mathcal{L}\bigl(y,f(x,\alpha)\bigr)\le
B,\alpha\in\Lambda$, that is the loss is totally bounded, then with
probability at least $1-\eta$ the inequality $$\label{eqn:pac}
    R(\alpha)\le
    R_{\text{emp}}(\alpha)+\frac{B\varepsilon}{2}\Biggl(1+\sqrt{1+\frac{4R_{\text{emp}(\alpha)}}{B\varepsilon}}\Biggr)$$
holds true simultaneously for all functions of the set
$\mathcal{L}\bigl(y,f(x,\alpha)\bigr)$. where $$\label{eq:vareps}
    \varepsilon = 4\frac{h\biggl(\ln\frac{2M}{h}+1\biggr)-\ln\eta}{M}$$
and $h$ is the VC-dimension of the set of functions.

On the RHS of equation [\[eqn:pac\]](#eqn:pac){reference-type="ref"
reference="eqn:pac"}, the term other than the empirical risk
$R_{\text{emp}}(\alpha)$ is called the *structural risk*.

Another important result from Vapnik that will be needed to continue our
discussion of MCMs is: (Vapnik) Let vectors $x\in X$ belong to a sphere
of radius $R$. Then the set of $\Delta$-margin separating hyperplanes
has the VC-dimension $h$ bounded by the inequality $$\label{eqn:vcbound}
    h\le\min\left(\left[\frac{R}{\Delta}\right]^2,n\right)+1.$$ where
$n$ is the feature dimension of the data. For more, refer [@slt], but
for our discussion this is sufficient.

Minimal Complexity Machines
---------------------------

### Motivation

We want to minimize the actual **risk**, $R(\alpha)$, in order to
generalize prediction on future samples. We can do this by tightening
the bound on the RHS of
equation [\[eqn:pac\]](#eqn:pac){reference-type="ref"
reference="eqn:pac"}. We can optimize the empirical risk, and we can
optimize the structural risk. Note that in structural risk, the only
thing we can alter is $\varepsilon$. We need to minimize $\varepsilon$
to tighten the bound on the risk. And in the expansion of $\varepsilon$
we can only alter $h$ assuming the training sample size is as large as
we can get. $\varepsilon\propto h$. Thus to reduce $\varepsilon$, we
want to minimize $h$.

Now let's look at figure [1](#fig:1){reference-type="ref"
reference="fig:1"}. Imagine we have the data points in $\mathbb{R}^1$.
The blue dot shows the optimal hyperplane that classifies the points
with maximum margin. $R$ is the radius of the hypersphere and $\Delta$
is the margin. According to
equation [\[eqn:vcbound\]](#eqn:vcbound){reference-type="ref"
reference="eqn:vcbound"}, we'd like to find the hyperplane such that $R$
is minimized while $\Delta$ is maximized simultaneously.

![[\[fig:1\]]{#fig:1 label="fig:1"}Data points
$\in\mathbb{R}^1$](/home/mridul/Desktop/iitd_rishi_laptop/backups/ELL888/margin_func.png){#fig:1
width=".7\\linewidth"}

![[\[fig:2\]]{#fig:2 label="fig:2"}The red arrows are operations on
margin and blue arrows on
radius](/home/mridul/Desktop/iitd_rishi_laptop/backups/ELL888/margin_func2.png){#fig:2
width=".7\\linewidth"}

### MCM formulations

The mathematical formulations of MCM [@MCM] starts by defining a
mathematical quantity $h_{MCM}$ as
$$h_{MCM}=\frac{\max_{i=1,2,\dotsc,M}\lVert u^Tx_i+v\rVert}{\min_{i=1,2,\dotsc,M}\lVert u^Tx_i+v\rVert}$$
where it is assumed that the separating hyperplane is $u^Tx+v=0$. It is
assumed that the data is linearly separable.

Augmenting the data vectors to have a feature whose value is always 1,
and concatenating the weight vector, we have $\hat{x}_i\gets\{x_i;1\}$
and $\hat{u}\gets \{u;v\}$. Then the hyperplane goes through the origin
in the $\mathbb{R}^{(n+1)}$ space.

Then, the margin $\Delta$ is given by
$$\Delta=\min_{i=1,2,\dotsc,M}\frac{\lVert\hat{u}^T\hat{x}_i\rVert}{\lVert\hat{u}\rVert}$$
and the radius is just $R=\max_{i=1,2,\dotsc,M}\lVert\hat{x}_i\rVert$.
So, the ration of interest $\dfrac{R}{\Delta}$ is given by:
$$\frac{R}{\Delta}=\frac{\max_{i=1,2,\dotsc,M}\lVert\hat{x}_i\rVert}{\min_{i=1,2,\dotsc,M}\frac{\lVert\hat{u}^T\hat{x}_i\rVert}{\lVert\hat{u}\rVert}}=\frac{\max_{i=1,2,\dotsc,M}\lVert\hat{u}\rVert\lVert\hat{x}_i\rVert}{\min_{i=1,2,\dotsc,M}\lVert\hat{u}^T\hat{x}_i\rVert}$$
And using the Cauchy-Schwarz inequality ($\lVert a^Tb\rVert\le\lVert
a\rVert\lvert b\rVert$)
$$\frac{R}{\Delta}\ge\frac{\max_{i=1,2,\dotsc,M}\lVert\hat{u}^T\hat{x}_i\rVert}{\min_{i=1,2,\dotsc,M}\lVert\hat{u}^T\hat{x}_i\rVert}=\frac{\max_{i=1,2,\dotsc,M}\lVert u^Tx_i+v\rVert}{\min_{i=1,2,\dotsc,M}\lVert
    u^Tx_i+v\rVert}$$ Thus $$\begin{aligned}
    h_{MCM}&\le\frac{R}{\Delta}\\
    \Rightarrow
    h_{MCM}^2&\le\left(\frac{R}{\Delta}\right)^2<1+\left(\frac{R}{\Delta}\right)^2\end{aligned}$$
From equation [\[vcbound\]](#vcbound){reference-type="ref"
reference="vcbound"} we have for large dimensional data:
$$h\le 1+\left(\frac{R}{\Delta}\right)^2$$ Thus
$\exists\beta\in\mathbb{R}^+,$ such that $h\le\beta h_{MCM}^2$. Also
since, $h_{MCM}^2\ge 1$ and VC-dimension satisfies $h\ge 1$,
$\exists\alpha\in\mathbb{R},\alpha>0$ such that $\alpha h_{MCM}^2\le h$.
Combining the two we have
$\exists \alpha,\beta > 0, \alpha,\beta\in\mathbb{R}$ such that
$$\alpha h_{MCM}^2\le h\le\beta h_{MCM}^2$$ That is $h^2_{MCM}$ is an
exact bound on the VC dimension $h$. And since the data is linearly
separable, $u^Tx_i+v\ge 0$ if $y_i=1$ and $u^Tx_i+v\le 0$ if $y_i=-1$.
Thus $\lVert u^Tx_i+v\rVert$ can be written as $y_i(u^Tx_i+v)$. Thus the
machine capacity can be minimized by keeping $h_{MCM}^2$ as small as
possible.
$$\underset{u,v}{\operatorname{minimize}}\;\; h_{MCM}=\frac{\max_{i=1,\dotsc,M}y_i(u^Tx_i+v)}{\min_{i=1,\dotsc,M}y_i(u^Tx_i+v)}$$
Further the authors show that the above formulation can be further
simplified by writing: $$\begin{aligned}
    &h_{MCM}=\frac{g}{l}\\
    &\min_{u,v,g,l}\frac{g}{l}\quad\text{subject to}\\
    &g\ge y_i(u^Tx_i+v),\quad i=1,\dotsc,M\\
    &l\le y_i(u^Tx_i+v),\quad i=1,\dotsc,M\end{aligned}$$ Using
Charnes-Cooper transformation, introducing $p=\frac{1}{l}$
$$\begin{aligned}
    &\min_{u,v,g,l,p}g\cdot p\quad\text{subject to}\\
    &g\cdot p\ge y_i(p\cdot u^Tx_i+p\cdot v),\quad i=1,\dotsc,M\\
    &l\cdot p\le y_i(p\cdot u^Tx_i+p\cdot v),\quad i=1,\dotsc,M\\
    &p\cdot l=1\end{aligned}$$ Denoting
$w\stackrel{\Delta}{=}p\cdot u,b\stackrel{\Delta}{=}p\cdot v$ and noting
that $p\cdot l=1$ $$\begin{aligned}
    &\label{eq:1}\min_{w,b,h}h\quad\text{subject to}\\
    &\label{eq:2}h\ge y_i(w^Tx_i+b),\quad i=1,\dotsc,M\\
    &\label{eq:3}1\le y_i(w^Tx_i+b),\quad i=1,\dotsc,M\end{aligned}$$
Equations [\[eq:1\]](#eq:1){reference-type="ref"
reference="eq:1"}-[\[eq:3\]](#eq:3){reference-type="ref"
reference="eq:3"} define the Minimal Complexity Machine (MCM). And it is
trained by solving the Linear Programming Problem defined above.

### Generalizing the MCM

The MCM above is further generalized to allow for classification errors
by introducing slack variables. $$\begin{aligned}
    &\min_{w,b,h,q}h+C\cdot\sum_{i=1}^Mq_i\\
    &h\ge y_i(w^Tx_i+b)+q_i,\quad i=1,\dotsc,M\\
    &1\le y_i(w^Tx_i+b)+q_i,\quad i=1,\dotsc,M\\
    &q_i\ge 0\quad i=1,\dotsc,M\end{aligned}$$ And for the non-linear
case using Kernels $$\begin{aligned}
    &\min_{w,b,h,q}h+C\cdot\sum_{i=1}^Mq_i\\
    &h\ge y_i(w^T\phi(x_i)+b)+q_i,\quad i=1,\dotsc,M\\
    &1\le y_i(w^T\phi(x_i)+b)+q_i,\quad i=1,\dotsc,M\\
    &q_i\ge 0\quad i=1,\dotsc,M\end{aligned}$$ where $\phi(\cdot)$ is
the mapping function that maps input to a higher dimensional space,
where the inputs are assumed to be linearly separable with some errors.
As of yet, this doesn't use a kernel function $K(\cdot,\cdot)$. Assume
that $K$ is a kernel function corresponding to the map $\phi(\cdot)$;
$\phi(x_j)^T\phi(x_k)=K(x_j,x_k)$. Also, since $\phi(x_i), i=1,\dotsc,M$
forms a basis for the feature space in which $w$ lies, $w$ can be
written as a linear combination of the basis vectors $$\begin{aligned}
    &w=\sum_{j=1}^M\lambda_j\phi(x_j)\\
    &\Rightarrow
    w^T\phi(x_i)=\sum_{j=1}^M\lambda_j\phi(x_j)^T\phi(x_i)=\sum_{j=1}^M\lambda_jK(x_j,x_i)\end{aligned}$$
Now, the MCM formulation can be rewritten using $K(\cdot,\cdot)$ as
$$\begin{aligned}
    &\label{eq:mcmstart}\min_{\lambda,b,h,q}h+C\cdot\sum_{i=1}^Mq_i\quad\text{subject to}\\
    &\label{eq:upbound}h\ge y_i\left(\sum_{j=1}^M\lambda_jK(x_j,x_i)+b\right)+q_i,\quad i=1,\dotsc,M\\
    &1\le y_i\left(\sum_{j=1}^M\lambda_jK(x_j,x_i)+b\right)+q_i,\quad i=1,\dotsc,M\\
    &\label{eq:mcmend}q_i\ge 0\quad i=1,\dotsc,M\end{aligned}$$ After
optimizing, those vectors $x_i$ for which the corresponding $\lambda_i$
is non zero form the support vectors that support $w$.
$SV=\{i\;\lvert\;\lambda_i\ne 0\}$. One last thing that needs to be
defined explicitly is how to predict on new data once the optimization
is complete. The class is assigned as:
$$y_{\text{pred}}=\operatorname{sign}\left(\sum_{i\in
SV}\lambda_iK(x_i,x_{\text{test}})+b\right)$$ This completes the MCM
formulation. One thing to note is Vapnik's SVM formulation was also
motivated from the fact that constraining the VC-dimension will improve
generalization, but Vapnik only does this by trying to maximize the
margin $\Delta$, while in MCM the ratio $R/\Delta$ is constrained to
control the complexity of the machine. This results in an extra
constraint bounding the maximum distance from the hyperplane from above
(eq. [\[eq:upbound\]](#eq:upbound){reference-type="ref"
reference="eq:upbound"}), thus providing actual theoretical guarantees.
It has also been experimentally showed that the numebr of support
vectors (and thus the size itself of MCM) is much smaller, allowing
these machines to be used in edge devices where neural networks can't be
used.

Kernel Optimization
-------------------

The key ideas for kernel optimization comes from [@amari]. When the data
is non-linearly separable, it is assumed that when projected to a space
of high enough dimension, then they'll be separable. But this is only
true when the mapping function is suitably chosen. There is no one
kernel that defeats all others. If the kernel is "wrong" for the current
task, the class separability in the feature space/image space might be
poorer than it was in input space.

### Motivation

Again let's consider the data shown in
figure [1](#fig:1){reference-type="ref" reference="fig:1"}. Even though
the data is already linearly separable, let's see what we can do to
improve the class separability. Consider
figures [3](#fig:3){reference-type="ref"
reference="fig:3"}-[5](#fig:5){reference-type="ref" reference="fig:5"}.
We can see that if we magnify around the feature space around the class
boundary (near hyperplane) while compress the space around the edges, we
can improve class separability thus improving generalization.

![[\[fig:3\]]{#fig:3 label="fig:3"}Identity map, to
compare](/home/mridul/Desktop/iitd_rishi_laptop/backups/ELL888/margin_func3.png){#fig:3
width=".5\\linewidth"}

![[\[fig:4\]]{#fig:4 label="fig:4"}A non-linear map that is steeper
between the classes, shallower
within](/home/mridul/Desktop/iitd_rishi_laptop/backups/ELL888/margin_func4.png){#fig:4
width=".5\\linewidth"}

![[\[fig:5\]]{#fig:5 label="fig:5"}This improves class separability, and
also compresses the data reducing
VC-dimension](/home/mridul/Desktop/iitd_rishi_laptop/backups/ELL888/margin_func5.png){#fig:5
width=".5\\linewidth"}

### Formulations

This is approximately what Amari *et al.* do. They do not compress
around the edges, but just focus on magnifying on the confusion region,
the boundary. That is they try to increase the margin even further by
having an appropriate distortion. Since the boundary is not generally
known, magnification is done close to support vectors, since they are
close to the boundary. These are called "empirical cores" in the context
of kernel optimization (since after optimization the actual support
vectors will be different).

The "magnification" is done by applying a *conformal transformation* to
the kernel. Conformal transformation of a space preserves angles but
might change lengths locally, which is just what we need, to magnify
without altering angular structures. Conformal transformation of the
kernel *by a factor $c(x)$* is given by:
$$\tilde{K}(x,x')=c(x)c(x')K(x,x')$$ This transformation function is
constructed in a data dependent way to be maximal around support
vectors. For more detail about how to arrive at this, and the underlying
Riemannian metric tensor based theory, refer [@amari].

This work is taken further by Xiong *et al.* [@xiong]. Until now we have
been talking about "class separability" subjectively, but Xiong *et
al.* solidify this by giving an actual mathematical formulation of class
separability that can be used to guide the kernel optimization. But
first define the factor function they use $$\begin{aligned}
    &k(x,y)=c(x)c(y)k_0(x,y)\\
    &c(x)=\alpha_0+\sum_{i=1}^{n_{ec}}\alpha_ik_1(x,a_i)\end{aligned}$$
$k_0(\cdot,\cdot)$ is a basic kernel, like Gaussian.
$k_1(x,a_i)=\exp(-\gamma\lVert x-a_i\rVert^2)$ and $a_i$ are "empirical
cores". $\alpha_i$'s are called combination coefficients. It is easy to
write the matrix notation of these by defining $K\text{ and }K_0$ as the
matrix corresponding to $k(\cdot,\cdot)$ and $k_0(\cdot,\cdot)$
respectively. $$K=[c(x_i)c(x_j)k_0(x_i,x_j)]_{M\times M}=CK_0C$$ where
$C=diag(c(x_1),c(x_2),\dotsc,c(x_M))$. Furthermore $$\label{eq:alk}
    \vec{c}=\begin{pmatrix}
        1&k_1(x_1,a_1)&\dots&k_1(x_1,a_{n_{ec}})\\
        1&k_2(x_2,a_1)&\dots&k_1(x_2,a_{n_{ec}})\\
        \vdots&\vdots&\ddots&\vdots\\
        1&k_M(x_M,a_1)&\dots&k_1(x_M,a_{n_{ec}})\\
        \end{pmatrix}\begin{pmatrix}
            \alpha_0\\
            \alpha_1\\
            \vdots\\
            \alpha_{n_{ec}}
    \end{pmatrix}\stackrel{\Delta}{=}K_1\vec{\alpha}$$ where $K_1$ is an
$M\times(n_{ec}+1)$ matrix.

Next, they use a measure of class separability called *Fisher scalar* to
define their own Kernel based class separability metric. The Fisher
scalar is defined as $$\begin{aligned}
    &J=\frac{\operatorname{tr} S_b}{\operatorname{tr} S_w}\\
    &S_b=\frac{1}{M}\sum_{i=1}^2M_i(\bar{z}_i-\bar{z})(\bar{z}_i-\bar{z})^T\\
    &S_w=\frac{1}{M}\sum_{i=1}^2\sum_{j=1}^{M_i}(z_j^i-\bar{z}_i)(z_j^i-\bar{z}_i)^T\end{aligned}$$
where $i$ indexes through the two classes, $M_i$ is the number of
samples in the $i^\text{th}$ class. $\{z_j\}_{j=1}^M$ are the images of
the training data in the empirical feature space. $M_1+M_2=M$.
$\bar{z},\bar{z}_1,\bar{z}_2$ are the centers of the entire training
data and those of class 1 and 2 resepctively in the empirical feature
space. $z_j^i$ is the $j^\text{th}$ data sample in $i^\text{th}$ class.

![[\[fig:6\]]{#fig:6 label="fig:6"}This picture is taken from Xiong *et
al.* [@xiong]. It shows how a kernel can harm class separability. (a)
Input. (b) Two dimensional projection of empirical feature space for
second order polynomial kernel. (c) Two dimensional projection of
Gaussian
kernel.](/home/mridul/Desktop/iitd_rishi_laptop/backups/ELL888/class_sep.png){#fig:6
width=".8\\linewidth"}

This can be further simplified by ordering the data points such that the
first $M_1$ points belong to class 1, $y_i=-1,i\le M_1$ and the
remaining belong to class 2. Then kernel matrix can be written in a
block form as:
$$K=\begin{pmatrix}K_{11}&K_{12}\\K_{21}&K_{22}\end{pmatrix}$$ $K_{11}$
is an $M_1\times M_1$ submatrix of $K$ that corresponds to class 1.
Similarly $K_{22}$ is an $M_2\times M_2$ sized matrix corresponding to
class 2. $K_{12}\text{ and }K_{21}$ are
$M_1\times M_2\text{ and }M_2\times M_1$ respectively. Next define
"between-class" and "within-class" kernel scatter matrices
$B\text{ and }W$ as: $$\begin{aligned}
    \label{eq:B}B=&\begin{pmatrix}
        \frac{1}{M_1}K_{11}&0\\
        0&\frac{1}{M_2}K_{22}
        \end{pmatrix}-\begin{pmatrix}
            \frac{1}{M}K_{11}&\frac{1}{M}K_{12}\\
            \frac{1}{M}K_{21}&\frac{1}{M}K_{22}
    \end{pmatrix}\\
    \label{eq:W}W=&\begin{pmatrix}
        k_{11} & 0 & \dots & 0\\
        0 & k_{22} & \dots & 0\\
        \vdots & \vdots & \ddots & \vdots\\
        0 & 0 & \dots & k_{MM}
        \end{pmatrix}-\begin{pmatrix}\frac{1}{M_1}K_{11}&0\\0&\frac{1}{M_2}K_{22}\end{pmatrix}\end{aligned}$$
Also, denote $B_0$ and $W_0$ as the between-class and within-class
kernel scatter matrices corresponding to the basic kernel $K_0$. (Xiong
*et al.*) Let $1_k$ be the $k$-dimensional vector whose entries are all
equal to one. Then $$\label{eq:gep}
    J=\frac{1_M^TB1_M}{1_M^TW1_M}=\frac{\vec{c}^TB_0\vec{c}}{\vec{c}^TW_0\vec{c}}$$
Proof: refer [@xiong] Xiong *et al.* then optimize the kernel so that
the class separability metric $J$ is maximized using a gradient based
approach. Here we diverge to the approach used by Jayadeva *et
al.* in [@keropt]. Instead of solving as a gradient based optimization,
they identify the Rayleigh Quotient in
equation [\[eq:gep\]](#eq:gep){reference-type="ref" reference="eq:gep"}
and naturally propose it as a Generalized Eigenvalue Problem and try to
solve simultaneously for $\gamma$ and $\vec{c}$ in
$$B_0\vec{c}=\gamma W_0\vec{c}$$ where $\gamma$ is the set of
eigenvalues. This is further rewritten using
equation [\[eq:alk\]](#eq:alk){reference-type="ref" reference="eq:alk"}
as $[K_1^TB_0K_1]\vec{\alpha}=\gamma[K_1^T(W_0+DI)K_1]\vec{\alpha}$. $D$
is a small constant added to the diagonal values for stability, $I$ is
identity matrix. Let $K_1^TB_0K_1\stackrel{\Delta}{=}P,
[K_1^T(W_0+DI)K_1]\stackrel{\Delta}{=}Q$. So we have
$$\label{eq:GEP}P\vec{\alpha}=\gamma Q\vec{\alpha}$$

From the solution we can get $\vec{\alpha}$ (corresponding to maximum
eigenvalue $\gamma$), which in turn gives $\vec{c}$, which in turn gives
the conformal map transformed kernel $K$. And, finally we have enough
theory to discuss Kernel optimization for MCMs. Note, all the previous
theory regarding kernel optimization was in context of support vector
machines (SVMs).

Kernel optimization using conformal maps for the minimal complexity machine [@keropt]
=====================================================================================

We have basically every ingredient necessary to talk about this: we have
the motivation to use MCMs, we have the motivation and the method to
perform kernel optimization, we have the statistical learning theory
backing up our intuition that separating out classes would be better.
Now let's talk about how kernel optimization is done in the context of
MCMs. It's just a simple algorithm:

1.  Transform the dataset to zero mean and unit variance.

2.  Solve the optimization defined in
    equations [\[eq:mcmstart\]](#eq:mcmstart){reference-type="ref"
    reference="eq:mcmstart"}-[\[eq:mcmend\]](#eq:mcmend){reference-type="ref"
    reference="eq:mcmend"} using the basic kernel to get support vectors
    for which $\lambda_i \ne 0$ (or to account for computational
    limitations, $\lvert\lambda_i\rvert > \varepsilon$ where
    $\varepsilon$ is a small constant.

3.  Compute $W_0$, $B_0$ (eq [\[eq:W\]](#eq:W){reference-type="ref"
    reference="eq:W"},[\[eq:B\]](#eq:B){reference-type="ref"
    reference="eq:B"})

4.  Solve the generalized eigenvalue problem in
    equation [\[eq:GEP\]](#eq:GEP){reference-type="ref"
    reference="eq:GEP"} to find $\vec{\alpha}$ corresponding to largest
    eigen value.

5.  Use $\vec{\alpha}$ to compute the scaling factor $\vec{c}$

6.  Use the scaling factor to compute the optimized kernel
    $k(x_i,x_j)=c(x_i)c(x_j)k_0(x_i,x_j)$

7.  Use the optimized kernel matrix $K(\cdot,\cdot)$ corresponding to
    kernel function $k(\cdot,\cdot)$ to train a new MCM (that is again
    solve equations [\[eq:mcmstart\]](#eq:mcmstart){reference-type="ref"
    reference="eq:mcmstart"}-[\[eq:mcmend\]](#eq:mcmend){reference-type="ref"
    reference="eq:mcmend"}.)

8.  Use this new MCM on test data.

Results in [@keropt] shows that MCM with optimized kernels perform
better when measured by accuracy by a big margin for most of the
benchmark datasets as compared to vanilla MCM or SVM. The maximum number
of samples in any of these datasets was 1000. When applied on datasets
of larger size, the MCM's do not perform much better, which is expected
from equation [\[eq:vareps\]](#eq:vareps){reference-type="ref"
reference="eq:vareps"}. As $M$, the training sample size plays a much
larger role in reducing $\varepsilon$ and in turn reducing the
structural risk, it doesn't help a lot in big data setting to focus on
structural risk minimization. But MCMs still provide a sparser model to
be deployed on power constrained devices.

10 Amari SI, Wu S. . Neural Networks. 1999 Jul 1;12(6):783-9. Vapnik VN.
IEEE transactions on neural networks. 1999 Sep;10(5):988-99. Jayadeva.
Neurocomputing. 2015 Feb 3;149:683-9. Xiong H, Swamy MN, Ahmad MO. IEEE
transactions on neural networks. 2005 Mar 7;16(2):460-74. Badge S, Soman
S, Chandra S, Jayadeva. Engineering Applications of Artificial
Intelligence. 2021 Nov 1;106:104493.
