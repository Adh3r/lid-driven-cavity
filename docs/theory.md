# Theory and numerical method

## 1. Problem

A square cavity of side $L$ is filled with an incompressible Newtonian fluid.
The lid ($y = L$) slides at constant speed $U$ in $+x$; the other three walls are
fixed. With $L$, $U$ and $L/U$ as scales, the only parameter is the Reynolds
number $Re = UL/\nu$, and every quantity in this repository is
non-dimensional.

$$
\nabla\cdot\mathbf u = 0, \qquad
\frac{\partial \mathbf u}{\partial t} + (\mathbf u\cdot\nabla)\mathbf u
= -\nabla p + \frac{1}{Re}\nabla^2 \mathbf u .
$$

## 2. Stream function-vorticity formulation

In 2-D, continuity is satisfied identically by a stream function $\psi$:
$u = \partial\psi/\partial y$, $v = -\partial\psi/\partial x$. Taking the curl
of the momentum equation removes the pressure and leaves the transport
equation of the vorticity $\xi = \partial v/\partial x - \partial u/\partial y$
(vortex stretching vanishes in 2-D):

$$
\frac{\partial \xi}{\partial t} + u\frac{\partial \xi}{\partial x} + v\frac{\partial \xi}{\partial y}
= \frac{1}{Re}\nabla^2\xi, \qquad \nabla^2\psi = -\xi .
$$

Two scalar unknowns replace three ($u$, $v$, $p$), and there is no
pressure-velocity coupling to handle. The price is that the wall vorticity is
not known and has to be derived from $\psi$.

**Sign convention.** With this definition the clockwise primary vortex has
$\xi < 0$ and $\psi < 0$. Ghia et al. (1982) tabulate
$\omega = \partial u/\partial y - \partial v/\partial x$, so their vorticity has
the opposite sign; magnitudes are compared.

## 3. Boundary conditions

The walls form one streamline, $\psi = 0$. No slip also fixes the normal
derivative: $\partial\psi/\partial n = 0$ on fixed walls and
$\partial\psi/\partial y = 1$ on the lid. Expanding $\psi$ one node into the
fluid,

$$
\psi_1 = \psi_w + h\left.\frac{\partial\psi}{\partial n}\right|_w
+ \frac{h^2}{2}\left.\frac{\partial^2\psi}{\partial n^2}\right|_w + O(h^3),
$$

and using $\xi_w = -\nabla^2\psi|_w = -\partial^2\psi/\partial n^2|_w$ (the
tangential second derivative of $\psi$ vanishes along the wall) gives **Thom's
formula**

$$
\xi_w = -\frac{2\psi_1}{h^2} \quad\text{(fixed walls)}, \qquad
\xi_w = -\frac{2\psi_1}{h^2} - \frac{2}{h} \quad\text{(lid)} .
$$

It is locally first-order accurate, but the solution converges at second
order, as the grid study in the README confirms (observed order 2.0).

**Corner singularities.** At the two lid corners the wall velocity jumps from 0
to 1, so the vorticity is unbounded there. The singularity is integrable and
local: it does not spoil convergence elsewhere, but vorticity colour scales
are clipped in the figures.

## 4. Discretisation

Uniform grid of $N\times N$ nodes, $h = 1/(N-1)$.

- **Diffusion and Poisson:** 5-point second-order Laplacian. The Poisson
  equation is solved exactly with a sparse LU factorisation (computed once per
  grid). The original code used Jacobi iterations capped at 3000 per step.
- **Advection, central:** $u\,(\xi_{i+1}-\xi_{i-1})/2h$, second order.
- **Advection, first-order upwind** (the original code). Its truncation error
  is diffusive. The modified equation is

  $$
  \frac{\partial \xi}{\partial t} + \mathbf u\cdot\nabla\xi
  = \frac{1}{Re}\nabla^2\xi + \frac{|u|h}{2}\frac{\partial^2\xi}{\partial x^2}
  + \frac{|v|h}{2}\frac{\partial^2\xi}{\partial y^2} + O(h^2),
  $$

  that is, a numerical viscosity $\nu_{num} \approx |u|h/2$. Relative to the
  physical one, $\nu_{num}/\nu = Re\,|u|\,h/2$, the cell Péclet number over two.
  On the original 101² grid at $Re = 1000$ it is up to 5: the scheme solves a
  much more viscous flow. At $Re = 100$ it is 0.5 or less, which is why the
  original results agreed with the benchmark only at low $Re$.

## 5. Solvers

**Direct steady solver (Newton).** The steady discrete equations are
$2(N-2)^2$ algebraic equations in the interior $\psi$ and $\xi$, with the wall
vorticity eliminated through Thom's formula. Their Jacobian is assembled
exactly as a sparse matrix, and each Newton step is one sparse LU solve. The
iteration converges quadratically; high $Re$ is reached by continuation, using
each converged solution as the initial guess of the next $Re$. On fine grids
the continuation runs on a grid of half the resolution, and its interpolated
solution starts a single Newton solve (grid sequencing). The result is
the discrete steady solution to round-off, so grid studies are not polluted by
a time-marching stopping tolerance.

**Time marching.** For transients, three-stage strong-stability-preserving
Runge-Kutta (SSP-RK3), with the Poisson equation solved at every stage. Two
stability limits apply:

- *Advection:* the eigenvalues of central differences are imaginary, and
  forward Euler is unstable for any $\Delta t$ there. RK3 is stable for
  $\Delta t \le \sqrt 3\, h/(|u|+|v|)$; the solver uses $0.5\,h/(|u|+|v|)$.
- *Diffusion:* forward Euler on the 5-point Laplacian requires
  $\Delta t \le h^2 Re/4$; the solver uses $0.2\,h^2 Re$.

The original code used a fixed $\Delta t = 10^{-3}$. At $Re = 10$ the diffusive
limit is $2.5\times10^{-4}$, so it diverged in 13 steps. Its stated range of
$Re = 10^{-2}$ to $10^3$ would have needed $\Delta t \approx 2.5\times10^{-8}$
at the low end.

## 6. Range of validity

The square cavity flow is steady up to a Hopf bifurcation at $Re \approx 8000$
(Auteri, Parolini & Quartapelle 2002; Bruneau & Saad 2006). The Newton solver
still returns a steady solution beyond it, but that branch is unstable, and
time marching is needed to follow the periodic flow. The pipe-flow criterion
$Re < 2300$, quoted in the original presentation, does not apply to a cavity.

## 7. Extension: obstacle that does not touch the walls

A body inside the flow is a streamline, $\psi = \psi_b$, but $\psi_b$ is
unknown: it equals the wall value only if no fluid passes between the body and
the walls. It is fixed by requiring the pressure to be single-valued:
$\oint_C \nabla p\cdot d\mathbf s = 0$ around the body. With the steady momentum
equation in rotational form,

$$
\nabla\!\left(p + \tfrac12|\mathbf u|^2\right) = \mathbf u\times\xi\,\mathbf e_z
- \frac{1}{Re}\nabla\times(\xi\,\mathbf e_z),
$$

the condition on a closed loop $C$ in the fluid is

$$
\oint_C \left(v\xi - \frac{1}{Re}\frac{\partial\xi}{\partial y}\right)dx
+ \left(-u\xi + \frac{1}{Re}\frac{\partial\xi}{\partial x}\right)dy = 0 .
$$

It becomes one extra equation for the extra unknown $\psi_b$ in the Newton
system. The loop runs through fluid nodes at a fixed distance (1/16) from the
body. On the body surface itself, $\partial p/\partial s = -Re^{-1}\,\partial\xi/\partial n$,
but evaluating $\partial\xi/\partial n \approx (\xi_1 - \xi_w)/h$ there is
inconsistent: Thom's $\xi_w$ carries an $O(h)$ error, so the derivative carries
an $O(1)$ error. A first implementation that did this gave a $\psi_b$ that kept
drifting as the grid was refined. With the loop, $\psi_b$ converges at second
order.

Setting $\psi_b = 0$, as the obstacle variant of the original project did,
treats the body as if it were connected to the walls. It leaves a pressure
jump around the body, as if a pump were placed inside it, and produces a
completely different flow pattern (see the README figure).

## 8. Limitations

- The pressure field is not computed (it can be recovered from $\psi$ and $\xi$
  with a pressure Poisson equation).
- Uniform grid: boundary layers at high $Re$ would be resolved more cheaply with
  wall-clustered grids.
- Central advection at cell Péclet numbers above 2 can produce wiggles. None
  are visible at $Re \le 1000$ on the grids used, and the benchmark errors
  confirm it, but higher $Re$ would need a bounded high-order scheme.

## References

- U. Ghia, K. N. Ghia, C. T. Shin, *High-Re solutions for incompressible flow using
  the Navier-Stokes equations and a multigrid method*, J. Comput. Phys. 48 (1982)
  387-411.
- O. Botella, R. Peyret, *Benchmark spectral results on the lid-driven cavity flow*,
  Comput. Fluids 27 (1998) 421-433.
- F. Auteri, N. Parolini, L. Quartapelle, *Numerical investigation on the stability
  of singular driven cavity flow*, J. Comput. Phys. 183 (2002) 1-25.
- C.-H. Bruneau, M. Saad, *The 2D lid-driven cavity problem revisited*, Comput. Fluids
  35 (2006) 326-348.
- A. Thom, *The flow past circular cylinders at low speeds*, Proc. R. Soc. A 141
  (1933) 651-669.
- S. Gottlieb, C.-W. Shu, *Total variation diminishing Runge-Kutta schemes*, Math.
  Comput. 67 (1998) 73-85.
