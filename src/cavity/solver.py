"""Stream function-vorticity solver for the 2-D lid-driven cavity.

Non-dimensional incompressible Navier-Stokes equations (length L, velocity U,
Re = U L / nu) in the stream function-vorticity form:

    d(xi)/dt + u d(xi)/dx + v d(xi)/dy = (1/Re) lap(xi)     vorticity transport
    lap(psi) = -xi                                         stream function
    u = d(psi)/dy,  v = -d(psi)/dx                          velocity

on the unit square, with no slip on every wall and the lid (y = 1) moving at
u = 1. Pressure does not appear, and continuity holds by construction.

Discretisation (uniform grid of N x N nodes, spacing h = 1/(N-1)):
- Diffusion: second-order central differences.
- Advection: second-order central differences ('central'), or first-order
  upwind ('upwind'), the scheme of the original course project. Upwinding adds
  a numerical viscosity of order |u| h / 2, which at high Re is larger than the
  physical one.
- Wall vorticity: Thom's formula, xi_w = -2 psi_1 / h^2 (-2 U / h on the lid).
- Stream function: the Poisson equation is solved exactly at every stage with
  a sparse LU factorisation computed once.
- Time: third-order strong-stability-preserving Runge-Kutta (SSP-RK3), needed
  for central advection (explicit Euler is unconditionally unstable for pure
  central advection); forward Euler is kept for the original scheme. The time
  step adapts to the advective (CFL) and diffusive limits.

The solution is marched in pseudo-time to the steady state, declared when the
largest time derivative of the vorticity falls below `tol`.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu

SCHEMES = ('central', 'upwind')
INTEGRATORS = ('rk3', 'euler')


@dataclass(frozen=True)
class CavityConfig:
    """Parameters of one simulation (non-dimensional)."""

    re: float
    n: int = 129
    scheme: str = 'central'
    integrator: str | None = None  # default: 'rk3' for central, 'euler' for upwind
    cfl: float = 0.5  # fraction of the advective limit h / (|u| + |v|)
    diffusion_number: float = 0.2  # dt <= diffusion_number * h^2 * Re (2-D explicit limit 0.25)
    tol: float = 1e-6  # steady state: max |d xi / dt| < tol
    t_max: float = 400.0
    max_steps: int = 2_000_000

    def __post_init__(self):
        if self.scheme not in SCHEMES:
            raise ValueError(f"scheme must be one of {SCHEMES}")
        if self.integrator is not None and self.integrator not in INTEGRATORS:
            raise ValueError(f"integrator must be one of {INTEGRATORS}")
        if self.scheme == 'central' and self.integrator == 'euler':
            raise ValueError("central advection with forward Euler is unconditionally unstable")
        if self.n < 5:
            raise ValueError("n must be at least 5")

    @property
    def method(self) -> str:
        return self.integrator or ('rk3' if self.scheme == 'central' else 'euler')

    @property
    def h(self) -> float:
        return 1.0 / (self.n - 1)


@dataclass
class CavityResult:
    """Steady (or last) state of a simulation."""

    config: CavityConfig
    x: np.ndarray
    y: np.ndarray
    psi: np.ndarray  # [j, i] = (y, x)
    xi: np.ndarray
    u: np.ndarray
    v: np.ndarray
    t: float
    steps: int
    converged: bool
    wall_time: float
    history: dict = field(default_factory=dict)


class PoissonSolver:
    """Exact solver of lap(psi) = -xi with psi = 0 on the boundary (5-point stencil)."""

    def __init__(self, n: int):
        m = n - 2
        h = 1.0 / (n - 1)
        d = sp.diags([1.0, -2.0, 1.0], [-1, 0, 1], shape=(m, m))
        eye = sp.identity(m)
        lap = (sp.kron(eye, d) + sp.kron(d, eye)) / h**2
        self.m = m
        self._lu = splu(lap.tocsc())

    def solve(self, xi: np.ndarray) -> np.ndarray:
        psi = np.zeros_like(xi)
        psi[1:-1, 1:-1] = self._lu.solve(-xi[1:-1, 1:-1].ravel()).reshape(self.m, self.m)
        return psi


def velocities(psi: np.ndarray, h: float, lid: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """u = d(psi)/dy, v = -d(psi)/dx (central differences inside, no slip on the walls)."""
    u = np.zeros_like(psi)
    v = np.zeros_like(psi)
    u[1:-1, 1:-1] = (psi[2:, 1:-1] - psi[:-2, 1:-1]) / (2 * h)
    v[1:-1, 1:-1] = -(psi[1:-1, 2:] - psi[1:-1, :-2]) / (2 * h)
    u[-1, :] = lid
    return u, v


def apply_wall_vorticity(xi: np.ndarray, psi: np.ndarray, h: float, lid: float = 1.0) -> None:
    """Thom's formula on the four walls (psi = 0 on the boundary)."""
    xi[-1, 1:-1] = -2.0 * psi[-2, 1:-1] / h**2 - 2.0 * lid / h
    xi[0, 1:-1] = -2.0 * psi[1, 1:-1] / h**2
    xi[1:-1, 0] = -2.0 * psi[1:-1, 1] / h**2
    xi[1:-1, -1] = -2.0 * psi[1:-1, -2] / h**2


def vorticity_rate(xi, u, v, h, re, scheme):
    """d(xi)/dt at the interior nodes."""
    c = xi[1:-1, 1:-1]
    w, e, s, nn = xi[1:-1, :-2], xi[1:-1, 2:], xi[:-2, 1:-1], xi[2:, 1:-1]
    uc, vc = u[1:-1, 1:-1], v[1:-1, 1:-1]
    diffusion = (e + w + nn + s - 4.0 * c) / (re * h**2)
    if scheme == 'central':
        advection = uc * (e - w) / (2 * h) + vc * (nn - s) / (2 * h)
    else:
        advection = (np.maximum(uc, 0) * (c - w) + np.minimum(uc, 0) * (e - c)) / h + (
            np.maximum(vc, 0) * (c - s) + np.minimum(vc, 0) * (nn - c)
        ) / h
    return diffusion - advection


def solve(
    config: CavityConfig,
    initial: CavityResult | None = None,
    verbose: bool = False,
    callback=None,
    callback_every: float = 0.5,
) -> CavityResult:
    """March the cavity in time to steady state.

    Args:
        config:         simulation parameters.
        initial:        optional previous solution on the same grid.
        verbose:        print progress every 1000 steps.
        callback:       optional function callback(t, psi, xi) called every
                        callback_every time units (e.g. to record animation frames).
    """
    n, h, re = config.n, config.h, config.re
    poisson = PoissonSolver(n)
    xi = initial.xi.copy() if initial is not None and initial.xi.shape == (n, n) else np.zeros((n, n))
    psi = poisson.solve(xi)
    apply_wall_vorticity(xi, psi, h)
    u, v = velocities(psi, h)

    def stage(xi_in):
        """Poisson solve, wall vorticity and interior rate for one RK stage."""
        p = poisson.solve(xi_in)
        apply_wall_vorticity(xi_in, p, h)
        uu, vv = velocities(p, h)
        return vorticity_rate(xi_in, uu, vv, h, re, config.scheme), p, uu, vv

    hist_t, hist_res, hist_ek = [], [], []
    t, steps, converged = 0.0, 0, False
    next_frame = 0.0
    start = time.perf_counter()
    while t < config.t_max and steps < config.max_steps:
        rate, psi, u, v = stage(xi)
        residual = float(np.abs(rate).max())
        if steps % 50 == 0:
            hist_t.append(t)
            hist_res.append(residual)
            hist_ek.append(kinetic_energy(u, v, h))
            if verbose and steps % 1000 == 0:
                print(f"step {steps:7d}  t={t:8.3f}  max|dxi/dt|={residual:.2e}")
        if callback is not None and t >= next_frame:
            callback(t, psi, xi)
            next_frame += callback_every
        if not np.isfinite(residual):
            raise FloatingPointError(f"solution diverged at step {steps} (t = {t:.4f})")
        if residual < config.tol and steps > 0:
            converged = True
            break

        speed = np.abs(u).max() + np.abs(v).max()
        dt = min(config.cfl * h / max(speed, 1e-12), config.diffusion_number * h**2 * re)
        if config.method == 'euler':
            xi[1:-1, 1:-1] += dt * rate
        else:
            x1 = xi.copy()
            x1[1:-1, 1:-1] += dt * rate
            r1 = stage(x1)[0]
            x2 = xi.copy()
            x2[1:-1, 1:-1] = 0.75 * xi[1:-1, 1:-1] + 0.25 * (x1[1:-1, 1:-1] + dt * r1)
            r2 = stage(x2)[0]
            xi[1:-1, 1:-1] = xi[1:-1, 1:-1] / 3.0 + 2.0 / 3.0 * (x2[1:-1, 1:-1] + dt * r2)
        t += dt
        steps += 1

    psi = poisson.solve(xi)
    apply_wall_vorticity(xi, psi, h)
    u, v = velocities(psi, h)
    grid = np.linspace(0.0, 1.0, n)
    return CavityResult(
        config=config,
        x=grid,
        y=grid,
        psi=psi,
        xi=xi,
        u=u,
        v=v,
        t=t,
        steps=steps,
        converged=converged,
        wall_time=time.perf_counter() - start,
        history={'t': np.array(hist_t), 'residual': np.array(hist_res), 'kinetic_energy': np.array(hist_ek)},
    )


def kinetic_energy(u: np.ndarray, v: np.ndarray, h: float) -> float:
    """Non-dimensional kinetic energy per unit depth, 1/2 integral of (u^2 + v^2) (trapezoidal rule)."""
    w = np.ones(u.shape[0])
    w[0] = w[-1] = 0.5
    weights = np.outer(w, w) * h**2
    return 0.5 * float(np.sum(weights * (u**2 + v**2)))
