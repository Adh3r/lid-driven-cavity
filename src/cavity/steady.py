"""Direct steady-state solver: Newton's method on the discrete equations.

The steady discrete stream function-vorticity system (same discretisation as
solver.py: 5-point Laplacians, central or first-order upwind advection, Thom
wall vorticity) is a set of 2 (N-2)^2 non-linear algebraic equations in the
interior values of psi and xi:

    F1 = lap_h(psi) + xi = 0
    F2 = (1/Re) lap_h(xi) - (u Dx(xi) + v Dy(xi)) = 0,   u = Dy(psi), v = -Dx(psi)

where the wall vorticity entering lap_h(xi) and the advection stencils is
Thom's formula, linear in the first interior psi. The Jacobian is assembled
exactly as a sparse matrix and each Newton step is one sparse LU solve, so the
discrete steady solution is found to machine precision in a few iterations,
with no pseudo-time and no stopping tolerance in time. High Re is reached by
continuation (each solution is the initial guess of the next Re).
"""

from __future__ import annotations

import time

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve

from .solver import CavityConfig, CavityResult, apply_wall_vorticity, velocities


class _Operators:
    """Sparse operators on the interior unknowns, built once per grid."""

    def __init__(self, n: int):
        h = 1.0 / (n - 1)
        m = n - 2
        self.n, self.m, self.h, self.size = n, m, h, m * m
        ids = np.arange(n * n).reshape(n, n)
        interior = ids[1:-1, 1:-1].ravel()

        # P: interior vector -> full grid (zeros on the boundary)
        self.P = sp.csr_matrix((np.ones(m * m), (interior, np.arange(m * m))), shape=(n * n, m * m))
        # R: full grid -> interior
        self.R = self.P.T.tocsr()

        # Thom wall vorticity: xi_wall = -2 psi_adjacent / h^2 (+ lid term, constant)
        rows, cols = [], []
        for j in range(1, n - 1):
            rows += [ids[j, 0], ids[j, -1]]
            cols += [ids[j, 1], ids[j, -2]]
        for i in range(1, n - 1):
            rows += [ids[0, i], ids[-1, i]]
            cols += [ids[1, i], ids[-2, i]]
        B_full = sp.csr_matrix((np.full(len(rows), -2.0 / h**2), (rows, cols)), shape=(n * n, n * n))
        self.B = (B_full @ self.P).tocsr()  # psi_interior -> wall xi
        self.lid = np.zeros(n * n)
        self.lid[ids[-1, 1:-1]] = -2.0 / h

        def stencil(offsets):
            """Full-grid operator evaluated at interior nodes: sum c * f(node + offset)."""
            r, c, val = [], [], []
            for (dj, di), coef in offsets.items():
                r.append(np.arange(m * m))
                c.append(ids[1 + dj : n - 1 + dj, 1 + di : n - 1 + di].ravel())
                val.append(np.full(m * m, coef))
            return sp.csr_matrix(
                (np.concatenate(val), (np.concatenate(r), np.concatenate(c))), shape=(m * m, n * n)
            )

        self.L = stencil(
            {(0, 0): -4 / h**2, (0, 1): 1 / h**2, (0, -1): 1 / h**2, (1, 0): 1 / h**2, (-1, 0): 1 / h**2}
        )
        self.Dx = stencil({(0, 1): 1 / (2 * h), (0, -1): -1 / (2 * h)})
        self.Dy = stencil({(1, 0): 1 / (2 * h), (-1, 0): -1 / (2 * h)})
        self.Dxm = stencil({(0, 0): 1 / h, (0, -1): -1 / h})  # backward
        self.Dxp = stencil({(0, 1): 1 / h, (0, 0): -1 / h})  # forward
        self.Dym = stencil({(0, 0): 1 / h, (-1, 0): -1 / h})
        self.Dyp = stencil({(1, 0): 1 / h, (0, 0): -1 / h})
        self.Lp = (self.L @ self.P).tocsr()


def _residual_and_jacobian(ops: _Operators, z: np.ndarray, re: float, scheme: str, jacobian: bool = True):
    k = ops.size
    psi_i, xi_i = z[:k], z[k:]
    psi_f = ops.P @ psi_i
    xi_f = ops.P @ xi_i + ops.B @ psi_i + ops.lid
    u = ops.Dy @ psi_f
    v = -(ops.Dx @ psi_f)
    if scheme == 'central':
        ax_op = sp.diags(u) @ ops.Dx
        ay_op = sp.diags(v) @ ops.Dy
        dxi_x = ops.Dx @ xi_f
        dxi_y = ops.Dy @ xi_f
    else:  # first-order upwind
        up, um, vp, vm = np.maximum(u, 0), np.minimum(u, 0), np.maximum(v, 0), np.minimum(v, 0)
        ax_op = sp.diags(up) @ ops.Dxm + sp.diags(um) @ ops.Dxp
        ay_op = sp.diags(vp) @ ops.Dym + sp.diags(vm) @ ops.Dyp
        hu, hv = (u > 0).astype(float), (v > 0).astype(float)
        dxi_x = hu * (ops.Dxm @ xi_f) + (1 - hu) * (ops.Dxp @ xi_f)
        dxi_y = hv * (ops.Dym @ xi_f) + (1 - hv) * (ops.Dyp @ xi_f)
    adv_op = (ax_op + ay_op).tocsr()

    f1 = ops.Lp @ psi_i + xi_i
    f2 = (ops.L @ xi_f) / re - adv_op @ xi_f
    F = np.concatenate([f1, f2])
    if not jacobian:
        return F, None

    # d(xi_full)/d(xi_int) = P, d(xi_full)/d(psi_int) = B; du/dpsi = Dy P, dv/dpsi = -Dx P
    diff_minus_adv = ops.L / re - adv_op
    j21 = diff_minus_adv @ ops.B - sp.diags(dxi_x) @ (ops.Dy @ ops.P) + sp.diags(dxi_y) @ (ops.Dx @ ops.P)
    j22 = diff_minus_adv @ ops.P
    J = sp.bmat([[ops.Lp, sp.identity(k)], [j21, j22]], format='csc')
    return F, J


def solve_steady(
    re: float,
    n: int = 129,
    scheme: str = 'central',
    initial: CavityResult | None = None,
    tol: float = 1e-12,
    max_iter: int = 30,
    verbose: bool = False,
) -> CavityResult:
    """Steady solution at Reynolds number `re` by damped Newton iteration.

    Args:
        initial: solution to start from (same n); otherwise the fluid at rest.
        tol:     relative convergence: max-norm of the residual below tol times
                 the residual of the fluid at rest (the lid forcing), a fixed
                 scale that grows like 1/h^3 on finer grids.
    """
    start = time.perf_counter()
    config = CavityConfig(re=re, n=n, scheme=scheme)
    ops = _Operators(n)
    k = ops.size
    if initial is not None and initial.psi.shape == (n, n):
        z = np.concatenate([ops.R @ initial.psi.ravel(), ops.R @ initial.xi.ravel()])
    else:
        z = np.zeros(2 * k)
    scale = float(np.abs(_residual_and_jacobian(ops, np.zeros(2 * k), re, scheme, jacobian=False)[0]).max())
    history = []
    converged = False
    for it in range(max_iter):
        F, J = _residual_and_jacobian(ops, z, re, scheme)
        norm = float(np.abs(F).max())
        history.append(norm)
        if verbose:
            print(f"  Re={re:g} N={n} it={it} |F|={norm:.3e} (relative {norm / scale:.1e})")
        if norm < tol * scale:
            converged = True
            break
        dz = spsolve(J, -F)
        # Backtracking line search on the residual norm
        step = 1.0
        while step > 1e-3:
            trial = z + step * dz
            trial_norm = float(
                np.abs(_residual_and_jacobian(ops, trial, re, scheme, jacobian=False)[0]).max()
            )
            if trial_norm < norm or step < 0.1:
                break
            step *= 0.5
        z = z + step * dz

    psi = (ops.P @ z[:k]).reshape(n, n)
    xi = (ops.P @ z[k:]).reshape(n, n)
    apply_wall_vorticity(xi, psi, config.h)
    u, v = velocities(psi, config.h)
    grid = np.linspace(0.0, 1.0, n)
    return CavityResult(
        config=config,
        x=grid,
        y=grid,
        psi=psi,
        xi=xi,
        u=u,
        v=v,
        t=np.inf,
        steps=len(history),
        converged=converged,
        wall_time=time.perf_counter() - start,
        history={'newton_residual': np.array(history)},
    )


def _interpolate(result: CavityResult, n: int) -> CavityResult:
    """Bicubic interpolation of psi and xi onto an n x n grid (initial guess only)."""
    from scipy.interpolate import RectBivariateSpline

    fine = np.linspace(0.0, 1.0, n)
    psi = RectBivariateSpline(result.y, result.x, result.psi)(fine, fine)
    xi = RectBivariateSpline(result.y, result.x, result.xi)(fine, fine)
    return CavityResult(result.config, fine, fine, psi, xi, psi * 0, psi * 0, np.inf, 0, False, 0.0)


def _plain_continuation(targets, n, scheme, re_step_factor, verbose):
    results: dict[float, CavityResult] = {}
    prev = solve_steady(1.0, n=n, scheme=scheme, verbose=verbose)
    re_now = 1.0
    for target in targets:
        while re_now < target:
            factor = re_step_factor
            while True:
                re_next = min(target, re_now * factor)
                trial = solve_steady(re_next, n=n, scheme=scheme, initial=prev, verbose=verbose)
                if trial.converged:
                    break
                factor = factor**0.5
                if factor < 1.01:
                    raise RuntimeError(f"continuation stalled at Re = {re_now} (N = {n})")
            prev, re_now = trial, re_next
        results[target] = prev
    return results


def continuation(
    re_targets,
    n: int = 129,
    scheme: str = 'central',
    re_step_factor: float = 2.5,
    verbose: bool = False,
    coarse_n: int = 129,
) -> dict[float, CavityResult]:
    """Solve the Re values in re_targets by continuation from Re = 1.

    Each solution is the initial guess of the next; consecutive Re differ by at
    most re_step_factor, and the step is halved (geometrically) whenever Newton
    fails to converge. On grids finer than coarse_n, the continuation runs on
    the grid of half the resolution (recursively) and its solution, interpolated,
    is the initial guess of a single Newton solve per target (grid sequencing);
    if that solve fails, the continuation is repeated on the fine grid.
    """
    targets = sorted(float(r) for r in re_targets)
    if n <= coarse_n or (n - 1) % 2:
        return _plain_continuation(targets, n, scheme, re_step_factor, verbose)
    coarse = continuation(targets, (n + 1) // 2, scheme, re_step_factor, verbose, coarse_n)
    results: dict[float, CavityResult] = {}
    for target in targets:
        res = solve_steady(
            target, n=n, scheme=scheme, initial=_interpolate(coarse[target], n), verbose=verbose
        )
        if not res.converged:
            return _plain_continuation(targets, n, scheme, re_step_factor, verbose)
        results[target] = res
    return results
