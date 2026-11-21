"""Lid-driven cavity with a square obstacle that does not touch the walls.

The obstacle surface is a streamline, psi = psi_b, but psi_b is not known in
advance: it differs from the value on the cavity walls (0) whenever fluid
circulates between the obstacle and the walls. It is fixed by requiring the
pressure to be single-valued around the obstacle: the closed integral of
grad(p) . ds vanishes on any loop C in the fluid that encloses it. With the
steady momentum equation in rotational form,

    grad(p + |u|^2 / 2) = u x xi e_z - (1/Re) curl(xi e_z),

the condition becomes

    closed integral over C of (v xi - xi_y / Re) dx + (-u xi + xi_x / Re) dy = 0,

which is added as one extra equation for the extra unknown psi_b in the Newton
system of steady.py. C is a square loop at a fixed distance (1/16) from the
obstacle, through fluid nodes only. Evaluating the condition on the obstacle
surface itself, as d(xi)/dn = (xi_fluid - xi_wall)/h, would be inconsistent:
Thom's wall vorticity has an O(h) error, so the derivative has an O(1) error
that does not vanish as the grid is refined. Setting psi_b = 0 instead (the obstacle variant of the
original course project) treats the obstacle as if it were connected to the
walls and generally gives a pressure that is not single-valued.

The obstacle is a rectangle of grid nodes; Thom's formula gives the vorticity
on its faces (the average of the two face formulas at its corners).
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve


@dataclass
class ObstacleResult:
    re: float
    n: int
    x: np.ndarray
    y: np.ndarray
    psi: np.ndarray
    xi: np.ndarray
    solid: np.ndarray  # bool mask of obstacle nodes
    psi_body: float
    pressure_circulation: float  # closed integral of grad(p) . ds on the loop around the obstacle
    converged: bool
    wall_time: float


class _ObstacleOperators:
    def __init__(self, n: int, box: tuple[float, float, float, float]):
        h = 1.0 / (n - 1)
        x0, x1, y0, y1 = box
        i0, i1 = int(round(x0 / h)), int(round(x1 / h))
        j0, j1 = int(round(y0 / h)), int(round(y1 / h))
        if not (1 < i0 < i1 < n - 2 and 1 < j0 < j1 < n - 2):
            raise ValueError("the obstacle must lie strictly inside the cavity, away from the walls")
        self.n, self.h, self.box_idx = n, h, (i0, i1, j0, j1)
        ids = np.arange(n * n).reshape(n, n)
        solid = np.zeros((n, n), bool)
        solid[j0 : j1 + 1, i0 : i1 + 1] = True
        fluid = np.zeros((n, n), bool)
        fluid[1:-1, 1:-1] = True
        fluid &= ~solid
        self.solid, self.fluid = solid, fluid
        fj, fi = np.nonzero(fluid)
        self.fj, self.fi = fj, fi
        k = len(fj)
        self.k = k
        fidx = ids[fj, fi]
        self.P = sp.csr_matrix((np.ones(k), (fidx, np.arange(k))), shape=(n * n, k))
        self.R = self.P.T.tocsr()
        self.E = sp.csr_matrix(
            (np.ones(solid.sum()), (ids[solid], np.zeros(solid.sum(), int))), shape=(n * n, 1)
        )

        # Wall vorticity (Thom): rows on the outer walls and on the obstacle surface.
        rows, cols, vals = [], [], []
        rows_b, vals_b = [], []
        for j in range(1, n - 1):
            rows += [ids[j, 0], ids[j, -1]]
            cols += [ids[j, 1], ids[j, -2]]
            vals += [-2 / h**2, -2 / h**2]
        for i in range(1, n - 1):
            rows += [ids[0, i], ids[-1, i]]
            cols += [ids[1, i], ids[-2, i]]
            vals += [-2 / h**2, -2 / h**2]
        # obstacle faces: (node, outward fluid neighbour)
        self.faces = []
        for i in range(i0, i1 + 1):
            self.faces += [((j0, i), (j0 - 1, i)), ((j1, i), (j1 + 1, i))]
        for j in range(j0, j1 + 1):
            self.faces += [((j, i0), (j, i0 - 1)), ((j, i1), (j, i1 + 1))]
        count = {}
        for node, _ in self.faces:
            count[node] = count.get(node, 0) + 1  # 2 at the corners
        for node, nb in self.faces:
            w = 1.0 / count[node]
            rows.append(ids[node])
            cols.append(ids[nb])
            vals.append(-2 / h**2 * w)
            rows_b.append(ids[node])
            vals_b.append(2 / h**2 * w)
        self.Bfull = sp.csr_matrix((vals, (rows, cols)), shape=(n * n, n * n))
        self.B = (self.Bfull @ self.P).tocsr()
        self.Bb = sp.csr_matrix((vals_b, (rows_b, np.zeros(len(rows_b), int))), shape=(n * n, 1))
        self.lid = np.zeros(n * n)
        self.lid[ids[-1, 1:-1]] = -2 / h

        def stencil(offsets):
            r, c, v = [], [], []
            for (dj, di), coef in offsets.items():
                r.append(np.arange(k))
                c.append(ids[fj + dj, fi + di])
                v.append(np.full(k, coef))
            return sp.csr_matrix(
                (np.concatenate(v), (np.concatenate(r), np.concatenate(c))), shape=(k, n * n)
            )

        self.L = stencil(
            {(0, 0): -4 / h**2, (0, 1): 1 / h**2, (0, -1): 1 / h**2, (1, 0): 1 / h**2, (-1, 0): 1 / h**2}
        )
        self.Dx = stencil({(0, 1): 1 / (2 * h), (0, -1): -1 / (2 * h)})
        self.Dy = stencil({(1, 0): 1 / (2 * h), (-1, 0): -1 / (2 * h)})

        # Pressure loop C: a square at distance d from the obstacle, traversed
        # counter-clockwise, trapezoidal weights. Each node gets weights for the
        # dx terms (bottom +, top -) and the dy terms (right +, left -).
        d = max(2, int(round(0.0625 / h)))
        a0, a1, b0, b1 = i0 - d, i1 + d, j0 - d, j1 + d
        if not (a0 >= 2 and b0 >= 2 and a1 <= n - 3 and b1 <= n - 3):
            raise ValueError("the obstacle is too close to the walls for the pressure loop")
        wx = np.zeros(n * n)
        wy = np.zeros(n * n)
        for i in range(a0, a1 + 1):
            w = h * (0.5 if i in (a0, a1) else 1.0)
            wx[ids[b0, i]] += w
            wx[ids[b1, i]] -= w
        for j in range(b0, b1 + 1):
            w = h * (0.5 if j in (b0, b1) else 1.0)
            wy[ids[j, a1]] += w
            wy[ids[j, a0]] -= w
        self.wx, self.wy = wx, wy
        # derivatives on the full grid (central), used only at loop nodes
        full_dx = sp.diags([-1 / (2 * h), 1 / (2 * h)], [-1, 1], shape=(n * n, n * n)).tocsr()
        full_dy = sp.diags([-1 / (2 * h), 1 / (2 * h)], [-n, n], shape=(n * n, n * n)).tocsr()
        self.fdx, self.fdy = full_dx, full_dy


def _pressure_loop(ops: _ObstacleOperators, psi_full, xi_full, re):
    """Closed integral of grad(p) . ds on the loop, and its gradients with
    respect to xi_full and psi_full (row vectors)."""
    u = ops.fdy @ psi_full
    v = -(ops.fdx @ psi_full)
    xx, xy = ops.fdx @ xi_full, ops.fdy @ xi_full
    value = float(ops.wx @ (v * xi_full - xy / re) + ops.wy @ (-u * xi_full + xx / re))
    g_xi = ops.wx * v - (ops.fdy.T @ ops.wx) / re - ops.wy * u + (ops.fdx.T @ ops.wy) / re
    g_psi = -(ops.fdx.T @ (ops.wx * xi_full)) - (ops.fdy.T @ (ops.wy * xi_full))
    return value, g_xi, g_psi


def _system(ops: _ObstacleOperators, z, re, free_body: bool, psi_body_fixed: float):
    k = ops.k
    psi_f, xi_f = z[:k], z[k : 2 * k]
    psi_b = z[2 * k] if free_body else psi_body_fixed
    psi_full = ops.P @ psi_f + (ops.E @ np.array([psi_b]))
    xi_full = ops.P @ xi_f + ops.B @ psi_f + (ops.Bb @ np.array([psi_b])) + ops.lid
    u = ops.Dy @ psi_full
    v = -(ops.Dx @ psi_full)
    adv = (sp.diags(u) @ ops.Dx + sp.diags(v) @ ops.Dy).tocsr()
    dxx, dxy = ops.Dx @ xi_full, ops.Dy @ xi_full
    f1 = ops.L @ psi_full + xi_f
    f2 = (ops.L @ xi_full) / re - adv @ xi_full
    op = ops.L / re - adv
    LP = (ops.L @ ops.P).tocsr()
    j11, j12 = LP, sp.identity(k)
    j21 = op @ ops.B - sp.diags(dxx) @ (ops.Dy @ ops.P) + sp.diags(dxy) @ (ops.Dx @ ops.P)
    j22 = op @ ops.P
    blocks_F = [f1, f2]
    if not free_body:
        return np.concatenate(blocks_F), sp.bmat([[j11, j12], [j21, j22]], format='csc'), xi_full, psi_full
    j13 = ops.L @ ops.E
    j23 = op @ ops.Bb - sp.diags(dxx) @ (ops.Dy @ ops.E) + sp.diags(dxy) @ (ops.Dx @ ops.E)
    f3, g_xi, g_psi = _pressure_loop(ops, psi_full, xi_full, re)
    j31 = (g_xi @ ops.B + g_psi @ ops.P).reshape(1, -1)
    j32 = (g_xi @ ops.P).reshape(1, -1)
    j33 = np.array([[g_xi @ (ops.Bb.toarray().ravel()) + g_psi @ (ops.E.toarray().ravel())]])
    f3 = np.array([f3])
    J = sp.bmat([[j11, j12, j13], [j21, j22, j23], [j31, j32, j33]], format='csc')
    return np.concatenate([f1, f2, f3]), J, xi_full, psi_full


def solve_obstacle(
    re: float,
    n: int = 129,
    box=(0.375, 0.625, 0.375, 0.625),
    psi_body: float | None = None,
    tol: float = 1e-12,
    max_iter: int = 40,
    initial: ObstacleResult | None = None,
) -> ObstacleResult:
    """Steady flow around a rectangular obstacle box = (x0, x1, y0, y1).

    psi_body=None solves for the obstacle's stream function value from the
    single-valued pressure condition; a number fixes it (psi_body=0 reproduces
    the original project's assumption).
    """
    start = time.perf_counter()
    ops = _ObstacleOperators(n, box)
    free = psi_body is None
    k = ops.k
    size = 2 * k + (1 if free else 0)
    if initial is not None and initial.n == n:
        z = np.concatenate(
            [ops.R @ initial.psi.ravel(), ops.R @ initial.xi.ravel()]
            + ([np.array([initial.psi_body])] if free else [])
        )
    else:
        z = np.zeros(size)
    fixed = 0.0 if free else float(psi_body)
    scale = float(np.abs(_system(ops, np.zeros(size), re, free, fixed)[0]).max())
    converged = False
    for _ in range(max_iter):
        F, J, _, _ = _system(ops, z, re, free, fixed)
        norm = float(np.abs(F).max())
        if norm < tol * scale:
            converged = True
            break
        dz = spsolve(J, -F)
        step = 1.0
        while step > 0.05 and float(np.abs(_system(ops, z + step * dz, re, free, fixed)[0]).max()) >= norm:
            step *= 0.5
        z = z + step * dz
    _, _, xi_full, psi_full = _system(ops, z, re, free, fixed)
    psi_b = float(z[2 * k]) if free else fixed
    circulation = _pressure_loop(ops, psi_full, xi_full, re)[0]
    grid = np.linspace(0, 1, n)
    return ObstacleResult(
        re=re,
        n=n,
        x=grid,
        y=grid,
        psi=psi_full.reshape(n, n),
        xi=xi_full.reshape(n, n),
        solid=ops.solid,
        psi_body=psi_b,
        pressure_circulation=circulation,
        converged=converged,
        wall_time=time.perf_counter() - start,
    )


def obstacle_continuation(
    re: float,
    n: int = 129,
    box=(0.375, 0.625, 0.375, 0.625),
    psi_body: float | None = None,
    factor: float = 2.5,
) -> ObstacleResult:
    """Reach Re by continuation from Re = 1."""
    res = solve_obstacle(1.0, n, box, psi_body)
    current = 1.0
    while current < re:
        current = min(re, current * factor)
        res = solve_obstacle(current, n, box, psi_body, initial=res)
        if not res.converged:
            raise RuntimeError(f"no convergence at Re = {current}")
    return res
