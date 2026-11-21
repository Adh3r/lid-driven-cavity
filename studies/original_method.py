"""What the original course code got right and wrong, quantified.

Faithful headless replica of original/cavity_original.py (101 x 101 grid,
forward Euler with fixed dt = 1e-3, first-order upwind advection, Jacobi
iterations for the Poisson equation capped at 3000 per step, steady state when
max |xi^{n+1} - xi^n| < 1e-7, t <= 100), compared with Ghia et al. (1982) and
with the revised solver on the same grid.

    python studies/original_method.py
"""

import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'src'))

from cavity import continuation, summary  # noqa: E402
from cavity.benchmarks import GHIA_PRIMARY_VORTEX, GHIA_U, GHIA_Y  # noqa: E402


def original(re, n=101, dt=1e-3, t_final=100.0, steady_tol=1e-7, max_jacobi=3000, tol_jacobi=1e-5):
    h, nu = 1.0 / (n - 1), 1.0 / re
    psi, xi, u, v = (np.zeros((n, n)) for _ in range(4))
    steps, start = 0, time.perf_counter()
    while steps * dt < t_final:
        xi_old = xi.copy()
        xi[-1, 1:-1] = -2.0 * (psi[-2, 1:-1] + h) / h**2
        xi[0, 1:-1] = -2.0 * psi[1, 1:-1] / h**2
        xi[1:-1, 0] = -2.0 * psi[1:-1, 1] / h**2
        xi[1:-1, -1] = -2.0 * psi[1:-1, -2] / h**2
        c, w, e, s, nn = xi[1:-1, 1:-1], xi[1:-1, :-2], xi[1:-1, 2:], xi[:-2, 1:-1], xi[2:, 1:-1]
        uc, vc = u[1:-1, 1:-1], v[1:-1, 1:-1]
        with np.errstate(over='ignore', invalid='ignore'):
            lap = (e + w + nn + s - 4 * c) / h**2
            adv = (np.maximum(uc, 0) * (c - w) + np.minimum(uc, 0) * (e - c)) / h + (
                np.maximum(vc, 0) * (c - s) + np.minimum(vc, 0) * (nn - c)
            ) / h
            xi[1:-1, 1:-1] = c + dt * (nu * lap - adv)
        if not np.isfinite(xi).all():
            return {'status': f'diverged at step {steps} (t = {steps * dt:.3f})'}
        for k in range(max_jacobi):
            new = psi.copy()
            new[1:-1, 1:-1] = 0.25 * (
                psi[1:-1, 2:] + psi[1:-1, :-2] + psi[2:, 1:-1] + psi[:-2, 1:-1] + h**2 * xi[1:-1, 1:-1]
            )
            done = k % 50 == 0 and np.abs(new - psi).max() < tol_jacobi
            psi = new
            if done:
                break
        u[1:-1, 1:-1] = (psi[2:, 1:-1] - psi[:-2, 1:-1]) / (2 * h)
        v[1:-1, 1:-1] = -(psi[1:-1, 2:] - psi[1:-1, :-2]) / (2 * h)
        u[-1, :] = 1.0
        if steps % 50 == 0 and steps > 0 and np.abs(xi - xi_old).max() < steady_tol:
            break
        steps += 1
    y = np.linspace(0, 1, n)
    u_err = (
        np.abs(np.interp(GHIA_Y[::-1], y, u[:, n // 2])[::-1] - np.array(GHIA_U[re])).max()
        if re in GHIA_U
        else None
    )
    converged = steps * dt < t_final
    return {
        'status': 'converged' if converged else f'not steady at t = {t_final:g}',
        'psi_min': psi.min(),
        'u_max_err': u_err,
        'wall_s': time.perf_counter() - start,
    }


def main() -> None:
    print(
        f"{'Re':>6} | {'original code (101x101, upwind, dt = 1e-3)':52} | revised (101x101, central, Newton)"
    )
    for re in (10, 100, 400, 1000):
        o = original(re)
        line = f"{re:>6} | {o['status']:22}"
        if 'psi_min' in o:
            err = f", u err {o['u_max_err']:.3f}" if o['u_max_err'] is not None else ''
            line += f" psi_min {o['psi_min']:.4f}{err}"
        r = summary(continuation([re], n=101)[float(re)])
        rev = f"psi_min {r['psi_min']:.4f}"
        if 'u_max_err' in r:
            rev += f", u err {r['u_max_err']:.3f}"
        ghia = f" (Ghia {GHIA_PRIMARY_VORTEX[re][0]:.4f})" if re in GHIA_PRIMARY_VORTEX else ''
        print(f"{line:61} | {rev}{ghia}", flush=True)


if __name__ == '__main__':
    main()
