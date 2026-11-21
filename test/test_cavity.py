"""Verification and validation tests (small grids, a few seconds each)."""

import numpy as np
import pytest

from cavity import CavityConfig, continuation, solve, solve_steady, summary, vortex_centre
from cavity.benchmarks import GHIA_PRIMARY_VORTEX
from cavity.solver import PoissonSolver, kinetic_energy, velocities

# -----------------------------------------------------------------------------
# Verification: discrete operators
# -----------------------------------------------------------------------------


@pytest.mark.parametrize('n', [17, 33])
def test_poisson_solver_is_second_order(n):
    # Manufactured solution psi = sin(pi x) sin(pi y): lap(psi) = -2 pi^2 psi = -xi.
    def error(m):
        x = np.linspace(0, 1, m)
        X, Y = np.meshgrid(x, x)
        exact = np.sin(np.pi * X) * np.sin(np.pi * Y)
        return np.abs(PoissonSolver(m).solve(2 * np.pi**2 * exact) - exact).max()

    order = np.log2(error(n) / error(2 * n - 1))
    assert order == pytest.approx(2.0, abs=0.1)


def test_velocities_satisfy_continuity_and_no_slip():
    n = 33
    x = np.linspace(0, 1, n)
    X, Y = np.meshgrid(x, x)
    psi = (X * (1 - X) * Y * (1 - Y)) ** 2  # psi = 0 and grad(psi) = 0 on the walls
    u, v = velocities(psi, 1 / (n - 1))
    h = 1 / (n - 1)
    div = (u[1:-1, 2:] - u[1:-1, :-2] + v[2:, 1:-1] - v[:-2, 1:-1]) / (2 * h)
    assert np.abs(div[1:-1, 1:-1]).max() < 1e-12  # discrete continuity holds exactly
    assert (u[-1, :] == 1).all() and (u[0, :] == 0).all() and (v[:, 0] == 0).all()


def test_kinetic_energy_of_uniform_flow_is_one_half():
    n = 9
    assert kinetic_energy(np.ones((n, n)), np.zeros((n, n)), 1 / (n - 1)) == pytest.approx(0.5)


# -----------------------------------------------------------------------------
# The two solvers share the discretisation
# -----------------------------------------------------------------------------


def test_newton_converges_quadratically():
    res = solve_steady(100, n=33)
    r = res.history['newton_residual']
    assert res.converged
    ratios = [np.log(r[i + 1]) / np.log(r[i]) for i in range(len(r) - 3, len(r) - 1) if r[i] < 1]
    assert all(q > 1.6 for q in ratios)  # error_{k+1} ~ error_k^2


@pytest.mark.parametrize('scheme', ['central', 'upwind'])
def test_time_marching_reaches_the_newton_steady_state(scheme):
    n, re = 25, 100
    marched = solve(CavityConfig(re=re, n=n, scheme=scheme, tol=1e-9))
    direct = solve_steady(re, n=n, scheme=scheme)
    assert marched.converged
    assert np.abs(marched.psi - direct.psi).max() < 1e-7


# -----------------------------------------------------------------------------
# Validation against Ghia, Ghia & Shin (1982)
# -----------------------------------------------------------------------------


def test_re100_matches_ghia():
    s = summary(continuation([100], n=65)[100.0])
    psi_g, xg, yg = GHIA_PRIMARY_VORTEX[100]
    assert s['psi_min'] == pytest.approx(psi_g, rel=0.01)
    assert (s['x_c'], s['y_c']) == pytest.approx((xg, yg), abs=0.01)
    assert s['u_max_err'] < 0.01 and s['v_max_err'] < 0.01


def test_central_scheme_is_second_order_and_upwind_first_order():
    # Observed order of psi_min at Re = 400 on grids 33, 65, 129 (in the
    # asymptotic range; 17-33-65 is not yet).
    def order(scheme):
        f = [vortex_centre(continuation([400], n=n, scheme=scheme)[400.0])[0] for n in (33, 65, 129)]
        return np.log2(abs(f[0] - f[1]) / abs(f[1] - f[2]))

    assert order('central') == pytest.approx(2.0, abs=0.2)
    assert order('upwind') == pytest.approx(1.0, abs=0.2)


def test_upwind_numerical_viscosity_weakens_the_vortex_at_high_re():
    central = vortex_centre(continuation([1000], n=129)[1000.0])[0]
    upwind = vortex_centre(continuation([1000], n=129, scheme='upwind')[1000.0])[0]
    ghia = GHIA_PRIMARY_VORTEX[1000][0]
    assert abs(upwind - ghia) > 3 * abs(central - ghia)


# -----------------------------------------------------------------------------
# Robustness: the cases the original code could not run
# -----------------------------------------------------------------------------


def test_low_reynolds_is_stable_with_the_adaptive_time_step():
    # The original fixed dt = 1e-3 diverged at Re = 10 (diffusive limit h^2 Re / 4 = 2.5e-4).
    res = solve(CavityConfig(re=10, n=33))
    assert res.converged and np.isfinite(res.psi).all()


def test_creeping_flow_is_symmetric_and_inertia_breaks_it_linearly():
    # Stokes flow (Re -> 0) is symmetric about x = 1/2: psi(x, y) = psi(1 - x, y).
    # Inertia breaks the symmetry in proportion to Re.
    def asymmetry(re):
        psi = solve_steady(re, n=33).psi
        return np.abs(psi - psi[:, ::-1]).max() / np.abs(psi).max()

    assert asymmetry(1e-6) < 1e-7
    assert asymmetry(1e-3) / asymmetry(1e-6) == pytest.approx(1e3, rel=0.01)


def test_central_with_forward_euler_is_rejected():
    with pytest.raises(ValueError):
        CavityConfig(re=100, scheme='central', integrator='euler')


# -----------------------------------------------------------------------------
# Extension: square obstacle
# -----------------------------------------------------------------------------


def test_obstacle_stream_function_makes_the_pressure_single_valued():
    from cavity.obstacle import obstacle_continuation

    free = obstacle_continuation(100, n=65)
    fixed = obstacle_continuation(100, n=65, psi_body=0.0)
    assert free.converged and fixed.converged
    assert abs(free.pressure_circulation) < 1e-10
    assert abs(fixed.pressure_circulation) > 1e-2  # psi_b = 0 leaves a pressure jump
    assert -0.1 < free.psi_body < 0  # the obstacle sits inside the clockwise primary vortex


def test_obstacle_stream_function_converges_at_second_order():
    from cavity.obstacle import obstacle_continuation

    f = [obstacle_continuation(400, n=n).psi_body for n in (33, 65, 129)]
    assert np.log2(abs(f[0] - f[1]) / abs(f[1] - f[2])) == pytest.approx(2.0, abs=0.3)
