"""Post-processing: centre-line profiles, vortex centres and benchmark errors."""

from __future__ import annotations

import numpy as np
from scipy.interpolate import RectBivariateSpline
from scipy.optimize import minimize

from .benchmarks import GHIA_U, GHIA_V, GHIA_V_MISPRINTS, GHIA_X, GHIA_Y
from .solver import CavityResult


def _spline(result: CavityResult, field: np.ndarray) -> RectBivariateSpline:
    # field is indexed [j, i] = (y, x)
    return RectBivariateSpline(result.y, result.x, field, kx=3, ky=3)


def centreline_u(result: CavityResult, y: np.ndarray = GHIA_Y) -> np.ndarray:
    """u(x = 0.5, y), interpolated with a bicubic spline."""
    return _spline(result, result.u).ev(y, np.full_like(y, 0.5))


def centreline_v(result: CavityResult, x: np.ndarray = GHIA_X) -> np.ndarray:
    """v(x, y = 0.5), interpolated with a bicubic spline."""
    return _spline(result, result.v).ev(np.full_like(x, 0.5), x)


def vortex_centre(
    result: CavityResult, region=(0.0, 1.0, 0.0, 1.0), kind: str = 'min'
) -> tuple[float, float, float]:
    """Extremum of psi in region (x0, x1, y0, y1), located to sub-grid accuracy.

    Returns (psi_extremum, x, y). kind='min' for the primary vortex (clockwise,
    psi < 0), 'max' for the counter-rotating corner vortices.
    """
    x0, x1, y0, y1 = region
    sign = 1.0 if kind == 'min' else -1.0
    ix = (result.x >= x0) & (result.x <= x1)
    iy = (result.y >= y0) & (result.y <= y1)
    sub = sign * result.psi[np.ix_(iy, ix)]
    j, i = np.unravel_index(np.argmin(sub), sub.shape)
    guess = (result.x[ix][i], result.y[iy][j])
    spline = _spline(result, result.psi)
    res = minimize(
        lambda p: sign * spline.ev(p[1], p[0]),
        guess,
        method='Nelder-Mead',
        options={'xatol': 1e-7, 'fatol': 1e-12},
    )
    x, y = res.x
    return float(spline.ev(y, x)), float(x), float(y)


def vorticity_at(result: CavityResult, x: float, y: float) -> float:
    return float(_spline(result, result.xi).ev(y, x))


def summary(result: CavityResult) -> dict:
    """Primary and corner vortices, vorticity at the primary centre, and errors
    against Ghia et al. (1982) when the Reynolds number is tabulated."""
    re = result.config.re
    psi_min, xc, yc = vortex_centre(result, kind='min')
    out = {
        'Re': re,
        'N': result.config.n,
        'scheme': result.config.scheme,
        'psi_min': psi_min,
        'x_c': xc,
        'y_c': yc,
        'xi_c': vorticity_at(result, xc, yc),
        'psi_BR': vortex_centre(result, (0.6, 1.0, 0.0, 0.4), 'max')[0],
        'psi_BL': vortex_centre(result, (0.0, 0.4, 0.0, 0.4), 'max')[0],
    }
    key = int(re) if float(re).is_integer() else None
    if key in GHIA_U:
        out['u_max_err'] = float(np.abs(centreline_u(result) - np.array(GHIA_U[key])).max())
        keep = np.ones(len(GHIA_X), dtype=bool)
        keep[GHIA_V_MISPRINTS.get(key, [])] = False
        err_v = np.abs(centreline_v(result) - np.array(GHIA_V[key]))[keep]
        out['v_max_err'] = float(err_v.max())
    return out
