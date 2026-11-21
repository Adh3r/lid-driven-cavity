"""Figures: flow field, centre-line profiles against Ghia et al., grid convergence."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

from .benchmarks import GHIA_U, GHIA_V, GHIA_V_MISPRINTS, GHIA_X, GHIA_Y
from .solver import CavityResult

# Categorical series colours in fixed order (blue, orange, aqua, yellow) and a
# blue <-> red diverging map with a neutral grey midpoint for the signed vorticity.
SERIES = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100']
INK, MUTED, SURFACE = '#2b2b2a', '#77766f', '#fcfcfb'
DIVERGING = LinearSegmentedColormap.from_list(
    'blue_grey_red', ['#104281', '#2a78d6', '#86b6ef', '#f0efec', '#f19a99', '#e34948', '#8f1d1d']
)

# Stream-function contour levels of Ghia et al. (1982), Table III: the primary
# vortex (negative) and the much weaker counter-rotating corner vortices (positive).
PSI_LEVELS = [
    -0.1175,
    -0.115,
    -0.11,
    -0.1,
    -0.09,
    -0.07,
    -0.05,
    -0.03,
    -0.01,
    -1e-4,
    -1e-5,
    -1e-7,
    1e-8,
    1e-7,
    1e-6,
    1e-5,
    5e-5,
    1e-4,
    2.5e-4,
    5e-4,
    1e-3,
    1.5e-3,
    3e-3,
]


def _style(ax):
    ax.set_facecolor(SURFACE)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=INK)
    ax.grid(True, color='#e4e3df', linewidth=0.6)
    ax.set_axisbelow(True)


def plot_flow(result: CavityResult, ax=None, vort_limit: float = 5.0):
    """Vorticity (filled, diverging) with streamlines (psi contours)."""
    if ax is None:
        _, ax = plt.subplots(figsize=(5.2, 5.2))
    X, Y = np.meshgrid(result.x, result.y)
    # The lid corners are singular (u jumps from 0 to 1), so |xi| is unbounded
    # there; the colour scale is clipped to show the interior flow.
    xi = np.clip(result.xi, -vort_limit, vort_limit)
    cf = ax.contourf(
        X,
        Y,
        xi,
        levels=np.linspace(-vort_limit, vort_limit, 41),
        cmap=DIVERGING,
        norm=TwoSlopeNorm(0.0, -vort_limit, vort_limit),
        extend='both',
    )
    neg = sorted(lv for lv in PSI_LEVELS if lv < 0)
    pos = sorted(lv for lv in PSI_LEVELS if lv > 0)
    ax.contour(X, Y, result.psi, levels=neg, colors=INK, linewidths=0.6, negative_linestyles='solid')
    ax.contour(X, Y, result.psi, levels=pos, colors=INK, linewidths=0.6, linestyles='dashed')
    ax.set_aspect('equal')
    ax.set_xticks([0, 0.5, 1])
    ax.set_yticks([0, 0.5, 1])
    ax.set_xlabel('x / L')
    ax.set_ylabel('y / L')
    ax.set_title(f'Re = {result.config.re:g}', color=INK)
    return cf


def plot_profiles(results: dict, path: str | None = None):
    """u(x=0.5, y) and v(x, y=0.5) of several Re against Ghia et al. (1982)."""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.8))
    for k, (re, res) in enumerate(sorted(results.items())):
        c = SERIES[k % len(SERIES)]
        mid = res.config.n // 2
        a1.plot(res.u[:, mid], res.y, color=c, linewidth=2, label=f'Re = {re:g}, this solver')
        a2.plot(res.x, res.v[mid, :], color=c, linewidth=2, label=f'Re = {re:g}, this solver')
        key = int(re)
        if key in GHIA_U:
            a1.plot(
                GHIA_U[key], GHIA_Y, 'o', mfc=SURFACE, mec=c, ms=8, mew=1.6, label=f'Re = {re:g}, Ghia et al.'
            )
            keep = np.ones(len(GHIA_X), bool)
            keep[GHIA_V_MISPRINTS.get(key, [])] = False
            a2.plot(
                GHIA_X[keep],
                np.array(GHIA_V[key])[keep],
                'o',
                mfc=SURFACE,
                mec=c,
                ms=8,
                mew=1.6,
                label=f'Re = {re:g}, Ghia et al.',
            )
    a1.set_xlabel('u / U on the vertical centre line')
    a1.set_ylabel('y / L')
    a2.set_xlabel('x / L')
    a2.set_ylabel('v / U on the horizontal centre line')
    for ax in (a1, a2):
        _style(ax)
    a1.legend(frameon=False, fontsize=8, loc='lower right')
    fig.suptitle(
        f'Centre-line velocity profiles ({next(iter(results.values())).config.n}² grid) '
        'vs Ghia, Ghia & Shin (1982)',
        color=INK,
    )
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=130, facecolor='white')
        plt.close(fig)
    return fig


def plot_flow_panel(results: dict, path: str | None = None):
    """Streamlines and vorticity for several Re side by side, one shared colour bar."""
    fig, axes = plt.subplots(1, len(results), figsize=(4.4 * len(results), 4.9))
    axes = np.atleast_1d(axes)
    cf = None
    for ax, (_, res) in zip(axes, sorted(results.items()), strict=True):
        cf = plot_flow(res, ax)
    fig.subplots_adjust(bottom=0.2, wspace=0.25)
    cax = fig.add_axes([0.25, 0.07, 0.5, 0.03])
    cb = fig.colorbar(cf, cax=cax, orientation='horizontal', ticks=[-5, -2.5, 0, 2.5, 5])
    cb.set_label('vorticity ξ L / U (clipped at ±5; solid lines ψ < 0, dashed ψ > 0)', color=INK)
    if path:
        fig.savefig(path, dpi=130, facecolor='white', bbox_inches='tight')
        plt.close(fig)
    return fig
