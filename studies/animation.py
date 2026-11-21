"""Spin-up of the cavity flow at Re = 1000 (time marching, 129 x 129 grid).

Writes docs/figures/spin_up.gif (vorticity and streamlines while the primary
vortex forms) and docs/figures/transient_history.png (residual and kinetic
energy against time).

    python studies/animation.py
"""

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'src'))

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402

from cavity import CavityConfig, CavityResult, solve  # noqa: E402
from cavity.plotting import INK, SERIES, _style, plot_flow  # noqa: E402
from cavity.solver import velocities  # noqa: E402

FIG = os.path.join(ROOT, 'docs', 'figures')


def main(re: float = 1000.0, n: int = 129, t_end: float = 40.0) -> None:
    frames = []
    config = CavityConfig(re=re, n=n, t_max=t_end)
    res = solve(
        config, callback=lambda t, psi, xi: frames.append((t, psi.copy(), xi.copy())), callback_every=0.5
    )
    print(f'{len(frames)} frames, t = {res.t:.1f}, {res.wall_time:.0f} s')

    grid = np.linspace(0, 1, n)
    fig, ax = plt.subplots(figsize=(4.6, 4.6))

    def draw(k):
        ax.clear()
        t, psi, xi = frames[k]
        u, v = velocities(psi, config.h)
        snap = CavityResult(config, grid, grid, psi, xi, u, v, t, 0, False, 0.0)
        plot_flow(snap, ax)
        ax.set_title(f'Re = {re:g}, t U / L = {t:4.1f}', color=INK)

    anim = FuncAnimation(fig, draw, frames=len(frames))
    anim.save(os.path.join(FIG, 'spin_up.gif'), writer=PillowWriter(fps=8), dpi=72)
    plt.close(fig)

    h = res.history
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(6.4, 5.6), sharex=True)
    a1.semilogy(h['t'], h['residual'], color=SERIES[0], lw=2)
    a1.set_ylabel('max |∂ξ/∂t|')
    a2.plot(h['t'], h['kinetic_energy'], color=SERIES[1], lw=2)
    a2.set_ylabel('kinetic energy E / (ρ U² L²)')
    a2.set_xlabel('time t U / L')
    for a in (a1, a2):
        _style(a)
    fig.suptitle(f'Transient from rest, Re = {re:g} ({n}² grid)', color=INK)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, 'transient_history.png'), dpi=130, facecolor='white')
    print('figures written to docs/figures/')


if __name__ == '__main__':
    main()
