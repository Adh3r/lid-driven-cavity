"""Grid convergence study and Richardson extrapolation.

For Re = 100 and Re = 1000, solves the steady problem on N = 33, 65, 129, 257
(and 513 with --fine) with central advection and with the original
first-order upwind scheme. For each, the primary-vortex stream function psi_min:

- observed order of accuracy p from three consecutive grids,
      p = log2(|f_4h - f_2h| / |f_2h - f_h|);
- Richardson extrapolation f_h + (f_h - f_2h) / (2^p - 1) with p = 2 (central),
  compared with the spectral solution of Botella & Peyret (1998) at Re = 1000.

Writes docs/results/grid_convergence.csv and docs/figures/grid_convergence.png.

    python studies/grid_convergence.py [--fine]
"""

import csv
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'src'))

import matplotlib.pyplot as plt  # noqa: E402

from cavity import continuation, vortex_centre  # noqa: E402
from cavity.benchmarks import BOTELLA_PEYRET_1000  # noqa: E402
from cavity.plotting import INK, MUTED, SERIES, _style  # noqa: E402


def main(fine: bool = False) -> None:
    grids = [33, 65, 129, 257] + ([513] if fine else [])
    rows = []
    for re in (100, 1000):
        for scheme in ('central', 'upwind'):
            for n in grids:
                if scheme == 'upwind' and n > 257:
                    continue
                res = continuation([re], n=n, scheme=scheme)[float(re)]
                psi, xc, yc = vortex_centre(res)
                rows.append(
                    {
                        'Re': re,
                        'scheme': scheme,
                        'N': n,
                        'h': 1 / (n - 1),
                        'psi_min': psi,
                        'x_c': xc,
                        'y_c': yc,
                        'wall_s': res.wall_time,
                    }
                )
                print(
                    f"Re={re:5d} {scheme:8s} N={n:4d} psi_min={psi:.7f} ({res.wall_time:.1f} s)", flush=True
                )

    out = os.path.join(ROOT, 'docs', 'results')
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, 'grid_convergence.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    print('\nObserved order and Richardson extrapolation of psi_min')
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    labels = {'central': 'second-order central', 'upwind': 'first-order upwind (original scheme)'}
    for k, (re, scheme) in enumerate(
        [(1000, 'central'), (1000, 'upwind'), (100, 'central'), (100, 'upwind')]
    ):
        sel = [r for r in rows if r['Re'] == re and r['scheme'] == scheme]
        h = np.array([r['h'] for r in sel])
        f = np.array([r['psi_min'] for r in sel])
        orders = [np.log2(abs(f[i] - f[i + 1]) / abs(f[i + 1] - f[i + 2])) for i in range(len(f) - 2)]
        p = 2.0 if scheme == 'central' else 1.0
        extrap = f[-1] + (f[-1] - f[-2]) / (2**p - 1)
        print(
            f"Re={re} {scheme}: observed orders {np.round(orders, 2).tolist()}, "
            f"Richardson (p={p:g}) psi_min = {extrap:.7f}"
        )
        if re == 1000:
            ref = BOTELLA_PEYRET_1000[0]
            if scheme == 'central':
                print(f"   Botella & Peyret (1998): {ref:.7f}  (difference {abs(extrap - ref):.1e})")
            ax.loglog(
                h,
                np.abs(f - ref),
                'o-',
                color=SERIES[k],
                lw=2,
                ms=8,
                mfc='white',
                mew=1.6,
                label=labels[scheme],
            )
    hh = np.array([1 / 32, 1 / 512])
    ax.loglog(hh, 0.6 * hh, color=MUTED, lw=1, ls='--')
    ax.loglog(hh, 1.5 * hh**2, color=MUTED, lw=1, ls=':')
    ax.text(1 / 32 * 1.08, 0.6 / 32, 'slope 1', color=MUTED, fontsize=9, va='center')
    ax.text(1 / 32 * 1.08, 1.5 / 32**2, 'slope 2', color=MUTED, fontsize=9, va='center')
    ax.set_xticks([1 / 512, 1 / 256, 1 / 128, 1 / 64, 1 / 32])
    ax.set_xticklabels(['1/512', '1/256', '1/128', '1/64', '1/32'])
    ax.minorticks_off()
    ax.set_xlim(1 / 700, 1 / 20)
    ax.set_xlabel('grid spacing h / L')
    ax.set_ylabel('|ψ_min − ψ_min (Botella & Peyret)|')
    ax.set_title('Grid convergence of the primary vortex, Re = 1000', color=INK)
    _style(ax)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(ROOT, 'docs', 'figures', 'grid_convergence.png'), dpi=130, facecolor='white')
    print('figure written to docs/figures/grid_convergence.png')


if __name__ == '__main__':
    main(fine='--fine' in sys.argv)
