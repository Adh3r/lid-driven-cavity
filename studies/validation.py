"""Validation against Ghia, Ghia & Shin (1982) and Botella & Peyret (1998).

Solves Re = 100, 400 and 1000 directly (Newton) on 129^2 and 257^2 grids with
central advection, and with the original first-order upwind scheme on 129^2.
Writes docs/results/validation.csv and the README figures.

    python studies/validation.py
"""

import csv
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'src'))

from cavity import continuation, summary  # noqa: E402
from cavity.benchmarks import (  # noqa: E402
    BOTELLA_PEYRET_1000,
    GHIA_PRIMARY_VORTEX,
    GHIA_SECONDARY,
    GHIA_VORTICITY_CENTRE,
)
from cavity.plotting import plot_flow_panel, plot_profiles  # noqa: E402

RES = [100, 400, 1000]
OUT = os.path.join(ROOT, 'docs', 'results')
FIG = os.path.join(ROOT, 'docs', 'figures')


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(FIG, exist_ok=True)
    rows = []
    fine = None
    for n, scheme in ((129, 'upwind'), (129, 'central'), (257, 'central')):
        results = continuation(RES, n=n, scheme=scheme)
        if (n, scheme) == (257, 'central'):
            fine = results
        for re, res in results.items():
            s = summary(res)
            g_psi, g_x, g_y = GHIA_PRIMARY_VORTEX[re]
            rows.append(
                {
                    'Re': re,
                    'N': n,
                    'scheme': scheme,
                    'psi_min': s['psi_min'],
                    'x_c': s['x_c'],
                    'y_c': s['y_c'],
                    '|xi_c|': abs(s['xi_c']),
                    'psi_BR': s['psi_BR'],
                    'psi_BL': s['psi_BL'],
                    'u_max_err': s['u_max_err'],
                    'v_max_err': s['v_max_err'],
                    'ghia_psi_min': g_psi,
                    'ghia_x_c': g_x,
                    'ghia_y_c': g_y,
                    'ghia_|xi_c|': GHIA_VORTICITY_CENTRE[re],
                    'ghia_psi_BR': GHIA_SECONDARY[re]['BR'][0],
                    'ghia_psi_BL': GHIA_SECONDARY[re]['BL'][0],
                    'wall_s': res.wall_time,
                }
            )
            print(
                f"Re={re:5g} N={n} {scheme:8s} psi_min={s['psi_min']:.6f} (Ghia {g_psi}) "
                f"u_err={s['u_max_err']:.4f} v_err={s['v_max_err']:.4f}"
            )
    with open(os.path.join(OUT, 'validation.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"Botella & Peyret (1998), Re = 1000: psi_min = {BOTELLA_PEYRET_1000[0]}")

    plot_profiles(fine, os.path.join(FIG, 'centreline_profiles.png'))
    plot_flow_panel(fine, os.path.join(FIG, 'flow_fields.png'))
    print('figures written to docs/figures/')


if __name__ == '__main__':
    main()
