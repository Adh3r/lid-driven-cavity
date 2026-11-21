"""Cavity with a square obstacle (side 0.25, centred) - extension.

1. Grid convergence of the obstacle's stream function psi_b, fixed by the
   single-valued pressure condition (Re = 100 and 400, N = 33 ... 257).
2. Comparison with the assumption of the original project (psi_b = 0, the
   obstacle treated as if connected to the walls), which leaves a non-zero
   pressure jump around the obstacle.

Writes docs/results/obstacle.csv and docs/figures/obstacle.png.

    python studies/obstacle.py
"""

import csv
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'src'))

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

from cavity.obstacle import obstacle_continuation  # noqa: E402
from cavity.plotting import INK  # noqa: E402

BOX = (0.375, 0.625, 0.375, 0.625)


def main() -> None:
    rows = []
    for re in (100, 400):
        vals = []
        for n in (33, 65, 129, 257):
            r = obstacle_continuation(re, n=n, box=BOX)
            vals.append(r.psi_body)
            rows.append(
                {
                    'Re': re,
                    'N': n,
                    'psi_body': r.psi_body,
                    'pressure_jump': r.pressure_circulation,
                    'wall_s': r.wall_time,
                }
            )
            print(
                f"Re={re} N={n:3d} psi_b={r.psi_body:.7f} pressure jump={r.pressure_circulation:.1e}",
                flush=True,
            )
        orders = [np.log2(abs(vals[i] - vals[i + 1]) / abs(vals[i + 1] - vals[i + 2])) for i in range(2)]
        print(
            f"   observed order {np.round(orders, 2).tolist()}, "
            f"Richardson psi_b = {vals[-1] + (vals[-1] - vals[-2]) / 3:.6f}"
        )

    os.makedirs(os.path.join(ROOT, 'docs', 'results'), exist_ok=True)
    with open(os.path.join(ROOT, 'docs', 'results', 'obstacle.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    free = obstacle_continuation(400, n=257, box=BOX)
    fixed = obstacle_continuation(400, n=257, box=BOX, psi_body=0.0)
    print(
        f"Re=400 N=257: psi_b free {free.psi_body:.5f} (jump {free.pressure_circulation:.1e}); "
        f"psi_b = 0 jump {fixed.pressure_circulation:.4f}"
    )

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.9))
    titles = [f'ψ_body = {free.psi_body:.4f} from single-valued pressure', 'ψ_body = 0 (original assumption)']
    for ax, res, title in zip(axes, (free, fixed), titles, strict=True):
        X, Y = np.meshgrid(res.x, res.y)
        psi = np.ma.masked_where(res.solid, res.psi)
        levels = np.linspace(-0.11, -0.005, 15)
        ax.contour(X, Y, psi, levels=levels, colors=INK, linewidths=0.7, negative_linestyles='solid')
        ax.contour(
            X, Y, psi, levels=[1e-5, 1e-4, 5e-4, 1e-3], colors=INK, linewidths=0.7, linestyles='dashed'
        )
        x0, x1, y0, y1 = BOX
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, facecolor='#c3c2b7', edgecolor=INK, lw=1))
        ax.set_aspect('equal')
        ax.set_xticks([0, 0.5, 1])
        ax.set_yticks([0, 0.5, 1])
        ax.set_xlabel('x / L')
        ax.set_title(title, color=INK, fontsize=10)
    axes[0].set_ylabel('y / L')
    fig.suptitle('Lid-driven cavity with a square obstacle, Re = 400 (257² grid): streamlines', color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(
        os.path.join(ROOT, 'docs', 'figures', 'obstacle.png'), dpi=130, facecolor='white', bbox_inches='tight'
    )
    print('figure written to docs/figures/obstacle.png')


if __name__ == '__main__':
    main()
