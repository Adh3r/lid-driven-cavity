"""Command line: python -m cavity --re 1000 --n 129 [--scheme upwind] [--transient] [--plot flow.png]"""

import argparse
import time

from .diagnostics import summary
from .solver import CavityConfig, solve
from .steady import continuation


def main() -> None:
    p = argparse.ArgumentParser(description='Lid-driven cavity flow (stream function-vorticity).')
    p.add_argument('--re', type=float, default=1000.0, help='Reynolds number U L / nu')
    p.add_argument('--n', type=int, default=129, help='grid nodes per side')
    p.add_argument(
        '--scheme',
        choices=['central', 'upwind'],
        default='central',
        help="advection scheme ('upwind' = the original first-order scheme)",
    )
    p.add_argument(
        '--transient',
        action='store_true',
        help='march in time to steady state instead of solving it directly (Newton)',
    )
    p.add_argument('--plot', metavar='PNG', help='save the flow field to this file')
    args = p.parse_args()
    start = time.perf_counter()

    if args.transient:
        result = solve(CavityConfig(re=args.re, n=args.n, scheme=args.scheme), verbose=True)
    else:
        result = continuation([args.re], n=args.n, scheme=args.scheme)[args.re]
    s = summary(result)
    print(
        f"Re = {s['Re']:g}, {s['N']}x{s['N']} grid, {s['scheme']} advection, "
        f"{'converged' if result.converged else 'NOT converged'} in {time.perf_counter() - start:.1f} s"
    )
    print(
        f"primary vortex: psi = {s['psi_min']:.6f} at ({s['x_c']:.4f}, {s['y_c']:.4f}), "
        f"vorticity {s['xi_c']:.4f}"
    )
    print(f"corner vortices: psi_BR = {s['psi_BR']:.3e}, psi_BL = {s['psi_BL']:.3e}")
    if 'u_max_err' in s:
        print(f"max |error| vs Ghia et al. (1982): u {s['u_max_err']:.4f}, v {s['v_max_err']:.4f}")
    if args.plot:
        import matplotlib.pyplot as plt

        from .plotting import plot_flow

        fig, ax = plt.subplots(figsize=(5.2, 5.2))
        plot_flow(result, ax)
        fig.savefig(args.plot, dpi=130, facecolor='white', bbox_inches='tight')
        print(f'figure written to {args.plot}')


if __name__ == '__main__':
    main()
