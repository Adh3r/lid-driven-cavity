"""Lid-driven cavity flow: stream function-vorticity finite-difference solver."""

from .diagnostics import centreline_u, centreline_v, summary, vortex_centre
from .solver import CavityConfig, CavityResult, solve
from .steady import continuation, solve_steady

__all__ = [
    'CavityConfig',
    'CavityResult',
    'centreline_u',
    'centreline_v',
    'continuation',
    'solve',
    'solve_steady',
    'summary',
    'vortex_centre',
]
__version__ = '1.0.0'
