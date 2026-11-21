"""Reference solutions for the lid-driven cavity (lid moving in +x at y = 1).

- U. Ghia, K. N. Ghia & C. T. Shin, "High-Re solutions for incompressible flow
  using the Navier-Stokes equations and a multigrid method", Journal of
  Computational Physics 48, 387-411 (1982). Table I (u on the vertical line
  x = 0.5), Table II (v on the horizontal line y = 0.5) and the primary vortex
  of Tables III-V (129 x 129 grid, second-order upwind with defect correction).
- O. Botella & R. Peyret, "Benchmark spectral results on the lid-driven cavity
  flow", Computers & Fluids 27(4), 421-433 (1998): spectral (Chebyshev)
  solution at Re = 1000, converged to about 7 digits. Their lid moves in -x,
  so the vortex abscissa is mirrored here (x -> 1 - x).
"""

import numpy as np

GHIA_Y = np.array(
    [
        1.0000,
        0.9766,
        0.9688,
        0.9609,
        0.9531,
        0.8516,
        0.7344,
        0.6172,
        0.5000,
        0.4531,
        0.2813,
        0.1719,
        0.1016,
        0.0703,
        0.0625,
        0.0547,
        0.0000,
    ]
)
GHIA_U = {
    100: [
        1.0,
        0.84123,
        0.78871,
        0.73722,
        0.68717,
        0.23151,
        0.00332,
        -0.13641,
        -0.20581,
        -0.21090,
        -0.15662,
        -0.10150,
        -0.06434,
        -0.04775,
        -0.04192,
        -0.03717,
        0.0,
    ],
    400: [
        1.0,
        0.75837,
        0.68439,
        0.61756,
        0.55892,
        0.29093,
        0.16256,
        0.02135,
        -0.11477,
        -0.17119,
        -0.32726,
        -0.24299,
        -0.14612,
        -0.10338,
        -0.09266,
        -0.08186,
        0.0,
    ],
    1000: [
        1.0,
        0.65928,
        0.57492,
        0.51117,
        0.46604,
        0.33304,
        0.18719,
        0.05702,
        -0.06080,
        -0.10648,
        -0.27805,
        -0.38289,
        -0.29730,
        -0.22220,
        -0.20196,
        -0.18109,
        0.0,
    ],
}

GHIA_X = np.array(
    [
        1.0000,
        0.9688,
        0.9609,
        0.9531,
        0.9453,
        0.9063,
        0.8594,
        0.8047,
        0.5000,
        0.2344,
        0.2266,
        0.1563,
        0.0938,
        0.0781,
        0.0703,
        0.0625,
        0.0000,
    ]
)
GHIA_V = {
    100: [
        0.0,
        -0.05906,
        -0.07391,
        -0.08864,
        -0.10313,
        -0.16914,
        -0.22445,
        -0.24533,
        0.05454,
        0.17527,
        0.17507,
        0.16077,
        0.12317,
        0.10890,
        0.10091,
        0.09233,
        0.0,
    ],
    400: [
        0.0,
        -0.12146,
        -0.15663,
        -0.19254,
        -0.22847,
        -0.23827,
        -0.44993,
        -0.38598,
        0.05188,
        0.30174,
        0.30203,
        0.28124,
        0.22965,
        0.20920,
        0.19713,
        0.18360,
        0.0,
    ],
    1000: [
        0.0,
        -0.21388,
        -0.27669,
        -0.33714,
        -0.39188,
        -0.51550,
        -0.42665,
        -0.31966,
        0.02526,
        0.32235,
        0.33075,
        0.37095,
        0.32627,
        0.30353,
        0.29012,
        0.27485,
        0.0,
    ],
}

# Ghia Table II, Re = 400, x = 0.9063 lists v = -0.23827, which breaks the
# smooth profile between its neighbours (-0.22847 at x = 0.9453 and -0.44993 at
# x = 0.8594). Every other point of the profile is reproduced to within 0.004
# by the converged solutions here, which give v = -0.388 at that abscissa, so
# the entry is treated as a misprint and left out of the error metrics.
GHIA_V_MISPRINTS = {400: [5]}  # indices into GHIA_X

# Vorticity sign convention: this code uses xi = dv/dx - du/dy, so the
# clockwise primary vortex has xi < 0. Ghia et al. tabulate omega = du/dy - dv/dx
# (positive at the primary vortex); compare magnitudes.

# Primary vortex: stream function minimum and its location (x, y).
GHIA_PRIMARY_VORTEX = {
    100: (-0.103423, 0.6172, 0.7344),
    400: (-0.113909, 0.5547, 0.6055),
    1000: (-0.117929, 0.5313, 0.5625),
}
BOTELLA_PEYRET_1000 = (-0.1189366, 1 - 0.4692, 0.5652)

# Vorticity at the primary vortex centre (Ghia Table V).
GHIA_VORTICITY_CENTRE = {100: 3.16646, 400: 2.29469, 1000: 2.04968}

# Secondary corner vortices, Ghia Table V: (psi_max, x, y).
GHIA_SECONDARY = {
    100: {'BL': (1.74877e-6, 0.0313, 0.0391), 'BR': (1.25374e-5, 0.9453, 0.0625)},
    400: {'BL': (1.41951e-5, 0.0508, 0.0469), 'BR': (6.42352e-4, 0.8906, 0.1250)},
    1000: {'BL': (2.31129e-4, 0.0859, 0.0781), 'BR': (1.75102e-3, 0.8594, 0.1094)},
}
