# Revision of the original course project

The course solver (`original/cavity_original.py`, January 2026) was re-run
headless (`studies/original_method.py`) and compared with Ghia et al. (1982).
The physics and the derivation were right; the numerics limited it to low
Reynolds numbers, and nothing in the original work showed it, because it was
never compared with a benchmark.

## What the original code gave (101 × 101 grid, dt = 10⁻³)

| Re | Original code | Max error in u (centre line) | ψ_min (Ghia) |
|---|---|---|---|
| 10 | **diverged** in 13 steps | — | — |
| 100 | converged | 0.010 | −0.1034 (−0.1034) |
| 400 | converged | 0.085 | −0.1061 (−0.1139) |
| 1000 | **no steady state** by t = 100 | 0.212 | −0.0979 (−0.1179) |

The revised solver on the same 101² grid gives centre-line errors of 0.004,
0.009 and 0.024, and on 257² ψ_min = −0.1181 at Re = 1000 (Botella & Peyret:
−0.1189). The Richardson extrapolation of 257² and 513² matches the spectral
value to 2 × 10⁻⁶.

## Issues and corrections

| # | Issue in the original | Effect | Correction |
|---|---|---|---|
| O1 | **First-order upwind advection.** Numerical viscosity of order \|u\|h/2, up to 5 times the physical viscosity at Re = 1000 on 101². | The simulated flow is far more viscous than intended: primary vortex 17 % weak at Re = 1000. Results look plausible but are wrong. | Second-order central advection (upwind kept as an option for comparison). Observed order 2.0 vs 0.95 (grid study). |
| O2 | **Fixed time step with explicit Euler.** Stability needs dt ≤ h²Re/4 (diffusion) and a CFL limit (advection). | Diverges at Re = 10; the stated range "Re = 10⁻² to 10³" was not reachable. | Adaptive time step from both limits; SSP-RK3 (central advection is unstable with forward Euler at any dt); direct Newton solver for steady states at any Re. |
| O3 | **Steady-state test on max \|Δξ\| per step.** It depends on dt and on the Jacobi tolerance. | At Re = 1000 the criterion was never met in 100 time units. | Test on max \|∂ξ/∂t\| (time marching); residual of the discrete equations to round-off (Newton). |
| O4 | **Jacobi iterations for the Poisson equation, capped at 3000 per step.** | Dominates the run time and leaves an inexact ψ at every step. | Exact sparse LU solve, factorised once; steady states by Newton with grid sequencing (Re = 1000 on 257² in about 15 s). |
| O5 | **No validation.** Ghia et al. (1982) was in the project folder but never used. | Errors O1–O3 went unnoticed. | Centre-line velocities, primary and corner vortices, and vorticity at the vortex centre compared with Ghia; ψ_min with Botella & Peyret; tests in CI. |
| O6 | **No grid study.** | Accuracy unknown. | Grids 33 to 513, observed order, Richardson extrapolation. |
| O7 | **Obstacle variant: ψ = 0 on a block that does not touch the walls.** | Treats the block as connected to the walls. The pressure is not single-valued around it, and the flow pattern is completely different. | ψ_body is an extra unknown fixed by ∮ ∇p · ds = 0 (see `theory.md`, §7). |
| O8 | **"Laminar flow, Re < 2300"** in the presentation. | The pipe-flow criterion does not apply to a cavity. | The cavity is steady up to a Hopf bifurcation at Re ≈ 8000. |
| O9 | **Kinetic energy reported in joules.** | Units inconsistent with the non-dimensional model. | Non-dimensional energy per unit depth, E/(ρU²L²). |
| O10 | **Rainbow (`jet`) colour map.** | Distorts perceived magnitude and is not safe for colour-blind readers. | Diverging blue–grey–red map for the signed vorticity. |
| O11 | **Two copies of the solver** (vectorised and loop-based), parameters hard-coded, interactive plotting inside the time loop. | Hard to test or reuse. | One package with a CLI, a direct and a transient solver, diagnostics and plotting separated, 15 tests. |

## Kept from the original

The formulation, the boundary conditions (Thom's formula) and the
qualitative picture of the flow (primary vortex, corner eddies, the vortex
centre moving towards the middle of the cavity as Re grows) were correct and
are unchanged.
