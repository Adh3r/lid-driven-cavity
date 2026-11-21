# Original course submission

`cavity_original.py` is the final code of the course project (Mecànica de
Fluids, EETAC-UPC, January 2026), kept unmodified for reference: comments in
Spanish, interactive plotting, explicit Euler with a fixed time step,
first-order upwind advection and a Jacobi Poisson solver.

What it got right and wrong, measured against Ghia et al. (1982), is in
[`../docs/revision.md`](../docs/revision.md); `studies/original_method.py`
reproduces it headless. The solver in `src/cavity/` replaces it.
