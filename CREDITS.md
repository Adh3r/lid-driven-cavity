# Credits

## Authors

The original project was developed for the course *Mecànica de Fluids*
(EETAC – Universitat Politècnica de Catalunya, January 2026) by:

- **Adam Ruiz Herrero** ([@Adh3r](https://github.com/Adh3r))
- **Dènia Cabrera** ([@deniacabrera](https://github.com/deniacabrera))
- **Elena Alegre** ([@elenalegre](https://github.com/elenalegre))

The team derived the stream function-vorticity formulation and its
discretisation, wrote the original solver (`original/cavity_original.py`), and
explored Re = 10 to 15,000 and the obstacle variants.

The November 2026 revision (validated solver, Newton steady solver, grid
convergence study, obstacle extension, tests and documentation) was done by
Adam Ruiz Herrero. [`docs/revision.md`](docs/revision.md) lists what changed
from the original and why.

## Reference data

- U. Ghia, K. N. Ghia & C. T. Shin (1982), *J. Comput. Phys.* 48, 387–411:
  centre-line velocities and vortex properties (Tables I, II and V), stored in
  `src/cavity/benchmarks.py`.
- O. Botella & R. Peyret (1998), *Comput. Fluids* 27, 421–433: spectral value
  of the primary vortex at Re = 1000.

## Development assistance

The revision was done with help from Claude (Anthropic) as a coding assistant.
