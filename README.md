# PPSL-MOBO

Code for **Dynamic Multi-Objective Optimization** with Parametric Pareto Set Learning (PPSL).

The repository is structured for simplicity and readability. It contains:

- **mobo/**: Files for surrogate model definition and training (borrowed from the [PSL-MOBO](https://github.com/Xi-L/PSL-MOBO) repository).
- **problems/**: Implementation of functions for:
  - Multi-objective optimization with shared components.
  - Dynamic multi-objective optimization problems.
- **results/**: Contains raw experimental results (only for the review period; will be deleted post-review).

### Key Files:
- **baselines_mobo.py**: MOBO methods implemented as benchmarks.
- **experiment_dmop.py**: Main experiment file for dynamic multi-objective optimization problems.
- **experiment_mop_sc.py**: Main experiment file for multi-objective optimization with shared components.
- **model.py**: Definitions of Parametric Pareto Set Learning (PPSL) models.
- **trainer.py**: Training methods for PPSL, including:
  - Randomly distributed parameters.
  - Fixed parameters.
- **utils.py**: Provides gradient computations for smooth Chebyshev scalarization functions.
- **setup.py**: Package setup file.

### License
This repository is licensed under the terms specified in the `LICENSE` file.