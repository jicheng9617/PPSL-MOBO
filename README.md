# PPSL-MOBO

Code for **Parametric Pareto Set Learning (PPSL) for Expensive Multi-Objective Optimization**.

The repository is structured for simplicity and readability. It contains:

- <code>**mobo/**</code>: Files for surrogate model definition and training (borrowed from the [PSL-MOBO](https://github.com/Xi-L/PSL-MOBO) repository).
- <code>**problems/**:</code> Implementation of functions for:
  - Multi-objective optimization with shared components.
  - Dynamic multi-objective optimization problems.
- <code>**results/**</code>: Contains raw experimental results (only for the review period; will be deleted post-review).

### Key Files:
- <code>**baselines_mobo.py**</code>: MOBO methods implemented as benchmarks.
- <code>**experiment_dmop.py**</code>: Main experiment file for dynamic multi-objective optimization problems.
- <code>**experiment_mop_sc.py**</code>: Main experiment file for multi-objective optimization with shared components.
- <code>**model.py**</code>: Definitions of Parametric Pareto Set Learning (PPSL) models.
- <code>**trainer.py**</code>: Training methods for PPSL, including:
  - Randomly distributed parameters.
  - Fixed parameters.
- <code>**utils.py**</code>: Provides gradient computations for smooth Tchebycheff scalarization functions.
- <code>**setup.py**</code>: Package setup file.

### License
This repository is licensed under the terms specified in the `LICENSE` file.