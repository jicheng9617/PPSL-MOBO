# PPSL-MOBO

Code for the AAAI 2026 paper: *Parametric Pareto Set Learning for Expensive Multi-Objective Optimization*.

## Repository Overview

- `mobo/`: Contains modules for Gaussian Process surrogate models and their training, building upon the [PSL-MOBO](https://github.com/Xi-L/PSL-MOBO) codebase.
- `problems/`: A collection of synthetic test problems used in our experiments. This includes:
  - Multi-objective problems with shared components.
  - Dynamic multi-objective optimization problems (DMOPs).
- `results/`: Directory containing the raw data from our experiments. Note: This will be removed after the review process.

## Core Implementation

- `model.py`: Defines the neural network architecture for the Parametric Pareto Set Learner.
- `trainer.py`: Handles the training loop for the PPSL model under different parameterization strategies.
- `baselines_mobo.py`: A unified implementation of competitive MOBO algorithms used as benchmarks.
- `experiment_mop_sc.py`: Runs the primary experiments for problems with shared components.
- `experiment_dmop.py`: Runs the primary experiments for dynamic problems.
- `utils.py`: Provides helper functions, notably for computing gradients of the smooth Tchebycheff scalarization.

## Cite
If you use this code, please cite the paper:
