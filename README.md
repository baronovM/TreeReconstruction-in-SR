# Spanning Tree Reconstruction in Symbolic Regression

This project studies how reliably spanning-tree methods recover symbolic formulas from noisy adjacency matrices. Each experiment encodes a known equation as a typed expression tree, perturbs its adjacency matrix, and measures recovery accuracy across noise levels. The simulated data used for parameter fitting stay noise-free.

The experiments cover a forced Van der Pol oscillator, the Lotka–Volterra system, the Lorenz system, and magnetohydrodynamics (MHD) with synthetic fields. The core recovery methods are DFS, BFS, and Prim. Installing `pcst_fast` enables the k-MST variants.

## Quick start

From the repository root, using Python 3.12 on Windows:

```powershell
py -3.12 -m venv .venv
.venv/Scripts/python -m pip install deap sympy zss numpy scipy matplotlib networkx pydot tqdm dysts==0.96
.venv/Scripts/python -m unittest
.venv/Scripts/python -m demos.vdp --quick
```

Install Graphviz and make sure `dot` is on your `PATH` to render expression trees. The quick run saves plots and NumPy results to `figures/quick/`. Use `demos.lv`, `demos.lorenz`, or `demos.mhd` to select another model, and omit `--quick` to run with the model's full defaults. The Lorenz experiment uses the reference equations provided by `dysts` to generate trajectories and derivative targets.

## Repository layout

- `symreg/` contains the tree representation, reconstruction algorithms, fitting routines, and experiment code.
- `tests/` contains the `unittest` suite.
- `demos/` contains experiment entry points and small examples. Run `python -m demos.trees` to see a formula converted to an adjacency matrix and back.
