# Applied Convex Optimization 

A collection of notes and numerical models addressing problems in quantitative finance, risk assessment, and metric learning. This repository demonstrates the transition from theoretical convex optimization to practical implementation, utilizing disciplined convex programming (DCP) to solve non-linear financial and statistical problems.

## Overview

The models in this repository rely on conic programming (SDP, SOCP, RSOCP) and custom interior-point solvers to tackle scenarios where standard heuristic methods fall short. The architecture strictly separates the mathematical core logic from exploratory analysis, focusing on clean, reproducible research environments and algorithmic stability.

## Project Structure

```text
applied-convex-optimization/
├── notebooks/                               # Exploratory research notes and visual analysis
│   ├── 01_analytic_center_newton_convergence.ipynb
│   ├── 02_worst_case_portfolio_loss_lp.ipynb
│   └── 03_mahalanobis_metric_learning_sdp.ipynb
├── src/                                     # Isolated, production-ready optimization modules
│   ├── __init__.py
│   ├── optimize_barrier.py                  # Log-barrier Newton/GD solvers
│   ├── portfolio_tail_risk.py               # Copula LP optimization
│   └── metric_learning.py                   # SDP/RSOCP metric learning classes
├── pyproject.toml                           # Dependency specifications
└── README.md
```

## Key Implementations

### 1. Worst-Case Portfolio Loss via Linear Programming
Standard risk models often rely on joint normality assumptions, which critically underestimate tail risk. This module implements a non-parametric approach to compute the absolute upper bound of portfolio loss probability. By discretizing the return space and optimizing the empirical joint distribution under exact marginal and covariance constraints, it demonstrates how worst-case tail risk can significantly exceed Gaussian estimates (e.g., 9.66% vs. 1.80%).
* **Techniques:** Linear Programming (LP), Copula Optimization.
* **Stack:** `cvxpy`, `scipy.stats`.

### 2. Quadratic Metric Learning via Semidefinite Programming
Finding the correct distance metric between multivariate feature vectors is essential for regime clustering and similarity search. This module learns a data-driven Mahalanobis-like pseudo-metric from noisy pairwise distances. The Mean Squared Error (MSE) minimization is cast as a convex conic program, leveraging Semidefinite and Rotated Second-Order Cone constraints to avoid local minima and ensure robust out-of-sample generalization.
* **Techniques:** Semidefinite Programming (SDP), Rotated Second-Order Cone Programming (RSOCP).
* **Stack:** `cvxpy` (SCS/ECOS solvers).

### 3. Analytic Center Computation: Phase Transitions in Interior-Point Methods
A benchmark of first-order and second-order optimization algorithms for finding the analytic center of high-dimensional polytopes. The custom implementation evaluates the exact analytical gradient and Hessian, highlighting the rapid phase transition of Newton's Method (with backtracking line search) from a damped sequence to terminal quadratic convergence.
* **Techniques:** Newton's Method, Gradient Descent, Logarithmic Barrier Functions, Backtracking Line Search.
* **Stack:** `numpy`, `scipy.linalg`.

## Quick Start

This project uses `uv` for deterministic, lightning-fast dependency management.

```bash
# Clone the repository
git clone https://github.com/Pavelpuff/applied-convex-optimization.git
cd applied-convex-optimization

# Initialize the environment and install dependencies using uv
uv sync

# Run the Jupyter environment to view the research notes
uv run jupyter notebook
```