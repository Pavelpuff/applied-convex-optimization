import math
from typing import Any

import cvxpy as cp
import numpy as np
from scipy.stats import norm


def optimize_worst_case_copula(
    mu_vec: tuple[float, float],
    sigma_vec: tuple[float, float],
    rho: float,
    grid_min: float = -20.0,
    grid_max: float = 50.0,
    grid_size: int = 100,
) -> dict[str, Any]:
    """
    Computes the worst-case probability of loss for a two-asset portfolio
    using Linear Programming to optimize the empirical joint distribution.
    """
    r = np.linspace(grid_min, grid_max, grid_size)

    # 1. Compute discrete marginals
    p_mar = np.zeros((2, grid_size))
    for k in range(2):
        unnormalized = np.exp(-((r - mu_vec[k]) ** 2) / (2 * sigma_vec[k] ** 2))
        p_mar[k] = unnormalized / np.sum(unnormalized)

    # 2. Setup LP variables and parameters
    R1, R2 = np.meshgrid(r, r, indexing="ij")
    M = np.multiply(R1 - mu_vec[0], R2 - mu_vec[1])
    Mask = R1 + R2 <= 0

    P = cp.Variable((grid_size, grid_size), nonneg=True)

    # 3. Define constraints
    constraints = [
        cp.sum(P, axis=1) == p_mar[0],
        cp.sum(P, axis=0) == p_mar[1],
        cp.sum(cp.multiply(P, M)) == rho * sigma_vec[0] * sigma_vec[1],
    ]

    # 4. Solve maximizing the loss probability
    obj = cp.Maximize(cp.sum(cp.multiply(P, Mask)))
    prob = cp.Problem(obj, constraints)
    prob.solve(solver=cp.SCS)  # Using SCS or ECOS for stability

    # 5. Compute analytical Gaussian baseline
    variance = (
        sigma_vec[0] ** 2 + sigma_vec[1] ** 2 + 2 * rho * sigma_vec[0] * sigma_vec[1]
    )
    joint_prob = norm.cdf(0, sum(mu_vec), math.sqrt(variance))

    return {
        "worst_case_prob": prob.value,
        "gaussian_prob": joint_prob,
        "joint_distribution": P.value,
        "grid": r,
        "R1_grid": R1,
        "R2_grid": R2,
        "marginals": p_mar,
    }
