import numpy as np


class AnalyticCenterOptimizer:
    """
    Solves for the analytic center of the linear inequality system A*x <= 1 and |x| <= 1
    via logarithmic barrier minimization.
    """

    def __init__(self, A: np.ndarray):
        """
        Initializes the optimizer with the constraint matrix A.
        """
        self.A = A
        self.m, self.n = A.shape

    def objective(self, x: np.ndarray) -> float:
        """Computes the logarithmic barrier objective function."""
        if np.any(1 - self.A @ x <= 0) or np.any(1 - x**2 <= 0):
            return np.inf
        return -np.sum(np.log(1 - self.A @ x)) - np.sum(np.log(1 - x**2))

    def gradient(self, x: np.ndarray) -> np.ndarray:
        """Computes the exact analytical gradient."""
        d = 1.0 / (1 - self.A @ x)
        v = 2 * x / (1 - x**2)
        return self.A.T @ d + v

    def hessian(self, x: np.ndarray) -> np.ndarray:
        """Computes the exact analytical Hessian."""
        d = 1.0 / (1 - self.A @ x)
        w = 2 * (1 + x**2) / (1 - x**2) ** 2
        return self.A.T @ np.diag(d**2) @ self.A + np.diag(w)

    def optimize_gd(
        self,
        x0: np.ndarray,
        alpha: float = 0.1,
        beta: float = 0.5,
        delta: float = 1e-5,
        max_iters: int = 2000,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Runs Gradient Descent with Backtracking Line Search."""
        x = x0.copy()
        f_vals, step_lengths = [], []

        for _ in range(max_iters):
            f_val = self.objective(x)
            g = self.gradient(x)
            if np.linalg.norm(g) ** 2 <= delta:
                f_vals.append(f_val)
                break

            f_vals.append(f_val)
            dx = -g
            t = 1.0
            while self.objective(x + t * dx) >= f_val + alpha * t * np.dot(g.T, dx):
                t *= beta

            x = x + t * dx
            step_lengths.append(t)

        return x, np.array(f_vals), np.array(step_lengths)

    def optimize_newton(
        self,
        x0: np.ndarray,
        alpha: float = 0.1,
        beta: float = 0.5,
        delta: float = 1e-5,
        max_iters: int = 100,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Runs Newton's Method with Backtracking Line Search."""
        x = x0.copy()
        f_vals, step_lengths = [], []

        for _ in range(max_iters):
            f_val = self.objective(x)
            g = self.gradient(x)
            H = self.hessian(x)
            dx = np.linalg.solve(H, -g)
            lambda_sq = np.dot(g.T, -dx)

            if lambda_sq <= delta:
                f_vals.append(f_val)
                break

            f_vals.append(f_val)
            t = 1.0
            while self.objective(x + t * dx) >= f_val + alpha * t * np.dot(g.T, dx):
                t *= beta

            x = x + t * dx
            step_lengths.append(t)

        return x, np.array(f_vals), np.array(step_lengths)
