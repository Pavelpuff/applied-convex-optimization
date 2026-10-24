import cvxpy as cp
import numpy as np


class MahalanobisMetricLearner:
    """
    Learns a quadratic pseudo-metric (Mahalanobis distance matrix P)
    from pairwise distance measurements using Semidefinite Programming (SDP).
    """

    def __init__(self):
        self.P_opt_: np.ndarray | None = None

    def fit(
        self, X: np.ndarray, Y: np.ndarray, d: np.ndarray
    ) -> MahalanobisMetricLearner:
        """
        Fits the metric matrix P to minimize Mean Squared Error (MSE)
        against observed distances.
        """
        N, n = X.shape

        # Define PSD matrix variable
        P = cp.Variable((n, n), PSD=True)
        z = X - Y

        objective_terms = []
        for i in range(N):
            z_i = z[i]
            w_i = z_i.T @ P @ z_i

            # The RSOCP formulation is implicitly handled by cvxpy's cp.sqrt
            term = d[i] ** 2 - 2 * d[i] * cp.sqrt(w_i) + w_i
            objective_terms.append(term)

        objective = cp.Minimize(cp.sum(objective_terms) / N)
        problem = cp.Problem(objective)

        # SCS is highly robust for SDP/SOCP problems
        problem.solve(solver=cp.SCS)

        self.P_opt_ = P.value
        return self

    def predict(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """
        Predicts distances between pairs in X and Y using the learned metric.
        """
        if self.P_opt_ is None:
            raise ValueError("Model must be fitted before calling predict.")

        z = X - Y
        N = X.shape[0]
        d_pred = np.zeros(N)

        for i in range(N):
            w_i = z[i].T @ self.P_opt_ @ z[i]
            # Max with 0 guarantees numerical stability for pseudo-metrics
            d_pred[i] = np.sqrt(max(w_i, 0.0))

        return d_pred

    def score(self, X: np.ndarray, Y: np.ndarray, d_true: np.ndarray) -> float:
        """
        Computes Mean Squared Error (MSE) of the predictions.
        """
        d_pred = self.predict(X, Y)
        return np.mean((d_true - d_pred) ** 2)
