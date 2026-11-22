from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp

from .. import metrics
from .market import softmax_ref


@dataclass
class AnchorFit:
    success: bool
    message: str
    objective: float
    iterations: int
    gradient_inf: float


class CanonicalAnchorLR:
    """Canonical eta=b+A z_market+B x multinomial model.

    The two eta columns are ordered [H, A], while probabilities are returned
    in the project order [A, D, H]. Historical features are expected to have
    already been transformed by a training-only preprocessor; market logits
    are passed untouched.
    """

    VARIANTS = {"fixed", "free", "soft", "zero", "tied_zero"}

    def __init__(self, variant="soft", lambda_h=1.0, lambda_a=1.0,
                 maxiter=30000, gtol=1e-7, *, lambda_0=0.0):
        if variant not in self.VARIANTS:
            raise ValueError(f"unknown anchoring variant: {variant}")
        self.variant = variant
        self.lambda_h = float(lambda_h)
        self.lambda_a = float(lambda_a)
        self.lambda_0 = float(lambda_0)
        if any(not np.isfinite(v) or v < 0 for v in
               (self.lambda_h, self.lambda_a, self.lambda_0)):
            raise ValueError("penalty strengths must be finite and nonnegative")
        self.maxiter = int(maxiter)
        self.gtol = float(gtol)
        self.b = None
        self.A = None
        self.B = None
        self.fit_info: AnchorFit | None = None

    @staticmethod
    def _unpack(theta, p, variant):
        b = theta[:2]
        off = 2
        if variant == "fixed":
            A = np.eye(2)
        else:
            A = theta[off:off + 4].reshape(2, 2)
            off += 4
        B = theta[off:].reshape(2, p)
        return b, A, B

    def _pack(self, b, A, B):
        if self.variant == "fixed":
            return np.concatenate([b, B.ravel()])
        return np.concatenate([b, A.ravel(), B.ravel()])

    @staticmethod
    def _eta(X, z, b, A, B):
        return b[None, :] + np.asarray(z, float) @ A.T + np.asarray(X, float) @ B.T

    def objective_and_grad(self, theta, X, z, y):
        X = np.asarray(X, float)
        z = np.asarray(z, float)
        yi = metrics.encode_target(np.asarray(y))
        n, p = X.shape
        b, A, B = self._unpack(theta, p, self.variant)
        eta = self._eta(X, z, b, A, B)
        logits = np.stack([eta[:, 1], np.zeros(n), eta[:, 0]], axis=1)
        logp = logits - logsumexp(logits, axis=1, keepdims=True)
        prob = np.exp(logp)
        ce = float(-np.mean(logp[np.arange(n), yi]))
        d_eta = np.column_stack([
            prob[:, 2] - (yi == 2),  # H
            prob[:, 0] - (yi == 0),  # A
        ]) / n
        grad_b = d_eta.sum(axis=0)
        grad_A = d_eta.T @ z
        grad_B = d_eta.T @ X
        value = ce + self.lambda_h * float(np.sum(B * B))
        grad_B = grad_B + 2.0 * self.lambda_h * B
        if self.variant == "soft":
            value += self.lambda_a * float(np.sum((A - np.eye(2)) ** 2))
            grad_A = grad_A + 2.0 * self.lambda_a * (A - np.eye(2))
        elif self.variant in {"zero", "tied_zero"}:
            l0 = self.lambda_h if self.variant == "tied_zero" else self.lambda_0
            value += l0 * float(np.sum(A * A))
            grad_A = grad_A + 2.0 * l0 * A
        if self.variant == "fixed":
            grad = np.concatenate([grad_b, grad_B.ravel()])
        else:
            grad = np.concatenate([grad_b, grad_A.ravel(), grad_B.ravel()])
        return float(value), grad

    def fit(self, X, z, y):
        X = np.asarray(X, float)
        z = np.asarray(z, float)
        if X.ndim != 2 or z.shape != (len(X), 2):
            raise ValueError("X must be (N,p) and z must be (N,2)")
        p = X.shape[1]
        theta0 = self._pack(np.zeros(2), np.eye(2), np.zeros((2, p)))

        def fun(theta):
            return self.objective_and_grad(theta, X, z, y)

        result = minimize(fun, theta0, jac=True, method="L-BFGS-B",
                          options={"maxiter": self.maxiter, "gtol": self.gtol,
                                   "ftol": 1e-12, "maxls": 50})
        self.b, self.A, self.B = self._unpack(result.x, p, self.variant)
        grad_inf = float(np.max(np.abs(result.jac))) if result.jac is not None else float("nan")
        self.fit_info = AnchorFit(bool(result.success), str(result.message),
                                  float(result.fun), int(getattr(result, "nit", 0)), grad_inf)
        if not result.success or not np.isfinite(result.fun):
            raise RuntimeError(f"CanonicalAnchorLR failed: {result.message}")
        if self.variant == "fixed" and not np.array_equal(self.A, np.eye(2)):
            raise AssertionError("Fixed variant changed A")
        return self

    def predict_proba(self, X, z):
        if self.b is None or self.A is None or self.B is None:
            raise RuntimeError("model not fit")
        eta = self._eta(X, z, self.b, self.A, self.B)
        p = softmax_ref(eta)
        metrics.assert_valid_proba(p, context=f"CanonicalAnchorLR/{self.variant}")
        return p

    def diagnostics(self):
        if self.A is None or self.B is None or self.b is None or self.fit_info is None:
            raise RuntimeError("model not fit")
        return {
            "variant": self.variant,
            "b": self.b.tolist(),
            "A": self.A.tolist(),
            "A_minus_I_fro": float(np.linalg.norm(self.A - np.eye(2))),
            "A_fro": float(np.linalg.norm(self.A)),
            "B_fro": float(np.linalg.norm(self.B)),
            "objective": self.fit_info.objective,
            "optimizer_success": self.fit_info.success,
            "optimizer_message": self.fit_info.message,
            "optimizer_iterations": self.fit_info.iterations,
            "gradient_inf_norm": self.fit_info.gradient_inf,
        }


def anchor_penalty(A, B, lambda_h, lambda_a, variant, *, lambda_0=0.0):
    """Return only the regularization term for unit-test inspection."""
    B = np.asarray(B, float)
    value = float(lambda_h) * float(np.sum(B * B))
    if variant == "soft":
        value += float(lambda_a) * float(np.sum((np.asarray(A, float) - np.eye(2)) ** 2))
    elif variant in {"zero", "tied_zero"}:
        l0 = float(lambda_h) if variant == "tied_zero" else float(lambda_0)
        value += l0 * float(np.sum(np.asarray(A, float) ** 2))
    return value
