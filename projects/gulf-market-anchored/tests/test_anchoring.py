from __future__ import annotations

import numpy as np

from src import metrics
from src.models.anchoring import CanonicalAnchorLR, anchor_penalty
from src.models.market import reference_logits, softmax_ref
from src.preprocessing import Preprocessor


def _data(seed=7, n=90, p=4):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p)); z = rng.normal(scale=0.4, size=(n, 2))
    y = np.array(["A", "D", "H"] * (n // 3))
    return X, z, y


def test_class_and_reference_order_roundtrip():
    p = np.array([[.2, .3, .5], [.55, .25, .2]])
    assert metrics.class_order() == ["A", "D", "H"]
    assert np.allclose(softmax_ref(reference_logits(p)), p)


def test_fixed_keeps_identity_exactly():
    X, z, y = _data()
    m = CanonicalAnchorLR("fixed", lambda_h=.3).fit(X, z, y)
    assert np.array_equal(m.A, np.eye(2))
    assert np.all(np.isfinite(m.predict_proba(X, z)))


def test_free_does_not_penalize_A():
    X, z, y = _data()
    m = CanonicalAnchorLR("free", lambda_h=.3)
    b = np.zeros(2); A0 = np.eye(2); B = np.zeros((2, X.shape[1]))
    t0 = m._pack(b, A0, B)
    t1 = t0.copy(); t1[2:6] = np.array([4.0, -2.0, 1.5, 3.0])
    assert np.isclose(anchor_penalty(A0, B, .3, 9.0, "free"),
                      anchor_penalty(t1[2:6].reshape(2, 2), B, .3, 9.0, "free"))


def test_soft_anchor_penalty_zero_at_identity_and_lambda_zero_matches_free():
    X, z, y = _data()
    B = np.ones((2, X.shape[1]))
    assert anchor_penalty(np.eye(2), B, .3, 10.0, "soft") == .3 * np.sum(B * B)
    free = CanonicalAnchorLR("free", lambda_h=.3).fit(X, z, y)
    soft = CanonicalAnchorLR("soft", lambda_h=.3, lambda_a=0.0).fit(X, z, y)
    assert np.allclose(free.predict_proba(X, z), soft.predict_proba(X, z), atol=1e-7)


def test_market_logits_are_not_standardized_with_history():
    X, z, _ = _data()
    z0 = z.copy()
    pp = Preprocessor(scale=True).fit(X)
    _ = pp.transform(X)
    assert np.array_equal(z, z0)


def test_intercepts_are_unpenalized():
    X, _, _ = _data()
    A = np.eye(2); B = np.zeros((2, X.shape[1]))
    assert anchor_penalty(A, B, 3.0, 5.0, "soft") == 0.0


def test_gradient_matches_finite_difference():
    X, z, y = _data(n=24, p=3)
    m = CanonicalAnchorLR("soft", lambda_h=.2, lambda_a=.7)
    rng = np.random.default_rng(9)
    theta = rng.normal(scale=.1, size=2 + 4 + 2 * X.shape[1])
    value, grad = m.objective_and_grad(theta, X, z, y)
    eps = 1e-6
    numeric = np.empty_like(grad)
    for i in range(len(theta)):
        a = theta.copy(); b = theta.copy(); a[i] += eps; b[i] -= eps
        numeric[i] = (m.objective_and_grad(a, X, z, y)[0] -
                      m.objective_and_grad(b, X, z, y)[0]) / (2 * eps)
    assert np.isfinite(value)
    assert np.max(np.abs(grad - numeric)) < 2e-5


def test_optimizer_converges_and_probabilities_normalize():
    X, z, y = _data()
    m = CanonicalAnchorLR("soft", lambda_h=.1, lambda_a=.5).fit(X, z, y)
    assert m.fit_info.success
    assert m.fit_info.gradient_inf < 1e-5
    metrics.assert_valid_proba(m.predict_proba(X, z))
