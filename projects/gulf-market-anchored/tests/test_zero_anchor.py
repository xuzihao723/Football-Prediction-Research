import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.models.anchoring import CanonicalAnchorLR
from src.models.market import softmax_ref, bookmaker_prior
from src.experiments.prematch_lr_completion import prepared, aligned, boundary_audit
from src.experiments.anchor_continuum import _bootstrap
from src.statistics import paired_cluster_bootstrap


def data():
    rng = np.random.default_rng(102)
    X, z = rng.normal(size=(120, 3)), rng.normal(size=(120, 2))
    probs = softmax_ref(z + X[:, :2] * .2)
    y = np.array([rng.choice(3, p=p) for p in probs])
    return X, z, y


def test_zero_endpoint_determinism_and_probabilities():
    X, z, y = data()
    zero = CanonicalAnchorLR('zero', lambda_h=.3, lambda_0=0).fit(X, z, y)
    free = CanonicalAnchorLR('free', lambda_h=.3).fit(X, z, y)
    again = CanonicalAnchorLR('zero', lambda_h=.3, lambda_0=0).fit(X, z, y)
    p = zero.predict_proba(X, z)
    np.testing.assert_allclose(p, free.predict_proba(X, z), atol=1e-10, rtol=0)
    np.testing.assert_array_equal(p, again.predict_proba(X, z))
    assert np.isfinite(p).all()
    np.testing.assert_allclose(p.sum(axis=1), 1, atol=1e-14)


def test_tied_zero_uses_one_shared_penalty():
    X, z, y = data()
    tied = CanonicalAnchorLR('tied_zero', lambda_h=.3).fit(X, z, y)
    zero = CanonicalAnchorLR('zero', lambda_h=.3, lambda_0=.3).fit(X, z, y)
    np.testing.assert_allclose(tied.predict_proba(X, z), zero.predict_proba(X, z), atol=1e-10, rtol=0)


def test_zero_penalty_drives_A_toward_zero():
    X, z, y = data()
    norms = [np.linalg.norm(CanonicalAnchorLR('zero', lambda_h=.3, lambda_0=v).fit(X, z, y).A)
             for v in [0., .01, 1., 100.]]
    assert all(a > b for a, b in zip(norms, norms[1:]))
    assert norms[-1] < .002


def test_independent_penalty_blocks_and_intercept():
    X, z, y = data()
    m = CanonicalAnchorLR('zero', lambda_h=0, lambda_0=0)
    b, A, B = np.array([1., -2.]), np.array([[.4, .3], [-.2, .1]]), np.ones((2, 3))*.2
    theta = m._pack(b, A, B)
    v0, g0 = m.objective_and_grad(theta, X, z, y)
    for lh, l0 in [(0., .7), (.3, 0.)]:
        q = CanonicalAnchorLR('zero', lambda_h=lh, lambda_0=l0)
        v, g = q.objective_and_grad(theta, X, z, y)
        np.testing.assert_allclose(v-v0, lh*np.sum(B**2)+l0*np.sum(A**2), atol=1e-14)
        np.testing.assert_allclose(g-g0, m._pack(np.zeros(2), 2*l0*A, 2*lh*B), atol=1e-14)
    theta = m._pack(b, np.zeros((2, 2)), B)
    v, g = CanonicalAnchorLR('zero', lambda_h=0, lambda_0=100).objective_and_grad(theta, X, z, y)
    vb, gb = m.objective_and_grad(theta, X, z, y)
    assert v == vb
    np.testing.assert_array_equal(g, gb)


def test_zero_gradient_and_order():
    X, z, y = data()
    m = CanonicalAnchorLR('zero', lambda_h=.3, lambda_0=.7)
    theta = np.linspace(-.3, .3, 12)
    _, analytic = m.objective_and_grad(theta, X, z, y)
    eps = 1e-6
    numeric = []
    for v in np.eye(len(theta))*eps:
        numeric.append((m.objective_and_grad(theta+v, X, z, y)[0] -
                        m.objective_and_grad(theta-v, X, z, y)[0])/(2*eps))
    np.testing.assert_allclose(analytic, numeric, atol=1e-8, rtol=0)
    m.b, m.A, m.B = np.zeros(2), np.eye(2), np.zeros((2, 3))
    expected = np.exp(np.c_[z[:, 1], np.zeros(len(z)), z[:, 0]])
    expected /= expected.sum(axis=1, keepdims=True)
    np.testing.assert_allclose(m.predict_proba(X, z), expected, atol=1e-14)


def test_pipeline_train_only_scaling_and_unscaled_market():
    tr = pd.DataFrame({'history': [1., 3., np.nan], 'match_datetime': ['2020-01-01']*3,
        'odds_ft_home_team_win': [1.2, 2., 3.], 'odds_ft_draw': [4., 3., 2.],
        'odds_ft_away_team_win': [6., 4., 2.]})
    va = tr.copy(); va['history'] = [1000., np.nan, 4000.]; va['match_datetime'] = '2020-02-01'
    pp, Xtr, Xva, ztr, zva = prepared(tr, va, ['history'])
    assert pp.imputer.statistics_[0] == 2.
    assert pp.scaler.mean_[0] == 2.
    assert Xva[0, 0] > 100 and Xva[1, 0] == 0.
    np.testing.assert_array_equal(ztr, bookmaker_prior(tr)[1])
    np.testing.assert_array_equal(zva, bookmaker_prior(va)[1])
    assert Xtr.shape[1] == 1  # no market columns in preprocessing
    with pytest.raises(AssertionError):
        prepared(va, tr, ['history'])


@pytest.mark.parametrize('field', ['fixture_key', 'fold', 'league', 'season', 'actual_class'])
def test_alignment_rejects_every_metadata_mismatch(field):
    a = pd.DataFrame({'fixture_key': ['a', 'b'], 'fold': [0, 0], 'league': ['L', 'L'],
        'season': ['S', 'S'], 'actual_class': ['A', 'H'], 'p_A': [.3, .3], 'p_D': [.3, .3], 'p_H': [.4, .4]})
    b = a.copy(); b.loc[0, field] = 1 if field == 'fold' else 'D'
    with pytest.raises(AssertionError): aligned(a, b)
    aligned(a, a.iloc[::-1])


def test_existing_three_variants_unchanged_on_synthetic_data():
    # Snapshot made before extending the shared canonical module; no real outer fits.
    path = Path(__file__).parents[1]/'results/anchor_continuum/prematch_lr/anchoring_before.txt'
    mod = types.ModuleType('src.models.anchoring_before')
    mod.__package__ = 'src.models'; sys.modules[mod.__name__] = mod
    exec(compile(path.read_text(encoding='utf-8'), str(path), 'exec'), mod.__dict__)
    X, z, y = data()
    for v in ['fixed', 'free', 'soft']:
        old = mod.CanonicalAnchorLR(v, .3, .2).fit(X, z, y)
        new = CanonicalAnchorLR(v, .3, .2).fit(X, z, y)
        np.testing.assert_array_equal(old.predict_proba(X, z), new.predict_proba(X, z))


def test_bootstrap_reuses_same_sampling_and_boundary_detection():
    d = np.arange(12)/100.; clusters = np.repeat(['a', 'b', 'c'], 4)
    assert _bootstrap(d, clusters, n_boot=100) == paired_cluster_bootstrap(d, np.zeros(12), clusters, n_boot=100)
    rows = [{'lambda_h': v, 'score': s} for v, s in [(1., .9), (3., .8), (10., .7)]]
    assert boundary_audit(rows, 'lambda_h')['boundary_limited']
    rows[0]['score'] = .6
    assert boundary_audit(rows, 'lambda_h')['boundary_limited']


def test_optimizer_failure_is_not_silenced():
    X, z, y = data()
    with pytest.raises(RuntimeError):
        CanonicalAnchorLR('zero', lambda_0=.1, maxiter=0).fit(X, z, y)
