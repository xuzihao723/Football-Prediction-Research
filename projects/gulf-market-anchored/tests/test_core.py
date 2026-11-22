import numpy as np
from src.models.market import softmax_ref,reference_logits
from src.models.residual import ResidualLR,ResidualXGB
from src.models.concat import ConcatLR
from src import metrics

def test_reference_roundtrip():
    p=np.array([[.2,.3,.5],[.55,.25,.2]])
    assert np.allclose(softmax_ref(reference_logits(p)),p)

def test_residual_null_identity():
    z=np.array([[.2,-.1],[.4,.3],[-.2,.1]])
    X=np.random.default_rng(1).normal(size=(3,4)); m=ResidualLR.zero(4)
    assert np.allclose(m.predict_proba(X,z),softmax_ref(z))

def test_residual_scale_zero_is_market():
    rng=np.random.default_rng(2); X=rng.normal(size=(60,3)); z=rng.normal(scale=.2,size=(60,2)); y=np.array([0,1,2]*20)
    m=ResidualLR(alpha=1,residual_scale=0).fit(X,z,y)
    assert np.allclose(m.predict_proba(X,z),softmax_ref(z))

def test_lr_probability_validity():
    rng=np.random.default_rng(3); X=rng.normal(size=(90,4)); z=rng.normal(size=(90,2))*.15; y=np.array([0,1,2]*30)
    for m in [ResidualLR(alpha=1).fit(X,z,y),ConcatLR(alpha=1).fit(X,z,y)]: metrics.assert_valid_proba(m.predict_proba(X,z))

def test_xgb_rounds_uses_best_iteration_plus_one():
    rng=np.random.default_rng(4); X=rng.normal(size=(120,3)); z=rng.normal(scale=.1,size=(120,2)); y=np.array([0,1,2]*40)
    m=ResidualXGB(num_boost_round=30).fit(X[:90],z[:90],y[:90],eval_set=(X[90:],z[90:],y[90:]),early_stopping_rounds=3)
    assert m.best_rounds>=1
    bi=getattr(m.booster,'best_iteration',None)
    if bi is not None: assert m.best_rounds==bi+1
