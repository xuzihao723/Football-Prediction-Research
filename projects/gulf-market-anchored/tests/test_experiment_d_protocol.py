import numpy as np
from src.config import protocol
from src.models.generic_ml import make_classifier
from src.experiments.experiment_d import _fnn_fit_temporal


def test_fnn_generic_classifier_has_no_random_internal_early_stopping():
    m=make_classifier('FNN',{'hidden_layer_sizes':(8,), 'alpha':0.001, 'learning_rate_init':0.001})
    assert m.early_stopping is False
    assert m.shuffle is False


def test_fnn_temporal_helper_returns_training_derived_epoch():
    rng=np.random.default_rng(0)
    X=rng.normal(size=(90,5)); y=np.array([0,1,2]*30)
    _,epochs,p=_fnn_fit_temporal(X[:60],y[:60],X[60:],y[60:],{'hidden_layer_sizes':[8],'alpha':0.001,'learning_rate_init':0.001})
    assert epochs>=1
    assert p.shape==(30,3)
    assert np.allclose(p.sum(axis=1),1.0)


def test_nested_fallbacks_are_protocol_frozen():
    p=protocol()
    assert 'hybrid_early_origin_fallback' in p
    assert 'd_nested_early_origin_fallback' in p
    assert p['hybrid_early_origin_fallback']['XGB']['rounds']>0
    assert p['d_nested_early_origin_fallback']['FNN_epochs']>0
