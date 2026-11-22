import numpy as np
import pandas as pd
import pytest
from tools import final_design_execution as e
from src.models.xgb_v4 import base_margin
from src.models.market import bookmaker_prior
from src.experiments.common import load_data
from src.pooled_split import pooled_outer_split,assert_temporal,pooled_inner_temporal_split

def test_populations_and_nested_diagonal_grid():
    d=e.source();original=load_data()
    for k in range(5):
        a,b=e.outer_split(d,k);x,y=pooled_outer_split(original,k)
        assert a.fixture_key.tolist()==x.loc[bookmaker_prior(x)[2],'fixture_key'].tolist()
        assert b.fixture_key.tolist()==y.loc[bookmaker_prior(y)[2],'fixture_key'].tolist()
        for sp in pooled_inner_temporal_split(a,min_train_seasons=1):assert_temporal(a.iloc[sp['train_idx']],a.iloc[sp['val_idx']])
    zero={(c['lambda_h'],c['market_penalty']) for c in e.grid('Zero-LR')}
    assert all((c['lambda_h'],c['lambda_h']) in zero for c in e.grid('Tied-Zero-LR'))

def test_oof_data_flow_query_labels_never_used_for_hpo_or_fit(monkeypatch):
    d=e.source();tr,te=e.outer_split(d,0);calls=[]
    def tune(train,alg,origin,**kw):
        calls.append(('tune',set(train.fixture_key)));return {'params':{},'rounds':1,'fallback':True}
    def fit(train,query,alg,origin,params,rounds):
        assert_temporal(train,query);assert not set(train.fixture_key)&set(query.fixture_key)
        calls.append(('fit',set(train.fixture_key)))
        return np.tile([.2,.3,.5],(len(query),1)),1,{}
    monkeypatch.setattr(e,'tuned',tune);monkeypatch.setattr(e,'fit_predict',fit)
    oof,trace=e.anchor_oof(tr,'Fixed-LR');assert len(oof)==1022
    for t in trace:assert not set(t['producer_keys'])&set(t['query_keys'])
    assert all(not keys&set(te.fixture_key) for _,keys in calls)
    assert set(oof.fixture_key)<set(tr.fixture_key)

def test_half_time_information_and_market_branch():
    d=e.source().head(10)
    for alg in e.ALGORITHMS:
        X,z=e.design(d,'HT',alg)
        assert X.shape[1]==32+(3 if alg=='MC-XGB' else 2 if alg.startswith('Joint') else 0)
        changed=d.copy();changed['target_match_result']='H'
        V,q=e.design(changed,'HT',alg);np.testing.assert_array_equal(X,V);np.testing.assert_array_equal(z,q)

def test_KL_zero_early_stopping_native_endpoint():
    d=e.source();a,_=e.outer_split(d,0);sp=pooled_inner_temporal_split(a,min_train_seasons=2)[0]
    tr=a.iloc[sp['train_idx']];va=a.iloc[sp['val_idx']]
    c=e.grid('Fixed-XGB')[0]
    p,n,_=e.fit_predict(tr,va,'Fixed-XGB','PRE',c,validation=True)
    q,m,_=e.fit_predict(tr,va,'KL-XGB','PRE',c,validation=True)
    assert n==m;np.testing.assert_allclose(p,q,atol=2e-5,rtol=0)

def test_inner_only_true_label_early_stop_for_KL(monkeypatch):
    original=e.xgb.train
    def wrapper(params,dm,**kw):
        assert params['eval_metric']=='mlogloss'
        assert kw['early_stopping_rounds']==40
        assert kw['evals'][0][1]=='validation'
        assert kw['obj'] is not None
        return original(params,dm,**kw)
    monkeypatch.setattr(e.xgb,'train',wrapper)
    d=e.source();a,_=e.outer_split(d,0);sp=pooled_inner_temporal_split(a,min_train_seasons=2)[0]
    c=e.grid('KL-XGB')[2]
    p,_,_=e.fit_predict(a.iloc[sp['train_idx']],a.iloc[sp['val_idx']],'KL-XGB','HT',c,validation=True)
    assert np.isfinite(p).all()
