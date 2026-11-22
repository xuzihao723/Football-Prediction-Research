import numpy as np
import pandas as pd
import pytest
import xgboost as xgb
from src.experiments.standard_baselines import OrdinaryModel,matrix,aligned,grids
from src.experiments.common import load_data,history_cols
from src.pooled_split import pooled_outer_split,pooled_inner_temporal_split,assert_temporal
from src.models.market import bookmaker_prior

def toy():
    rng=np.random.default_rng(31);return rng.normal(size=(90,4)),np.array(['A','D','H']*30)

def test_standard_LR_train_only_preprocess_and_class_mapping():
    X,y=toy();X[0,0]=np.nan
    m=OrdinaryModel('LR',{'C':.1}).fit(X,y)
    assert list(m.model.classes_)==[0,1,2]
    assert m.imputer.statistics_[0]==np.nanmedian(X[:,0])
    mean=m.scaler.mean_.copy();p=m.predict_proba(np.full((2,4),1000.))
    np.testing.assert_array_equal(mean,m.scaler.mean_)
    assert np.isfinite(p).all();np.testing.assert_allclose(p.sum(1),1)

def test_market_is_ordinary_feature_not_offset():
    d=pd.DataFrame({'hist':[1.,2.],'odds_ft_home_team_win':[2.,3.],'odds_ft_draw':[3.,3.], 'odds_ft_away_team_win':[4.,2.]})
    X=matrix(d,['hist'],True);assert X.shape==(2,3)
    np.testing.assert_array_equal(X[:,1:],bookmaker_prior(d)[1])
    assert matrix(d,['hist'],False).shape==(2,1)
    d['target_match_result']=['H','A'];np.testing.assert_array_equal(X,matrix(d,['hist'],True))

def test_XGB_never_receives_base_margin(monkeypatch):
    X,y=toy();original=xgb.train
    def checked(params,dm,**kw):
        assert dm.get_base_margin().size==0
        assert 'obj' not in kw
        for val,_ in kw['evals']:assert val.get_base_margin().size==0
        return original(params,dm,**kw)
    monkeypatch.setattr(xgb,'train',checked)
    m=OrdinaryModel('XGB',grids()['XGB'][0],rounds=8).fit(X[:60],y[:60],validation=(X[60:],y[60:]))
    assert 1<=m.best_rounds<=8
    p=m.predict_proba(X[60:]);assert p.shape==(30,3);np.testing.assert_allclose(p.sum(1),1,atol=1e-6)

def test_refit_never_early_stops_on_query_labels():
    X,y=toy();m=OrdinaryModel('XGB',grids()['XGB'][0],rounds=7).fit(X,y)
    assert m.best_rounds==7 and m.diagnostics['stopping']=='refit fixed inner-selected budget'
    repeat=OrdinaryModel('XGB',grids()['XGB'][0],rounds=7).fit(X,y)
    np.testing.assert_array_equal(m.predict_proba(X),repeat.predict_proba(X))

def test_population_and_temporal_preprocessing_inputs():
    d=load_data()
    for k in range(5):
        tr,te=pooled_outer_split(d,k);tr=tr.loc[bookmaker_prior(tr)[2]].reset_index(drop=True);te=te.loc[bookmaker_prior(te)[2]]
        assert len(te)==[544,530,527,594,594][k];assert_temporal(tr,te)
        assert len(history_cols(k))==29
        for s in pooled_inner_temporal_split(tr,min_train_seasons=2):
            a,b=tr.iloc[s['train_idx']],tr.iloc[s['val_idx']];assert_temporal(a,b)
            assert not (set(a.fixture_key)|set(b.fixture_key))&set(te.fixture_key)

def test_pair_alignment_rejects_label_change():
    d=pd.DataFrame({'fixture_key':['x'],'fold':[0],'league':['Q'],'season':['S'],'actual_class':['A']})
    bad=d.copy();bad.actual_class='H'
    with pytest.raises(AssertionError):aligned(d,bad)
