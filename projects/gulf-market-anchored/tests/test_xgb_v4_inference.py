import numpy as np
import pandas as pd
from scipy.special import softmax
from src.statistics import paired_cluster_bootstrap,per_obs_metric
from src.models.xgb_v4 import soft_targets,soft_loss_grad
from tools.analyze_xgb_v4 import bootstrap_values,holm,f1

def test_aggregated_bootstrap_matches_original_cluster_resampling():
    y=np.array(['A','D','H']*8); rng=np.random.default_rng(51)
    a=softmax(rng.normal(size=(24,3)),axis=1); b=softmax(rng.normal(size=(24,3)),axis=1)
    def frame(p):
        return pd.DataFrame({'actual_class':y,'league':['L']*24,'season':np.repeat(['S0','S1','S2'],8),'p_A':p[:,0],'p_D':p[:,1],'p_H':p[:,2]})
    ga,gb=frame(a),frame(b); clusters=(ga.league+'|'+ga.season).to_numpy(); clist=np.unique(clusters)
    draws=np.random.default_rng(42).choice(3,size=(200,3),replace=True)
    weights=np.eye(3,dtype=int)[draws].sum(axis=1)
    va,vb=bootstrap_values(ga,weights,clist),bootstrap_values(gb,weights,clist)
    for name in ['log_loss','brier','acc']:
        original=paired_cluster_bootstrap(per_obs_metric(y,a,name),per_obs_metric(y,b,name),clusters,n_boot=200,seed=42)
        np.testing.assert_allclose(np.percentile(va[name]-vb[name],[2.5,97.5]),[original['ci_low'],original['ci_high']],atol=1e-14)
    np.testing.assert_allclose(holm([.01,.04,.03]),[.03,.06,.06])

def test_soft_cross_entropy_equals_CE_plus_KL_up_to_constant():
    rng=np.random.default_rng(5); m=rng.normal(size=(18,3)); q=softmax(rng.normal(size=(18,3)),axis=1)
    y=np.tile([0,1,2],6); rho=.25; p=softmax(m,axis=1)
    original=-(1-rho)*np.log(p[np.arange(18),y]).sum()+rho*np.sum(q*np.log(q/p))
    surrogate=soft_loss_grad(m,soft_targets(y,q,rho))[0]
    np.testing.assert_allclose(original,surrogate+rho*np.sum(q*np.log(q)),atol=1e-12)

def test_group_interaction_aggregation_indexing():
    ix=np.arange(2*3*5*5).reshape(2,3,5,5).astype(float); jj=[0,2]
    aggregate=(ix[:,2,4,jj]-ix[:,1,4,jj]).sum(axis=1)
    manual=np.array([sum(ix[i,2,4,k]-ix[i,1,4,k] for k in jj) for i in range(2)])
    np.testing.assert_array_equal(aggregate,manual)
