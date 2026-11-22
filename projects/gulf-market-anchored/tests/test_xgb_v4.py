import numpy as np
import pytest
from scipy.special import softmax
from scipy.optimize import minimize
from src.models.xgb_v4 import (base_margin,draw_reference,a_objective,soft_targets,
    soft_loss_grad,TreeCorrection,market_context,feature_names,CONTEXT,StructuredXGB)
from src.experiments.common import history_cols

def sample():
    rng=np.random.default_rng(42); X=rng.normal(size=(180,3)); z=rng.normal(size=(180,2))
    y=np.array([rng.choice(3,p=p) for p in softmax(base_margin(z)+np.c_[X[:,0],X[:,1],X[:,2]],axis=1)])
    return X,z,y

def model(X,z,y,rho=None):
    return TreeCorrection({'max_depth':2,'eta':.05,'min_child_weight':5,'lambda':5},40,['x0','x1','x2']).fit(X,base_margin(z),y,rho=rho,q=softmax(base_margin(z),axis=1))

def test_reference_mapping():
    _,z,_=sample(); p=softmax(base_margin(z),axis=1)
    np.testing.assert_allclose(np.log(p[:,2]/p[:,1]),z[:,0],atol=1e-14)
    np.testing.assert_allclose(np.log(p[:,0]/p[:,1]),z[:,1],atol=1e-14)

@pytest.mark.parametrize('regime',['zero','soft','free'])
def test_A_gradient_and_endpoints(regime):
    X,z,y=sample(); a=np.array([.6,.2,-.1,.8]); tree=np.c_[X[:,0],X[:,1],X[:,2]]
    fun=lambda a:a_objective(a,z,tree,y,regime,.3)
    _,g=fun(a); numeric=[(fun(a+d)[0]-fun(a-d)[0])/2e-6 for d in np.eye(4)*1e-6]
    np.testing.assert_allclose(g,numeric,atol=2e-6)
    v0,g0=a_objective(a,z,tree,y,'free',0)
    v,g=a_objective(a,z,tree,y,'zero',0)
    assert v==v0; np.testing.assert_array_equal(g,g0)
    if regime!='free':
        center=np.zeros(4) if regime=='zero' else np.eye(2).ravel()
        opt=minimize(a_objective,a,args=(z,tree,y,regime,1000),jac=True,method='L-BFGS-B')
        assert opt.success; assert np.linalg.norm(opt.x-center)<.001

def test_soft_target_derivatives_and_hessian_majorizer():
    X,z,y=sample(); q=softmax(base_margin(z),axis=1); target=soft_targets(y,q,.25)
    np.testing.assert_allclose(target.sum(axis=1),1)
    m=base_margin(z); _,g,h=soft_loss_grad(m,target)
    for row,col in [(0,0),(10,2),(15,1)]:
        e=np.zeros_like(m); e[row,col]=1e-5
        numeric=(soft_loss_grad(m+e,target)[0]-soft_loss_grad(m-e,target)[0])/2e-5
        assert abs(g[row,col]-numeric)<2e-6
    for p,diagonal in zip(q,h):
        exact=np.diag(p)-np.outer(p,p)
        assert np.linalg.eigvalsh(np.diag(diagonal)-exact).min()>-1e-10
    for rho in [-.1,1.,1.1]:
        with pytest.raises(ValueError):soft_targets(y,q,rho)

def test_rho_zero_native_endpoint_and_repeatability():
    X,z,y=sample(); native=model(X,z,y); kl=model(X,z,y,0.); again=model(X,z,y,0.)
    np.testing.assert_allclose(native.probabilities(X,base_margin(z)),kl.probabilities(X,base_margin(z)),atol=2e-5,rtol=0)
    np.testing.assert_array_equal(kl.tree_only(X),again.tree_only(X))

def test_tree_margin_SHAP_and_interaction_reconstruction():
    X,z,y=sample(); tree=model(X,z,y); X=X[:15]; z=z[:15]
    t=tree.tree_only(X); final=tree.margins(X,base_margin(z))
    np.testing.assert_allclose(final-t,base_margin(z),atol=2e-5,rtol=0)
    s=tree.shap(X); np.testing.assert_allclose(s.sum(axis=2),t,atol=2e-5)
    for k,c in enumerate([2,0]):
        np.testing.assert_allclose((s[:,c,:]-s[:,1,:]).sum(axis=1),draw_reference(t)[:,k],atol=2e-5)
    interactions=tree.shap(X,True)
    np.testing.assert_allclose(interactions.sum(axis=3),s,atol=2e-5)
    np.testing.assert_allclose(interactions.sum(axis=(2,3)),t,atol=2e-5)

def test_direction_free_context_and_whitelist():
    p=np.array([[.2,.3,.5],[1/3,1/3,1/3]]); odds=1/(1.1*p)
    c=market_context(p,odds)
    np.testing.assert_allclose(c[0,1],.2); np.testing.assert_allclose(c[:,2],.1)
    np.testing.assert_allclose(c[1,0],1)
    for perm in [[2,0,1],[1,2,0]]:np.testing.assert_allclose(c,market_context(p[:,perm],odds[:,perm]))
    hist=history_cols(0); assert feature_names(hist,True,True)[-3:]==CONTEXT
    for forbidden in ['p_H','odds_ft_draw','market_logit','favourite_identity']:
        with pytest.raises(ValueError):feature_names(hist+[forbidden])

def test_fixed_block_exact_identity_deterministic():
    X,z,y=sample(); params={'max_depth':1,'eta':.05,'min_child_weight':5,'lambda':5}
    a=StructuredXGB('fixed',0,params,40,['x0','x1','x2']).fit(X,z,y)
    b=StructuredXGB('fixed',0,params,40,['x0','x1','x2']).fit(X,z,y,initial=np.zeros((2,2)))
    assert a.converged and b.converged; np.testing.assert_array_equal(a.A,np.eye(2))
    np.testing.assert_array_equal(a.tree.tree_only(X),b.tree.tree_only(X))
