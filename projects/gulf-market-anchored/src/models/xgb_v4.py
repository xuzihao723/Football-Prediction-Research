"""Additive v4 nonlinear models. No imports of legacy fit implementations."""
import json
import numpy as np
import xgboost as xgb
from scipy.special import softmax, logsumexp
from scipy.optimize import minimize
from ..metrics import encode_target, assert_valid_proba
from ..feature_registry import get_registry

CONTEXT = ['market_entropy', 'market_top_two_gap', 'market_overround']

def feature_names(history, halftime=False, context=False):
    r = get_registry()
    allowed = set(r.full_history()+r.league_context_columns)
    if not set(history) <= allowed:
        raise ValueError('unapproved history feature')
    names = list(history)+(r.ht_score_columns if halftime else [])+(CONTEXT if context else [])
    if len(names)!=len(set(names)):
        raise ValueError('duplicate feature')
    return names

def market_context(p, odds):
    p, odds = np.asarray(p,float), np.asarray(odds,float)
    assert_valid_proba(p)
    if np.any(~np.isfinite(odds)) or np.any(odds<=1): raise ValueError('invalid odds')
    ordered = np.sort(p,axis=1)
    return np.c_[-np.sum(p*np.log(np.clip(p,1e-15,1)),axis=1)/np.log(3),
                 ordered[:,-1]-ordered[:,-2], np.sum(1/odds,axis=1)-1]

def base_margin(z, A=None):
    u=np.asarray(z,float) @ (np.eye(2) if A is None else np.asarray(A)).T
    return np.c_[u[:,1], np.zeros(len(u)), u[:,0]]

def draw_reference(t):
    return np.stack([t[:,2]-t[:,1], t[:,0]-t[:,1]],axis=1)

def a_objective(a,z,tree,y,regime,penalty):
    A=np.asarray(a).reshape(2,2); m=base_margin(z,A)+tree
    yi=encode_target(np.asarray(y)); lp=m-logsumexp(m,axis=1,keepdims=True)
    p=np.exp(lp); loss=float(-lp[np.arange(len(yi)),yi].mean())
    grad=np.c_[p[:,2]-(yi==2), p[:,0]-(yi==0)].T @ z / len(yi)
    delta=A if regime=='zero' else A-np.eye(2)
    if regime in ('zero','soft'):
        loss+=penalty*np.sum(delta**2); grad+=2*penalty*delta
    return float(loss),grad.ravel()

def soft_targets(y,q,rho):
    if not 0<=rho<1: raise ValueError('rho must be in [0,1)')
    q=np.asarray(q,float); assert_valid_proba(q)
    return (1-rho)*np.eye(3)[encode_target(np.asarray(y))]+rho*q

def soft_loss_grad(m,target):
    lp=m-logsumexp(m,axis=1,keepdims=True); p=np.exp(lp)
    return float(-np.sum(target*lp)),p-target,np.maximum(2*p*(1-p),1e-6)

class TreeCorrection:
    def __init__(self,params,rounds,names):
        self.params={'objective':'multi:softprob','num_class':3,'tree_method':'hist',
            'seed':42,'nthread':1,'subsample':1.,'colsample_bytree':1.,
            'alpha':0.,'gamma':0.,'base_score':.5,**params}
        self.rounds=int(rounds); self.names=list(names)
    def matrix(self,X,margin=None,y=None):
        X=np.asarray(X,float)
        if X.shape[1]!=len(self.names): raise ValueError('tree feature count')
        return xgb.DMatrix(X,label=None if y is None else encode_target(np.asarray(y)),
            base_margin=np.zeros((len(X),3)) if margin is None else margin,
            feature_names=self.names,nthread=1)
    def fit(self,X,margin,y,rho=None,q=None):
        d=self.matrix(X,margin,y); obj=None
        if rho is not None:
            target=soft_targets(y,q,rho).astype(np.float32)
            def obj(pred,dm):
                _,g,h=soft_loss_grad(pred,target)
                return g,h
        self.booster=xgb.train(self.params,d,num_boost_round=self.rounds,obj=obj,verbose_eval=False)
        return self
    def tree_only(self,X):
        return self.booster.predict(self.matrix(X),output_margin=True,strict_shape=True)
    def margins(self,X,margin):
        return self.booster.predict(self.matrix(X,margin),output_margin=True,strict_shape=True)
    def probabilities(self,X,margin):
        p=softmax(self.margins(X,margin),axis=1); assert_valid_proba(p); return p
    def shap(self,X,interactions=False):
        kw={'pred_interactions':True} if interactions else {'pred_contribs':True}
        return self.booster.predict(self.matrix(X),strict_shape=True,**kw)
    def complexity(self,n):
        # Sum native L2 leaf and gamma structure costs / N. alpha is fixed to zero.
        lam=float(self.params.get('lambda',1)); gamma=float(self.params.get('gamma',0))
        def cost(node):
            if 'leaf' in node:return .5*lam*node['leaf']**2+gamma
            return sum(cost(c) for c in node['children'])
        return sum(cost(json.loads(s)) for s in self.booster.get_dump(dump_format='json'))/n

class StructuredXGB:
    def __init__(self,regime,penalty,params,rounds,names,max_blocks=20,objective_tol=1e-6,A_tol=1e-4):
        self.regime,self.penalty=regime,penalty
        self.params,self.rounds,self.names=params,rounds,names
        self.max_blocks,self.objective_tol,self.A_tol=max_blocks,objective_tol,A_tol
    def fit(self,X,z,y,initial=None):
        A=np.eye(2) if initial is None else np.asarray(initial,float).copy()
        if self.regime=='fixed': A=np.eye(2)
        self.trajectory=[]; previous=None; self.converged=False
        for block in range(self.max_blocks):
            tree=TreeCorrection(self.params,self.rounds,self.names).fit(X,base_margin(z,A),y)
            t=tree.tree_only(X)
            before=a_objective(A.ravel(),z,t,y,self.regime,self.penalty)[0]+tree.complexity(len(X))
            old=A.copy()
            if self.regime!='fixed':
                opt=minimize(a_objective,A.ravel(),args=(z,t,y,self.regime,self.penalty),jac=True,
                    method='L-BFGS-B',options={'ftol':1e-12,'gtol':1e-7,'maxiter':2000})
                if not opt.success: raise RuntimeError('A step failed: '+str(opt.message))
                A=opt.x.reshape(2,2)
            value=a_objective(A.ravel(),z,t,y,self.regime,self.penalty)[0]+tree.complexity(len(X))
            nll=a_objective(A.ravel(),z,t,y,'free',0)[0]
            movement=float(np.linalg.norm(A-old))
            self.trajectory.append({'block':block,'A':A.tolist(),'A_norm':float(np.linalg.norm(A)),
                'A_I_norm':float(np.linalg.norm(A-np.eye(2))),'nll':nll,'objective':value,
                'tree_block_objective':before,'tree_cost':tree.complexity(len(X)),
                'penalty':value-nll-tree.complexity(len(X)), 'A_change':movement,
                'objective_change':None if previous is None else value-previous,'params':self.params})
            self.tree,self.A=tree,A
            if previous is not None and before>previous+1e-5:
                self.failure='nonmonotone tree refit'; return self
            if self.regime=='fixed' or (previous is not None and abs(value-previous)<=self.objective_tol and movement<=self.A_tol):
                self.converged=True; return self
            previous=value
        self.failure='maximum blocks without convergence'; return self
