from __future__ import annotations
import numpy as np, xgboost as xgb
from scipy.optimize import minimize
from scipy.special import expit, logsumexp
from .. import metrics
from .market import softmax_ref

class ResidualLR:
    def __init__(self,alpha=1.0,residual_scale=1.0,gate=False):
        self.alpha=float(alpha); self.residual_scale=float(residual_scale); self.gate=bool(gate)
        self.theta=None; self.gate_theta=None
    @staticmethod
    def zero(n_features):
        m=ResidualLR(); m.theta=np.zeros(2+2*n_features); return m
    def fit(self,X,z,y):
        X=np.asarray(X,float); z=np.asarray(z,float); yi=metrics.encode_target(np.asarray(y)); d=X.shape[1]
        def obj(th):
            bH,bA=th[:2]; bh=th[2:2+d]; ba=th[2+d:2+2*d]
            zz=np.stack([z[:,0]+bH+X@bh,z[:,1]+bA+X@ba],axis=1)
            logits=np.stack([zz[:,1],np.zeros(len(X)),zz[:,0]],axis=1)
            ce=np.mean(logsumexp(logits,axis=1)-logits[np.arange(len(yi)),yi])
            return float(ce+self.alpha*(bh@bh+ba@ba))
        r=minimize(obj,np.zeros(2+2*d),method="L-BFGS-B",options={"maxiter":30000,"maxfun":150000,"ftol":1e-9})
        if not r.success or not np.isfinite(r.fun): raise RuntimeError(f"ResidualLR failed: {r.message}")
        self.theta=r.x; return self
    def residual(self,X):
        X=np.asarray(X,float); d=X.shape[1]; bH,bA=self.theta[:2]; bh=self.theta[2:2+d]; ba=self.theta[2+d:2+2*d]
        return np.stack([bH+X@bh,bA+X@ba],axis=1)
    def fit_gate(self,X,z,y):
        """Optional low-capacity entropy gate, train-side only. Base residual must already be fit."""
        p=softmax_ref(np.asarray(z,float)); H=-(p*np.log(np.clip(p,1e-12,1))).sum(axis=1)/np.log(3)
        r=self.residual(X); yi=metrics.encode_target(np.asarray(y)); z=np.asarray(z,float)
        def obj(ab):
            g=expit(ab[0]+ab[1]*H); zz=z+g[:,None]*r
            logits=np.stack([zz[:,1],np.zeros(len(zz)),zz[:,0]],axis=1)
            return float(np.mean(logsumexp(logits,axis=1)-logits[np.arange(len(yi)),yi]))
        q=minimize(obj,np.zeros(2),method="L-BFGS-B")
        if not q.success: raise RuntimeError("Residual gate fit failed")
        self.gate_theta=q.x; self.gate=True; return self
    def predict_proba(self,X,z):
        z=np.asarray(z,float); r=self.residual(X); scale=self.residual_scale
        if self.gate:
            p=softmax_ref(z); H=-(p*np.log(np.clip(p,1e-12,1))).sum(axis=1)/np.log(3)
            if self.gate_theta is None: raise RuntimeError("gate enabled but not fit")
            scale=expit(self.gate_theta[0]+self.gate_theta[1]*H)
            return softmax_ref(z+scale[:,None]*r)
        return softmax_ref(z+float(scale)*r)

class ResidualXGB:
    """XGBoost correction model using market raw logits as base_margin."""
    def __init__(self,params=None,num_boost_round=300):
        self.params={"objective":"multi:softprob","num_class":3,"eval_metric":"mlogloss","max_depth":2,"eta":0.05,"min_child_weight":10,"subsample":0.8,"colsample_bytree":0.8,"lambda":5.0,"alpha":0.0,"gamma":0.0,"seed":42,"nthread":4}
        if params: self.params.update(params)
        self.num_boost_round=int(num_boost_round); self.booster=None; self.best_rounds=None
    @staticmethod
    def _margin(z): return np.stack([z[:,1],np.zeros(len(z)),z[:,0]],axis=1)
    def fit(self,X,z,y,eval_set=None,early_stopping_rounds=None):
        d=xgb.DMatrix(np.asarray(X,float),label=metrics.encode_target(np.asarray(y)),base_margin=self._margin(np.asarray(z,float)))
        ev=[]
        if eval_set is not None:
            Xe,ze,ye=eval_set; de=xgb.DMatrix(np.asarray(Xe,float),label=metrics.encode_target(np.asarray(ye)),base_margin=self._margin(np.asarray(ze,float))); ev=[(de,"val")]
        self.booster=xgb.train(self.params,d,num_boost_round=self.num_boost_round,evals=ev,early_stopping_rounds=early_stopping_rounds,verbose_eval=False)
        bi=getattr(self.booster,"best_iteration",None); self.best_rounds=(int(bi)+1) if bi is not None else self.num_boost_round
        return self
    def predict_proba(self,X,z):
        d=xgb.DMatrix(np.asarray(X,float),base_margin=self._margin(np.asarray(z,float)))
        p=self.booster.predict(d,iteration_range=(0,self.best_rounds) if self.best_rounds else (0,0)).reshape(-1,3)
        p=p/p.sum(axis=1,keepdims=True); metrics.assert_valid_proba(p,context="ResidualXGB"); return p
