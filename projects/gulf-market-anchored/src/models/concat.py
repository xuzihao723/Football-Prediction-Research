from __future__ import annotations
import numpy as np, xgboost as xgb
from scipy.optimize import minimize
from scipy.special import logsumexp
from .. import metrics

class ConcatLR:
    def __init__(self,alpha=1.0): self.alpha=float(alpha); self.theta=None
    def fit(self,X,z,y):
        Q=np.hstack([np.asarray(X,float),np.asarray(z,float)]); yi=metrics.encode_target(np.asarray(y)); d=Q.shape[1]
        def obj(th):
            bH,bA=th[:2]; wh=th[2:2+d]; wa=th[2+d:2+2*d]
            h=bH+Q@wh; a=bA+Q@wa; logits=np.stack([a,np.zeros(len(Q)),h],axis=1)
            return float(np.mean(logsumexp(logits,axis=1)-logits[np.arange(len(yi)),yi])+self.alpha*(wh@wh+wa@wa))
        r=minimize(obj,np.zeros(2+2*d),method="L-BFGS-B",options={"maxiter":30000,"maxfun":150000,"ftol":1e-9})
        if not r.success: raise RuntimeError(f"ConcatLR failed: {r.message}")
        self.theta=r.x; return self
    def predict_proba(self,X,z):
        Q=np.hstack([np.asarray(X,float),np.asarray(z,float)]); d=Q.shape[1]; bH,bA=self.theta[:2]; wh=self.theta[2:2+d]; wa=self.theta[2+d:2+2*d]
        logits=np.stack([bA+Q@wa,np.zeros(len(Q)),bH+Q@wh],axis=1); return np.exp(logits-logsumexp(logits,axis=1,keepdims=True))

class ConcatXGB:
    def __init__(self,params=None,num_boost_round=300):
        self.params={"objective":"multi:softprob","num_class":3,"eval_metric":"mlogloss","max_depth":2,"eta":0.05,"min_child_weight":10,"subsample":0.8,"colsample_bytree":0.8,"lambda":5.0,"alpha":0.0,"gamma":0.0,"seed":42,"nthread":4}
        if params:self.params.update(params)
        self.num_boost_round=int(num_boost_round); self.booster=None; self.best_rounds=None
    def _x(self,X,z): return np.hstack([np.asarray(X,float),np.asarray(z,float)])
    def fit(self,X,z,y,eval_set=None,early_stopping_rounds=None):
        d=xgb.DMatrix(self._x(X,z),label=metrics.encode_target(np.asarray(y))); ev=[]
        if eval_set is not None:
            Xe,ze,ye=eval_set; de=xgb.DMatrix(self._x(Xe,ze),label=metrics.encode_target(np.asarray(ye))); ev=[(de,"val")]
        self.booster=xgb.train(self.params,d,num_boost_round=self.num_boost_round,evals=ev,early_stopping_rounds=early_stopping_rounds,verbose_eval=False)
        bi=getattr(self.booster,"best_iteration",None); self.best_rounds=(int(bi)+1) if bi is not None else self.num_boost_round; return self
    def predict_proba(self,X,z):
        p=self.booster.predict(xgb.DMatrix(self._x(X,z)),iteration_range=(0,self.best_rounds) if self.best_rounds else (0,0)).reshape(-1,3); return p/p.sum(axis=1,keepdims=True)
