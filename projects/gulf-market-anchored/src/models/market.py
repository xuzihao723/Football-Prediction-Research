from __future__ import annotations
import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp
from ..feature_registry import get_registry
from .. import metrics

EPS=1e-12

def bookmaker_prior(df):
    cols=get_registry().market_prior_columns
    X=df[cols].to_numpy(float)
    valid=np.all(np.isfinite(X)&(X>1.0),axis=1)
    q=np.full_like(X,np.nan,dtype=float)
    q[valid]=1.0/X[valid]
    p=q/q.sum(axis=1,keepdims=True)
    # config order is H,D,A; return A,D,H
    p_adh=np.stack([p[:,2],p[:,1],p[:,0]],axis=1)
    z=np.stack([np.log(p[:,0]/p[:,1]),np.log(p[:,2]/p[:,1])],axis=1)
    return p_adh,z,valid

def softmax_ref(z):
    z=np.asarray(z,float); logits=np.stack([z[:,1],np.zeros(len(z)),z[:,0]],axis=1)
    return np.exp(logits-logsumexp(logits,axis=1,keepdims=True))

def reference_logits(p):
    p=np.clip(np.asarray(p,float),EPS,1); return np.stack([np.log(p[:,2]/p[:,1]),np.log(p[:,0]/p[:,1])],axis=1)

class RecalibratedMarket:
    """z = z_market + b + league fixed effects; Qatar reference via two dummies."""
    def __init__(self): self.theta=None
    def fit(self,z,league_ctx,y):
        z=np.asarray(z,float); L=np.asarray(league_ctx,float); yi=metrics.encode_target(np.asarray(y)); d=L.shape[1]
        def obj(th):
            b=th[:2]; W=th[2:].reshape(2,d)
            zz=z + b[None,:] + L@W.T
            logits=np.stack([zz[:,1],np.zeros(len(zz)),zz[:,0]],axis=1)
            return float(np.mean(logsumexp(logits,axis=1)-logits[np.arange(len(yi)),yi]))
        r=minimize(obj,np.zeros(2+2*d),method="L-BFGS-B")
        if not r.success or not np.isfinite(r.fun): raise RuntimeError(f"RecalibratedMarket fit failed: {r.message}")
        self.theta=r.x; return self
    def predict_proba(self,z,league_ctx):
        L=np.asarray(league_ctx,float); d=L.shape[1]; b=self.theta[:2]; W=self.theta[2:].reshape(2,d)
        return softmax_ref(np.asarray(z,float)+b[None,:]+L@W.T)
