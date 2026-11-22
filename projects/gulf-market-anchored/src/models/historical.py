from __future__ import annotations
import numpy as np, xgboost as xgb
from sklearn.linear_model import LogisticRegression
from .. import metrics

class HistoricalLR:
    def __init__(self,C=1.0): self.C=float(C); self.model=None
    def fit(self,X,y):
        self.model=LogisticRegression(C=self.C,max_iter=20000,tol=1e-7).fit(np.asarray(X,float),metrics.encode_target(np.asarray(y)))
        if list(self.model.classes_)!=[0,1,2]: raise AssertionError(f"class incomplete/order wrong: {self.model.classes_}")
        return self
    def predict_proba(self,X): return self.model.predict_proba(np.asarray(X,float))

class HistoricalXGB:
    def __init__(self,params=None,num_boost_round=300):
        self.params={"objective":"multi:softprob","num_class":3,"eval_metric":"mlogloss","max_depth":2,"eta":0.05,"min_child_weight":10,"subsample":0.8,"colsample_bytree":0.8,"lambda":5.0,"alpha":0.0,"gamma":0.0,"seed":42,"nthread":4}
        if params:self.params.update(params)
        self.num_boost_round=int(num_boost_round); self.booster=None; self.best_rounds=None
    def fit(self,X,y,eval_set=None,early_stopping_rounds=None):
        d=xgb.DMatrix(np.asarray(X,float),label=metrics.encode_target(np.asarray(y))); ev=[]
        if eval_set is not None:
            Xe,ye=eval_set; de=xgb.DMatrix(np.asarray(Xe,float),label=metrics.encode_target(np.asarray(ye))); ev=[(de,"val")]
        self.booster=xgb.train(self.params,d,num_boost_round=self.num_boost_round,evals=ev,early_stopping_rounds=early_stopping_rounds,verbose_eval=False)
        bi=getattr(self.booster,"best_iteration",None); self.best_rounds=(int(bi)+1) if bi is not None else self.num_boost_round; return self
    def predict_proba(self,X):
        p=self.booster.predict(xgb.DMatrix(np.asarray(X,float)),iteration_range=(0,self.best_rounds) if self.best_rounds else (0,0)).reshape(-1,3); return p/p.sum(axis=1,keepdims=True)
