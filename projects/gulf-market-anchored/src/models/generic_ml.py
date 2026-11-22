from __future__ import annotations
import warnings
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from .. import metrics

def make_classifier(name,params=None,seed=42):
    p=dict(params or {})
    if name=="LR": return LogisticRegression(max_iter=20000,random_state=seed,**p)
    if name=="RF": return RandomForestClassifier(random_state=seed,n_jobs=-1,**p)
    if name=="XGB": return XGBClassifier(objective="multi:softprob",num_class=3,eval_metric="mlogloss",random_state=seed,n_jobs=4,**p)
    if name=="LightGBM": return LGBMClassifier(objective="multiclass",num_class=3,random_state=seed,verbosity=-1,**p)
    if name=="CatBoost": return CatBoostClassifier(loss_function="MultiClass",random_seed=seed,verbose=False,allow_writing_files=False,**p)
    if name=="FNN": return MLPClassifier(max_iter=1000,early_stopping=False,shuffle=False,random_state=seed,**p)
    raise ValueError(name)
def fit_classifier(name,X,y,params=None,seed=42):
    m=make_classifier(name,params,seed); yi=metrics.encode_target(np.asarray(y)); m.fit(np.asarray(X,float),yi)
    if list(m.classes_)!=[0,1,2]: raise RuntimeError(f"{name}: missing class/order {m.classes_}")
    return m
def predict_classifier(model,X):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="X does not have valid feature names.*", category=UserWarning)
        p=np.asarray(model.predict_proba(np.asarray(X,float)),float)
    metrics.assert_valid_proba(p,context="generic classifier"); return p

def smart_vote(prob_by_name:dict[str,np.ndarray],validation_accuracy:dict[str,float],top_k=3):
    if top_k < 1:
        raise ValueError("top_k must be >= 1")
    if set(validation_accuracy) - set(prob_by_name):
        raise KeyError("validation_accuracy contains model(s) without probabilities")
    vals={k:float(v) for k,v in validation_accuracy.items()}
    if not vals or not all(np.isfinite(v) and v >= 0.0 for v in vals.values()):
        raise ValueError("validation accuracies must be finite and non-negative")
    # Deterministic tie-break by model name after descending validation accuracy.
    names=sorted(vals,key=lambda n:(-vals[n],n))[:min(top_k,len(vals))]
    w=np.array([vals[n] for n in names],float)
    if not np.isfinite(w).all() or w.sum() <= 0:
        raise RuntimeError("Smart Voting weights are undefined (all validation accuracies are zero/non-finite)")
    w=w/w.sum()
    shapes={np.asarray(prob_by_name[n]).shape for n in names}
    if len(shapes)!=1:
        raise ValueError(f"Smart Voting member probability shapes differ: {sorted(shapes)}")
    p=sum(a*np.asarray(prob_by_name[n],float) for a,n in zip(w,names))
    metrics.assert_valid_proba(p,context="Smart Voting")
    return p,{"members":names,"weights":w.tolist()}
