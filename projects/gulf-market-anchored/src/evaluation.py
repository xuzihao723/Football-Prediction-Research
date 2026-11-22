from __future__ import annotations
import numpy as np, pandas as pd
from .feature_registry import get_registry
from . import metrics

def compute_odds_mask(df):
    X=df[get_registry().market_prior_columns].to_numpy(float); return np.all(np.isfinite(X)&(X>1),axis=1)

def compute_tseq_mask(df):
    r=get_registry(); m=compute_odds_mask(df)
    for cols in (r.ht_score_columns,r.ht_non_score_columns):
        if cols:
            r.validate_frame(df,cols,"T_seq"); m &= df[cols].notna().all(axis=1).to_numpy()
    return m

def assert_matched_fixture(id_sets,names=None):
    if not id_sets:return
    base=list(id_sets[0])
    if len(base)!=len(set(base)): raise AssertionError("base fixture keys are not unique")
    for i,x in enumerate(id_sets[1:],1):
        y=list(x); n=(names[i] if names else str(i))
        if len(y)!=len(set(y)): raise AssertionError(f"{n}: duplicate fixture_key")
        if y!=base: raise AssertionError(f"{n}: matched fixture order/set differs")

def evaluate(y,p): return metrics.compute_all_metrics(np.asarray(y),np.asarray(p,float))
