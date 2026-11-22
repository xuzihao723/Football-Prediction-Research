from __future__ import annotations
import pandas as pd
from .common import load_data,split_fold,tune_fit,predict_tuned,odds_subset
from ..feature_registry import get_registry
from ..outputs import prediction_frame
from ..hpo import is_smoke

ARCH=["Concat-LR","Residual-LR","Concat-XGB","Residual-XGB"]

def run_fold(df,fold):
    tr,te=split_fold(df,fold); reg=get_registry(); teo,_,_=odds_subset(te); rows=[];diag=[]
    full=reg.with_context(reg.full_history())
    specs=[("Full",full)]
    groups = ["Elo"] if is_smoke() else reg.order
    for g in groups:
        minus=[c for c in reg.full_history() if c not in set(reg.groups[g])]
        specs.append((f"Minus-{g}",reg.with_context(minus)))
    for label,cols in specs:
        for kind in ARCH:
            tune,pp,m=tune_fit(kind,tr,cols,min_train_seasons=2); p=predict_tuned(kind,pp,m,teo,cols); rows.append(prediction_frame(teo,p,fold,kind,label)); diag.append({"fold":fold,"representation":label,"method":kind,"hpo":tune.__dict__})
    return pd.concat(rows,ignore_index=True),diag

def run(data=None,folds=range(5)):
    df=load_data() if data is None else data; rr=[];dd=[]
    for f in folds:
        r,d=run_fold(df,int(f)); rr.append(r);dd.extend(d)
    return pd.concat(rr,ignore_index=True),dd
