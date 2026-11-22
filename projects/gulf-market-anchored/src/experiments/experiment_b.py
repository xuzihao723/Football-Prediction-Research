from __future__ import annotations
import pandas as pd
from .common import load_data,split_fold,history_cols,tune_fit,predict_tuned,odds_subset
from ..feature_registry import get_registry
from ..models.market import RecalibratedMarket
from ..preprocessing import Preprocessor
from ..outputs import prediction_frame
from .. import metrics

METHODS=["Bookmaker","Recalibrated-Market","Historical-LR","Historical-XGB","Concat-LR","Concat-XGB","Residual-LR","Residual-XGB"]

def run_fold(df,fold,representation="selected"):
    tr,te=split_fold(df,fold); cols=history_cols(fold,representation); reg=get_registry()
    tro,ptr,ztr=odds_subset(tr); teo,pte,zte=odds_subset(te); rows=[prediction_frame(teo,pte,fold,"Bookmaker",representation)]; diag=[]
    Ltr=tro[reg.league_context_columns].to_numpy(float); Lte=teo[reg.league_context_columns].to_numpy(float)
    rm=RecalibratedMarket().fit(ztr,Ltr,tro[reg.target_col].to_numpy()); pr=rm.predict_proba(zte,Lte); rows.append(prediction_frame(teo,pr,fold,"Recalibrated-Market",representation))
    for kind in ["Historical-LR","Historical-XGB","Concat-LR","Concat-XGB","Residual-LR","Residual-XGB"]:
        # Historical methods are also evaluated on T_odds for exact matched B comparison.
        tune,pp,m=tune_fit(kind,tr,cols,min_train_seasons=2)
        p=predict_tuned(kind,pp,m,teo,cols); rows.append(prediction_frame(teo,p,fold,kind,representation)); diag.append({"fold":fold,"method":kind,"hpo":tune.__dict__})
    base=list(rows[0].fixture_key)
    for r in rows[1:]:
        if list(r.fixture_key)!=base: raise AssertionError(f"B matched-set failure fold={fold}")
    return pd.concat(rows,ignore_index=True),diag

def run(data=None,folds=range(5),include_full7=True):
    df=load_data() if data is None else data; rr=[]; dd=[]
    reps=["selected"]+( ["full7"] if include_full7 else [] )
    for rep in reps:
        for f in folds:
            r,d=run_fold(df,int(f),rep); rr.append(r); dd.extend(d)
    return pd.concat(rr,ignore_index=True),dd
