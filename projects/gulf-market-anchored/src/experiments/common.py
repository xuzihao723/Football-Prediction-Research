from __future__ import annotations
import json
import numpy as np, pandas as pd
from ..config import DATA_DIR,CONFIG_DIR
from ..feature_registry import get_registry
from ..pooled_split import pooled_outer_split, development_smoke_split
from ..preprocessing import Preprocessor
from ..hpo import tune_model,fit_tuned,is_smoke
from ..models.market import bookmaker_prior
from .. import metrics

DATA_FILE=DATA_DIR/"modeling"/"ModelingData_harmonized.csv"

def load_data(path=None):
    p=path or DATA_FILE
    if not p.exists(): raise FileNotFoundError(f"Missing modeling data: {p}")
    d=pd.read_csv(p,low_memory=False)
    reg=get_registry(); reg.validate_frame(d,[reg.fixture_key_col,reg.target_col,"league","season"],"model data")
    if d[reg.fixture_key_col].duplicated().any(): raise AssertionError("fixture_key not globally unique")
    return d

def selection_cfg(): return json.loads((CONFIG_DIR/"sk_selection_frozen.json").read_text(encoding="utf-8"))
def history_cols(fold,representation="selected"):
    r=get_registry(); cols=r.full_history() if representation=="full7" else r.columns_for_fold(fold,selection_cfg())
    return r.with_context(cols)
def split_fold(df,fold): return development_smoke_split(df) if is_smoke() else pooled_outer_split(df,fold)

def odds_subset(df):
    p,z,m=bookmaker_prior(df); return df.loc[m].reset_index(drop=True),p[m],z[m]

def predict_tuned(kind,pp,model,df,cols):
    X=pp.transform(df[cols])
    if kind.startswith("Historical"): p=model.predict_proba(X)
    else:
        _,z,m=bookmaker_prior(df)
        if not m.all(): raise AssertionError(f"{kind}: prediction dataframe must already be odds-valid")
        p=model.predict_proba(X,z)
    metrics.assert_valid_proba(p,context=kind); return p

def tune_fit(kind,train_df,cols,min_train_seasons=2):
    t=tune_model(kind,train_df,cols,min_train_seasons=min_train_seasons); pp,m=fit_tuned(kind,train_df,cols,t); return t,pp,m
