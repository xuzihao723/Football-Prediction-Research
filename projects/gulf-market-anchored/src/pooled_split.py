from __future__ import annotations
import numpy as np, pandas as pd
from .config import protocol

PRECISION_COL="timestamp_precision"; DATE_ONLY="date_only"

def seasons(): return list(protocol()["seasons"])
def outer_folds(): return list(protocol()["outer_folds"])

def match_time(df: pd.DataFrame) -> pd.Series:
    if "match_datetime" in df.columns:
        t=pd.to_datetime(df["match_datetime"],errors="coerce",utc=True)
    else:
        t=pd.Series(pd.NaT,index=df.index,dtype="datetime64[ns, UTC]")
    if "date" in df.columns:
        d=pd.to_datetime(df["date"],errors="coerce",utc=True)
        t=t.fillna(d)
    if t.isna().any(): raise AssertionError(f"match_time missing for {int(t.isna().sum())} rows")
    return t

def legal_time(df: pd.DataFrame) -> pd.Series:
    t=match_time(df)
    if PRECISION_COL not in df.columns: return t
    mask=df[PRECISION_COL].astype(str).eq(DATE_ONLY)
    if mask.any():
        day=t.dt.floor("D")+pd.Timedelta(days=1)
        t=t.where(~mask,day)
    return t

def forecast_boundary(df: pd.DataFrame): return match_time(df).min()

def assert_temporal(train_df,test_df,ctx=""):
    if len(train_df)==0 or len(test_df)==0: raise AssertionError(f"{ctx}: empty split")
    a=legal_time(train_df); b=match_time(test_df)
    if not a.max()<b.min(): raise AssertionError(f"{ctx}: max legal train {a.max()} >= min query {b.min()}")

def _trim(nominal,test):
    bound=forecast_boundary(test); keep=legal_time(nominal)<bound
    return nominal.loc[keep].reset_index(drop=True), int((~keep).sum()), bound

def pooled_outer_split(df, fold:int):
    cfg=outer_folds()[fold]
    te=df[df["season"].astype(str).isin(cfg["outer_test"])].reset_index(drop=True)
    nom=df[df["season"].astype(str).isin(cfg["outer_train"])].reset_index(drop=True)
    tr,_,_=_trim(nom,te); assert_temporal(tr,te,f"outer fold {fold}")
    return tr,te

def pooled_inner_temporal_split(train_df,min_train_seasons:int=3):
    work=train_df.reset_index(drop=True); season=work["season"].astype(str)
    present=[s for s in seasons() if s in set(season)]
    out=[]
    for end in range(min_train_seasons,len(present)):
        va_s=present[end]; tr_s=present[:end]
        va=np.flatnonzero((season==va_s).to_numpy()); nom=np.flatnonzero(season.isin(tr_s).to_numpy())
        if len(va)==0 or len(nom)==0: continue
        bound=forecast_boundary(work.iloc[va]); keep=(legal_time(work.iloc[nom])<bound).to_numpy(); tr=nom[np.flatnonzero(keep)]
        if len(tr)==0: continue
        assert_temporal(work.iloc[tr],work.iloc[va],f"inner {tr_s}->{va_s}")
        out.append({"train_idx":tr.tolist(),"val_idx":va.tolist(),"train_seasons":tr_s,"val_seasons":[va_s],"nominal_train_n":len(nom),"cutoff_removed_n":len(nom)-len(tr),"legal_train_n":len(tr),"val_n":len(va),"boundary":str(bound)})
    return out

def temporal_batches(df: pd.DataFrame):
    """Yield positional batches that cannot consume one another's outcomes.

    Exact timestamps share a batch.  If *any* fixture on a calendar day is date-only,
    every fixture on that day is batched together.  This is intentionally conservative:
    an unknown kickoff time must never be ordered before/after an exact-time fixture on
    the same day.  The rule is shared by Elo, Poisson/DC and HT structural updates.
    """
    work=df.reset_index(drop=True)
    t=match_time(work)
    precision=work.get(PRECISION_COL,pd.Series("exact",index=work.index)).astype(str)
    date_only=precision.eq(DATE_ONLY)
    day=t.dt.floor("D")
    whole_day=set(day[date_only].tolist())
    key=[]
    for i,ts in enumerate(t):
        d=day.iloc[i]
        key.append(("D",str(d.date())) if d in whole_day else ("T",ts.isoformat()))
    first={}
    for i,k in enumerate(key):
        first.setdefault(k, day.iloc[i] if k[0]=="D" else t.iloc[i])
    for k in sorted(first,key=lambda x:first[x]):
        yield np.array([i for i,v in enumerate(key) if v==k],dtype=int)


def development_smoke_split(df: pd.DataFrame):
    """Development-only pseudo outer split used by smoke tests.

    Uses S1-S4 -> S5 and the same actual-date cutoff as formal outer folds.  It never
    touches the definitive S6-S10 test seasons.  Market-dependent smoke HPO is allowed
    to use a one-season minimum inside this development sandbox (handled by hpo.py).
    """
    ss=seasons()
    train_seasons=ss[:4]
    test_season=ss[4]
    te=df[df["season"].astype(str).eq(test_season)].reset_index(drop=True)
    nom=df[df["season"].astype(str).isin(train_seasons)].reset_index(drop=True)
    tr,_,_=_trim(nom,te)
    assert_temporal(tr,te,"development smoke S1-S4->S5")
    return tr,te
