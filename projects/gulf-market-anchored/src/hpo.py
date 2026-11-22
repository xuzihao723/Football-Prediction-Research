from __future__ import annotations
from dataclasses import dataclass
import itertools, json, numpy as np, pandas as pd
from .config import CONFIG_DIR
from .pooled_split import pooled_inner_temporal_split
from .preprocessing import Preprocessor
from . import metrics
from .models.market import bookmaker_prior
from .models.historical import HistoricalLR,HistoricalXGB
from .models.concat import ConcatLR,ConcatXGB
from .models.residual import ResidualLR,ResidualXGB

@dataclass
class TuneResult:
    params: dict
    score: float
    rounds: int|None=None
    n_splits: int=0

_SMOKE = False

def set_smoke(flag: bool) -> None:
    global _SMOKE
    _SMOKE = bool(flag)

def is_smoke() -> bool:
    return _SMOKE

def _raw_cfg() -> dict:
    return json.loads((CONFIG_DIR/"hpo_final_models.json").read_text(encoding="utf-8"))

def _shrink_config(c: dict) -> dict:
    """Deterministic reduced budget for smoke: keeps the plumbing path (candidate loop,
    fit, predict, early stopping) intact while cutting grid size and boost rounds."""
    out = {}
    for k, v in c.items():
        if k == "XGB":
            x = dict(v)
            x["configs"] = v["configs"][:1]
            x["max_rounds"] = 60
            x["early_stopping_rounds"] = 8
            out[k] = x
        elif k == "Residual-LR":
            out[k] = {"alpha": v["alpha"][:1], "residual_scale": [1.0], "gate": [False]}
        elif isinstance(v, dict):
            out[k] = {kk: (vv[:1] if isinstance(vv, list) else vv) for kk, vv in v.items()}
        else:
            out[k] = v
    return out

def load_config() -> dict:
    c = _raw_cfg()
    return _shrink_config(c) if _SMOKE else c

def _cfg(): return load_config()
def _assert_classes(y,ctx=""):
    u=set(metrics.encode_target(np.asarray(y)).tolist())
    if u!={0,1,2}: raise RuntimeError(f"{ctx}: train block missing class(es): {sorted(u)}")
def _wmean(vals,weights): return float(np.average(np.asarray(vals,float),weights=np.asarray(weights,float)))

def _xgb_params(c):
    h=_cfg()["XGB"]
    return {**c,"subsample":h["subsample"],"colsample_bytree":h["colsample_bytree"],"gamma":h["xgb_gamma"],"alpha":h["reg_alpha"]}

def tune_model(kind:str, train_df:pd.DataFrame, cols:list[str], min_train_seasons:int=2)->TuneResult:
    """Nested temporal HPO inside one outer-train/origin. No fallback if no legal split."""
    target="target_match_result"
    # Market-dependent models are tuned only on odds-valid rows. Pre-filter before the inner
    # seasonal split so early odds-free seasons (e.g. 2015-16/2016-17) never produce an empty
    # train/val odds block.
    market_dep = kind not in {"Historical-LR","Historical-XGB"}
    work = train_df
    if market_dep:
        _,_,mv = bookmaker_prior(train_df)
        work = train_df.loc[mv].reset_index(drop=True)
    effective_min = 1 if _SMOKE and market_dep else min_train_seasons
    splits=pooled_inner_temporal_split(work,min_train_seasons=effective_min)
    if not splits: raise RuntimeError(f"{kind}: no legal inner temporal HPO split")
    if kind=="Historical-LR": candidates=[{"C":x} for x in _cfg()[kind]["C"]]
    elif kind=="Concat-LR": candidates=[{"alpha":x} for x in _cfg()[kind]["alpha"]]
    elif kind=="Residual-LR":
        c=_cfg()[kind]; candidates=[]
        for a in c["alpha"]:
            for rs in c.get("residual_scale",[1.0]): candidates.append({"alpha":a,"residual_scale":rs,"gate":False})
            if True in c.get("gate",[]): candidates.append({"alpha":a,"residual_scale":1.0,"gate":True})
    elif kind in {"Historical-XGB","Concat-XGB","Residual-XGB"}: candidates=[_xgb_params(x) for x in _cfg()["XGB"]["configs"]]
    else: raise ValueError(kind)
    all_scores=[]; all_rounds=[]
    for cand in candidates:
        sc=[]; ww=[]; rr=[]
        for sp in splits:
            tr=work.iloc[sp["train_idx"]].reset_index(drop=True); va=work.iloc[sp["val_idx"]].reset_index(drop=True)
            if market_dep:
                _,ztr,_=bookmaker_prior(tr); _,zva,_=bookmaker_prior(va)
                if len(tr)==0 or len(va)==0: raise RuntimeError(f"{kind}: empty odds-valid HPO block")
            _assert_classes(tr[target],kind)
            pp=Preprocessor(scale=kind.endswith("LR")); Xtr=pp.fit_transform(tr[cols]); Xva=pp.transform(va[cols]); ytr=tr[target].to_numpy(); yva=va[target].to_numpy()
            if kind=="Historical-LR": m=HistoricalLR(**cand).fit(Xtr,ytr); p=m.predict_proba(Xva)
            elif kind=="Concat-LR": m=ConcatLR(**cand).fit(Xtr,ztr,ytr); p=m.predict_proba(Xva,zva)
            elif kind=="Residual-LR":
                m=ResidualLR(**cand).fit(Xtr,ztr,ytr)
                if cand.get("gate",False): m.fit_gate(Xtr,ztr,ytr)
                p=m.predict_proba(Xva,zva)
            else:
                H=_cfg()["XGB"]; kw={"params":cand,"num_boost_round":H["max_rounds"]}
                if kind=="Historical-XGB": m=HistoricalXGB(**kw).fit(Xtr,ytr,eval_set=(Xva,yva),early_stopping_rounds=H["early_stopping_rounds"]); p=m.predict_proba(Xva)
                elif kind=="Concat-XGB": m=ConcatXGB(**kw).fit(Xtr,ztr,ytr,eval_set=(Xva,zva,yva),early_stopping_rounds=H["early_stopping_rounds"]); p=m.predict_proba(Xva,zva)
                else: m=ResidualXGB(**kw).fit(Xtr,ztr,ytr,eval_set=(Xva,zva,yva),early_stopping_rounds=H["early_stopping_rounds"]); p=m.predict_proba(Xva,zva)
                rr.append(m.best_rounds)
            metrics.assert_valid_proba(p,context=f"HPO {kind}"); sc.append(metrics.log_loss(yva,p)); ww.append(len(yva))
        all_scores.append(_wmean(sc,ww)); all_rounds.append(int(np.median(rr)) if rr else None)
    j=int(np.argmin(all_scores)); return TuneResult(candidates[j],float(all_scores[j]),all_rounds[j],len(splits))

def fit_tuned(kind:str, train_df, cols, tune: TuneResult):
    target="target_match_result"; d=train_df.reset_index(drop=True)
    if kind.startswith("Historical"):
        fitdf=d; z=None
    else:
        _,z,mask=bookmaker_prior(d); fitdf=d.loc[mask].reset_index(drop=True); z=z[mask]
    _assert_classes(fitdf[target],kind); pp=Preprocessor(scale=kind.endswith("LR")); X=pp.fit_transform(fitdf[cols]); y=fitdf[target].to_numpy()
    if kind=="Historical-LR": m=HistoricalLR(**tune.params).fit(X,y)
    elif kind=="Concat-LR": m=ConcatLR(**tune.params).fit(X,z,y)
    elif kind=="Residual-LR":
        m=ResidualLR(**tune.params).fit(X,z,y)
        if tune.params.get("gate",False): m.fit_gate(X,z,y)
    elif kind=="Historical-XGB": m=HistoricalXGB(params=tune.params,num_boost_round=tune.rounds or 300).fit(X,y)
    elif kind=="Concat-XGB": m=ConcatXGB(params=tune.params,num_boost_round=tune.rounds or 300).fit(X,z,y)
    elif kind=="Residual-XGB": m=ResidualXGB(params=tune.params,num_boost_round=tune.rounds or 300).fit(X,z,y)
    else: raise ValueError(kind)
    return pp,m
