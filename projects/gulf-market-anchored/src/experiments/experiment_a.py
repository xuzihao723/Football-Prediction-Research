from __future__ import annotations
import numpy as np, pandas as pd
from .common import load_data,split_fold,history_cols,tune_fit,predict_tuned
from ..models.structural import fit_elo_fixed_h,elo_proba_online,poisson_proba_rolling,dc_proba_rolling,fit_dixon_coles,dc_proba,DC_XI_GRID,ELO_H_GRID
from ..pooled_split import pooled_inner_temporal_split
from .. import metrics
from ..outputs import prediction_frame

METHODS=["Elo","Independent-Poisson","Dixon-Coles","Historical-LR","Historical-XGB"]


def _tune_elo_h(train):
    from ..hpo import is_smoke
    splits=pooled_inner_temporal_split(train,min_train_seasons=2)
    if not splits:
        raise RuntimeError("Elo: no legal temporal split for h tuning")
    vals=[]
    grid=[0.0,60.0] if is_smoke() else ELO_H_GRID
    for h in grid:
        sc=[]; w=[]
        for sp in splits:
            tr=train.iloc[sp["train_idx"]].reset_index(drop=True); va=train.iloc[sp["val_idx"]].reset_index(drop=True)
            try:
                p=elo_proba_online(fit_elo_fixed_h(tr,h),va)
            except Exception:
                sc.append(1e9); w.append(len(va)); continue
            sc.append(metrics.log_loss(va["target_match_result"].to_numpy(),p)); w.append(len(va))
        vals.append(float(np.average(sc,weights=w)))
    return float(grid[int(np.argmin(vals))])

def _tune_xi(train):
    from ..hpo import is_smoke
    if is_smoke():
        return DC_XI_GRID[0]
    splits=pooled_inner_temporal_split(train,min_train_seasons=2)
    if not splits:
        raise RuntimeError("Dixon-Coles: no legal temporal split for xi tuning")
    vals=[]
    for xi in DC_XI_GRID:
        sc=[]; w=[]
        for sp in splits:
            tr=train.iloc[sp["train_idx"]]; va=train.iloc[sp["val_idx"]]
            try: p=dc_proba(fit_dixon_coles(tr,xi),va)
            except Exception: sc.append(1e9); w.append(len(va)); continue
            sc.append(metrics.log_loss(va["target_match_result"].to_numpy(),p)); w.append(len(va))
        vals.append(float(np.average(sc,weights=w)))
    if not np.isfinite(vals).any() or min(vals) >= 1e9:
        raise RuntimeError("Dixon-Coles: every xi candidate failed on the legal temporal HPO origins")
    return float(DC_XI_GRID[int(np.argmin(vals))])

def run_fold(df,fold,representation="selected"):
    tr,te=split_fold(df,fold); cols=history_cols(fold,representation); rows=[]; diagnostics=[]
    for kind in ["Historical-LR","Historical-XGB"]:
        tune,pp,m=tune_fit(kind,tr,cols,min_train_seasons=2); p=predict_tuned(kind,pp,m,te,cols)
        rows.append(prediction_frame(te,p,fold,kind,representation)); diagnostics.append({"fold":fold,"method":kind,"hpo":tune.__dict__})
    for league,gte in te.groupby("league",sort=False):
        gtr=tr[tr.league==league].reset_index(drop=True); gte=gte.reset_index(drop=True)
        if len(gtr)==0: raise RuntimeError(f"A fold {fold}: no structural train for {league}")
        h=_tune_elo_h(gtr); pe=elo_proba_online(fit_elo_fixed_h(gtr,h),gte); rows.append(prediction_frame(gte,pe,fold,"Elo","structural"))
        pp=poisson_proba_rolling(gtr,gte); rows.append(prediction_frame(gte,pp,fold,"Independent-Poisson","structural"))
        xi=_tune_xi(gtr); pdx=dc_proba_rolling(gtr,gte,xi); rows.append(prediction_frame(gte,pdx,fold,"Dixon-Coles","structural"))
        home_col="home_team_key" if "home_team_key" in gte.columns else "home_team_name"; away_col="away_team_key" if "away_team_key" in gte.columns else "away_team_name"
        tr_home="home_team_key" if "home_team_key" in gtr.columns else "home_team_name"; tr_away="away_team_key" if "away_team_key" in gtr.columns else "away_team_name"
        known=set(gtr[tr_home].astype(str)) | set(gtr[tr_away].astype(str))
        unseen=((~gte[home_col].astype(str).isin(known))|(~gte[away_col].astype(str).isin(known))).sum()
        diagnostics.append({"fold":fold,"league":league,"method":"structural","elo_h":h,"dc_xi":xi,"initial_unseen_fixtures":int(unseen)})
    return pd.concat(rows,ignore_index=True),diagnostics

def run(data=None,folds=range(5),representation="selected"):
    df=load_data() if data is None else data; allr=[]; diag=[]
    for f in folds:
        r,d=run_fold(df,int(f),representation); allr.append(r); diag.extend(d)
    return pd.concat(allr,ignore_index=True),diag
