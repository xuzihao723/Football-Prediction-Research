from __future__ import annotations
import numpy as np, pandas as pd
from .common import load_data,split_fold,history_cols,tune_fit,predict_tuned
from ..hpo import TuneResult, fit_tuned, load_config, is_smoke
from ..feature_registry import get_registry
from ..evaluation import compute_tseq_mask, assert_matched_fixture
from ..models.market import bookmaker_prior,reference_logits
from ..models.concat import ConcatLR,ConcatXGB
from ..models.structural import conditional_poisson_proba_rolling
from ..preprocessing import Preprocessor
from ..pooled_split import pooled_inner_temporal_split
from ..outputs import prediction_frame
from ..config import protocol
from .. import metrics


def _stage_tune(train,ht_cols,backend):
    cfg=load_config()
    splits=pooled_inner_temporal_split(train,min_train_seasons=1 if is_smoke() else 2)
    if backend=="LR": cand=[{"alpha":a} for a in cfg["Concat-LR"]["alpha"]]
    else: cand=cfg["XGB"]["configs"]
    if not splits:
        fb=protocol()["hybrid_early_origin_fallback"]
        if backend=="LR": return {"alpha":float(fb["LR"]["alpha"])},None,True
        idx=int(fb["XGB"]["config_index"]); return dict(cand[idx]),int(fb["XGB"]["rounds"]),True
    scores=[]; rounds=[]
    for c in cand:
        ss=[]; ww=[]; rr=[]
        for sp in splits:
            tr=train.iloc[sp["train_idx"]]; va=train.iloc[sp["val_idx"]]
            if tr["target_match_result"].nunique()<3: raise RuntimeError("Hybrid stage train missing class")
            pp=Preprocessor(scale=(backend=="LR")); Xtr=pp.fit_transform(tr[ht_cols]); Xva=pp.transform(va[ht_cols]); ztr=tr[["_hyb_zH","_hyb_zA"]].to_numpy(); zva=va[["_hyb_zH","_hyb_zA"]].to_numpy()
            if backend=="LR": m=ConcatLR(**c).fit(Xtr,ztr,tr.target_match_result.to_numpy()); p=m.predict_proba(Xva,zva)
            else:
                H=cfg["XGB"]; params={**c,"subsample":H["subsample"],"colsample_bytree":H["colsample_bytree"],"gamma":H["xgb_gamma"],"alpha":H["reg_alpha"]}
                m=ConcatXGB(params=params,num_boost_round=H["max_rounds"]).fit(Xtr,ztr,tr.target_match_result.to_numpy(),eval_set=(Xva,zva,va.target_match_result.to_numpy()),early_stopping_rounds=H["early_stopping_rounds"]); p=m.predict_proba(Xva,zva); rr.append(m.best_rounds)
            ss.append(metrics.log_loss(va.target_match_result.to_numpy(),p)); ww.append(len(va))
        scores.append(float(np.average(ss,weights=ww))); rounds.append(int(np.median(rr)) if rr else None)
    j=int(np.argmin(scores)); return cand[j],rounds[j],False

def _fit_stage(train,ht_cols,backend,params,rounds):
    pp=Preprocessor(scale=(backend=="LR")); X=pp.fit_transform(train[ht_cols]); z=train[["_hyb_zH","_hyb_zA"]].to_numpy(); y=train.target_match_result.to_numpy()
    if backend=="LR": m=ConcatLR(**params).fit(X,z,y)
    else:
        H=load_config()["XGB"]; pars={**params,"subsample":H["subsample"],"colsample_bytree":H["colsample_bytree"],"gamma":H["xgb_gamma"],"alpha":H["reg_alpha"]}; m=ConcatXGB(params=pars,num_boost_round=rounds or 300).fit(X,z,y)
    return pp,m

def _frozen_early_oof_tune(backend):
    fb=protocol()["hybrid_early_origin_fallback"]
    if backend=="LR":
        x=fb["LR"]
        return TuneResult({"alpha":float(x["alpha"]),"residual_scale":float(x["residual_scale"]),"gate":bool(x["gate"])},score=float("nan"),rounds=None,n_splits=0)
    H=load_config()["XGB"]
    c=dict(H["configs"][int(fb["XGB"]["config_index"])]); c.update({"subsample":H["subsample"],"colsample_bytree":H["colsample_bytree"],"gamma":H["xgb_gamma"],"alpha":H["reg_alpha"]})
    return TuneResult(c,score=float("nan"),rounds=int(fb["XGB"]["rounds"]),n_splits=0)

def _residual_oof(train_seq,fold,backend,history):
    """Temporal OOF pre-match prior for Hybrid. If an earliest OOF origin has only two
    market seasons and therefore cannot itself host another season-level HPO split,
    use the explicitly frozen conservative pre-test fallback above. This is logged,
    never selected from the OOF validation labels, and is not a silent fallback.
    """
    kind=f"Residual-{backend}"; work=train_seq.reset_index(drop=True); oof=np.full((len(work),3),np.nan); fallback_origins=[]
    for sp in pooled_inner_temporal_split(work,min_train_seasons=1 if is_smoke() else 2):
        tr=work.iloc[sp["train_idx"]].reset_index(drop=True); va=work.iloc[sp["val_idx"]].reset_index(drop=True)
        try:
            tune,pp,m=tune_fit(kind,tr,history,min_train_seasons=2)
        except RuntimeError as e:
            if "no legal inner temporal HPO split" not in str(e): raise
            tune=_frozen_early_oof_tune(backend); pp,m=fit_tuned(kind,tr,history,tune); fallback_origins.append(sp.get("val_seasons",[]))
        oof[sp["val_idx"]]=predict_tuned(kind,pp,m,va,history)
    ok=np.isfinite(oof).all(axis=1); out=work.loc[ok].copy().reset_index(drop=True); z=reference_logits(oof[ok]); out["_hyb_zH"]=z[:,0]; out["_hyb_zA"]=z[:,1]
    if len(out)==0: raise RuntimeError("Hybrid produced no temporal OOF training rows")
    return out,fallback_origins

def run_fold(df,fold,representation="selected"):
    tr,te=split_fold(df,fold); reg=get_registry(); hist=history_cols(fold,representation)
    tr_seq=tr.loc[compute_tseq_mask(tr)].reset_index(drop=True); te_seq=te.loc[compute_tseq_mask(te)].reset_index(drop=True)
    if len(tr_seq)==0 or len(te_seq)==0: raise RuntimeError(f"C fold {fold}: empty T_seq")
    rows=[]; diag=[]
    # Pre-match rows on the exact same T_seq test fixtures.
    pmarket,_,m=bookmaker_prior(te_seq)
    if not m.all(): raise AssertionError("T_seq must be odds-valid")
    rows.append(prediction_frame(te_seq,pmarket,fold,"Market","T_seq"))
    for backend in ["LR","XGB"]:
        for arch in ["Concat","Residual"]:
            kind=f"{arch}-{backend}"; tune,pp,model=tune_fit(kind,tr_seq,hist,min_train_seasons=2); p=predict_tuned(kind,pp,model,te_seq,hist); rows.append(prediction_frame(te_seq,p,fold,f"Pre-{kind}",representation)); diag.append({"fold":fold,"method":f"Pre-{kind}","hpo":tune.__dict__})
        for stage_name,ht_cols in [("HT-S",reg.ht_score_columns),("HT-F",reg.ht_score_columns+reg.ht_non_score_columns)]:
            reg.validate_frame(tr_seq,ht_cols,stage_name)
            fullcols=hist+ht_cols
            for arch in ["Concat","Residual"]:
                kind=f"{arch}-{backend}"; tune,pp,model=tune_fit(kind,tr_seq,fullcols,min_train_seasons=2); p=predict_tuned(kind,pp,model,te_seq,fullcols); label=("Full-Concat" if arch=="Concat" else "Direct-Residual")+f"-{backend}-{stage_name}"; rows.append(prediction_frame(te_seq,p,fold,label,representation)); diag.append({"fold":fold,"method":label,"hpo":tune.__dict__})
            # Hybrid: upstream pre Residual is temporal OOF in HT training.
            oof,fallback_origins=_residual_oof(tr_seq,fold,backend,hist); params,rounds,stage_fallback=_stage_tune(oof,ht_cols,backend); pp2,m2=_fit_stage(oof,ht_cols,backend,params,rounds)
            pre_kind=f"Residual-{backend}"; tpre,ppre,mpre=tune_fit(pre_kind,tr_seq,hist,min_train_seasons=2); p_pre=predict_tuned(pre_kind,ppre,mpre,te_seq,hist); zte=reference_logits(p_pre); Xte=pp2.transform(te_seq[ht_cols]); p_h=m2.predict_proba(Xte,zte); metrics.assert_valid_proba(p_h,context="Hybrid")
            label=f"Hybrid-{backend}-{stage_name}"; rows.append(prediction_frame(te_seq,p_h,fold,label,representation)); diag.append({"fold":fold,"method":label,"stage_params":params,"stage_rounds":rounds,"upstream_hpo":tpre.__dict__,"early_oof_fallback_origins":fallback_origins,"stage_fallback_used":stage_fallback})
    # Conditional Poisson reference: league-specific and prequential across the full
    # test chronology.  Non-T_seq completed fixtures still update later structural history.
    cp_parts=[]
    for league,gte in te_seq.groupby("league",sort=False):
        gtr=tr[tr.league==league].reset_index(drop=True)
        gchron=te[te.league==league].reset_index(drop=True)
        gteg=gte.reset_index(drop=True)
        p=conditional_poisson_proba_rolling(gtr,gteg,chronology_df=gchron)
        cp_parts.append(prediction_frame(gteg,p,fold,"Conditional-Poisson","T_seq"))
    cp_frame=pd.concat(cp_parts,ignore_index=True).set_index("fixture_key").loc[te_seq["fixture_key"].astype(str).tolist()].reset_index()
    rows.append(cp_frame)
    assert_matched_fixture([list(r.fixture_key) for r in rows], names=[r["method"].iloc[0] for r in rows])
    out=pd.concat(rows,ignore_index=True)
    return out,diag

def run(data=None,folds=range(5)):
    df=load_data() if data is None else data; rr=[]; dd=[]
    for f in folds:
        r,d=run_fold(df,int(f)); rr.append(r); dd.extend(d)
    return pd.concat(rr,ignore_index=True),dd
