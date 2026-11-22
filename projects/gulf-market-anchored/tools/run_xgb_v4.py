"""Only independently gated v4 branches; frozen predictions are never refit."""
import argparse,json,time,hashlib,itertools
from pathlib import Path
import numpy as np
import pandas as pd
from src import metrics
from src.config import PROJECT_ROOT as ROOT
from src.experiments.common import load_data,history_cols,odds_subset
from src.pooled_split import pooled_outer_split,pooled_inner_temporal_split,assert_temporal
from src.evaluation import compute_tseq_mask
from src.preprocessing import Preprocessor
from src.feature_registry import get_registry
from src.models.market import bookmaker_prior
from src.models.xgb_v4 import TreeCorrection,base_margin,market_context,feature_names,draw_reference,CONTEXT
from src.models.anchoring import CanonicalAnchorLR
from src.outputs import prediction_frame

OUT=ROOT/'results/xgb_v4'
CFG=json.loads((OUT/'protocol/frozen_protocol_v4.json').read_text())
def dump(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf8') as f:json.dump(obj,f,indent=2,ensure_ascii=False,allow_nan=False)

def design(df,hist,origin,family):
    names=feature_names(hist,origin=='HT',family=='MC')
    aux=hist+(get_registry().ht_score_columns if origin=='HT' else [])
    X=df[aux].to_numpy(float); p,z,valid=bookmaker_prior(df)
    assert valid.all()
    if family=='MC':X=np.c_[X,market_context(p,df[get_registry().market_prior_columns].to_numpy(float))]
    assert X.shape[1]==len(names)
    return X,p,z,names

def blocks(train,hist,origin,family):
    result=[]
    for sp in pooled_inner_temporal_split(train,min_train_seasons=2):
        tr=train.iloc[sp['train_idx']]; va=train.iloc[sp['val_idx']]; assert_temporal(tr,va)
        x,p,z,names=design(tr,hist,origin,family); v,pv,zv,_=design(va,hist,origin,family)
        pp=Preprocessor(scale=family=='LR'); x=pp.fit_transform(x); v=pp.transform(v)
        result.append((x,p,z,tr.target_match_result.to_numpy(),v,pv,zv,va.target_match_result.to_numpy()))
    if not result:raise RuntimeError('no legal temporal splits')
    return result,names

def fit(family,c,x,p,z,y,names):
    if family=='LR':
        return CanonicalAnchorLR(c['regime'],lambda_h=c['lambda_h'],lambda_a=c['market_penalty'],lambda_0=c['market_penalty']).fit(x,z,y)
    model=TreeCorrection(c['params'],c['rounds'],names)
    return model.fit(x,base_margin(z),y,rho=c.get('rho') if family=='KL' else None,q=p)

def predict(model,family,x,z):
    return model.predict_proba(x,z) if family=='LR' else model.probabilities(x,base_margin(z))

def candidates(family,regime=None):
    if family=='LR':
        penalties=CFG['market_penalties'] if regime in ['zero','soft'] else [0.]
        return [{'regime':regime,'lambda_h':h,'market_penalty':a} for h,a in itertools.product(CFG['lr_lambda_h'],penalties)]
    return [{**c,**({'rho':rho} if family=='KL' else {})} for c in CFG['tree_candidates'] for rho in (CFG['rho_grid'] if family=='KL' else [None])]

def tune(train,hist,origin,family,regime):
    data,names=blocks(train,hist,origin,family); surface=[]
    for c in candidates(family,regime):
        scores=[]; times=[]
        for x,p,z,y,v,pv,zv,yv in data:
            start=time.perf_counter(); m=fit(family,c,x,p,z,y,names)
            scores.append(metrics.log_loss(yv,predict(m,family,v,zv))); times.append(time.perf_counter()-start)
        surface.append({'candidate':c,'score':float(np.average(scores,weights=[len(d[-1]) for d in data])),
            'split_scores':scores,'split_weights':[len(d[-1]) for d in data],'seconds':times})
    best=min(surface,key=lambda s:s['score'])
    return best['candidate'],surface

def shap_save(model,X,test,fold,method,family,origin):
    names=model.names; s=model.shap(X); tree=model.tree_only(X)
    if not np.allclose(s.sum(axis=2),tree,atol=2e-5,rtol=0):raise AssertionError('SHAP reconstruction')
    refs=np.stack([s[:,2,:]-s[:,1,:],s[:,0,:]-s[:,1,:]],axis=1)
    if not np.allclose(refs.sum(axis=2),draw_reference(tree),atol=2e-5,rtol=0):raise AssertionError('reference SHAP')
    records=[]; reg=get_registry(); groupmap={c:g for g,cols in reg.groups.items() for c in cols}
    groupmap.update({c:'LeagueContext' for c in reg.league_context_columns})
    groupmap.update({c:'HTScore' for c in reg.ht_score_columns}); groupmap.update({c:'MarketContext' for c in CONTEXT})
    for j,name in enumerate(names+['__bias__']):
        records.append(pd.DataFrame({'fixture_key':test.fixture_key.to_numpy(),'fold':fold,'league':test.league.to_numpy(),
            'season':test.season.to_numpy(),'method':method,'origin':origin,'feature':name,'group':groupmap.get(name,'Bias'),
            'phi_A':s[:,0,j],'phi_D':s[:,1,j],'phi_H':s[:,2,j], 'phi_HD':refs[:,0,j],'phi_AD':refs[:,1,j]}))
    directory={'Fixed':'ms_xgb','MC':'mc_xgb','KL':'kl_xgb'}[family]
    pd.concat(records,ignore_index=True).to_csv(OUT/f'shap/{directory}/{method}_fold{fold}.csv',index=False,mode='x')
    if family=='MC':
        ix=model.shap(X,True)
        if not np.allclose(ix.sum(axis=3),s,atol=2e-5,rtol=0):raise AssertionError('interaction sum')
        inter=[]
        for context in CONTEXT:
            j=names.index(context)
            for group in list(reg.groups)+(['HTScore'] if origin=='HT' else []):
                jj=[k for k,n in enumerate(names) if groupmap[n]==group]
                for ref,c in [('HD',2),('AD',0)]:
                    vals=(ix[:,c,j,jj]-ix[:,1,j,jj]).sum(axis=1) if jj else np.zeros(len(X))
                    inter.append({'method':method,'origin':origin,'fold':fold,'context':context,'group':group,
                        'reference':ref,'selected_features':len(jj),'N':len(X),
                        'mean_absolute_group_interaction':float(np.abs(vals).mean()),
                        'convention':'one off-diagonal cell per feature pair, no doubling'})
        dump(OUT/f'shap/mc_xgb/{method}_fold{fold}_interactions.json',inter)
    return {'tree_reconstruction_max_error':float(np.max(np.abs(s.sum(axis=2)-tree))),
            'reference_reconstruction_max_error':float(np.max(np.abs(refs.sum(axis=2)-draw_reference(tree))))}

def run(family,origin,regime=None):
    gates=json.loads((OUT/'engineering/gates.json').read_text())
    gate={'Fixed':'Fixed','MC':'MC','KL':'KL_PRE','LR':'HT_LR'}[family]
    assert gates[gate]['passed']
    if family=='KL' and origin!='PRE':raise RuntimeError('sequential family failed OOF gate')
    method=(f'Canonical-{regime.title()}-LR-HT' if family=='LR' else
        {'Fixed':'MS-XGB-Fixed','MC':'MC-XGB','KL':'KL-XGB'}[family]+'-'+origin)
    directory={'Fixed':'ms_xgb','MC':'mc_xgb','KL':'kl_xgb','LR':'lr_ht'}[family]
    dest=OUT/directory/('prematch' if origin=='PRE' else 'halftime') if family!='LR' else OUT/directory
    dest.mkdir(parents=True,exist_ok=True); df=load_data()
    legacy=pd.read_csv(ROOT/'results/raw'/('experiment_b_oos.csv' if origin=='PRE' else 'experiment_c_oos.csv'))
    base=legacy[(legacy.method==('Bookmaker' if origin=='PRE' else 'Market')) & (legacy.representation==('selected' if origin=='PRE' else 'T_seq'))]
    for fold in range(5):
        target=dest/f'{method}_fold{fold}.csv'
        if target.exists():raise FileExistsError(target)
        started=time.perf_counter(); train,test=pooled_outer_split(df,fold)
        if origin=='HT':train=train.loc[compute_tseq_mask(train)]; test=test.loc[compute_tseq_mask(test)]
        train,_,_=odds_subset(train); test,_,_=odds_subset(test); hist=history_cols(fold)
        c,surface=tune(train,hist,origin,family,regime)
        assert_temporal(train,test)
        x,p,z,names=design(train,hist,origin,family); v,pv,zv,_=design(test,hist,origin,family)
        pp=Preprocessor(scale=family=='LR'); x=pp.fit_transform(x); v=pp.transform(v)
        m=fit(family,c,x,p,z,train.target_match_result.to_numpy(),names)
        prob=predict(m,family,v,zv); frame=prediction_frame(test,prob,fold,method,'selected')
        expected=base[base.fold==fold].reset_index(drop=True)
        for key in ['fixture_key','fold','league','season','actual_class']:
            if frame[key].tolist()!=expected[key].tolist():raise AssertionError('alignment '+key)
        if family!='LR':
            shapdiag=shap_save(m,v,test,fold,method,family,origin)
            m.booster.save_model(dest/f'{method}_fold{fold}.ubj')
            fitdiag={'A':np.eye(2).tolist(),'A_norm':float(np.sqrt(2)),'A_I_norm':0.,'rounds':m.rounds,'SHAP':shapdiag,
                'train_log_loss':metrics.log_loss(train.target_match_result.to_numpy(),predict(m,family,x,z))}
        else:fitdiag=m.diagnostics()
        # Selected fit replay on same training only; compare OOS without using labels.
        replay=fit(family,c,x,p,z,train.target_match_result.to_numpy(),names)
        repeat=float(np.max(np.abs(predict(replay,family,v,zv)-prob)))
        if repeat>1e-10:raise AssertionError('selected-fit deterministic replay')
        dump(dest/f'{method}_fold{fold}.json',{'method':method,'origin':origin,'fold':fold,'selected':c,
            'HPO_surface':surface,'features':names,'train_keys':train.fixture_key.tolist(),'test_keys':test.fixture_key.tolist(),
            'train_n':len(train),'test_n':len(test),'seed':42,'fit':fitdiag,'repeat_max_error':repeat,
            'preprocessor':{'imputer':pp.imputer.statistics_.tolist(),'scale':family=='LR'},
            'seconds':time.perf_counter()-started})
        frame.to_csv(target,index=False,mode='x')
        print(method,fold,'complete',c,'seconds',round(time.perf_counter()-started,2),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--branch',choices=['trees','lr'],required=True)
    a=parser.parse_args()
    if a.branch=='trees':
        for family,origin in [('Fixed','PRE'),('Fixed','HT'),('MC','PRE'),('MC','HT'),('KL','PRE')]:run(family,origin)
    else:
        for regime in ['zero','free','soft','fixed']:run('LR','HT',regime)
