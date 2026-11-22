"""Final approved direct/sequential and score-only robustness study. Additive."""
import argparse,hashlib,json,itertools,time,warnings
from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.special import softmax
from scipy.optimize import minimize
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from src.config import PROJECT_ROOT as ROOT
from src import metrics
from src.experiments.common import load_data,history_cols
from src.experiments.standard_baselines import OrdinaryModel
from src.models.anchoring import CanonicalAnchorLR
from src.models.xgb_v4 import base_margin,soft_targets,soft_loss_grad,market_context
from src.models.market import bookmaker_prior,reference_logits
from src.feature_registry import get_registry
from src.pooled_split import pooled_outer_split,pooled_inner_temporal_split,assert_temporal,legal_time,match_time
from src.outputs import prediction_frame
from src.experiments import experiment_d as old_d

OUT=ROOT/'results/final_framework'
ALGORITHMS=['Joint-LR','Zero-LR','Free-LR','Soft-LR','Fixed-LR','Joint-XGB','Fixed-XGB','MC-XGB','KL-XGB']
REG={'Zero-LR':'zero','Free-LR':'free','Soft-LR':'soft','Fixed-LR':'fixed','Tied-Zero-LR':'tied_zero'}
HP=[.01,.1,.3,1.,3.,10.];MP=[0.,.001,.01,.1,.3,1.,3.,10.,100.];RHO=[0.,.05,.1,.25,.5,.75,.9]
BASECOLS=history_cols(0);HTCOLS=get_registry().ht_score_columns
GENERIC=['RF','LightGBM','CatBoost','FNN']

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(path,obj):
    p=OUT/path;p.parent.mkdir(parents=True,exist_ok=True)
    if isinstance(obj,pd.DataFrame):obj.to_csv(p,index=False,mode='x')
    else:
        with p.open('x',encoding='utf8') as f:json.dump(obj,f,indent=2,ensure_ascii=False,allow_nan=False)
def grid(alg):
    c=json.loads((ROOT/'configs/hpo_final_models.json').read_text())
    if alg=='Joint-LR':return [{'C':x} for x in c['Historical-LR']['C']]
    if alg in REG:return [{'lambda_h':h,'market_penalty':a} for h in HP for a in (MP if alg in ['Zero-LR','Soft-LR'] else [0.])]
    if alg in GENERIC:
        original=old_d._candidate_params(alg)
        return [original[i] for i in sorted(set(np.linspace(0,len(original)-1,4).astype(int)))]
    return [{'tree':p,'rho':rho} for p in c['XGB']['configs'] for rho in (RHO if alg=='KL-XGB' else [0.])]

def clean_json(x):
    if isinstance(x,dict):return {str(k):clean_json(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [clean_json(v) for v in x]
    if isinstance(x,np.ndarray):return clean_json(x.tolist())
    if isinstance(x,(float,np.floating)):return float(x) if np.isfinite(x) else None
    if isinstance(x,np.integer):return int(x)
    return x

def design(d,origin,alg):
    cols=BASECOLS+(HTCOLS if origin=='HT' else [])
    X=d[cols].to_numpy(float);z=d[['_zH','_zA']].to_numpy(float)
    if alg=='MC-XGB':X=np.c_[X,d[['_entropy','_gap','_overround']].to_numpy(float)]
    if alg.startswith('Joint') or alg in GENERIC:X=np.c_[X,z]
    return X,z

def fit_predict(train,query,alg,origin,params,rounds=None,validation=False):
    assert_temporal(train,query,'final fit/query')
    X,z=design(train,origin,alg);V,q=design(query,origin,alg)
    y=train.target_match_result.to_numpy()
    if set(y)!={'A','D','H'}:raise RuntimeError('incomplete training classes')
    if alg=='Joint-LR':
        m=OrdinaryModel('LR',params).fit(X,y);return m.predict_proba(V),None,m.diagnostics
    imp=SimpleImputer(strategy='median',keep_empty_features=True).fit(X);X=imp.transform(X);V=imp.transform(V)
    if alg in REG:
        scale=StandardScaler().fit(X);X=scale.transform(X);V=scale.transform(V)
        m=CanonicalAnchorLR(REG[alg],lambda_h=params['lambda_h'],lambda_a=params['market_penalty'],lambda_0=params['market_penalty']).fit(X,z,y)
        return m.predict_proba(V,q),None,m.diagnostics()
    if alg in GENERIC:
        if alg=='FNN':
            scale=StandardScaler().fit(X);X=scale.transform(X);V=scale.transform(V)
        if validation:
            m,r,p=old_d._fit_eval(alg,X,y,V,query.target_match_result.to_numpy(),params)
        else:
            m=old_d._fit_final(alg,X,y,params,rounds);p=old_d.predict_classifier(m,V);r=rounds
        return p,r,{'rounds':r,'class_order':[0,1,2]}
    pars={'objective':'multi:softprob','num_class':3,'eval_metric':'mlogloss','tree_method':'hist','seed':42,'nthread':1,
        'subsample':.8,'colsample_bytree':.8,'alpha':0.,'gamma':0.,**params['tree']}
    yi=metrics.encode_target(y);isjoint=alg=='Joint-XGB'
    dm=xgb.DMatrix(X,label=yi,nthread=1,**({} if isjoint else {'base_margin':base_margin(z)}))
    dv=xgb.DMatrix(V,nthread=1,**({} if isjoint else {'base_margin':base_margin(q)}))
    if validation:dv.set_label(metrics.encode_target(query.target_match_result.to_numpy()))
    obj=None
    if alg=='KL-XGB':
        prior=softmax(base_margin(z),axis=1);t=soft_targets(y,prior,params['rho']).astype(np.float32)
        def obj(pred,dm):
            _,g,h=soft_loss_grad(pred,t);return g,h
    # Native eval_metric is true-label multiclass LL even for custom soft-target gradients.
    b=xgb.train(pars,dm,num_boost_round=800 if validation else int(rounds),obj=obj,
        evals=[(dv,'validation')] if validation else [],early_stopping_rounds=40 if validation else None,verbose_eval=False)
    nr=int(b.best_iteration)+1 if validation else int(rounds)
    p=b.predict(dv,iteration_range=(0,nr));p=np.asarray(p,float);metrics.assert_valid_proba(p)
    return p,nr,{'rounds':nr,'trained_rounds':b.num_boosted_rounds(),'A':None if isjoint else np.eye(2).tolist(),'rho':params['rho']}

def tuned(train,alg,origin,min_seasons=2,fallback=False):
    splits=pooled_inner_temporal_split(train,min_train_seasons=min_seasons)
    if not splits:
        if not fallback:raise RuntimeError('no HPO split')
        return {'params':grid(alg)[0],'rounds':100 if alg=='FNN' else 200 if 'XGB' in alg or alg in ['LightGBM','CatBoost'] else None,
            'fallback':True,'surface':[],'reason':'pre-frozen earliest producing origin only; no query labels used'}
    token=hashlib.sha256(json.dumps({'keys':train.fixture_key.tolist(),'alg':alg,'origin':origin,'min':min_seasons,
        'z':train[['_zH','_zA']].round(14).to_numpy().tolist()},sort_keys=True).encode()).hexdigest()
    cache=OUT/'cache'/f'{token}.json'
    if cache.exists():return json.loads(cache.read_text())
    surface=[]
    for c in grid(alg):
        ll=[];rr=[];nn=[]
        for sp in splits:
            a=train.iloc[sp['train_idx']].reset_index(drop=True);b=train.iloc[sp['val_idx']].reset_index(drop=True)
            p,r,_=fit_predict(a,b,alg,origin,c,validation=True);ll.append(metrics.log_loss(b.target_match_result.to_numpy(),p));nn.append(len(b))
            if r:rr.append(r)
        surface.append({'params':c,'score':float(np.average(ll,weights=nn)),'split_scores':ll,'split_N':nn,'rounds':rr})
    best=min(surface,key=lambda r:r['score'])
    result={'params':best['params'],'rounds':int(np.median(best['rounds'])) if best['rounds'] else None,'fallback':False,'surface':surface}
    save('cache/'+cache.name,result);return result

def source():
    d=load_data();p,z,v=bookmaker_prior(d);d=d.loc[v].copy();p=p[v];z=z[v]
    d['_zH']=z[:,0];d['_zA']=z[:,1]
    ctx=market_context(p,d[get_registry().market_prior_columns].to_numpy(float))
    d[['_entropy','_gap','_overround']]=ctx
    return d.reset_index(drop=True)

def outer_split(d,k):
    tr,te=pooled_outer_split(d,k)
    assert len(te)==[544,530,527,594,594][k]
    return tr,te

def anchor_oof(train,alg,origin='PRE'):
    parts=[];trace=[]
    for sp in pooled_inner_temporal_split(train,min_train_seasons=1):
        a=train.iloc[sp['train_idx']].reset_index(drop=True);b=train.iloc[sp['val_idx']].reset_index(drop=True)
        t=tuned(a,alg,origin,fallback=True);p,r,diag=fit_predict(a,b,alg,origin,t['params'],t['rounds'])
        x=b.copy();z=reference_logits(p);x['_zH']=z[:,0];x['_zA']=z[:,1];parts.append(x)
        trace.append({'producer_keys':a.fixture_key.tolist(),'query_keys':b.fixture_key.tolist(),'hpo':t,
            'producer_latest':str(legal_time(a).max()),'query_earliest':str(match_time(b).min()),'origin':origin})
    if not parts:raise RuntimeError('no OOF anchors')
    return pd.concat(parts,ignore_index=True),trace

def frozen_reference(k):
    cols=['fixture_key','league','season','fold','method','representation','actual_class','p_A','p_D','p_H']
    d=pd.read_csv(ROOT/'results/raw/experiment_b_oos.csv',usecols=cols)
    return d[(d.method=='Bookmaker')&(d.representation=='selected')&(d.fold==k)].reset_index(drop=True)

def experiment(train,test,k,alg,origin,route):
    key=f'{route}/{origin}/{alg}/fold{k}'
    if (OUT/(key+'.json')).exists():return pd.read_csv(OUT/(key+'.csv'))
    started=time.perf_counter();t=tuned(train,alg,origin,min_seasons=1 if route!='direct' and route!='robustness' else 2)
    p,_,diag=fit_predict(train,test,alg,origin,t['params'],t['rounds'])
    if k==0:
        replay,_,_=fit_predict(train,test,alg,origin,t['params'],t['rounds']);err=float(np.max(abs(replay-p)))
        if err>1e-8:raise RuntimeError('replay discrepancy')
    else:err=None
    frame=prediction_frame(test,p,k,f'{route}:{alg}:{origin}','selected')
    ref=frozen_reference(k)
    for c in ['fixture_key','league','season','actual_class']:assert frame[c].tolist()==ref[c].tolist()
    save(key+'.csv',frame)
    save(key+'.json',clean_json({'fold':k,'alg':alg,'origin':origin,'route':route,'train_keys':train.fixture_key.tolist(),'test_keys':test.fixture_key.tolist(),
        'N_train':len(train),'N_test':len(test),'hpo':t,'fit':diag,'replay_max_error':err,'seconds':time.perf_counter()-started}))
    print('DONE',key,'time',round(time.perf_counter()-started,1),flush=True)
    return frame

def freeze():
    OUT.mkdir(parents=True,exist_ok=False);d=source();counts=[]
    for k in range(5):
        tr,te=outer_split(d,k);sp=pooled_inner_temporal_split(tr,min_train_seasons=1)
        covered=set(i for s in sp for i in s['val_idx'])
        counts.append({'fold':k,'train':len(tr),'test':len(te),'OOF_N':len(covered),'warmup_N':len(tr)-len(covered)})
    cfg={'id':'final_framework','user_decisions':'no cards; LR four regimes plus ordinary Joint-LR at PRE/HT; XGB Fixed/MC/KL plus ordinary Joint; direct and sequential separately; seven learners score-only LogLoss',
        'features':BASECOLS,'HT':HTCOLS,'algorithms':ALGORITHMS,'grids':{a:grid(a) for a in ALGORITHMS+['Tied-Zero-LR']+GENERIC},
        'class_order':['A','D','H'],'odds':'available pre-match market prior, proportional de-vig; no odds timing claims',
        'population':counts,'outer':'unchanged five actual-date trimmed expanding folds; frozen2789 keys',
        'direct':'eta=A*z_market+r(history,HTscore); ordinaryJoint concatenates; directKL q=market',
        'sequential':'each algorithm own PRE producing model. HT eta=A*z_PRE_OOF+r(history,HTscore); Joint treats PRE logit as feature; sequentialKL q=PRE_OOF; original market entropy/gap/overround only for MC',
        'OOF':'min1 earlier season; upstream HPO min2; earliest inadequate HPO gets first frozen candidate with fixed200 boosting/100FNN; producer trained only earlier rows; no outer-selected parameter backfill',
        'warmup':'first odds season without temporal producing history omitted from HT correction training; ALL target test rows retained',
        'matched_route_control':'direct_matched uses exact same HT OOF-supported train keys but market prior; same HT HPO min1 as sequential; independent model fits. Primary direct remains full training min2',
        'sequential_comparison':'compare sequential vs direct_matched, not full-training direct as architecture-only effect',
        'XGB_common':'18 tree configs + inner max800/patience40; subsample/colsample .8; single CPU thread seed42; selected median rounds full refit; KL126 grid true-label mlogloss early stopping',
        'LR_common':'history-only standardized; canonical market natural; OrdinaryJoint all-input standardized; tied uses same solver as Zero; Zero/Soft grid includes .3/3 so tied path nested',
        'robustness':'LR/XGB reuse exact direct Joint PRE/HT; RF/LGB/Cat/FNN 4 deterministic evenly indexed candidates from old grid, model-specific tuning budgets explicit; no cards',
        'voting':'temporalOOF min1 nestedHPO min2+frozen earliest fallback; choose3 smallest pooledOOF LogLoss tie by name; nonnegative simplex weights optimizeOOF true-labelLL; HT includes score-only; no outer labels',
        'reporting':'both routes retained independent of outcome; final story chosen only after all valid outputs; no claim independent prospective confirmation',
        'inference':'same metadata keys, 15 league-season blocks,10000 seed42; raw percentile95CI; null-centered bootstrap p+Holm within origin/backbone core comparison family',
        'primary_pairs':'canonical four minus ordinaryJointLR per origin; FixedXGB minus ordinaryJointXGB; MC/KL minus FixedXGB; TiedZero minusZero PRE; sequential minus direct_matched by origin/backbone as secondary family',
        'no_E11':'do not make separate PRE-HT delta study; present origin tables',
        'stops':'any population mismatch/leakage/optimizer failure stops affected experiment; no grids changed by outer results',
        'no_manuscript_edit':True,'no_devig_sensitivity':True,'no_MCHR':True,'no_learnable_A_XGB':True}
    save('protocol.json',cfg);(OUT/'protocol_sha256.txt').write_text(sha(OUT/'protocol.json'))
    protect={str(p.relative_to(ROOT.parent)):sha(p) for p in ROOT.parent.rglob('*') if p.is_file() and OUT not in p.parents and '__pycache__' not in p.parts and '.pytest_cache' not in p.parts}
    save('before_hashes.json',protect);print(counts,flush=True)

def direct():
    d=source()
    for k in range(5):
        tr,te=outer_split(d,k)
        for origin in ['PRE','HT']:
            for alg in ALGORITHMS+(['Tied-Zero-LR'] if origin=='PRE' else []):
                # Exact unchanged canonical endpoints are reused, not refitted.
                old=None;oldname=None
                if origin=='PRE' and alg in ['Free-LR','Fixed-LR']:
                    old=ROOT/'results/anchor_continuum/canonical_prematch_oos.csv'
                    oldname={'Free-LR':'FreeMarket-LR','Fixed-LR':'Canonical-Fixed-LR'}[alg]
                elif origin=='PRE' and alg=='Tied-Zero-LR':
                    old=ROOT/'results/final_canonical/prematch/tied_zero_lr_oos.csv';oldname='Canonical-Tied-Zero-LR'
                elif origin=='HT' and alg in ['Free-LR','Fixed-LR']:
                    old=ROOT/f'results/xgb_v4/lr_ht/Canonical-{alg}-HT_fold{k}.csv';oldname=f'Canonical-{alg}-HT'
                if old is not None:
                    prior=pd.read_csv(old);prior=prior[(prior.method==oldname)&(prior.fold==k)].reset_index(drop=True)
                    assert prior.fixture_key.tolist()==te.fixture_key.tolist()
                    prior.method=f'direct:{alg}:{origin}';key=f'direct/{origin}/{alg}/fold{k}'
                    if not (OUT/(key+'.json')).exists():
                        save(key+'.csv',prior);save(key+'.json',{'fold':k,'alg':alg,'origin':origin,'route':'direct','reused_from':str(old.relative_to(ROOT))})
                    continue
                # Reuse established exactly same ordinary PRE path where configurations match.
                if origin=='PRE' and alg in ['Joint-LR','Joint-XGB']:
                    old=ROOT/'results/standard_prematch_baselines/folds'/f'Ordinary-{alg}_fold{k}.csv'
                    prior=pd.read_csv(old);prior.method=f'direct:{alg}:PRE'
                    key=f'direct/PRE/{alg}/fold{k}'
                    if not (OUT/(key+'.json')).exists():
                        save(key+'.csv',prior);save(key+'.json',{'fold':k,'alg':alg,'origin':origin,'route':'direct','reused_from':str(old.relative_to(ROOT))})
                    continue
                experiment(tr,te,k,alg,origin,'direct')

def sequential():
    d=source()
    for k in range(5):
        tr,te=outer_split(d,k)
        for alg in ALGORITHMS:
            key=f'anchors/{alg}/fold{k}'
            if (OUT/(key+'.csv')).exists():oof=pd.read_csv(OUT/(key+'.csv'))
            else:
                oof,trace=anchor_oof(tr,alg)
                save(key+'.csv',oof);save(key+'.json',trace)
            tp=tuned(tr,alg,'PRE');pre,_,_=fit_predict(tr,te,alg,'PRE',tp['params'],tp['rounds'])
            query=te.copy();z=reference_logits(pre);query['_zH']=z[:,0];query['_zA']=z[:,1]
            experiment(oof,query,k,alg,'HT','sequential')
            matched=tr.set_index('fixture_key').loc[oof.fixture_key].reset_index()
            experiment(matched,te,k,alg,'HT','direct_matched')

def robustness():
    d=source();members=['Joint-LR','Joint-XGB']+GENERIC
    for origin in ['PRE','HT']:
        for k in range(5):
            tr,te=outer_split(d,k);outerprobs={};oofprobs={};labels=None;keys=None;traces={}
            for alg in members:
                if alg in GENERIC:fr=experiment(tr,te,k,alg,origin,'robustness')
                else:fr=pd.read_csv(OUT/f'direct/{origin}/{alg}/fold{k}.csv')
                outerprobs[alg]=fr[['p_A','p_D','p_H']].to_numpy(float)
                oof,trace=anchor_oof(tr,alg,origin)
                if labels is None:labels=oof.target_match_result.to_numpy();keys=oof.fixture_key.tolist()
                else:assert keys==oof.fixture_key.tolist()
                oofprobs[alg]=softmax(base_margin(oof[['_zH','_zA']].to_numpy(float)),axis=1);traces[alg]=trace
            scores={a:metrics.log_loss(labels,p) for a,p in oofprobs.items()};chosen=sorted(scores,key=lambda a:(scores[a],a))[:3]
            y=metrics.encode_target(labels);P=np.stack([oofprobs[a][np.arange(len(y)),y] for a in chosen],axis=1)
            def loss(w):return -np.log(np.clip(P@w,1e-12,1)).mean()
            opt=minimize(loss,np.ones(3)/3,method='SLSQP',bounds=[(0,1)]*3,constraints={'type':'eq','fun':lambda w:w.sum()-1},options={'ftol':1e-12,'maxiter':2000})
            if not opt.success:raise RuntimeError('voting weights failed')
            w=np.maximum(opt.x,0);w/=w.sum();p=sum(wi*outerprobs[a] for wi,a in zip(w,chosen));metrics.assert_valid_proba(p)
            save(f'robustness/{origin}/SmartVoting/fold{k}.csv',prediction_frame(te,p,k,f'robustness:SmartVoting:{origin}','selected'))
            save(f'robustness/{origin}/SmartVoting/fold{k}.json',clean_json({'scores':scores,'members':chosen,'weights':w.tolist(),'OOF_keys':keys,'OOF_traces':traces,'success':bool(opt.success)}))
            print('VOTING',origin,k,chosen,w,flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['freeze','direct','sequential','robustness']);a=parser.parse_args()
    {'freeze':freeze,'direct':direct,'sequential':sequential,'robustness':robustness}[a.stage]()
