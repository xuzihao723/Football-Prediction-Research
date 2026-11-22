"""Ordinary sklearn multinomial LR and native XGBoost; no market offset."""
import argparse,hashlib,json,time,warnings,itertools
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning
from .. import metrics
from ..config import PROJECT_ROOT,CONFIG_DIR
from ..models.market import bookmaker_prior
from ..pooled_split import pooled_outer_split,pooled_inner_temporal_split,assert_temporal
from ..outputs import prediction_frame
from .common import load_data,history_cols

OUT=PROJECT_ROOT/'results/standard_prematch_baselines'
METHODS={'Ordinary-Historical-LR':('LR',False),'Ordinary-Joint-LR':('LR',True),
         'Ordinary-Historical-XGB':('XGB',False),'Ordinary-Joint-XGB':('XGB',True)}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(name,value):
    p=OUT/name;p.parent.mkdir(parents=True,exist_ok=True)
    if isinstance(value,pd.DataFrame):value.to_csv(p,index=False,mode='x')
    else:
        with p.open('x',encoding='utf8') as f:
            if isinstance(value,str):f.write(value)
            else:json.dump(value,f,indent=2,ensure_ascii=False,allow_nan=False)
def matrix(df,cols,market):
    X=df[cols].to_numpy(float)
    if market:
        _,z,valid=bookmaker_prior(df)
        if not valid.all():raise ValueError('joint model requires valid market odds')
        X=np.c_[X,z]
    if np.isinf(X).any():raise ValueError('infinite predictors')
    return X

class OrdinaryModel:
    def __init__(self,kind,params,rounds=800):
        self.kind=kind;self.params=params;self.rounds=rounds
    def fit(self,X,y,validation=None):
        yi=metrics.encode_target(np.asarray(y))
        if set(yi)!={0,1,2}:raise ValueError('training classes incomplete')
        self.imputer=SimpleImputer(strategy='median',keep_empty_features=True).fit(X)
        Z=self.imputer.transform(X);self.scaler=None
        if self.kind=='LR':
            self.scaler=StandardScaler().fit(Z);Z=self.scaler.transform(Z)
            self.model=LogisticRegression(C=self.params['C'],solver='lbfgs',l1_ratio=0.,
                fit_intercept=True,class_weight=None,tol=1e-7,max_iter=20000,random_state=42)
            with warnings.catch_warnings():
                warnings.simplefilter('error',ConvergenceWarning);self.model.fit(Z,yi)
            assert list(self.model.classes_)==[0,1,2]
            self.diagnostics={'converged':True,'iterations':self.model.n_iter_.tolist(),'classes':[0,1,2]}
        else:
            d=xgb.DMatrix(Z,label=yi,nthread=1);ev=[]
            if validation is not None:
                V,yv=validation;dv=xgb.DMatrix(self.imputer.transform(V),label=metrics.encode_target(np.asarray(yv)),nthread=1);ev=[(dv,'validation')]
            pars={'objective':'multi:softprob','num_class':3,'eval_metric':'mlogloss','seed':42,'nthread':1,
                'tree_method':'hist','subsample':.8,'colsample_bytree':.8,'alpha':0.,'gamma':0.,**self.params}
            self.model=xgb.train(pars,d,num_boost_round=self.rounds,evals=ev,
                early_stopping_rounds=40 if ev else None,verbose_eval=False)
            self.best_rounds=int(self.model.best_iteration)+1 if ev else self.rounds
            self.diagnostics={'best_rounds':self.best_rounds,'trained_rounds':self.model.num_boosted_rounds(),
                'stopping':'inner validation patience40' if ev else 'refit fixed inner-selected budget',
                'round_cap_reached':self.model.num_boosted_rounds()==self.rounds}
        return self
    def predict_proba(self,X):
        Z=self.imputer.transform(X)
        if self.scaler is not None:Z=self.scaler.transform(Z)
        p=self.model.predict_proba(Z) if self.kind=='LR' else self.model.predict(xgb.DMatrix(Z,nthread=1),iteration_range=(0,self.best_rounds))
        p=np.asarray(p,float);metrics.assert_valid_proba(p);return p

def grids():
    c=json.loads((CONFIG_DIR/'hpo_final_models.json').read_text())
    return {'LR':[{'C':v} for v in c['Historical-LR']['C']], 'XGB':c['XGB']['configs']}

def aligned(a,b):
    keys=['fixture_key','fold','league','season','actual_class']
    if a.fixture_key.duplicated().any() or b.fixture_key.duplicated().any():raise AssertionError('duplicate keys')
    a=a.sort_values('fixture_key').reset_index(drop=True);b=b.sort_values('fixture_key').reset_index(drop=True)
    if not a[keys].equals(b[keys]):raise AssertionError('paired metadata mismatch')
    return a,b

def tune(train,cols,kind,market):
    splits=pooled_inner_temporal_split(train,min_train_seasons=2);assert splits
    blocks=[]
    for sp in splits:
        a=train.iloc[sp['train_idx']];b=train.iloc[sp['val_idx']];assert_temporal(a,b)
        blocks.append((matrix(a,cols,market),a.target_match_result.to_numpy(),matrix(b,cols,market),b.target_match_result.to_numpy()))
    search=[]
    for params in grids()[kind]:
        scores=[];rounds=[];diags=[]
        for X,y,V,yv in blocks:
            m=OrdinaryModel(kind,params).fit(X,y,validation=(V,yv) if kind=='XGB' else None)
            scores.append(metrics.log_loss(yv,m.predict_proba(V)));diags.append(m.diagnostics)
            if kind=='XGB':rounds.append(m.best_rounds)
        search.append({'params':params,'weighted_LL':float(np.average(scores,weights=[len(x[3]) for x in blocks])),
            'split_LL':scores,'split_weights':[len(x[3]) for x in blocks],'rounds':rounds,'fit_diagnostics':diags})
    j=int(np.argmin([x['weighted_LL'] for x in search]));best=search[j]
    return {'selected_index':j,'selected_params':best['params'],
        'refit_rounds':int(np.median(best['rounds'])) if kind=='XGB' else None,'surface':search,'inner_splits':splits}

def freeze():
    OUT.mkdir(parents=True,exist_ok=False)
    d=load_data();frozen=pd.read_csv(PROJECT_ROOT/'results/raw/experiment_b_oos.csv',usecols=['method','representation','fixture_key','fold','league','season'])
    f=frozen[(frozen.method=='Bookmaker')&(frozen.representation=='selected')];assert len(f)==2789
    foldspec=[]
    for k in range(5):
        tr,te=pooled_outer_split(d,k);tr=tr.loc[bookmaker_prior(tr)[2]].reset_index(drop=True);te=te.loc[bookmaker_prior(te)[2]]
        assert te.fixture_key.tolist()==f[f.fold==k].fixture_key.tolist()
        foldspec.append({'fold':k,'train_N':len(tr),'test_N':len(te),'train_keys':tr.fixture_key.tolist(),'test_keys':te.fixture_key.tolist(),
            'history_cols':history_cols(k),'inner_splits':pooled_inner_temporal_split(tr,min_train_seasons=2)})
    protocol={'id':'standard_prematch_baselines','goal':'ordinary LR/XGB historical vs historical+market, fixed feature representation',
        'features':'existing selected29 columns incl league context; no feature selection/reengineering in this experiment',
        'models':METHODS,'grids':grids(),'objective':'multiclass log loss','class_order':['A','D','H'],
        'sample':'same odds-valid outer-train across all four methods; identical frozen2789 OOS keys',
        'market':'available pre-match market prior; two proportional-devig draw-reference logits appended as ordinary features',
        'LR':'sklearn multinomial lbfgs L2; C grid; unpenalized intercept; train median then StandardScaler on ALL inputs including logits by deliberate ordinary-pipeline design; not legacy reference-class ridge',
        'XGB':'native multiclass hist; no base_margin/custom objective; append market features; seed42/nthread1/subsample.8/colsample.8/alpha0/gamma0',
        'early_stopping':'inner validation only; max800/patience40; refit median selected inner best_rounds; no outer stopping',
        'fairness':'identical sample/splits within and across backbones; same feature-set budgets within backbone; LR6 vs XGB18 candidates explicitly unequal parameter counts, not equal compute claim',
        'selection':'validation-N weighted LL; first argmin; no outer-driven revisions',
        'replay':'each selected outer fit repeated with frozen params; compare predictions',
        'contrasts':'joint-historical within backbone; each model-market; Joint-XGB minus Joint-LR; first-minus-second, existing paired league-season10000 seed42; raw descriptive95%CI, no winner from marginal interval',
        'equivalence_to_legacy':'not assumed; conventional joint-LR changes scaling/parameterization; training historical subset is odds-matched',
        'limitations':'known repeated use of outer seasons; inherited feature/selection provenance; no prospective confirmation',
        'frozen_folds':foldspec,'data_sha256':sha(PROJECT_ROOT/'official_data/modeling/ModelingData_harmonized.csv')}
    save('protocol.json',protocol);save('protocol_sha256.txt',sha(OUT/'protocol.json'))
    protect={str(p.relative_to(PROJECT_ROOT.parent)):sha(p) for p in PROJECT_ROOT.parent.rglob('*') if p.is_file() and OUT not in p.parents and '__pycache__' not in p.parts and '.pytest_cache' not in p.parts}
    save('before_hashes.json',protect)
    print('frozen',sha(OUT/'protocol.json'),flush=True)

def run():
    cfg=json.loads((OUT/'protocol.json').read_text());assert cfg['grids']==grids()
    assert sha(OUT/'protocol.json')==(OUT/'protocol_sha256.txt').read_text()
    d=load_data();diags=[];preds=[]
    for k in range(5):
        tr,te=pooled_outer_split(d,k);tr=tr.loc[bookmaker_prior(tr)[2]].reset_index(drop=True);te=te.loc[bookmaker_prior(te)[2]].reset_index(drop=True)
        assert tr.fixture_key.tolist()==cfg['frozen_folds'][k]['train_keys']
        assert te.fixture_key.tolist()==cfg['frozen_folds'][k]['test_keys'];assert_temporal(tr,te)
        cols=history_cols(k);assert cols==cfg['frozen_folds'][k]['history_cols']
        for name,(kind,market) in METHODS.items():
            start=time.perf_counter();t=tune(tr,cols,kind,market)
            X=matrix(tr,cols,market);V=matrix(te,cols,market);y=tr.target_match_result.to_numpy()
            model=OrdinaryModel(kind,t['selected_params'],t['refit_rounds'] or 800).fit(X,y)
            p=model.predict_proba(V)
            again=OrdinaryModel(kind,t['selected_params'],t['refit_rounds'] or 800).fit(X,y)
            error=float(np.max(abs(again.predict_proba(V)-p)));assert error<=1e-10
            artifact=OUT/f'models/{name}_fold{k}.joblib';artifact.parent.mkdir(exist_ok=True)
            if artifact.exists():raise FileExistsError(artifact)
            joblib.dump(model,artifact);assert np.array_equal(joblib.load(artifact).predict_proba(V),p)
            frame=prediction_frame(te,p,k,name,'selected');preds.append(frame)
            save(f'folds/{name}_fold{k}.csv',frame)
            diag={'fold':k,'method':name,'information_set':'history+market' if market else 'history','features':cols+(['market_z_H','market_z_A'] if market else []),
                'training_N':len(tr),'test_N':len(te),'hpo':t,'refit':model.diagnostics,'replay_max_difference':error,
                'seconds':time.perf_counter()-start,'model_path':str(artifact.relative_to(PROJECT_ROOT)),
                'imputer_statistics':model.imputer.statistics_.tolist(),
                'scaler_mean':model.scaler.mean_.tolist() if model.scaler else None,
                'scaler_scale':model.scaler.scale_.tolist() if model.scaler else None}
            diags.append(diag);save(f'folds/{name}_fold{k}.json',diag)
            print(name,k,'done',t['selected_params'],'rounds',t['refit_rounds'],flush=True)
    save('oos_predictions.csv',pd.concat(preds,ignore_index=True));save('fold_diagnostics.json',diags)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['freeze','run']);args=parser.parse_args()
    {'freeze':freeze,'run':run}[args.stage]()
