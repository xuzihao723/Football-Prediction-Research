import hashlib,itertools,json,time
from pathlib import Path
import numpy as np
import pandas as pd
from src import metrics
from src.config import PROJECT_ROOT as ROOT
from src.statistics import per_obs_metric

OUT=ROOT/'results/xgb_v4'; P=['p_A','p_D','p_H']; KEYS=['fixture_key','fold','league','season','actual_class']
def write_csv(df,path):
    path.parent.mkdir(parents=True,exist_ok=True); df.to_csv(path,index=False,mode='x')
def dump(name,obj):
    with (OUT/name).open('x',encoding='utf8') as f:json.dump(obj,f,indent=2,ensure_ascii=False,allow_nan=False)
def f1(cm):
    tp=np.diagonal(cm,axis1=-2,axis2=-1); den=cm.sum(axis=-2)+cm.sum(axis=-1)
    return np.divide(2*tp,den,out=np.zeros_like(tp,dtype=float),where=den!=0).mean(axis=-1)
def bootstrap_values(g,weights,clist):
    y=metrics.encode_target(g.actual_class.to_numpy()); p=g[P].to_numpy(); pred=p.argmax(axis=1)
    clusters=(g.league+'|'+g.season).to_numpy()
    ns=np.array([(clusters==c).sum() for c in clist]); n=weights@ns
    result={}
    for metric in ['log_loss','brier','acc']:
        obs=per_obs_metric(g.actual_class.to_numpy(),p,metric)
        sums=np.array([obs[clusters==c].sum() for c in clist]); result[metric]=weights@sums/n
    conf=np.array([np.bincount(3*y[clusters==c]+pred[clusters==c],minlength=9) for c in clist])
    result['macro_f1']=f1((weights@conf).reshape(-1,3,3))
    return result
def holm(p):
    p=np.asarray(p); order=np.argsort(p); out=np.empty(len(p)); out[order]=np.minimum(1,np.maximum.accumulate(p[order]*(len(p)-np.arange(len(p)))))
    return out

def main():
    gates=json.loads((OUT/'engineering/gates.json').read_text())
    combined=[]; diagnostics=[]
    outputs=[('ms_xgb/prematch','ms_xgb_prematch_oos.csv'),('ms_xgb/halftime','ms_xgb_halftime_oos.csv'),
        ('mc_xgb/prematch','mc_xgb_prematch_oos.csv'),('mc_xgb/halftime','mc_xgb_halftime_oos.csv'),
        ('kl_xgb/prematch','kl_xgb_prematch_oos.csv'),('lr_ht','canonical_lr_halftime_oos.csv')]
    for sub,name in outputs:
        frames=[pd.read_csv(p) for p in sorted((OUT/sub).glob('*_fold*.csv'))]
        d=pd.concat(frames,ignore_index=True); d['origin']='PRE' if 'prematch' in sub else 'HT'
        for m,g in d.groupby('method'):
            assert len(g)==2789 and set(g.fold)==set(range(5))
        write_csv(d,OUT/sub/name); combined.append(d)
        diagnostics.extend(json.loads(p.read_text()) for p in sorted((OUT/sub).glob('*_fold*.json')))
    old=ROOT/'results/anchor_continuum'
    pre=pd.concat([pd.read_csv(old/'canonical_prematch_oos.csv'),pd.read_csv(old/'prematch_lr/zero_oos.csv')])
    pre['origin']='PRE'; pre['method']=pre.method.replace({'FreeMarket-LR':'Canonical-Free-LR','SoftAnchor-LR':'Canonical-Soft-LR'})+'-PRE'
    combined.append(pre)
    legacy=pd.read_csv(ROOT/'results/raw/experiment_b_oos.csv')
    base=legacy[(legacy.method=='Bookmaker')&(legacy.representation=='selected')].copy(); base['origin']='PRE'; combined.append(base)
    data=pd.concat(combined,ignore_index=True)
    groups={m:g.sort_values('fixture_key').reset_index(drop=True) for m,g in data.groupby('method')}
    b=groups['Bookmaker']; summary=[]; strata=[]
    for m,g in groups.items():
        assert g.fixture_key.is_unique and g[KEYS].equals(b[KEYS]); metrics.assert_valid_proba(g[P].to_numpy())
        summary.append({'method':m,'origin':g.origin.iloc[0],'N':len(g),**metrics.compute_all_metrics(g.actual_class.to_numpy(),g[P].to_numpy())})
        for dims in [['fold'],['league'],['league','season']]:
            for key,s in g.groupby(dims):
                strata.append({'method':m,'stratum':'|'.join(dims),'key':str(key),'N':len(s),**metrics.compute_all_metrics(s.actual_class.to_numpy(),s[P].to_numpy())})
    write_csv(pd.DataFrame(summary),OUT/'reports/xgb_metrics.csv'); write_csv(pd.DataFrame(strata),OUT/'reports/stratified_metrics.csv')
    write_csv(data,OUT/'reports/validated_oos_predictions.csv')
    clist=np.unique(b.league+'|'+b.season); assert len(clist)==15
    rng=np.random.default_rng(42); draws=rng.choice(len(clist),size=(10000,len(clist)),replace=True)
    weights=np.eye(len(clist),dtype=int)[draws].sum(axis=1)
    boot={m:bootstrap_values(g,weights,clist) for m,g in groups.items()}
    point={r['method']:r for r in summary}
    def contrast(a,b,family='exploratory',primary=False):
        rows=[]
        for metric in ['log_loss','brier','acc','macro_f1']:
            delta=point[a][metric]-point[b][metric]; samples=boot[a][metric]-boot[b][metric]
            rows.append({'first':a,'second':b,'metric':metric,'mean_diff':delta,
                'ci_low':float(np.percentile(samples,2.5)),'ci_high':float(np.percentile(samples,97.5)),
                'p_raw':float((1+np.sum(np.abs(samples-delta)>=abs(delta)))/10001),
                'n_clusters':15,'family':family,'primary':primary})
        return rows
    primary=[]
    for origin in ['PRE','HT']:
        primary+=contrast('MC-XGB-'+origin,'MS-XGB-Fixed-'+origin,'MC_'+origin,True)
        for r in ['Zero','Soft','Fixed']:
            primary+=contrast('Canonical-'+r+'-LR-'+origin,'Canonical-Free-LR-'+origin,'LR_'+origin,True)
    primary+=contrast('KL-XGB-PRE','MS-XGB-Fixed-PRE','KL_PRE',True)
    pdf=pd.DataFrame(primary); pdf['p_holm']=np.nan
    for (fam,metric),idx in pdf.groupby(['family','metric']).groups.items():pdf.loc[idx,'p_holm']=holm(pdf.loc[idx,'p_raw'])
    pdf['interpretation']='Log Loss primary; PRE LR iterative/exploratory; other metrics secondary'
    write_csv(pdf,OUT/'inference/xgb_primary_contrasts.csv')
    allrows=[]
    for origin in ['PRE','HT']:
        names=[m for m,g in groups.items() if g.origin.iloc[0]==origin]
        for a,b in itertools.combinations(names,2):allrows+=contrast(a,b)
    write_csv(pd.DataFrame(allrows),OUT/'inference/xgb_all_contrasts.csv')
    gains=[]
    for a in groups:
        if a.endswith('-PRE') and a[:-4]+'-HT' in groups:gains+=contrast(a,a[:-4]+'-HT','PRE_minus_HT')
    write_csv(pd.DataFrame(gains),OUT/'inference/pre_to_ht_gains.csv')
    # OOS calibration diagnosis, never fitted recalibration.
    bins=[]; calibration=[]
    for m,g in groups.items():
        es=[]
        for label,col in zip(['A','D','H'],P):
            ids=np.argsort(g[col].to_numpy(),kind='stable'); e=0.
            for k,idx in enumerate(np.array_split(ids,10)):
                q=g.iloc[idx]; pred=float(q[col].mean()); freq=float((q.actual_class==label).mean()); n=len(q)
                e+=n/len(g)*abs(pred-freq)
                bins.append({'method':m,'class':label,'bin':k,'N':n,'mean_probability':pred,'observed_frequency':freq})
            es.append(e); calibration.append({'method':m,'class':label,'ECE':e})
        calibration.append({'method':m,'class':'macro','ECE':float(np.mean(es))})
    write_csv(pd.DataFrame(bins),OUT/'reports/calibration_bins.csv'); write_csv(pd.DataFrame(calibration),OUT/'reports/calibration_summary.csv')
    dump('reports/fold_diagnostics.json',diagnostics)
    hpo=[]; surfaces=[]
    for d in diagnostics:
        best=min(d['HPO_surface'],key=lambda x:x['score'])
        hpo.append({'method':d['method'],'fold':d['fold'],'selected':json.dumps(d['selected']),
            'inner_log_loss':best['score'],'candidate_count':len(d['HPO_surface']),
            'train_n':d['train_n'],'test_n':d['test_n'],'seconds':d['seconds'],'repeat_max_error':d['repeat_max_error']})
        for s in d['HPO_surface']:surfaces.append({'method':d['method'],'fold':d['fold'],'candidate':json.dumps(s['candidate']),'score':s['score']})
    write_csv(pd.DataFrame(hpo),OUT/'reports/xgb_hpo_summary.csv'); write_csv(pd.DataFrame(surfaces),OUT/'reports/full_hpo_surface.csv')
    # SHAP: only tree correction; include bias in fixture output, exclude in importance.
    s=pd.concat([pd.read_csv(p) for branch in ['ms_xgb','mc_xgb','kl_xgb'] for p in (OUT/'shap'/branch).glob('*_fold*.csv')],ignore_index=True)
    write_csv(s,OUT/'shap/residual_shap_fixture_level.csv')
    fs=s[s.feature!='__bias__'].copy()
    for col in ['phi_HD','phi_AD']:fs['abs_'+col]=fs[col].abs()
    feat=fs.groupby(['method','origin','feature','group'])[['abs_phi_HD','abs_phi_AD']].mean().reset_index()
    write_csv(feat,OUT/'shap/residual_shap_feature_summary.csv'); write_csv(feat,OUT/'shap/residual_shap_draw_reference_summary.csv')
    gf=fs.groupby(['method','origin','fixture_key','league','group'])[['phi_HD','phi_AD']].sum().reset_index()
    gf['abs_HD']=gf.phi_HD.abs(); gf['abs_AD']=gf.phi_AD.abs()
    gs=gf.groupby(['method','origin','group'])[['abs_HD','abs_AD']].mean().reset_index()
    write_csv(gs,OUT/'shap/residual_shap_group_summary.csv')
    write_csv(gf.groupby(['method','origin','league','group'])[['abs_HD','abs_AD']].mean().reset_index(),OUT/'shap/residual_shap_league_summary.csv')
    changes=[]
    for stem in ['MS-XGB-Fixed','MC-XGB']:
        a=gs[gs.method==stem+'-PRE'].set_index('group'); b=gs[gs.method==stem+'-HT'].set_index('group')
        for group in a.index.intersection(b.index):changes.append({'model':stem,'group':group,
            'HT_minus_PRE_mean_abs_HD':float(b.loc[group,'abs_HD']-a.loc[group,'abs_HD']),
            'HT_minus_PRE_mean_abs_AD':float(b.loc[group,'abs_AD']-a.loc[group,'abs_AD'])})
    write_csv(pd.DataFrame(changes),OUT/'shap/pre_ht_group_changes.csv')
    interactions=[r for p in (OUT/'shap/mc_xgb').glob('*interactions.json') for r in json.loads(p.read_text())]
    write_csv(pd.DataFrame(interactions),OUT/'shap/mc_xgb_interaction_summary.csv')
    # hash confirmation includes both exact v3 bytes and manuscript.
    before=json.loads((OUT/'protocol/before_hashes.json').read_text()); changed=[]
    for rel,sha in before.items():
        p=ROOT.parent/rel
        if hashlib.sha256(p.read_bytes()).hexdigest()!=sha:changed.append(rel)
    permitted={'acml_final_revised/tools/freeze_xgb_v4.py'}
    unexpected=[p for p in changed if p.replace('\\','/') not in permitted]
    if unexpected:raise AssertionError('Unexpected old file changes: '+str(unexpected))
    dump('reports/integrity.json',{'old_file_changes':changed,'unexpected_changes':unexpected,'v3_and_manuscript_unchanged':True,
        'all_model_alignment_passed':True,'N':2789,'clusters':15})
    print(pd.DataFrame(summary).to_string(index=False),flush=True)
if __name__=='__main__':main()
