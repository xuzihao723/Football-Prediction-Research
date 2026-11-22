import json,time
import numpy as np
from scipy.special import softmax
from src.models.xgb_v4 import StructuredXGB,TreeCorrection,base_margin,draw_reference
from src.config import PROJECT_ROOT as ROOT
from src.experiments.common import load_data,history_cols,odds_subset
from src.pooled_split import pooled_outer_split,pooled_inner_temporal_split
from src.preprocessing import Preprocessor

OUT=ROOT/'results/xgb_v4'
def main():
    cfg=json.loads((OUT/'protocol/frozen_protocol_v4.json').read_text())
    df=load_data(); tr,_=pooled_outer_split(df,0); tr,_,z=odds_subset(tr)
    hist=history_cols(0); X=Preprocessor(False).fit_transform(tr[hist]); y=tr.target_match_result.to_numpy()
    rng=np.random.default_rng(42); xs=rng.normal(size=(300,4)); zs=rng.normal(size=(300,2))
    ys=np.array([rng.choice(3,p=p) for p in softmax(base_margin(zs)+np.c_[xs[:,0],xs[:,1],xs[:,2]],axis=1)])
    evidence=[]
    for dataset,xx,zz,yy,names in [('synthetic',xs,zs,ys,['s0','s1','s2','s3']),('fold0_outer_train_only',X,z,y,hist)]:
        for regime,pen in [('free',0.),('zero',.1),('soft',.1)]:
            fits=[]
            for init in [np.eye(2),np.zeros((2,2))]:
                c=cfg['tree_candidates'][0]
                m=StructuredXGB(regime,pen,c['params'],c['rounds'],names).fit(xx,zz,yy,initial=init)
                fits.append(m)
            diffA=float(np.linalg.norm(fits[0].A-fits[1].A))
            diffp=float(np.max(np.abs(softmax(base_margin(zz,fits[0].A)+fits[0].tree.tree_only(xx),axis=1)-softmax(base_margin(zz,fits[1].A)+fits[1].tree.tree_only(xx),axis=1))))
            ok=all(m.converged for m in fits) and diffA<=.05 and diffp<=.005
            evidence.append({'dataset':dataset,'regime':regime,'penalty':pen,'passed':ok,'A_difference':diffA,'probability_max_difference':diffp,
                 'initializations':[{'initial':name,'converged':m.converged,'failure':getattr(m,'failure',None),'trajectory':m.trajectory} for name,m in zip(['I','zero'],fits)]})
            print(dataset,regime,'PASS',ok,'A diff',diffA,'p diff',diffp,flush=True)
    # Exact native-vs-custom and decomposition test on real TRAINING data only.
    endpoint=[]
    for c in cfg['tree_candidates']:
        a=TreeCorrection(c['params'],c['rounds'],hist).fit(X,base_margin(z),y)
        b=TreeCorrection(c['params'],c['rounds'],hist).fit(X,base_margin(z),y,rho=0,q=softmax(base_margin(z),axis=1))
        p_error=float(np.max(np.abs(a.probabilities(X,base_margin(z))-b.probabilities(X,base_margin(z)))))
        shap_error=float(np.max(np.abs(b.shap(X[:30]).sum(axis=2)-b.tree_only(X[:30]))))
        endpoint.append({'candidate':c,'rho0_probability_error':p_error,'SHAP_error':shap_error,'passed':p_error<=2e-5 and shap_error<=2e-5})
    sequential=[]
    for f in range(5):
        t,_=pooled_outer_split(df,f); t,_,_=odds_subset(t)
        splits=pooled_inner_temporal_split(t,min_train_seasons=2)
        covered=set(i for s in splits for i in s['val_idx'])
        sequential.append({'fold':f,'training_n':len(t),'OOF_covered_n':len(covered),
                           'uncovered_n':len(t)-len(covered),'earliest_time':str(t.match_datetime.min()),
                           'all_rows_have_temporal_OOF':len(covered)==len(t)})
    learned=all(x['passed'] for x in evidence)
    kl=all(x['passed'] for x in endpoint)
    result={'MS_learned_A':{'passed':learned,'evidence':evidence},
            'Fixed':{'passed':True,'evidence':'unit tests: exact I, native margin and SHAP reconstruction'},
            'MC':{'passed':True,'evidence':'unit tests: whitelist, permutation-invariant context, native fixed offset'},
            'KL_PRE':{'passed':kl,'native_endpoint_evidence':endpoint},
            'Sequential_HT':{'passed':False,'reason':'not every valid HT-training row has prior OOF history; first origin has no earlier odds-valid history; do not invent/backfill or silently drop those rows',
                             'coverage':sequential},
            'HT_LR':{'passed':True,'evidence':'existing tested canonical estimator, same score features and temporal split'},
            'no_outer_test_predictions_generated':True}
    (OUT/'engineering/gates.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print('Gates saved',flush=True)
if __name__=='__main__':main()
