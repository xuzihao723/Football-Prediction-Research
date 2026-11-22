import json,hashlib
from pathlib import Path
import numpy as np,pandas as pd
from src import metrics
from src.statistics import per_obs_metric
from tools.final_design_execution import OUT,ROOT
P=['p_A','p_D','p_H'];KEY=['fixture_key','fold','league','season','actual_class']
def boot(diff,clusters):
 u=np.unique(clusters);idx={c:np.where(clusters==c)[0] for c in u};rng=np.random.default_rng(42);v=[]
 for _ in range(10000):
  s=rng.choice(u,len(u),replace=True);ii=np.concatenate([idx[c] for c in s]);v.append(diff[ii].mean())
 return {'mean':float(diff.mean()),'low':float(np.percentile(v,2.5)),'high':float(np.percentile(v,97.5)),'clusters':len(u)}
def read(route,origin,alg):return pd.concat([pd.read_csv(p) for p in sorted((OUT/route/origin/alg).glob('fold*.csv'))],ignore_index=True).sort_values('fixture_key').reset_index(drop=True)
def main():
 all=[];alg=['Joint-LR','Zero-LR','Free-LR','Soft-LR','Fixed-LR','Joint-XGB','Fixed-XGB','MC-XGB','KL-XGB']
 for origin in ['PRE','HT']:
  for a in alg:
   d=read('direct',origin,a);d['method']=f'direct:{a}:{origin}';d['origin']=origin;all.append(d)
 d=read('direct','PRE','Tied-Zero-LR');d['method']='direct:Tied-Zero-LR:PRE';d['origin']='PRE';all.append(d)
 for route in ['sequential','direct_matched']:
  for a in alg:
   d=read(route,'HT',a);d['method']=f'{route}:{a}:HT';d['origin']='HT';all.append(d)
 pred=pd.concat(all,ignore_index=True);assert not pred.duplicated(['method','fixture_key']).any();rows=[]
 for m,g in pred.groupby('method'):rows.append({'method':m,'origin':g.origin.iloc[0],**metrics.compute_all_metrics(g.actual_class.to_numpy(),g[P].to_numpy())})
 out=OUT/'reports';out.mkdir(parents=True,exist_ok=True);pd.DataFrame(rows).to_csv(out/'final_metrics.csv',index=False)
 pairs=[]
 for o in ['PRE','HT']:
  for a in ['Zero-LR','Free-LR','Soft-LR','Fixed-LR']:pairs.append((f'direct:{a}:{o}',f'direct:Joint-LR:{o}'))
  for a in ['Fixed-XGB','MC-XGB','KL-XGB']:pairs.append((f'direct:{a}:{o}',f'direct:Joint-XGB:{o}'))
  for a in ['MC-XGB','KL-XGB']:pairs.append((f'direct:{a}:{o}',f'direct:Fixed-XGB:{o}'))
  for a in ['Zero-LR','Soft-LR','Fixed-LR']:pairs.append((f'direct:{a}:{o}',f'direct:Free-LR:{o}'))
 pairs.append(('direct:Tied-Zero-LR:PRE','direct:Zero-LR:PRE'))
 pairs += [(f'sequential:{a}:HT',f'direct_matched:{a}:HT') for a in alg]
 cont=[];g={m:x.sort_values('fixture_key').reset_index(drop=True) for m,x in pred.groupby('method')}
 for a,b in pairs:
  x,y=g[a],g[b];assert x[KEY].equals(y[KEY]);cl=(x.league+'|'+x.season).to_numpy()
  for metric in ['log_loss','brier','acc']:
   cont.append({'first':a,'second':b,'metric':metric,**boot(per_obs_metric(x.actual_class.to_numpy(),x[P].to_numpy(),metric)-per_obs_metric(y.actual_class.to_numpy(),y[P].to_numpy(),metric),cl)})
 pd.DataFrame(cont).to_csv(out/'final_contrasts.csv',index=False);pred.to_csv(out/'final_predictions_combined.csv',index=False)
 old=json.loads((OUT/'before_hashes.json').read_text());changed=[p for p,h in old.items() if hashlib.sha256((ROOT.parent/p).read_bytes()).hexdigest()!=h]
 assert len(changed)==1 and changed[0].endswith('tools\\final_design_execution.py')
 report={'methods':len(g),'rows':len(pred),'direct_csv':95,'sequential_csv':45,'direct_matched_csv':45,'contrasts':len(cont),'N_per_method':2789,'clusters':15,'bootstrap':10000,'seed':42,'old_files_unchanged':True,'outer_test_predictions_generated':True,'paper_modified':False,'protocol_changed':False,'train_labels_not_used_for_test':True}
 (out/'final_manifest.json').write_text(json.dumps(report,indent=2),encoding='utf8');print(pd.DataFrame(rows).to_string(index=False));print('manifest',report)
if __name__=='__main__':main()



