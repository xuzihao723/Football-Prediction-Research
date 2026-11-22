import numpy as np
import pandas as pd
from tools.run_xgb_v4 import design,blocks
from src.experiments.common import load_data,history_cols,odds_subset
from src.pooled_split import pooled_outer_split,pooled_inner_temporal_split,assert_temporal
from src.evaluation import compute_tseq_mask

def test_exact_populations_and_outer_label_independent_design():
    df=load_data()
    for f in range(5):
        tr,te=pooled_outer_split(df,f); hist=history_cols(f)
        for origin in ['PRE','HT']:
            a=tr.loc[compute_tseq_mask(tr)] if origin=='HT' else tr
            b=te.loc[compute_tseq_mask(te)] if origin=='HT' else te
            a,_,_=odds_subset(a); b,_,_=odds_subset(b); assert_temporal(a,b)
            for family in ['Fixed','MC','KL','LR']:
                before=design(b,hist,origin,family)
                changed=b.copy(); changed['target_match_result']='D'
                after=design(changed,hist,origin,family)
                for x,y in zip(before[:3],after[:3]):np.testing.assert_array_equal(x,y)
                assert before[3]==after[3]
            for sp in pooled_inner_temporal_split(a,min_train_seasons=2):
                assert_temporal(a.iloc[sp['train_idx']],a.iloc[sp['val_idx']])
                assert not set(a.iloc[sp['train_idx']].fixture_key)&set(b.fixture_key)

def test_earliest_sequential_anchors_cannot_be_filled_without_policy_change():
    tr,_=pooled_outer_split(load_data(),0); tr,_,_=odds_subset(tr)
    sp=pooled_inner_temporal_split(tr,min_train_seasons=2)
    covered=set(i for s in sp for i in s['val_idx'])
    assert len(tr)-len(covered)==645
    first=pd.to_datetime(tr.match_datetime,utc=True).min()
    assert (pd.to_datetime(tr.match_datetime,utc=True)<first).sum()==0
