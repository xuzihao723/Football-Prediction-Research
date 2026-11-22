"""Corrected MCHR, isolated from invalid v1 and all predictive pipelines."""
from collections import defaultdict
import numpy as np
import pandas as pd
from ..pooled_split import temporal_batches, match_time
from ..models.market import bookmaker_prior

PAIRS={s:(f'home_team_{c}',f'away_team_{c}') for s,c in {
 'shots':'shots','sot':'shots_on_target','goals':'goal_count','corners':'corner_count',
 'fouls':'fouls','yellow_cards':'yellow_cards','red_cards':'red_cards','possession':'possession'}.items()}
IDS=['fixture_key','league','season','match_datetime','date','timestamp_precision','home_team_key','away_team_key']
STATES={'shots_for':'Shots','shots_against':'Shots','sot_for':'SOT','sot_against':'SOT',
 'possession_for':'Possession','corners_for':'Corners','corners_against':'Corners',
 'goals_for':'Goals','goals_against':'Goals','fouls_for':'Discipline','yellow_cards_for':'Discipline',
 'red_cards_for':'Discipline','market_surprise':'MarketSurprise'}
RATIOS={'shot_conversion':('goals_for','shots_for'),'sot_conversion':('goals_for','sot_for'),
 'goals_allowed_per_shot':('goals_against','shots_against'),'goals_allowed_per_sot':('goals_against','sot_against')}

def manifest():
    rows=[]
    for side in ['home','away']:
        for state in list(STATES)+list(RATIOS):
            deps=list(RATIOS[state]) if state in RATIOS else [state]
            for suffix,w in [('mean5',5),('mean10',10),('change5_10',10)]:
                # Ratio levels are explicitly ratios despite the stable meanW suffix.
                formula='sum(numerator)/sum(denominator)' if state in RATIOS else 'arithmetic mean'
                rows.append({'feature_name':f'{side}_mchr_{state}_{suffix}',
                  'family':'Efficiency' if state in RATIOS else STATES[state],'team_perspective':side,
                  'state':state,'component':suffix,'window':w,'dependencies':'|'.join(deps),
                  'formula':('W5 minus W10; ' if suffix.startswith('change') else '')+formula,
                  'missingness':'exact previous W team matches; any required NA => NA; ratio denominator zero => NA',
                  'timestamp_rule':'strictly earlier batch; date-only whole-day batch',
                  'requires_market_odds':state=='market_surprise','requires_model_state':False,
                  'supplier_sentinel_semantics':'UNKNOWN; approved INVALID_DATA->NA policy',
                  'units':'ratio' if state in RATIOS else 'points' if state=='market_surprise' else 'percent' if state.startswith('possession') else 'count per match'})
    return pd.DataFrame(rows)

def normalize(df):
    d=df.copy()
    if d.fixture_key.isna().any() or not d.fixture_key.is_unique:raise ValueError('nonunique fixture key')
    if 'status' in d and not d.status.eq('complete').all():raise ValueError('uncompleted fixture')
    d['match_datetime']=match_time(d)
    if 'date' not in d:d['date']=d.match_datetime.dt.strftime('%Y-%m-%d')
    if 'timestamp_precision' not in d:d['timestamp_precision']='datetime'
    if not d.timestamp_precision.isin(['datetime','exact','date_only']).all():raise ValueError('unknown precision')
    d=d.sort_values(['match_datetime','fixture_key'],kind='stable').reset_index(drop=True)
    reasons={}; log=[]
    def invalidate(col,mask,reason):
        for i in d.index[mask]:
            side='home' if col.startswith('home') else 'away'
            log.append({'fixture_key':d.at[i,'fixture_key'],'league':d.at[i,'league'],'season':d.at[i,'season'],
                'team':d.at[i,side+'_team_key'],'field':col,'raw_value':df.set_index('fixture_key').at[d.at[i,'fixture_key'],col],
                'classification':'INVALID_DATA','reason':reason,'action':'NA','supplier_sentinel_meaning':'UNKNOWN'})
            reasons[col][i]=reason
        d.loc[mask,col]=np.nan
    for stat,cols in PAIRS.items():
        for col in cols:
            v=pd.to_numeric(d[col],errors='raise').astype(float); d[col]=v
            reasons[col]=np.where(v.isna(),'raw_missing','').astype(object)
            if np.isinf(v).any():raise ValueError('infinite raw statistic')
            if stat!='possession' and ((v>=0)&v.ne(v.round())).any():raise ValueError('fractional event count')
            invalidate(col,v<0,'invalid_negative')
            if stat=='possession':invalidate(col,v>100,'invalid_possession_range')
    for side in ['home','away']:
        a,b=f'{side}_team_shots',f'{side}_team_shots_on_target'
        mask=d[b]>d[a]
        for col in [a,b]:invalidate(col,mask,'invalid_sot_gt_shots')
    h,a=PAIRS['possession']; valid=d[h].notna()&d[a].notna()
    if not np.allclose((d[h]+d[a])[valid],100):raise ValueError('possession semantics conflict')
    return d,reasons,pd.DataFrame(log)

def bilateral(h,a):return ((h,a),(a,h))

def construct(df):
    d,reasons,log=normalize(df); p,_,odds_ok=bookmaker_prior(d)
    team=[]
    for i,r in d.iterrows():
        for n,side in enumerate(['home','away']):
            q={'fixture_key':r.fixture_key,'league':r.league,'season':r.season,'time':r.match_datetime,
               'team':r[side+'_team_key'],'opponent':r[('away' if n==0 else 'home')+'_team_key'],'side':side}
            for stat,(hc,ac) in PAIRS.items():
                vals=bilateral(r[hc],r[ac])[n]; rs=bilateral(reasons[hc][i],reasons[ac][i])[n]
                for k,role in enumerate(['for','against']):q[stat+'_'+role]=vals[k]; q[stat+'_'+role+'_reason']=rs[k]
            gf,ga=q['goals_for'],q['goals_against']
            points=np.nan if pd.isna(gf) or pd.isna(ga) else 3. if gf>ga else 1. if gf==ga else 0.
            q['points']=points; q['market_expected_points']=3*p[i,2 if n==0 else 0]+p[i,1] if odds_ok[i] else np.nan
            q['market_surprise']=points-q['market_expected_points']
            q['market_surprise_reason']='historical_odds_missing' if not odds_ok[i] else 'raw_missing' if pd.isna(points) else ''
            team.append(q)
    history=defaultdict(list); values=[]; missing=[]; support=[]; contracts=manifest()
    by_fixture={(q['fixture_key'],q['side']):q for q in team}
    for batch_id,pos in enumerate(temporal_batches(d)):
        pending=[]
        for i in pos:
            r=d.iloc[i]; output=r[IDS].to_dict()
            for side in ['home','away']:
                q=by_fixture[(r.fixture_key,side)]; q['batch_id']=batch_id
                hist=history[(q['league'],q['team'])]
                if hist:assert all(h['time']<q['time'] and h['batch_id']<batch_id for h in hist)
                support.append({'fixture_key':r.fixture_key,'side':side,'prior_matches':len(hist),
                    'source_keys_W5':json_keys(hist[-5:]),'source_keys_W10':json_keys(hist[-10:])})
                for state in list(STATES)+list(RATIOS):
                    deps=RATIOS.get(state,(state,)); levels={}; causes={}
                    for w in [5,10]:
                        window=hist[-w:]; cause=set(); val=np.nan
                        if len(window)<w:cause.add(f'insufficient_W{w}_history')
                        else:
                            for h in window:
                                for dep in deps:
                                    if pd.isna(h[dep]):cause.add(h.get(dep+'_reason','raw_missing') or 'raw_missing')
                            if not cause:
                                if state in RATIOS:
                                    numerator=sum(h[deps[0]] for h in window); denom=sum(h[deps[1]] for h in window)
                                    if denom==0:cause.add('zero_denominator')
                                    else:val=numerator/denom
                                else:val=sum(h[state] for h in window)/w
                        levels[w]=val; causes[w]=cause
                    for suffix,val,cause in [('mean5',levels[5],causes[5]),('mean10',levels[10],causes[10]),
                         ('change5_10',levels[5]-levels[10],causes[5]|causes[10])]:
                        name=f'{side}_mchr_{state}_{suffix}'; output[name]=val
                        if pd.isna(val):missing.append({'fixture_key':r.fixture_key,'feature_name':name,'reason':'|'.join(sorted(cause))})
                pending.append(q)
            values.append(output)
        # Canonical key order for equal-time history ties; never input-row order.
        for q in sorted(pending,key=lambda v:(v['time'],v['fixture_key'],v['side'])):history[(q['league'],q['team'])].append(q)
    candidates=pd.DataFrame(values).sort_values('fixture_key').reset_index(drop=True)
    assert set(candidates)-set(IDS)==set(contracts.feature_name)
    assert len(candidates)==len(df)
    return candidates,contracts,pd.DataFrame(team),log,pd.DataFrame(missing),pd.DataFrame(support)

def json_keys(history):
    import json
    return json.dumps([h['fixture_key'] for h in history])
