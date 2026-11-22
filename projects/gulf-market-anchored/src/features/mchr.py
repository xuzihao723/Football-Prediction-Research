"""Strict, outcome-free candidate construction for MCHR audit.

Only completed matches strictly earlier than a target timestamp are consumed.
This module is diagnostic and is not imported by production experiments.
"""
from __future__ import annotations
import numpy as np, pandas as pd

RAW = {
 'goals_for':'target_home_team_goal_count', 'goals_against':'target_away_team_goal_count',
 'shots_for':'home_team_shots', 'shots_against':'away_team_shots',
 'sot_for':'home_team_shots_on_target', 'sot_against':'away_team_shots_on_target',
 'possession_for':'home_team_possession', 'possession_against':'away_team_possession',
 'corners_for':'home_team_corner_count', 'corners_against':'away_team_corner_count',
 'fouls_for':'home_team_fouls', 'fouls_against':'away_team_fouls',
 'yellow_for':'home_team_yellow_cards', 'yellow_against':'away_team_yellow_cards',
 'red_for':'home_team_red_cards', 'red_against':'away_team_red_cards',
}

def team_long(df: pd.DataFrame) -> pd.DataFrame:
    d=df.copy(); d['_dt']=pd.to_datetime(d['match_datetime'],utc=True,errors='coerce')
    d=d.sort_values(['league','_dt','fixture_key'],kind='stable').reset_index(drop=True)
    rows=[]
    for _,r in d.iterrows():
        common={'fixture_key':str(r['fixture_key']),'league':str(r['league']),'season':str(r['season']),'_dt':r['_dt']}
        for side,team,opp in [('home',r['home_team_key'],r['away_team_key']),('away',r['away_team_key'],r['home_team_key'])]:
            q=dict(common,team=str(team),opponent=str(opp),venue_role=side)
            for base,hcol in RAW.items():
                # RAW stores the home source column; derive the away source name.
                acol=hcol.replace('home_', 'away_', 1) if hcol.startswith('home_') else hcol
                q[base]=r[hcol] if side=='home' else r[acol]
            # Team points and market expected points are intrinsically historical after the match.
            y=str(r['target_match_result']); win=(y=='H' if side=='home' else y=='A'); draw=y=='D'
            q['points']=3. if win else 1. if draw else 0.
            pcols=['p_H','p_D','p_A']
            try:
                odds=np.array([float(r['odds_ft_home_team_win']),float(r['odds_ft_draw']),float(r['odds_ft_away_team_win'])])
                if np.all(np.isfinite(odds)&(odds>1)):
                    pp=(1/odds)/(1/odds).sum(); q['market_expected_points']=3*pp[0]+pp[1] if side=='home' else 3*pp[2]+pp[1]
                    q['market_surprise']=q['points']-q['market_expected_points']
                else:q['market_surprise']=np.nan
            except Exception:q['market_surprise']=np.nan
            rows.append(q)
    return pd.DataFrame(rows).sort_values(['league','team','_dt','fixture_key'],kind='stable').reset_index(drop=True)

def strict_state(long: pd.DataFrame, windows=(5,10)) -> pd.DataFrame:
    out=long.copy(); numeric=[c for c in RAW]+['points','market_surprise']
    for c in numeric:out[c]=pd.to_numeric(out[c],errors='coerce')
    g=out.groupby(['league','team'],sort=False)
    for c in numeric:
        for w in windows:
            # previous W team matches, complete only if every raw observation exists
            out[f'{c}_mean{w}']=g[c].transform(lambda s:s.shift(1).rolling(w,min_periods=w).mean())
            if c=='market_surprise': continue
            out[f'{c}_count{w}']=g[c].transform(lambda s:s.shift(1).rolling(w,min_periods=w).count())
        out[f'{c}_change5_10']=out[f'{c}_mean5']-out[f'{c}_mean10']
    # Aggregate-count efficiencies, never mean ratios.
    for w in windows:
        for num,den,name in [('goals_for','shots_for','shot_conversion'),('goals_for','sot_for','sot_conversion'),('goals_against','shots_against','goals_allowed_per_shot'),('goals_against','sot_against','goals_allowed_per_sot')]:
            ns=g[num].transform(lambda s:s.shift(1).rolling(w,min_periods=w).sum()); ds=g[den].transform(lambda s:s.shift(1).rolling(w,min_periods=w).sum())
            out[f'{name}{w}']=ns.where(ds>0)/ds.where(ds>0)
    out['shot_conversion_change5_10']=out['shot_conversion5']-out['shot_conversion10']
    out['sot_conversion_change5_10']=out['sot_conversion5']-out['sot_conversion10']
    out['goals_allowed_per_shot_change5_10']=out['goals_allowed_per_shot5']-out['goals_allowed_per_shot10']
    out['goals_allowed_per_sot_change5_10']=out['goals_allowed_per_sot5']-out['goals_allowed_per_sot10']
    return out

def pivot_fixture(states: pd.DataFrame, fixture_df: pd.DataFrame) -> pd.DataFrame:
    keys=fixture_df[['fixture_key','league','season','home_team_key','away_team_key','match_datetime']].copy(); keys['_dt']=pd.to_datetime(keys.match_datetime,utc=True)
    h=states.rename(columns={'team':'home_team_key'}); a=states.rename(columns={'team':'away_team_key'})
    h=h.add_prefix('home_mchr_'); a=a.add_prefix('away_mchr_')
    # Join prior states by fixture key/side; rows are already target fixture's own match record.
    h=h.rename(columns={'home_mchr_fixture_key':'fixture_key'}); a=a.rename(columns={'away_mchr_fixture_key':'fixture_key'})
    h=h[h['home_mchr_venue_role']=='home']; a=a[a['away_mchr_venue_role']=='away']
    out=keys.merge(h,on='fixture_key',how='left').merge(a,on='fixture_key',how='left',suffixes=('','_away'))
    return out
