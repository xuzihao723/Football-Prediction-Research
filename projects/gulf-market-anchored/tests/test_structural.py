"""Matched-subset masks and structural model legality (unseen teams, rolling no-drop)."""
from __future__ import annotations
import numpy as np
import pandas as pd

from src import metrics
from src.evaluation import compute_odds_mask, compute_tseq_mask
from src.models.structural import fit_poisson, PoissonModel


def _tiny_league_df():
    # 4 teams, 6 completed fixtures with full-time goals
    rows = []
    teams = ["A", "B", "C", "D"]
    scores = [(2, 1), (0, 0), (1, 3), (2, 2), (0, 1), (3, 0)]
    homes = [teams[i % 4] for i in range(6)]
    aways = [teams[(i + 1) % 4] for i in range(6)]
    for i in range(6):
        rows.append({
            "match_datetime": pd.Timestamp("2020-01-%02d 18:00:00" % (i + 1), tz="UTC"),
            "home_team_key": homes[i], "away_team_key": aways[i],
            "home_team_goal_count": scores[i][0], "away_team_goal_count": scores[i][1],
        })
    return pd.DataFrame(rows)


def test_poisson_unseen_team_falls_back_to_league_average():
    df = _tiny_league_df()
    m = fit_poisson(df)
    # unseen team "Z" has no attack/defence -> get(...,0.0)
    assert m.attack.get("Z", 0.0) == 0.0
    assert m.defence.get("Z", 0.0) == 0.0
    rec = pd.DataFrame({"home": ["A"], "away": ["Z"]})
    lh, la = m.lambda_home_away(rec)
    assert np.all(np.isfinite(lh)) and np.all(np.isfinite(la))
    assert lh[0] > 0 and la[0] > 0
    p = m.proba(rec)
    metrics.assert_valid_proba(p, context="unseen poisson")


def test_tseq_subset_of_todds():
    df = _tiny_league_df()
    # add odds columns (all valid) and HT columns
    df["odds_ft_home_team_win"] = 2.0
    df["odds_ft_draw"] = 3.3
    df["odds_ft_away_team_win"] = 3.9
    df["total_goals_at_half_time"] = np.nan
    df["home_team_goal_count_half_time"] = np.nan
    df["away_team_goal_count_half_time"] = np.nan
    df["home_team_first_half_cards"] = np.nan
    df["away_team_first_half_cards"] = np.nan
    odds = compute_odds_mask(df)
    tseq = compute_tseq_mask(df)
    # T_seq (odds + HT non-null) can only be a subset of T_odds
    assert set(np.where(tseq)[0]) <= set(np.where(odds)[0])
    # with HT all-NaN, T_seq is empty
    assert not tseq.any()
def test_conditional_poisson_rolling_uses_full_chronology_without_dropping_queries():
    from src.models.structural import conditional_poisson_proba_rolling
    tr=_tiny_league_df().copy()
    tr['fixture_key']=[f'tr{i}' for i in range(len(tr))]
    tr['home_team_goal_count_half_time']=[1,0,0,1,0,2]
    tr['away_team_goal_count_half_time']=[0,0,1,1,0,0]
    # Two same-time test fixtures; only one is a T_seq query, but both must be appended
    # only after the batch is predicted.
    te=pd.DataFrame([
        {'fixture_key':'q1','match_datetime':pd.Timestamp('2020-02-01 18:00:00',tz='UTC'),'timestamp_precision':'datetime',
         'home_team_key':'A','away_team_key':'B','home_team_goal_count':1,'away_team_goal_count':0,
         'home_team_goal_count_half_time':0,'away_team_goal_count_half_time':0},
        {'fixture_key':'nq','match_datetime':pd.Timestamp('2020-02-01 18:00:00',tz='UTC'),'timestamp_precision':'datetime',
         'home_team_key':'C','away_team_key':'D','home_team_goal_count':0,'away_team_goal_count':2,
         'home_team_goal_count_half_time':0,'away_team_goal_count_half_time':1},
        {'fixture_key':'q2','match_datetime':pd.Timestamp('2020-02-02 18:00:00',tz='UTC'),'timestamp_precision':'datetime',
         'home_team_key':'A','away_team_key':'C','home_team_goal_count':2,'away_team_goal_count':1,
         'home_team_goal_count_half_time':1,'away_team_goal_count_half_time':0},
    ])
    q=te[te.fixture_key.isin(['q1','q2'])].reset_index(drop=True)
    p=conditional_poisson_proba_rolling(tr,q,chronology_df=te)
    assert p.shape==(2,3)
    metrics.assert_valid_proba(p,context='conditional rolling test')

def test_elo_historical_forward_is_batch_order_invariant():
    from src.models.structural import _prior_rating_series
    rows = pd.DataFrame([
        {'match_datetime':pd.Timestamp('2020-03-01 18:00:00',tz='UTC'),'date':'2020-03-01','timestamp_precision':'datetime','time':pd.Timestamp('2020-03-01 18:00:00',tz='UTC'),'home':'A','away':'B','hg':2.0,'ag':0.0,'result':'H'},
        {'match_datetime':pd.Timestamp('2020-03-01 18:00:00',tz='UTC'),'date':'2020-03-01','timestamp_precision':'datetime','time':pd.Timestamp('2020-03-01 18:00:00',tz='UTC'),'home':'C','away':'D','hg':0.0,'ag':1.0,'result':'A'},
        {'match_datetime':pd.Timestamp('2020-03-08 18:00:00',tz='UTC'),'date':'2020-03-08','timestamp_precision':'datetime','time':pd.Timestamp('2020-03-08 18:00:00',tz='UTC'),'home':'A','away':'C','hg':1.0,'ag':1.0,'result':'D'},
    ])
    r1,d1=_prior_rating_series(rows.copy(),60.0)
    swapped=pd.concat([rows.iloc[[1,0]],rows.iloc[[2]]],ignore_index=True)
    r2,d2=_prior_rating_series(swapped,60.0)
    assert r1==r2
    # First-batch pre-match differences must be equal to the common 1500 prior regardless of order.
    assert np.allclose(d1[:2],[0.0,0.0])
    assert np.allclose(d2[:2],[0.0,0.0])
