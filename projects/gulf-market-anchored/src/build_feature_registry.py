"""One-time migration from the pre-consolidation config files to the final registry.
Runtime code never reads those old configs; only this migration utility does.
"""
from __future__ import annotations
import json
import pandas as pd
from .config import CONFIG_DIR, DATA_DIR, protocol
from .feature_registry import ALIASES, GROUP_ORDER

XG_REMOVED = {
 "home_xg_diff_5_proxy","away_xg_diff_5_proxy","xg_diff_mom_5_proxy_diff",
 "home_venue_xg_diff_5_proxy","away_venue_xg_diff_5_proxy","venue_xg_diff_5_proxy_diff"
}

def main():
    src=CONFIG_DIR/"feature_groups_harmonized.json"
    if not src.exists():
        raise FileNotFoundError(f"Expected migration input: {src}")
    g=json.loads(src.read_text(encoding="utf-8"))
    groups={ALIASES.get(k,k): [c for c in v if c not in XG_REMOVED] for k,v in g["groups"].items()}
    if list(groups) != GROUP_ORDER:
        groups={name:groups[name] for name in GROUP_ORDER}
    odds_path=CONFIG_DIR/"odds_features.json"
    avail_path=CONFIG_DIR/"feature_availability.json"
    odds=json.loads(odds_path.read_text(encoding="utf-8")) if odds_path.exists() else {"market_prior_columns":["odds_ft_home_team_win","odds_ft_draw","odds_ft_away_team_win"]}
    ht_score=[]; ht_non=[]
    if avail_path.exists():
        av=json.loads(avail_path.read_text(encoding="utf-8"))
        for c,v in av.get("columns",{}).items():
            if v.get("availability")=="HALF_TIME_SCORE": ht_score.append(c)
            if v.get("availability")=="HALF_TIME_NON_SCORE": ht_non.append(c)
    # Keep only HT columns actually persisted in the harmonized modeling file. Derived
    # columns (e.g. ht_goal_difference) are declared in the availability map but never
    # written into ModelingData_harmonized.csv and must not be referenced at runtime.
    modeling_csv = DATA_DIR/"modeling"/"ModelingData_harmonized.csv"
    present = set(pd.read_csv(modeling_csv, nrows=0).columns)
    ht_score = [c for c in ht_score if c in present]
    ht_non = [c for c in ht_non if c in present]
    features={}
    for group, cols in groups.items():
        for c in cols:
            features[c]={
                "feature_name":c,"group":group,"stage":"historical_pre_match",
                "selectable":True,"ablatable":True,"fixed_context":False,
                "harmonized_semantics":"canonical harmonized numeric pre-match feature",
                "preprocessing":"numeric; train-only imputation/scaling as learner requires",
                "leakage_status":"legal historical pre-match feature"
            }
    for c in protocol()["league_context_columns"]:
        features[c]={
            "feature_name":c,"group":"LeagueContext","stage":"context_pre_match",
            "selectable":False,"ablatable":False,"fixed_context":True,
            "harmonized_semantics":"league fixed-effect/context control",
            "preprocessing":"numeric indicator","leakage_status":"legal pre-match context"
        }
    for c in odds["market_prior_columns"]:
        features[c]={
            "feature_name":c,"group":"Market","stage":"market_pre_match",
            "selectable":False,"ablatable":False,"fixed_context":False,
            "harmonized_semantics":"decimal 1X2 pre-match market odds",
            "preprocessing":"normalized implied probability/reference logits",
            "leakage_status":"legal pre-match market information"
        }
    for c in ht_score:
        features[c]={
            "feature_name":c,"group":"HTScore","stage":"half_time_score",
            "selectable":False,"ablatable":False,"fixed_context":False,
            "harmonized_semantics":"observed half-time score state",
            "preprocessing":"numeric; train-only preprocessing",
            "leakage_status":"legal only at half-time"
        }
    for c in ht_non:
        features[c]={
            "feature_name":c,"group":"HTNonScore","stage":"half_time_non_score",
            "selectable":False,"ablatable":False,"fixed_context":False,
            "harmonized_semantics":"observed legal first-half non-score state",
            "preprocessing":"numeric; train-only preprocessing",
            "leakage_status":"legal only at half-time"
        }
    out={
      "schema":"feature_registry_final", "order":GROUP_ORDER, "groups":groups,
      "market_prior_columns":odds["market_prior_columns"],
      "ht_score_columns":ht_score, "ht_non_score_columns":ht_non,
      "league_context_columns":protocol()["league_context_columns"],
      "target_col":protocol()["target_col"], "fixture_key_col":protocol()["fixture_key_col"],
      "removed_xg_features":sorted(XG_REMOVED),
      "features":features
    }
    dest=CONFIG_DIR/"feature_registry_final.json"
    dest.write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding="utf-8")
    print(f"wrote {dest}; historical features={sum(map(len,groups.values()))}")
if __name__=="__main__": main()
