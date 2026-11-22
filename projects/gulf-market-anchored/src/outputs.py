from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
from .config import OUTPUT_DIR,TABLE_DIR
RAW_COLUMNS=["fixture_key","league","season","fold","method","representation","actual_class","p_A","p_D","p_H"]
def prediction_frame(df,p,fold,method,representation,target_col="target_match_result"):
    req=["fixture_key","league","season",target_col]; miss=[c for c in req if c not in df.columns]
    if miss: raise KeyError(f"raw output missing columns {miss}")
    if df["fixture_key"].duplicated().any(): raise AssertionError("fixture_key duplicated in prediction frame")
    return pd.DataFrame({"fixture_key":df.fixture_key.astype(str),"league":df.league.astype(str),"season":df.season.astype(str),"fold":fold,"method":method,"representation":representation,"actual_class":df[target_col].to_numpy(),"p_A":p[:,0],"p_D":p[:,1],"p_H":p[:,2]})
def write_raw(frame,name):
    out=OUTPUT_DIR/"raw"; out.mkdir(parents=True,exist_ok=True); p=out/f"{name}.csv"; frame.to_csv(p,index=False); return p
def write_json(obj,name):
    OUTPUT_DIR.mkdir(parents=True,exist_ok=True); p=OUTPUT_DIR/name; p.write_text(json.dumps(obj,indent=2,ensure_ascii=False,default=str),encoding="utf-8"); return p
