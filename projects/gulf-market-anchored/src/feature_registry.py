from __future__ import annotations
import hashlib, json
from pathlib import Path
from .config import CONFIG_DIR, protocol

GROUP_ORDER = ["Elo","Form","Standings","Venue","H2H","Rest","Draw"]
ALIASES = {
    "Form/momentum":"Form", "Venue history":"Venue", "Head-to-head":"H2H",
    "Rest/congestion/travel":"Rest", "Draw-oriented":"Draw"
}

class FeatureRegistry:
    def __init__(self, path: Path | None = None):
        self.path = path or CONFIG_DIR / "feature_registry_final.json"
        if not self.path.exists():
            raise FileNotFoundError(
                f"Missing authoritative registry {self.path}. Run: python -m src.build_feature_registry"
            )
        self.cfg = json.loads(self.path.read_text(encoding="utf-8"))
        self.groups = {ALIASES.get(k,k): list(v) for k,v in self.cfg["groups"].items()}
        self.order = [ALIASES.get(x,x) for x in self.cfg.get("order", GROUP_ORDER)]
        if self.order != GROUP_ORDER:
            raise AssertionError(f"Unexpected group order: {self.order}")
        seen=[]
        for g in self.order:
            if g not in self.groups: raise KeyError(f"Registry missing group {g}")
            seen.extend(self.groups[g])
        if len(seen) != len(set(seen)):
            dup=sorted({x for x in seen if seen.count(x)>1})
            raise AssertionError(f"Feature group overlap: {dup}")
        self.market_prior_columns=list(self.cfg["market_prior_columns"])
        self.ht_score_columns=list(self.cfg.get("ht_score_columns", []))
        self.ht_non_score_columns=list(self.cfg.get("ht_non_score_columns", []))
        self.league_context_columns=list(self.cfg.get("league_context_columns", protocol()["league_context_columns"]))
        self.target_col=self.cfg.get("target_col", protocol()["target_col"])
        self.fixture_key_col=self.cfg.get("fixture_key_col", protocol()["fixture_key_col"])

    def group_columns(self, groups: list[str] | tuple[str,...]) -> list[str]:
        out=[]
        for g in groups:
            g=ALIASES.get(g,g)
            out.extend(self.groups[g])
        return out
    def full_history(self): return self.group_columns(self.order)
    def columns_for_fold(self, fold:int, selection_cfg: dict) -> list[str]:
        groups=selection_cfg["selected_groups_by_fold"][str(fold)]
        return self.group_columns(groups)
    def with_context(self, cols): return list(cols)+[c for c in self.league_context_columns if c not in cols]
    def feature_hash(self, cols): return hashlib.sha256("|".join(sorted(cols)).encode()).hexdigest()[:16]
    def validate_frame(self, df, cols, context=""):
        miss=[c for c in cols if c not in df.columns]
        if miss: raise KeyError(f"{context}: missing columns: {miss}")

_REG=None
def get_registry():
    global _REG
    if _REG is None: _REG=FeatureRegistry()
    return _REG
