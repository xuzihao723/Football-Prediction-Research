from __future__ import annotations

import itertools
import warnings
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier, early_stopping as lgb_early_stopping, log_evaluation
from catboost import CatBoostClassifier

from .common import load_data, split_fold, history_cols
from ..config import seed, protocol
from ..feature_registry import get_registry
from ..evaluation import compute_tseq_mask, assert_matched_fixture
from ..models.market import bookmaker_prior
from ..models.generic_ml import fit_classifier, predict_classifier, smart_vote
from ..preprocessing import Preprocessor
from ..pooled_split import pooled_inner_temporal_split
from ..outputs import prediction_frame
from .. import metrics, hpo

BASE = ["LR", "RF", "XGB", "LightGBM", "CatBoost", "FNN"]
BOOSTING = {"XGB", "LightGBM", "CatBoost"}


def _candidate_params(name: str) -> list[dict]:
    c = hpo.load_config()
    if name == "LR":
        return [{"C": x} for x in c["Historical-LR"]["C"]]
    key = "XGBClassifier" if name == "XGB" else name
    spec = c[key]
    keys = list(spec)
    vals = [spec[k] for k in keys]
    combos = [dict(zip(keys, x)) for x in itertools.product(*vals)]
    return combos[:24]  # frozen deterministic cap


def _design(df: pd.DataFrame, base_cols: list[str], ht_cols: list[str]) -> np.ndarray:
    _, z, valid = bookmaker_prior(df)
    if not valid.all():
        raise AssertionError("D requires T_seq/odds-valid rows")
    parts = [df[base_cols].to_numpy(float), z]
    if ht_cols:
        parts.append(df[ht_cols].to_numpy(float))
    return np.hstack(parts)


def _inner_splits(train: pd.DataFrame) -> list[dict]:
    # Formal D uses >=2 earlier market seasons. Development-only smoke uses >=1 so
    # S3->S4 is available inside S1-S4 -> S5 without touching S6-S10.
    return pooled_inner_temporal_split(train, min_train_seasons=1 if hpo.is_smoke() else 2)


def _fnn_fit_temporal(Xtr, ytr, Xva, yva, params: dict) -> tuple[MLPClassifier, int, np.ndarray]:
    """Train an MLP epoch-by-epoch and stop only on a later temporal validation block."""
    max_epochs = 60 if hpo.is_smoke() else 500
    patience = 8 if hpo.is_smoke() else 30
    yi = metrics.encode_target(np.asarray(ytr))
    yv = np.asarray(yva)
    pars = dict(params)
    hidden = tuple(pars.pop("hidden_layer_sizes"))
    model = MLPClassifier(
        hidden_layer_sizes=hidden,
        max_iter=1,
        warm_start=False,
        shuffle=False,
        random_state=seed(),
        **pars,
    )
    best_loss = np.inf
    best_epoch = 1
    best_p = None
    stale = 0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        for epoch in range(1, max_epochs + 1):
            if epoch == 1:
                model.partial_fit(np.asarray(Xtr, float), yi, classes=np.array([0, 1, 2]))
            else:
                model.partial_fit(np.asarray(Xtr, float), yi)
            p = np.asarray(model.predict_proba(np.asarray(Xva, float)), float)
            metrics.assert_valid_proba(p, context="FNN temporal validation")
            ll = metrics.log_loss(yv, p)
            if ll < best_loss - 1e-8:
                best_loss, best_epoch, best_p, stale = ll, epoch, p.copy(), 0
            else:
                stale += 1
                if stale >= patience:
                    break
    if best_p is None:
        raise RuntimeError("FNN temporal early stopping failed")
    return model, best_epoch, best_p


def _fit_eval(name: str, Xtr, ytr, Xva, yva, params: dict):
    """Fit one candidate on earlier data, evaluate on later data, return best rounds."""
    yi = metrics.encode_target(np.asarray(ytr))
    yv_i = metrics.encode_target(np.asarray(yva))
    p = dict(params)
    if name in {"LR", "RF"}:
        m = fit_classifier(name, Xtr, ytr, p, seed())
        pv = predict_classifier(m, Xva)
        return m, None, pv
    if name == "FNN":
        return _fnn_fit_temporal(Xtr, ytr, Xva, yva, p)
    if name == "XGB":
        nmax = int(p.pop("n_estimators", 500))
        m = XGBClassifier(
            objective="multi:softprob", num_class=3, eval_metric="mlogloss",
            random_state=seed(), n_jobs=4, n_estimators=nmax,
            early_stopping_rounds=(8 if hpo.is_smoke() else 30), **p,
        )
        m.fit(np.asarray(Xtr, float), yi, eval_set=[(np.asarray(Xva, float), yv_i)], verbose=False)
        bi = getattr(m, "best_iteration", None)
        rounds = int(bi) + 1 if bi is not None else nmax
        pv = np.asarray(m.predict_proba(np.asarray(Xva, float)), float)
    elif name == "LightGBM":
        nmax = int(p.pop("n_estimators", 500))
        m = LGBMClassifier(
            objective="multiclass", num_class=3, random_state=seed(), verbosity=-1,
            n_estimators=nmax, **p,
        )
        m.fit(
            np.asarray(Xtr, float), yi,
            eval_set=[(np.asarray(Xva, float), yv_i)], eval_metric="multi_logloss",
            callbacks=[lgb_early_stopping(8 if hpo.is_smoke() else 30, verbose=False), log_evaluation(0)],
        )
        rounds = int(getattr(m, "best_iteration_", 0) or nmax)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="X does not have valid feature names.*", category=UserWarning)
            pv = np.asarray(m.predict_proba(np.asarray(Xva, float)), float)
    elif name == "CatBoost":
        nmax = int(p.pop("iterations", 500))
        m = CatBoostClassifier(
            loss_function="MultiClass", random_seed=seed(), verbose=False,
            allow_writing_files=False, iterations=nmax, **p,
        )
        m.fit(
            np.asarray(Xtr, float), yi,
            eval_set=(np.asarray(Xva, float), yv_i), use_best_model=True,
            early_stopping_rounds=(8 if hpo.is_smoke() else 30), verbose=False,
        )
        bi = m.get_best_iteration()
        rounds = int(bi) + 1 if bi is not None and bi >= 0 else nmax
        pv = np.asarray(m.predict_proba(np.asarray(Xva, float)), float)
    else:
        raise ValueError(name)
    metrics.assert_valid_proba(pv, context=f"D {name} temporal validation")
    return m, rounds, pv


def _fit_final(name: str, X, y, params: dict, rounds: int | None):
    """Refit on all legal training data using only training-derived rounds/epochs."""
    p = dict(params)
    if name == "FNN":
        epochs = int(rounds or protocol()["d_nested_early_origin_fallback"]["FNN_epochs"])
        yi = metrics.encode_target(np.asarray(y))
        hidden = tuple(p.pop("hidden_layer_sizes"))
        m = MLPClassifier(
            hidden_layer_sizes=hidden, max_iter=1, warm_start=False, shuffle=False,
            random_state=seed(), **p,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            for epoch in range(epochs):
                if epoch == 0:
                    m.partial_fit(np.asarray(X, float), yi, classes=np.array([0, 1, 2]))
                else:
                    m.partial_fit(np.asarray(X, float), yi)
        return m
    if name == "XGB":
        p["n_estimators"] = int(rounds or p.get("n_estimators", 500))
    elif name == "LightGBM":
        p["n_estimators"] = int(rounds or p.get("n_estimators", 500))
    elif name == "CatBoost":
        p["iterations"] = int(rounds or p.get("iterations", 500))
    return fit_classifier(name, X, y, p, seed())


def _tune(name: str, train: pd.DataFrame, base_cols: list[str], ht_cols: list[str]):
    splits = _inner_splits(train)
    if not splits:
        raise RuntimeError(f"D {name}: no inner temporal split")
    scale = name in {"LR", "FNN"}
    candidates = _candidate_params(name)
    scores, candidate_rounds = [], []
    for pars in candidates:
        ss, ww, rr = [], [], []
        for sp in splits:
            tr = train.iloc[sp["train_idx"]].reset_index(drop=True)
            va = train.iloc[sp["val_idx"]].reset_index(drop=True)
            if tr.target_match_result.nunique() < 3:
                raise RuntimeError("D inner train missing class")
            prep = Preprocessor(scale=scale)
            Xtr = prep.fit_transform(_design(tr, base_cols, ht_cols))
            Xva = prep.transform(_design(va, base_cols, ht_cols))
            _, rounds, pv = _fit_eval(name, Xtr, tr.target_match_result.to_numpy(), Xva, va.target_match_result.to_numpy(), pars)
            ss.append(metrics.log_loss(va.target_match_result.to_numpy(), pv))
            ww.append(len(va))
            if rounds is not None:
                rr.append(int(rounds))
        scores.append(float(np.average(ss, weights=ww)))
        candidate_rounds.append(int(np.median(rr)) if rr else None)
    j = int(np.argmin(scores))
    return {"params": candidates[j], "rounds": candidate_rounds[j], "score": scores[j], "n_splits": len(splits)}


def _fallback_tune(name: str) -> dict:
    """Protocol-frozen fallback used only when an early nested voting origin cannot HPO."""
    f = protocol()["d_nested_early_origin_fallback"]
    params = _candidate_params(name)[0]
    rounds = None
    if name == "XGB": rounds = int(f["XGB_rounds"])
    elif name == "LightGBM": rounds = int(f["LightGBM_rounds"])
    elif name == "CatBoost": rounds = int(f["CatBoost_rounds"])
    elif name == "FNN": rounds = int(f["FNN_epochs"])
    return {"params": params, "rounds": rounds, "score": float("nan"), "n_splits": 0, "fallback": True}


def _fit_from_tune(name: str, train: pd.DataFrame, test: pd.DataFrame, base_cols: list[str], ht_cols: list[str], tune: dict):
    scale = name in {"LR", "FNN"}
    prep = Preprocessor(scale=scale)
    Xtr = prep.fit_transform(_design(train, base_cols, ht_cols))
    Xte = prep.transform(_design(test, base_cols, ht_cols))
    model = _fit_final(name, Xtr, train.target_match_result.to_numpy(), tune["params"], tune.get("rounds"))
    p = predict_classifier(model, Xte)
    return p


def _nested_voting_accuracy(train: pd.DataFrame, base_cols: list[str], ht_cols: list[str]):
    """Generate strictly temporal OOF predictions for ensemble member selection/weights."""
    splits = _inner_splits(train)
    if not splits:
        raise RuntimeError("D Smart Voting: no temporal OOF origin")
    pred = {n: [] for n in BASE}
    actual = []
    fallback_counts = {n: 0 for n in BASE}
    for sp in splits:
        tr = train.iloc[sp["train_idx"]].reset_index(drop=True)
        va = train.iloc[sp["val_idx"]].reset_index(drop=True)
        actual.append(va.target_match_result.to_numpy())
        for name in BASE:
            try:
                tune = _tune(name, tr, base_cols, ht_cols)
            except RuntimeError as exc:
                if "no inner temporal split" not in str(exc):
                    raise
                tune = _fallback_tune(name)
                fallback_counts[name] += 1
            pred[name].append(_fit_from_tune(name, tr, va, base_cols, ht_cols, tune))
    y = np.concatenate(actual)
    vacc = {name: metrics.accuracy(y, np.vstack(pred[name])) for name in BASE}
    return vacc, fallback_counts


def run_fold(df: pd.DataFrame, fold: int):
    tr, te = split_fold(df, fold)
    reg = get_registry()
    base = history_cols(fold, "selected")
    ht = reg.ht_score_columns + reg.ht_non_score_columns
    tr = tr.loc[compute_tseq_mask(tr)].reset_index(drop=True)
    te = te.loc[compute_tseq_mask(te)].reset_index(drop=True)
    if len(tr) == 0 or len(te) == 0:
        raise RuntimeError(f"D fold {fold}: empty T_seq")

    rows, diag = [], []
    for info, htcols in [("Pre", []), ("HT-F", ht)]:
        probs = {}
        final_tunes = {}
        for name in BASE:
            tune = _tune(name, tr, base, htcols)
            p = _fit_from_tune(name, tr, te, base, htcols, tune)
            probs[name] = p
            final_tunes[name] = tune
            rows.append(prediction_frame(te, p, fold, name, info))
            diag.append({"fold": fold, "info": info, "method": name, "hpo": tune})

        vacc, fallback_counts = _nested_voting_accuracy(tr, base, htcols)
        pv, meta = smart_vote(probs, vacc, top_k=3)
        rows.append(prediction_frame(te, pv, fold, "Smart-Voting", info))
        diag.append({
            "fold": fold, "info": info, "method": "Smart-Voting",
            **meta, "oof_validation_accuracy": vacc,
            "nested_early_origin_fallback_counts": fallback_counts,
        })

    # Exact same fixture order for all seven methods within each information set.
    for info in ["Pre", "HT-F"]:
        subset = [r for r in rows if r["representation"].iloc[0] == info]
        assert_matched_fixture([list(r.fixture_key) for r in subset], names=[r.method.iloc[0] for r in subset])
    return pd.concat(rows, ignore_index=True), diag


def run(data=None, folds=range(5)):
    df = load_data() if data is None else data
    rr, dd = [], []
    for f in folds:
        r, d = run_fold(df, int(f))
        rr.append(r)
        dd.extend(d)
    return pd.concat(rr, ignore_index=True), dd
