"""Add only Zero-LR; reuse frozen canonical and legacy OOS artifacts.

Run --stage tune, review boundary diagnostics, then --stage predict and
--stage analyze. Files are exclusive-created, never silently overwritten.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .. import metrics
from ..config import OUTPUT_DIR, PROJECT_ROOT
from ..models.anchoring import CanonicalAnchorLR
from ..models.market import bookmaker_prior
from ..outputs import prediction_frame
from ..pooled_split import pooled_outer_split, pooled_inner_temporal_split, assert_temporal
from ..preprocessing import Preprocessor
from ..statistics import per_obs_metric, paired_cluster_bootstrap
from .common import load_data, history_cols, odds_subset
from .anchor_continuum import LAMBDA_H_GRID, LAMBDA_A_GRID

OUT = OUTPUT_DIR / 'anchor_continuum' / 'prematch_lr'
OLD = OUTPUT_DIR / 'anchor_continuum'
ALIGN = ['fixture_key', 'fold', 'league', 'season', 'actual_class']
PROB = ['p_A', 'p_D', 'p_H']
NAMES = ['Canonical-Zero-LR', 'FreeMarket-LR', 'SoftAnchor-LR', 'Canonical-Fixed-LR']
REFERENCES = ['Bookmaker', 'Recalibrated-Market', 'Concat-LR', 'Residual-LR']
CONFIG = {'lambda_h_grid': LAMBDA_H_GRID, 'lambda_0_grid': LAMBDA_A_GRID,
          'objective': 'mean NLL + lambda_h * sum(B**2) + lambda_0 * sum(A**2)',
          'seed': 42, 'class_order': ['A', 'D', 'H'], 'logit_order': ['H/D', 'A/D'],
          'selection': 'validation-N weighted mean Log Loss; first minimum',
          'gradient_inf_acceptance': 1e-4, 'boundary_clear_improvement': 1e-6,
          'devig': 'multiplicative', 'information_set': 'Experiment B selected pre-match'}


def dump(name, obj):
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / name).open('x', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, allow_nan=False)


def checked_fit(X, z, y, lh, l0):
    model = CanonicalAnchorLR('zero', lambda_h=lh, lambda_0=l0).fit(X, z, y)
    if model.fit_info.gradient_inf > CONFIG['gradient_inf_acceptance']:
        raise RuntimeError('Zero optimizer gradient exceeds prespecified tolerance')
    return model


def prepared(train, valid, cols):
    assert_temporal(train, valid, 'canonical zero inner/outer')
    pp = Preprocessor(scale=True)
    Xtr, Xva = pp.fit_transform(train[cols]), pp.transform(valid[cols])
    _, ztr, mtr = bookmaker_prior(train)
    _, zva, mva = bookmaker_prior(valid)
    if not (mtr.all() and mva.all()):
        raise AssertionError('expected odds-valid data')
    return pp, Xtr, Xva, ztr, zva


def boundary_audit(rows, name):
    grid = sorted({r[name] for r in rows})
    path = [{'penalty': v, 'best_inner_log_loss': min(r['score'] for r in rows if r[name] == v)}
            for v in grid]
    best = min(path, key=lambda r: r['best_inner_log_loss'])
    # Zero is an exact Free endpoint, not a truncated positive lower boundary.
    edge = best['penalty'] in (grid[0], grid[-1]) and best['penalty'] != 0
    neighbor = path[1] if best['penalty'] == grid[0] else path[-2]
    gain = neighbor['best_inner_log_loss'] - best['best_inner_log_loss'] if edge else 0.
    return {'parameter': name, 'profiled_path': path, 'best': best['penalty'],
            'positive_boundary_selected': edge, 'improvement_vs_neighbor': gain,
            'boundary_limited': bool(edge and gain > CONFIG['boundary_clear_improvement'])}


def tune():
    dump('zero_config.json', CONFIG)
    df = load_data()
    for fold in range(5):
        train, _ = pooled_outer_split(df, fold)
        train, _, _ = odds_subset(train)
        cols = history_cols(fold, 'selected')
        splits = pooled_inner_temporal_split(train, min_train_seasons=2)
        if not splits:
            raise RuntimeError('no legal inner split')
        blocks = []
        for sp in splits:
            tr, va = train.iloc[sp['train_idx']], train.iloc[sp['val_idx']]
            _, Xtr, Xva, ztr, zva = prepared(tr, va, cols)
            blocks.append((Xtr, Xva, ztr, zva, tr.target_match_result.to_numpy(),
                           va.target_match_result.to_numpy()))
        surface = []
        for lh, l0 in itertools.product(LAMBDA_H_GRID, LAMBDA_A_GRID):
            scores, weights, convergence = [], [], []
            for Xtr, Xva, ztr, zva, ytr, yva in blocks:
                model = checked_fit(Xtr, ztr, ytr, lh, l0)
                scores.append(metrics.log_loss(yva, model.predict_proba(Xva, zva)))
                weights.append(len(yva))
                convergence.append(model.diagnostics())
            surface.append({'lambda_h': lh, 'lambda_0': l0,
                            'score': float(np.average(scores, weights=weights)),
                            'split_scores': scores, 'split_weights': weights,
                            'convergence': convergence})
        best = min(surface, key=lambda r: r['score'])
        audit = [boundary_audit(surface, name) for name in ('lambda_h', 'lambda_0')]
        dump(f'zero_tuning_fold_{fold}.json', {
            'fold': fold, 'features': cols, 'train_n': len(train),
            'train_keys': train.fixture_key.tolist(), 'inner_splits': splits,
            'selected': {k: best[k] for k in ('lambda_h', 'lambda_0', 'score')},
            'surface': surface, 'boundary_audit': audit})
        print('TUNED', fold, {k: best[k] for k in ('lambda_h', 'lambda_0', 'score')}, flush=True)
        if any(a['boundary_limited'] for a in audit):
            raise RuntimeError(f'Boundary-limited HPO fold {fold}; review required before outer fitting')


def aligned(a, b):
    for frame in (a, b):
        if frame.fixture_key.isna().any() or frame.fixture_key.duplicated().any():
            raise AssertionError('missing/duplicate fixture key')
        if set(frame.actual_class) - set(metrics.class_order()):
            raise AssertionError('unknown class')
        metrics.assert_valid_proba(frame[PROB].to_numpy(float))
    a = a.sort_values('fixture_key').reset_index(drop=True)
    b = b.sort_values('fixture_key').reset_index(drop=True)
    if not a[ALIGN].equals(b[ALIGN]):
        raise AssertionError('fixture/fold/league/season/actual_class mismatch')
    return a, b


def predict():
    if (OUT / 'zero_oos.csv').exists():
        raise FileExistsError('Zero OOS already exists; no automatic rerun')
    if json.loads((OUT / 'zero_config.json').read_text()) != CONFIG:
        raise AssertionError('configuration changed after tuning')
    tuning = [json.loads((OUT / f'zero_tuning_fold_{f}.json').read_text()) for f in range(5)]
    if any(a['boundary_limited'] for t in tuning for a in t['boundary_audit']):
        raise RuntimeError('boundary audit requires review')
    df = load_data()
    legacy = pd.read_csv(OUTPUT_DIR / 'raw/experiment_b_oos.csv')
    base = legacy[(legacy.method == 'Bookmaker') & (legacy.representation == 'selected')]
    rows, diagnostics = [], []
    for t in tuning:
        fold, cols = t['fold'], t['features']
        tr, te = pooled_outer_split(df, fold)
        tr, _, _ = odds_subset(tr); te, _, _ = odds_subset(te)
        assert tr.fixture_key.tolist() == t['train_keys']
        assert cols == history_cols(fold, 'selected')
        pp, Xtr, Xte, ztr, zte = prepared(tr, te, cols)
        s = t['selected']
        m = checked_fit(Xtr, ztr, tr.target_match_result.to_numpy(), s['lambda_h'], s['lambda_0'])
        frame = prediction_frame(te, m.predict_proba(Xte, zte), fold, NAMES[0], 'selected')
        expected = base[base.fold == fold]
        aligned(frame, expected)
        if frame.fixture_key.tolist() != expected.fixture_key.tolist():
            raise AssertionError('serialized test order differs from frozen B')
        rows.append(frame)
        diagnostics.append({'fold': fold, 'method': NAMES[0], 'seed': 42,
            'lambda_h': s['lambda_h'], 'lambda_0': s['lambda_0'], 'lambda_a': 0.,
            'inner_validation_score': s['score'], 'fit': m.diagnostics(), 'B': m.B.tolist(),
            'features': cols, 'train_n': len(tr), 'test_n': len(te),
            'test_fixture_keys': te.fixture_key.tolist(),
            'preprocessing': {'imputation': pp.imputer.statistics_.tolist(),
                              'mean': pp.scaler.mean_.tolist(), 'scale': pp.scaler.scale_.tolist(),
                              'market_logits_scaled': False}})
        print('PREDICTED Zero fold', fold, flush=True)
    pd.concat(rows, ignore_index=True).to_csv(OUT / 'zero_oos.csv', index=False, mode='x')
    dump('zero_diagnostics.json', diagnostics)


def analyze():
    zero = pd.read_csv(OUT / 'zero_oos.csv')
    previous = pd.read_csv(OLD / 'canonical_prematch_oos.csv')
    legacy = pd.read_csv(OUTPUT_DIR / 'raw/experiment_b_oos.csv')
    reference = legacy[(legacy.representation == 'selected') & legacy.method.isin(REFERENCES)]
    combined = pd.concat([zero, previous, reference], ignore_index=True)
    assert set(combined.method) == set(NAMES + REFERENCES)
    groups = {m: g for m, g in combined.groupby('method')}
    base = groups['Bookmaker']
    summary = []
    for method, g in groups.items():
        a, _ = aligned(g, base)
        assert len(a) == 2789 and a.fold.nunique() == 5
        summary.append({'method': method, 'N': len(a),
                        **metrics.compute_all_metrics(a.actual_class.to_numpy(), a[PROB].to_numpy())})
    pd.DataFrame(summary).to_csv(OUT / 'metrics.csv', index=False, mode='x')
    pairs = list(itertools.combinations(NAMES, 2)) + list(itertools.product(NAMES, REFERENCES))
    contrasts = []
    for first, second in pairs:
        a, b = aligned(groups[first], groups[second])
        clusters = (a.league + '|' + a.season).to_numpy()
        assert len(np.unique(clusters)) == 15
        for metric in ('log_loss', 'brier', 'acc'):
            stats = paired_cluster_bootstrap(
                per_obs_metric(a.actual_class.to_numpy(), a[PROB].to_numpy(), metric),
                per_obs_metric(b.actual_class.to_numpy(), b[PROB].to_numpy(), metric),
                clusters, n_boot=10000, seed=42)
            contrasts.append({'first': first, 'second': second, 'metric': metric, **stats})
        print('PAIRED', first, '-', second, flush=True)
    pd.DataFrame(contrasts).to_csv(OUT / 'contrasts.csv', index=False, mode='x')
    dump('analysis.json', {'metrics': summary, 'contrasts': contrasts, 'n_boot': 10000,
         'seed': 42, 'n_clusters': 15, 'difference': 'first minus second',
         'alignment_columns': ALIGN, 'all_alignment_checks_passed': True,
         'ci': 'unadjusted paired percentile 95%; macro-F1 descriptive only'})
    olddiag = json.loads((OLD / 'canonical_prematch_diagnostics.json').read_text())
    audit, soft = [], []
    for d in olddiag:
        surface = [{**r, 'score': r['weighted_log_loss']} for r in d['tuning']['candidates']]
        audit.append({'method': d['method'], 'fold': d['fold'], **boundary_audit(surface, 'lambda_h')})
        if d['method'] == 'SoftAnchor-LR':
            soft.append(d)
    dump('existing_lambda_h_boundary_audit.json', audit)
    dump('soft_matrices_and_complete_hpo.json', soft)
    dump('artifact_sha256.json', {p.relative_to(PROJECT_ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
         for p in sorted(OUT.iterdir()) if p.is_file()})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=['tune', 'predict', 'analyze'], required=True)
    stage = parser.parse_args().stage
    {'tune': tune, 'predict': predict, 'analyze': analyze}[stage]()
