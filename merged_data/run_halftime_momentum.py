"""
run_halftime_momentum.py
对比 15-16 赛季半场预测：Baseline (FE5) vs Baseline+Momentum (FE5+Momentum)

核心逻辑：
  1. 加载两张表，按 match_id 取交集保证样本一致
  2. 分层时间划分：在每个联赛内部按时间顺序划分，然后合并
  3. 对同一组样本分别训练相同模型，对比 test 指标
  4. 输出对比表格 + CSV 报告

数据特点分析：
  - 总样本：1823场（五大联赛单赛季）
  - 类别分布：H 44.3%, A 29.7%, D 25.9%（不平衡）
  - 联赛差异：Spain主胜率最高(48.2%)，Germany客胜率最高(32.7%)
  - 时间跨度：2015-08-07 ~ 2016-05-15（282天）
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import (
    RandomForestClassifier, AdaBoostClassifier, 
    GradientBoostingClassifier, StackingClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, log_loss, classification_report
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.config.config import (
    IDENTIFIER_COLUMNS, TARGET_COLUMNS, TARGET_COLUMN,
    LABEL_ENCODING, MODEL_DISPLAY_NAMES
)
from src.features.prematch_features import handle_missing_values

# =====================================================
# 0. 配置参数（专门针对单赛季五大联赛数据优化）
# =====================================================

BASE_DIR = Path(r"d:/program/SoccerPre/merged_data")
BASELINE_PATH = BASE_DIR / "feature_engineer5" / "all_leagues_2015-2016.csv"
MOMENTUM_PATH = BASE_DIR / "all_leagues_2015-2016.csv"

MOMENTUM_ONLY_COLS = [
    "n_rows_period1",
    "eq_area_decay_tau600s", "eq_area_signed", "eq_max", "eq_mean",
    "eq_mean_last_600s", "eq_min", "eq_slope_last_third",
    "eq_slope_overall", "eq_value_at_end", "eq_zero_crossings",
    "int_area_decay_tau600s", "int_area_signed", "int_max", "int_mean",
    "int_mean_last_600s", "int_min", "int_slope_last_third",
    "int_slope_overall", "int_value_at_end", "int_zero_crossings",
    "league_name",
]

LEAKAGE_COLUMNS = [
    "home_ppg", "away_ppg",
    "home_team_goal_count", "away_team_goal_count", "total_goal_count",
    "full_time_result", "ft_result", "result",
    "home_team_goal_timings", "away_team_goal_timings",
    "home_team_possession", "away_team_possession",
    "home_team_shots", "away_team_shots",
    "home_team_shots_on_target", "away_team_shots_on_target",
    "home_team_shots_off_target", "away_team_shots_off_target",
    "home_team_corners", "away_team_corners",
    "home_team_corner_count", "away_team_corner_count",
    "home_team_fouls", "away_team_fouls",
    "home_team_yellow_cards", "away_team_yellow_cards",
    "home_team_red_cards", "away_team_red_cards",
    "home_team_cards", "away_team_cards",
    "home_team_goal_count_second_half", "away_team_goal_count_second_half",
    "total_goals_second_half",
    "home_team_second_half_cards", "away_team_second_half_cards",
    "home_team_yellow_cards_second_half", "away_team_yellow_cards_second_half",
    "home_team_red_cards_second_half", "away_team_red_cards_second_half",
    "home_team_shots_second_half", "away_team_shots_second_half",
    "home_team_shots_on_target_second_half", "away_team_shots_on_target_second_half",
    "home_team_corners_second_half", "away_team_corners_second_half",
    "home_team_fouls_second_half", "away_team_fouls_second_half",
    "team_a_xg", "team_b_xg",
]

TRAIN_RATIO = 0.60
VAL_RATIO = 0.20
TEST_RATIO = 0.20

# 模型参数（针对单赛季五大联赛数据优化）
# 关键考虑：
# 1. 数据量小(1823)：强正则化防止过拟合
# 2. 类别不平衡(H44.3%, A29.7%, D25.9%)：D类权重增加
# 3. 特征数多(200维)：特征采样/正则化
# 4. 联赛间差异：模型需要泛化到不同联赛
# 5. Momentum数据特点：高峰度、高相关性、长尾分布
MODEL_PARAMS_EXPERIMENT = {
    'random_forest': {
        'n_estimators': 150,
        'max_depth': 5,
        'min_samples_split': 10,
        'min_samples_leaf': 5,
        'max_features': 0.5,
        'random_state': 42,
        'class_weight': {0: 1.0, 1: 1.7, 2: 1.0},
        'bootstrap': True,
        'max_samples': 0.8
    },
    'logistic_regression': {
        'C': 0.3,
        'random_state': 42,
        'class_weight': {0: 1.0, 1: 1.7, 2: 1.0},
        'solver': 'lbfgs',
        'max_iter': 2000
    },
    'xgboost': {
        'n_estimators': 80,
        'max_depth': 3,
        'learning_rate': 0.05,
        'subsample': 0.7,
        'colsample_bytree': 0.6,
        'reg_lambda': 2.0,
        'reg_alpha': 0.5,
        'random_state': 42,
        'use_label_encoder': False,
        'eval_metric': 'mlogloss',
        'min_child_weight': 5,
        'gamma': 0.1
    },
    'svm': {
        'C': 0.5,
        'max_iter': 10000,
        'random_state': 42,
        'class_weight': {0: 1.0, 1: 1.7, 2: 1.0},
        'dual': 'auto'
    },
    'fnn': {
        'hidden_layer_sizes': (32, 16),
        'activation': 'relu',
        'solver': 'adam',
        'learning_rate_init': 0.0005,
        'max_iter': 300,
        'random_state': 42,
        'early_stopping': True,
        'validation_fraction': 0.2,
        'alpha': 0.01,
        'verbose': False
    },
    'adaboost': {
        'n_estimators': 150,
        'learning_rate': 0.1,
        'random_state': 42
    },
    'gradient_boosting': {
        'n_estimators': 100,
        'max_depth': 3,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'min_samples_split': 10,
        'min_samples_leaf': 5,
        'random_state': 42
    }
}

MODEL_DISPLAY_NAMES_EXPERIMENT = {
    'random_forest': 'Random Forest',
    'logistic_regression': 'Logistic Regression',
    'xgboost': 'XGBoost',
    'svm': 'SVM',
    'fnn': 'Feedforward NN',
    'adaboost': 'AdaBoost',
    'gradient_boosting': 'Gradient Boosting',
    'stacking': 'Stacking Ensemble'
}

RANDOM_SEED = 42


# =====================================================
# 1. 数据加载与对齐
# =====================================================

def load_and_align():
    df_base = pd.read_csv(BASELINE_PATH, encoding="utf-8-sig")
    df_mom = pd.read_csv(MOMENTUM_PATH, encoding="utf-8-sig")

    print(f"Baseline raw: {df_base.shape}")
    print(f"Momentum raw: {df_mom.shape}")

    df_base["match_id"] = df_base["match_id"].astype(str)
    df_mom["match_id"] = df_mom["match_id"].astype(str)

    common_ids = set(df_base["match_id"]) & set(df_mom["match_id"])
    print(f"Common match_id: {len(common_ids)}")

    df_base = df_base[df_base["match_id"].isin(common_ids)].copy()
    df_mom = df_mom[df_mom["match_id"].isin(common_ids)].copy()

    df_base = df_base.sort_values("match_id").reset_index(drop=True)
    df_mom = df_mom.sort_values("match_id").reset_index(drop=True)

    assert len(df_base) == len(df_mom), "Row count mismatch after alignment"
    assert (df_base["match_id"] == df_mom["match_id"]).all(), "match_id order mismatch"

    print(f"Aligned: {len(df_base)} rows")
    return df_base, df_mom


# =====================================================
# 2. 分层时间划分（关键改进）
# =====================================================

def stratified_time_split(df, train_ratio=TRAIN_RATIO, val_ratio=VAL_RATIO):
    """
    分层时间划分：在每个联赛内部按时间顺序划分，然后合并
    
    优势：
    1. 保证每个联赛在train/val/test中都有代表性
    2. 模拟真实预测场景（用早期数据预测后期比赛）
    3. 避免联赛分布不均导致的偏差
    
    参数：
    - df: 包含 'league' 和 'timestamp' 列的DataFrame
    - train_ratio: 训练集比例
    - val_ratio: 验证集比例
    """
    league_col = 'league' if 'league' in df.columns else 'league_name'
    time_col = 'timestamp' if 'timestamp' in df.columns else 'date_GMT'
    
    train_dfs = []
    val_dfs = []
    test_dfs = []
    
    leagues = df[league_col].unique()
    
    for lg in leagues:
        lg_df = df[df[league_col] == lg].copy()
        lg_df = lg_df.sort_values(time_col).reset_index(drop=True)
        
        n = len(lg_df)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))
        
        train_dfs.append(lg_df.iloc[:train_end].copy())
        val_dfs.append(lg_df.iloc[train_end:val_end].copy())
        test_dfs.append(lg_df.iloc[val_end:].copy())
    
    train = pd.concat(train_dfs, ignore_index=True)
    val = pd.concat(val_dfs, ignore_index=True)
    test = pd.concat(test_dfs, ignore_index=True)
    
    train = train.sort_values(time_col).reset_index(drop=True)
    val = val.sort_values(time_col).reset_index(drop=True)
    test = test.sort_values(time_col).reset_index(drop=True)
    
    return train, val, test


def simple_time_split(df, train_ratio=TRAIN_RATIO, val_ratio=VAL_RATIO):
    """简单时间划分（用于对比）"""
    sort_col = "timestamp" if "timestamp" in df.columns else "date_GMT"
    df_sorted = df.sort_values(sort_col).reset_index(drop=True)

    n = len(df_sorted)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train = df_sorted.iloc[:train_end].copy()
    val = df_sorted.iloc[train_end:val_end].copy()
    test = df_sorted.iloc[val_end:].copy()

    return train, val, test


# =====================================================
# 3. 特征工程
# =====================================================

def build_features(df, include_momentum=True):
    exclude = set(IDENTIFIER_COLUMNS + TARGET_COLUMNS + LEAKAGE_COLUMNS)

    if not include_momentum:
        exclude = exclude | set(MOMENTUM_ONLY_COLS)

    feature_cols = [
        c for c in df.columns
        if c not in exclude and (df[c].dtype != "object" or c == "ht_result")
    ]

    X = df[feature_cols].copy()

    if "ht_result" in X.columns:
        X["ht_result"] = X["ht_result"].map(LABEL_ENCODING)

    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce")

    y = df[TARGET_COLUMN].copy() if TARGET_COLUMN in df.columns else None
    return X, y


def encode_labels(y):
    le = LabelEncoder()
    le.fit(["H", "D", "A"])
    return le.transform(y), le


def prepare_splits(train_df, val_df, test_df, include_momentum=True):
    X_train, y_train = build_features(train_df, include_momentum)
    X_val, y_val = build_features(val_df, include_momentum)
    X_test, y_test = build_features(test_df, include_momentum)

    X_train = handle_missing_values(X_train)
    X_val = handle_missing_values(X_val)
    X_test = handle_missing_values(X_test)

    common_cols = list(
        set(X_train.columns) & set(X_val.columns) & set(X_test.columns)
    )
    common_cols.sort()

    X_train = X_train[common_cols]
    X_val = X_val[common_cols]
    X_test = X_test[common_cols]

    y_train_enc, le = encode_labels(y_train)
    y_val_enc, _ = encode_labels(y_val)
    y_test_enc, _ = encode_labels(y_test)

    return X_train, y_train_enc, X_val, y_val_enc, X_test, y_test_enc


# =====================================================
# 4. 模型训练与评估
# =====================================================

def train_model(model_type, X_train, y_train, X_val, y_val):
    params = MODEL_PARAMS_EXPERIMENT.get(model_type, {}).copy()

    if model_type == "random_forest":
        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)
    elif model_type == "logistic_regression":
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_train)
        X_va = scaler.transform(X_val)
        model = LogisticRegression(**params)
        model.fit(X_tr, y_train)
        model._scaler = scaler
        model._uses_scaler = True
        return model
    elif model_type == "xgboost":
        import xgboost as xgb
        model = xgb.XGBClassifier(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
    elif model_type == "svm":
        svm_params = params.copy()
        svm_params.pop("probability", None)
        base = LinearSVC(**svm_params)
        model = CalibratedClassifierCV(base, cv=3)
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_train)
        model.fit(X_tr, y_train)
        model._scaler = scaler
        model._uses_scaler = True
        return model
    elif model_type == "fnn":
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_train)
        model = MLPClassifier(**params)
        model.fit(X_tr, y_train)
        model._scaler = scaler
        model._uses_scaler = True
        return model
    elif model_type == "adaboost":
        base_estimator = DecisionTreeClassifier(max_depth=3, random_state=42)
        model = AdaBoostClassifier(
            estimator=base_estimator,
            **params
        )
        model.fit(X_train, y_train)
    elif model_type == "gradient_boosting":
        model = GradientBoostingClassifier(**params)
        model.fit(X_train, y_train)
    elif model_type == "stacking":
        estimators = [
            ('ada', AdaBoostClassifier(
                estimator=DecisionTreeClassifier(max_depth=3),
                n_estimators=150,
                learning_rate=0.1,
                random_state=42
            )),
            ('gb', GradientBoostingClassifier(
                n_estimators=100,
                max_depth=3,
                learning_rate=0.05,
                subsample=0.8,
                random_state=42
            )),
            ('rf', RandomForestClassifier(
                n_estimators=100,
                max_depth=5,
                class_weight={0: 1.0, 1: 1.7, 2: 1.0},
                random_state=42
            )),
        ]
        model = StackingClassifier(
            estimators=estimators,
            final_estimator=LogisticRegression(
                C=0.5,
                class_weight={0: 1.0, 1: 1.7, 2: 1.0},
                max_iter=1000,
                random_state=42
            ),
            cv=3,
            stack_method='predict_proba'
        )
        model.fit(X_train, y_train)
    else:
        raise ValueError(f"Unknown model: {model_type}")

    model._uses_scaler = False
    return model


def predict(model, X):
    if getattr(model, "_uses_scaler", False):
        X = model._scaler.transform(X)
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)
    return y_pred, y_proba


def evaluate(y_true, y_pred, y_proba):
    labels = [0, 1, 2]
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro", labels=labels),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", labels=labels),
        "precision_macro": precision_score(y_true, y_pred, average="macro", labels=labels),
        "recall_macro": recall_score(y_true, y_pred, average="macro", labels=labels),
        "log_loss": log_loss(y_true, y_proba, labels=labels),
    }


# =====================================================
# 5. 对比实验
# =====================================================

def run_comparison():
    print("=" * 70)
    print("Halftime Prediction: Baseline vs Baseline+Momentum (15-16)")
    print("=" * 70)
    print("\n【数据划分策略说明】")
    print("  - 数据量：1823场（单赛季15-16五大联赛）")
    print("  - 划分方式：分层时间划分（Stratified Time-based split）")
    print("  - 比例：Train=60%, Val=20%, Test=20%")
    print("  - 特点：在每个联赛内部按时间顺序划分，保证联赛代表性")
    print("  - 类别分布：H 44.3%, A 29.7%, D 25.9%（D为少数类）")
    print("=" * 70)

    df_base, df_mom = load_and_align()

    print("\n[2] Stratified time-order split ...")
    train_base, val_base, test_base = stratified_time_split(df_base)
    train_mom, val_mom, test_mom = stratified_time_split(df_mom)

    league_col = 'league' if 'league' in df_base.columns else 'league_name'
    
    print("\n  【各联赛划分详情】")
    for lg in sorted(df_base[league_col].unique()):
        tr = len(train_base[train_base[league_col] == lg])
        va = len(val_base[val_base[league_col] == lg])
        te = len(test_base[test_base[league_col] == lg])
        print(f"    {lg}: train={tr}, val={va}, test={te}")
    
    print(f"\n  Total: train={len(train_base)}, val={len(val_base)}, test={len(test_base)}")

    print("\n[3] Building features ...")
    X_tr_b, y_tr_b, X_va_b, y_va_b, X_te_b, y_te_b = prepare_splits(
        train_base, val_base, test_base, include_momentum=False
    )
    X_tr_m, y_tr_m, X_va_m, y_va_m, X_te_m, y_te_m = prepare_splits(
        train_mom, val_mom, test_mom, include_momentum=True
    )

    print(f"  Baseline features:  {X_tr_b.shape[1]}")
    print(f"  Momentum features:  {X_tr_m.shape[1]}")
    print(f"  Momentum extra:     {X_tr_m.shape[1] - X_tr_b.shape[1]}")

    model_types = ["random_forest", "logistic_regression", "xgboost", "svm", "fnn", 
                   "adaboost", "gradient_boosting", "stacking"]
    all_results = {}

    for model_type in model_types:
        display = MODEL_DISPLAY_NAMES_EXPERIMENT.get(model_type, model_type)
        print(f"\n{'='*60}")
        print(f"Model: {display}")
        print("=" * 60)

        results = {}
        for tag, X_tr, y_tr, X_va, y_va, X_te, y_te in [
            ("Baseline", X_tr_b, y_tr_b, X_va_b, y_va_b, X_te_b, y_te_b),
            ("+Momentum", X_tr_m, y_tr_m, X_va_m, y_va_m, X_te_m, y_te_m),
        ]:
            print(f"\n  [{tag}] Training ...")
            try:
                model = train_model(model_type, X_tr, y_tr, X_va, y_va)
                _, val_proba = predict(model, X_va)
                y_pred_te, y_proba_te = predict(model, X_te)
                test_metrics = evaluate(y_te, y_pred_te, y_proba_te)

                results[tag] = test_metrics
                print(f"  [{tag}] Test acc={test_metrics['accuracy']:.4f}  "
                      f"f1_macro={test_metrics['f1_macro']:.4f}  "
                      f"log_loss={test_metrics['log_loss']:.4f}")
            except Exception as e:
                print(f"  [{tag}] ERROR: {e}")
                results[tag] = None

        all_results[model_type] = results

    print_comparison_table(all_results)
    save_comparison_csv(all_results)

    print("\n" + "=" * 70)
    print("Comparison complete.")
    print("=" * 70)


# =====================================================
# 6. 输出
# =====================================================

def print_comparison_table(all_results):
    print("\n" + "=" * 90)
    print("COMPARISON: Baseline vs Baseline+Momentum  (Test Set)")
    print("=" * 90)

    metrics_keys = ["accuracy", "f1_macro", "f1_weighted", "precision_macro",
                    "recall_macro", "log_loss"]

    header = f"{'Model':<22} {'Dataset':<12}"
    for m in metrics_keys:
        header += f"{m:>16}"
    print(header)
    print("-" * len(header))

    for model_type, results in all_results.items():
        display = MODEL_DISPLAY_NAMES_EXPERIMENT.get(model_type, model_type)
        for tag in ["Baseline", "+Momentum"]:
            r = results.get(tag)
            if r is None:
                line = f"{display:<22} {tag:<12}"
                for _ in metrics_keys:
                    line += f"{'N/A':>16}"
                print(line)
                continue

            line = f"{display:<22} {tag:<12}"
            for m in metrics_keys:
                line += f"{r[m]:>16.4f}"
            print(line)

        delta_line = f"{'':>22} {'Delta':>12}"
        r_b = results.get("Baseline")
        r_m = results.get("+Momentum")
        if r_b and r_m:
            for m in metrics_keys:
                d = r_m[m] - r_b[m]
                sign = "+" if d > 0 else ""
                delta_line += f"{sign}{d:>15.4f}"
            print(delta_line)
        print()

    best_baseline = max(
        all_results.keys(),
        key=lambda k: (all_results[k].get("Baseline") or {}).get("f1_macro", 0),
    )
    best_momentum = max(
        all_results.keys(),
        key=lambda k: (all_results[k].get("+Momentum") or {}).get("f1_macro", 0),
    )
    b_f1 = (all_results[best_baseline].get("Baseline") or {}).get("f1_macro", 0)
    m_f1 = (all_results[best_momentum].get("+Momentum") or {}).get("f1_macro", 0)

    print(f"Best Baseline  F1-macro: {MODEL_DISPLAY_NAMES_EXPERIMENT.get(best_baseline, best_baseline)} = {b_f1:.4f}")
    print(f"Best +Momentum F1-macro: {MODEL_DISPLAY_NAMES_EXPERIMENT.get(best_momentum, best_momentum)} = {m_f1:.4f}")
    if b_f1 > 0:
        print(f"Improvement: {(m_f1 - b_f1)/b_f1*100:+.2f}%")


def save_comparison_csv(all_results):
    rows = []
    for model_type, results in all_results.items():
        display = MODEL_DISPLAY_NAMES_EXPERIMENT.get(model_type, model_type)
        for tag in ["Baseline", "+Momentum"]:
            r = results.get(tag)
            if r is None:
                continue
            row = {"model": display, "dataset": tag}
            row.update(r)
            rows.append(row)

        r_b = results.get("Baseline")
        r_m = results.get("+Momentum")
        if r_b and r_m:
            row = {"model": display, "dataset": "Delta"}
            for k in r_b:
                row[k] = r_m[k] - r_b[k]
            rows.append(row)

    df_out = pd.DataFrame(rows)
    out_path = BASE_DIR / "feature_engineer5" / "momentum_comparison_results.csv"
    df_out.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\nResults saved to: {out_path}")


# =====================================================
# 7. 数据划分合理性分析
# =====================================================

def analyze_split_rationality():
    print("\n" + "=" * 70)
    print("【数据划分合理性分析】")
    print("=" * 70)
    print("")
    print("1. 划分方式选择：分层时间划分 vs 简单时间划分")
    print("")
    print("   本实验采用分层时间划分（Stratified Time-based split）：")
    print("")
    print("   [+] 优点：")
    print("      - 保证每个联赛在train/val/test中都有代表性")
    print("      - 模拟真实预测场景（用早期数据预测后期比赛）")
    print("      - 避免联赛分布不均导致的偏差")
    print("      - 适应五大联赛时间高度重叠的特点")
    print("")
    print("   [-] 缺点：")
    print("      - 实现稍复杂")
    print("      - 联赛间的时间不同步（但影响较小）")
    print("")
    print("2. 比例选择：60/20/20")
    print("")
    print("   [+] 原因：")
    print("      - 单赛季数据量有限（~1823场），需要保证验证集和测试集有足够样本")
    print("      - 验证集增大有助于更稳定的超参数调整")
    print("      - 测试集增大提高结果的统计显著性")
    print("")
    print("   [+] 样本数（分层后）：")
    print("     - Train: 60% ≈ 1094场（每个联赛约220场）")
    print("     - Val:   20% ≈ 365场（每个联赛约73场）")
    print("     - Test:  20% ≈ 364场（每个联赛约73场）")
    print("")
    print("3. 模型参数优化要点")
    print("")
    print("   [+] 针对数据特点的调整：")
    print("      - 数据量小(1823)：强正则化防止过拟合")
    print("        * RF: max_depth=5, min_samples_leaf=5")
    print("        * XGB: max_depth=3, min_child_weight=5, gamma=0.1")
    print("        * LR: C=0.3 (更强正则化)")
    print("        * FNN: hidden=(32,16), alpha=0.01")
    print("")
    print("      - 类别不平衡(H44.3%, A29.7%, D25.9%)：")
    print("        * class_weight: {H:1.0, D:1.7, A:1.0}")
    print("        * D类权重=1.7x（基于H/D比例）")
    print("")
    print("      - 特征数多(200维)：")
    print("        * RF: max_features=0.5, max_samples=0.8")
    print("        * XGB: colsample_bytree=0.6, subsample=0.7")
    print("")
    print("4. 联赛间差异处理")
    print("")
    print("   [+] 五大联赛的目标变量分布差异：")
    print("      - Spain: 主胜率最高(48.2%)")
    print("      - Germany: 客胜率最高(32.7%)")
    print("      - France/England: 平局率最高(28.2-28.4%)")
    print("")
    print("   [+] 分层划分的优势：")
    print("      - 每个联赛的分布特点在train/val/test中保持一致")
    print("      - 模型学习到的是跨联赛的通用模式")
    print("")
    print("5. 潜在改进方向")
    print("")
    print("   [+] 可考虑的增强方案：")
    print("     1. 联赛作为特征：将联赛编码加入特征")
    print("     2. 联赛特定模型：为每个联赛训练独立模型")
    print("     3. 时间序列交叉验证：使用滑动窗口验证")
    print("     4. 特征选择：基于重要性筛选特征")
    print("")


if __name__ == "__main__":
    run_comparison()
    analyze_split_rationality()
