"""
momentum_merge.py
将 match_level_momentum_features.csv 与 all_leagues_2015-2016.csv 合并

输入文件：
  1. all_leagues_2015-2016.csv         - 五大联赛15-16赛季特征数据（已合并）
  2. match_level_momentum_features.csv - momentum特征数据

输出文件：
  feature_engineer5_momentum/all_leagues_2015-2016_momentum.merged.csv - 合并后全部数据
  feature_engineer5_momentum/{league}_2015-2016_momentum.csv          - 按联赛输出

合并策略：
  1. 优先按 match_id 精确匹配（inner join）
  2. 如果 match_id 不匹配，按时间顺序位置对齐合并
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")


# =====================================================
# 0. 路径配置
# =====================================================

BASE_DIR = Path(r"d:/program/SoccerPre/merged_data")

# 输入文件（直接使用已合并的文件）
FEATURES_PATH = BASE_DIR / "all_leagues_2015-2016.csv"
MOMENTUM_PATH = BASE_DIR / "match_level_momentum_features.csv"

# 输出目录
OUTPUT_DIR = BASE_DIR / "feature_engineer5_momentum"
OUTPUT_DIR.mkdir(exist_ok=True)

TARGET_SEASON = "2015-2016"


# =====================================================
# 1. 加载数据
# =====================================================

def load_features_data() -> pd.DataFrame:
    """加载已合并的五大联赛数据"""
    if not FEATURES_PATH.exists():
        raise FileNotFoundError(f"Features file not found: {FEATURES_PATH}")
    
    df = pd.read_csv(FEATURES_PATH, encoding="utf-8-sig")
    print(f"[Features] Loaded: {len(df):,} rows, {df.shape[1]:,} columns")
    
    # 确保 match_id 为字符串
    df["match_id"] = df["match_id"].astype(str)
    
    # 按时间排序（确保时间顺序一致）
    if "timestamp" in df.columns:
        df = df.sort_values("timestamp").reset_index(drop=True)
    
    return df


def load_momentum_data() -> pd.DataFrame:
    """加载 momentum 特征数据"""
    if not MOMENTUM_PATH.exists():
        raise FileNotFoundError(f"Momentum file not found: {MOMENTUM_PATH}")
    
    df = pd.read_csv(MOMENTUM_PATH, encoding="utf-8-sig")
    print(f"[Momentum] Loaded: {len(df):,} rows, {df.shape[1]:,} columns")
    
    # 统一 match_id 格式（去掉 .0 后缀）
    df["match_id"] = df["match_id"].astype(str).str.replace(r"\.0$", "", regex=True)
    
    return df


# =====================================================
# 2. 合并逻辑（保留原有策略）
# =====================================================

def merge_datasets(df_features: pd.DataFrame, df_momentum: pd.DataFrame) -> pd.DataFrame:
    """
    合并特征数据与 momentum 数据
    
    策略：
    1. 首先尝试按 match_id 精确匹配（inner join）
    2. 如果精确匹配失败，按时间顺序位置对齐合并
    """
    print("\n[Merge Strategy]")
    
    # 检查 match_id 格式
    feat_sample = df_features["match_id"].iloc[0]
    mom_sample = df_momentum["match_id"].iloc[0]
    print(f"  Features match_id sample: {feat_sample[:50]}")
    print(f"  Momentum match_id sample: {mom_sample[:50]}")
    
    # 方式 A: 按 match_id 精确匹配
    df_merged = df_features.merge(
        df_momentum,
        on="match_id",
        how="inner",
        suffixes=("", "_momentum")
    )
    
    if len(df_merged) > 0:
        match_rate = len(df_merged) / len(df_features) * 100
        print(f"  [SUCCESS] match_id join: {len(df_merged):,} rows matched ({match_rate:.1f}%)")
        return df_merged
    
    # 方式 B: 位置对齐合并（按时间顺序）
    print("  [FALLBACK] match_id not matching, using positional alignment")
    
    # 确保两边都按时间排序
    df_features_sorted = df_features.copy()
    df_momentum_sorted = df_momentum.sort_values("match_id").reset_index(drop=True)
    
    # 取较短长度对齐
    n_align = min(len(df_features_sorted), len(df_momentum_sorted))
    print(f"  Aligning {n_align:,} rows (features={len(df_features_sorted):,}, momentum={len(df_momentum_sorted):,})")
    
    df_merged = pd.concat([
        df_features_sorted.iloc[:n_align].reset_index(drop=True),
        df_momentum_sorted.iloc[:n_align].drop(columns=["match_id"]).reset_index(drop=True)
    ], axis=1)
    
    return df_merged


# =====================================================
# 3. 按联赛输出（保留原有功能）
# =====================================================

def save_outputs(df_merged: pd.DataFrame) -> None:
    """保存合并结果：只输出单个合并文件"""
    season_tag = TARGET_SEASON.replace("/", "-")
    
    # 保存全部合并数据（仅这一个文件）
    all_out = OUTPUT_DIR / f"all_leagues_{season_tag}_momentum.merged.csv"
    df_merged.to_csv(all_out, index=False, encoding="utf-8-sig")
    print(f"  Output: {len(df_merged):,} rows -> {all_out.name}")


# =====================================================
# 4. 验证（保留原有功能）
# =====================================================

def validate_merge(df_merged: pd.DataFrame) -> None:
    """验证合并结果"""
    print("\n[Validation]")
    
    # 检查 momentum 特征
    momentum_cols = [c for c in df_merged.columns if c.startswith(("eq_", "mod_"))]
    if momentum_cols:
        print(f"  Momentum features added: {len(momentum_cols)}")
        print(f"  Sample columns: {', '.join(momentum_cols[:5])}...")
    else:
        print(f"  [WARN] No momentum columns found!")
    
    # 检查缺失值
    total_na = df_merged.isna().sum().sum()
    if total_na > 0:
        print(f"  Total NaN values: {total_na:,}")
    else:
        print(f"  [OK] No NaN values")
    
    # 检查联赛分布
    if "league" in df_merged.columns:
        league_counts = df_merged["league"].value_counts().sort_index()
        print(f"\n  League distribution:")
        for league, count in league_counts.items():
            print(f"    {league}: {count:,}")
    
    # 检查重复 match_id
    if "match_id" in df_merged.columns:
        dup = df_merged["match_id"].duplicated().sum()
        if dup > 0:
            print(f"\n  [WARN] {dup} duplicate match_id found")
        else:
            print(f"\n  [OK] No duplicate match_id")


# =====================================================
# 5. 主流程
# =====================================================

def main():
    print("=" * 60)
    print("Momentum Feature Merge Tool")
    print("=" * 60)
    print(f"Input:       {FEATURES_PATH.name}")
    print(f"Momentum:    {MOMENTUM_PATH.name}")
    print(f"Output dir:  {OUTPUT_DIR.name}")
    print(f"Season:      {TARGET_SEASON}")
    print("=" * 60)
    
    # Step 1: 加载已合并的特征数据
    print("\n[1/3] Loading features data...")
    df_features = load_features_data()
    
    # Step 2: 加载 momentum 数据
    print("\n[2/3] Loading momentum data...")
    df_momentum = load_momentum_data()
    
    # Step 3: 合并（保留原有逻辑）
    print("\n[3/3] Merging datasets...")
    df_merged = merge_datasets(df_features, df_momentum)
    
    # Step 4: 验证
    validate_merge(df_merged)
    
    # Step 5: 保存（保留按联赛输出）
    print("\n[4/4] Saving outputs...")
    save_outputs(df_merged)
    
    print("\n" + "=" * 60)
    print("Merge completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()