# 足球比赛预测项目 - 改进代码使用指南

## 📋 目录
1. [改进内容总览](#改进内容总览)
2. [兼容性检查](#兼容性检查)
3. [使用方法](#使用方法)
4. [完整训练流程](#完整训练流程)

---

## 改进内容总览

### ✅ 已完成的7项改进

| 改进项 | 文件 | 赛前预测 | 半场预测 | 状态 |
|--------|------|----------|----------|------|
| 1. 数据划分策略 | `src/data/splitter.py` | ✅ | ✅ | 自动生效 |
| 2. 缺失值处理 | `src/features/prematch_features.py` | ✅ | ✅ | 自动生效 |
| 3. 模型参数优化 | `src/config/config.py` | ✅ | ✅ | 自动生效 |
| 4. 超参数调优 | `src/training/hyperparameter_tuning.py` | ⚠️ | ⚠️ | 需手动调用 |
| 5. 模型集成 | `src/training/ensemble.py` | ⚠️ | ⚠️ | 需手动调用 |
| 6. 评估指标 | `src/evaluation/metrics.py` | ✅ | ✅ | 自动生效 |
| 7. 类别不平衡 | `src/training/imbalance_handler.py` | ⚠️ | ⚠️ | 需手动调用 |

**图例**:
- ✅ 自动生效：无需修改代码，直接运行即可
- ⚠️ 需手动调用：需要修改训练脚本才能使用

---

## 兼容性检查

### ✅ 自动生效的改进（已兼容）

#### 1. 数据划分策略
```python
# src/data/splitter.py (已修改)
TRAIN_SEASONS = ["2015-2016", ..., "2020-2021"]  # 6个赛季
VAL_SEASONS = ["2021-2022", "2022-2023"]          # 2个赛季
TEST_SEASONS = ["2023-2024"]                       # 1个赛季
```
✅ **赛前和半场预测都使用相同的数据划分**

#### 2. 缺失值处理
```python
# src/features/prematch_features.py (已修改)
def handle_missing_values(X, strategy='smart'):
    # 智能分类处理策略
```
✅ **赛前预测**: `prematch_features.py` 调用 ✓
✅ **半场预测**: `halftime_features.py` 调用 ✓

#### 3. 模型参数优化
```python
# src/config/config.py (已修改)
MODEL_PARAMS = {
    'random_forest': {'max_depth': 10, ...},
    'xgboost': {'max_depth': 6, ...},
    ...
}
```
✅ **两者共用同一配置文件**

#### 4. 评估指标增强
```python
# src/evaluation/metrics.py (已修改)
def evaluate_classification(y_true, y_pred, y_proba):
    # 新增分类别指标和ROC-AUC
```
✅ **两者共用同一评估函数**

---

### ⚠️ 需要手动集成的改进

#### 1. 超参数调优
**当前状态**: 已实现但未集成到训练脚本

#### 2. 模型集成
**当前状态**: 已实现但未集成到训练脚本

#### 3. 类别不平衡处理
**当前状态**: 已实现但未集成到训练脚本

---

## 使用方法

### 方案A: 快速使用（使用默认改进）

**无需修改任何代码**，直接运行即可享受以下改进：
- ✅ 更合理的数据划分
- ✅ 智能缺失值处理
- ✅ 优化的模型参数
- ✅ 增强的评估指标（含ROC-AUC）

```bash
# 1. 赛前预测
python scripts/run_prematch.py --model xgboost --league England

# 2. 半场预测
python scripts/run_halftime.py --model xgboost --league England
```

---

### 方案B: 完整使用（包含高级功能）

需要修改训练脚本以使用超参数调优、模型集成等高级功能。

#### 步骤1: 创建增强版训练脚本

我为你创建两个新的训练脚本：

**1. 赛前预测增强版**
```python
# scripts/run_prematch_enhanced.py
```

**2. 半场预测增强版**
```python
# scripts/run_halftime_enhanced.py
```

这两个脚本包含：
- 超参数调优
- 模型集成（Voting/Stacking/RNN）
- 类别不平衡处理
- 完整的评估报告

---

## 完整训练流程

### 流程1: 基础训练（使用默认改进）

```bash
# 1. 数据准备（如果还没有feature_engineer6数据）
cd merged_data
python feature_engineer6.py

# 2. 赛前预测训练
python scripts/run_prematch.py --model xgboost --league England

# 3. 半场预测训练
python scripts/run_halftime.py --model xgboost --league England

# 4. 查看结果
# 报告保存在: outputs/reports/
```

**输出**:
- 混淆矩阵
- 分类别指标（Precision, Recall, F1）
- ROC曲线
- 每个类别的AUC

---

### 流程2: 高级训练（使用所有改进）

```bash
# 1. 使用超参数调优
python scripts/run_prematch_enhanced.py \
    --model xgboost \
    --league England \
    --tune \
    --tune-method grid

# 2. 使用模型集成
python scripts/run_prematch_enhanced.py \
    --model ensemble \
    --ensemble-type voting \
    --league England

# 3. 使用RNN模型
python scripts/run_prematch_enhanced.py \
    --model rnn \
    --rnn-type lstm \
    --league England

# 4. 完整流程（调优+集成+不平衡处理）
python scripts/run_prematch_enhanced.py \
    --model xgboost \
    --league England \
    --tune \
    --handle-imbalance \
    --optimize-threshold
```

---

## 详细使用示例

### 示例1: 使用智能缺失值处理

```python
# 已自动集成，无需修改
# 在 prematch_features.py 和 halftime_features.py 中自动调用

from src.features.prematch_features import handle_missing_values

# 自动使用智能策略
X = handle_missing_values(X, strategy='smart')
```

### 示例2: 使用超参数调优

```python
from src.training.hyperparameter_tuning import tune_hyperparameters_grid

# 网格搜索
best_model, best_params, le = tune_hyperparameters_grid(
    model_type='xgboost',
    X_train=X_train,
    y_train=y_train,
    cv=5,
    scoring='f1_macro'
)

# 随机搜索（更快）
from src.training.hyperparameter_tuning import tune_hyperparameters_random

best_model, best_params, le = tune_hyperparameters_random(
    model_type='xgboost',
    X_train=X_train,
    y_train=y_train,
    n_iter=50,
    cv=5
)
```

### 示例3: 使用模型集成

```python
from src.training.ensemble import create_voting_ensemble, create_stacking_ensemble

# Voting集成
voting_model = create_voting_ensemble(
    model_types=['random_forest', 'xgboost', 'logistic_regression'],
    voting='soft',
    weights=[2, 2, 1]  # 可选：设置权重
)
voting_model.train(X_train, y_train)
y_pred = voting_model.predict(X_test)
y_proba = voting_model.predict_proba(X_test)

# Stacking集成
stacking_model = create_stacking_ensemble(
    base_model_types=['random_forest', 'xgboost', 'svm'],
    meta_model_type='logistic_regression'
)
stacking_model.train(X_train, y_train, cv=5)
y_pred = stacking_model.predict(X_test)

# RNN模型
from src.training.ensemble import VanillaRNNClassifier, LSTMClassifier

rnn_model = VanillaRNNClassifier(
    rnn_units=64,
    dense_units=32,
    epochs=50
)
rnn_model.train(X_train, y_train, X_val, y_val, verbose=1)
y_pred = rnn_model.predict(X_test)
```

### 示例4: 处理类别不平衡

```python
from src.training.imbalance_handler import (
    recommend_imbalance_strategy,
    find_optimal_thresholds,
    adjust_decision_threshold
)

# 1. 获取推荐策略
recommendations = recommend_imbalance_strategy(y_train)

# 2. 训练时使用class_weight
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(
    class_weight='balanced',  # 根据推荐设置
    **other_params
)

# 3. 优化决策阈值
best_thresholds = find_optimal_thresholds(
    y_val, y_proba_val,
    metric='f1_macro'
)

# 4. 使用优化的阈值预测
y_pred_adjusted = adjust_decision_threshold(
    y_proba_test,
    thresholds=best_thresholds
)
```

### 示例5: 查看增强的评估指标

```python
from src.evaluation.metrics import evaluate_classification

# 自动包含所有新指标
metrics = evaluate_classification(y_test, y_pred, y_proba)

print(f"Accuracy: {metrics['accuracy']:.4f}")
print(f"F1-Macro: {metrics['f1_macro']:.4f}")
print(f"AUC-Macro: {metrics['auc_macro']:.4f}")

# 分类别指标
for label in ['H', 'D', 'A']:
    print(f"\nClass {label}:")
    print(f"  Precision: {metrics[f'precision_{label}']:.4f}")
    print(f"  Recall: {metrics[f'recall_{label}']:.4f}")
    print(f"  F1: {metrics[f'f1_{label}']:.4f}")
    print(f"  AUC: {metrics[f'auc_{label}']:.4f}")
```

---

## 常见问题

### Q1: 我需要重新训练所有模型吗？
**A**: 不需要。但建议重新训练以享受改进：
- 数据划分更合理（排除未完成赛季）
- 模型参数更优（特别是XGBoost）
- 缺失值处理更智能

### Q2: 如何选择使用哪些改进？
**A**: 建议顺序：
1. **必用**: 数据划分、缺失值处理、模型参数（自动生效）
2. **推荐**: 超参数调优（提升1-3%准确率）
3. **可选**: 模型集成（提升0.5-2%准确率，但训练慢）
4. **按需**: RNN模型（适合时序特征丰富的场景）

### Q3: 超参数调优需要多长时间？
**A**: 
- GridSearchCV: 30分钟 - 2小时（取决于参数网格大小）
- RandomizedSearchCV: 10-30分钟（推荐）

### Q4: 模型集成会提升多少性能？
**A**: 根据经验：
- Voting: +0.5-1.5% accuracy
- Stacking: +1-2% accuracy
- 但训练时间增加3-5倍

### Q5: 类别不平衡处理什么时候用？
**A**: 当你发现：
- 平局(D)的F1分数明显低于主胜(H)和客胜(A)
- 模型倾向于预测多数类
- 运行 `recommend_imbalance_strategy(y_train)` 查看建议

---

## 下一步建议

### 立即可做：
1. ✅ 运行 `feature_engineer6.py` 生成新特征
2. ✅ 使用默认改进重新训练模型
3. ✅ 对比新旧模型的性能

### 进阶优化：
4. ⚠️ 对最佳模型进行超参数调优
5. ⚠️ 尝试模型集成
6. ⚠️ 针对平局类别优化决策阈值

### 实验性：
7. 🔬 尝试RNN/LSTM模型
8. 🔬 分析特征重要性，进行特征选择
9. 🔬 使用SHAP解释模型预测

---

## 总结

**自动生效的改进（无需修改代码）**:
- ✅ 数据划分策略
- ✅ 智能缺失值处理
- ✅ 优化的模型参数
- ✅ 增强的评估指标

**需要手动集成的改进（需修改代码）**:
- ⚠️ 超参数调优
- ⚠️ 模型集成
- ⚠️ 类别不平衡处理

**推荐使用顺序**:
1. 先使用默认改进训练，建立baseline
2. 对比性能，确认改进有效
3. 根据需要逐步添加高级功能

所有改进都**完全兼容赛前预测和半场预测**！
