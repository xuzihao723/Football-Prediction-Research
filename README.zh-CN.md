# 足球预测研究

[English](README.md) · **简体中文**

[![研究项目检查](https://github.com/xuzihao723/Football-Prediction-Research/actions/workflows/research-checks.yml/badge.svg?branch=main)](https://github.com/xuzihao723/Football-Prediction-Research/actions/workflows/research-checks.yml)

## 项目简介

历史数据、市场概率和比赛进行中获得的信息，分别如何影响足球比赛预测？本仓库包含两个独立的科研项目，均预测全场胜平负，并分别维护 Python 环境与时序评估协议。

[科研项目](#科研项目) · [论文](#论文) · [快速开始](#快速开始) · [结果](#结果) · [可复现性](#可复现性) · [数据与引用](#数据与引用)

## 科研项目

| 项目 | 研究问题 | 实现 |
| --- | --- | --- |
| [欧洲五大联赛预测](projects/europe-big-five/README.md) | 比赛历史和上半场信息如何影响赛前及半场预测？ | 比赛级数据处理、训练和评估。独立的 StatsBomb 事件组件构建上半场动量特征，并运行 CatBoost 试点。 |
| [海湾联赛市场锚定预测](projects/gulf-market-anchored/README.md) | 如何用历史特征和半场比分更新赛前市场预测？ | 基于赔率的市场概率、市场锚定逻辑回归和结构化 XGBoost。对卡塔尔、沙特阿拉伯和阿联酋联赛进行五折扩展窗口时序评估。 |

## 论文

| 论文原题 | 期刊或会议与状态 |
| --- | --- |
| Time-Aware Football Outcome Prediction across the Top-Five European Leagues: Quantifying Pre-Match, Half-Time, and Event-Level Momentum Signals | **IJOCTA** · 稿件修订中 |
| Market-Anchored Football Forecasting: Source-Specific Regularization and Score-Based Half-Time Updating | **ICPRAM 2027** · 短文已录用 |

[论文信息](papers/README.md) · [论文与代码对应关系](docs/paper-code-mapping.md)

## 快速开始

克隆仓库后进入下表中的一个组件目录。使用 CPython 3.11，并为各组件创建独立环境。

~~~sh
git clone https://github.com/xuzihao723/Football-Prediction-Research.git
cd Football-Prediction-Research
~~~

| 组件目录 | 安装后的第一条检查命令 |
| --- | --- |
| `projects/europe-big-five/FootBallPredictionOnBig5` | `python scripts/run.py --help` |
| `projects/europe-big-five/FootBallRealPre/FootBallRealPre/FootBallRealTimePrediction` | `python build_momentum_features.py --help` |
| `projects/gulf-market-anchored` | `python -m src.run_experiments --help` |

**Linux / macOS**

~~~sh
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-validation.txt
~~~

**Windows PowerShell**

~~~powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-validation.txt
~~~

验证环境固定了 CPU 依赖版本。欧洲组件的验证测试使用合成数据。海湾完整测试套件还需要本地科研文件，可先运行下面的无数据测试子集。各平台要求和完整实验设置见复现指南。

~~~sh
# 在任一欧洲组件目录运行
python -m pytest -q tests

# 在海湾组件目录运行
python -m pytest -q tests/test_migration_basics.py tests/test_runner.py tests/test_temporal.py tests/test_temporal_regression.py
~~~

[复现指南](docs/reproducibility.md) · [CI 测试矩阵](.github/workflows/research-checks.yml)

## 结果

- **欧洲** — 优化后的特征构建程序为 **1,823 场比赛**生成与存档逐字节一致的上半场特征表。复现稿件最终预测表仍需要对应修订版的分折入口和逐场输出。
- **海湾** — 研究共包括 5,499 场比赛。在 **2,789 场配对的样本外比赛**中，Zero-LR 相比 Tied-Zero 的准确率高 **1.47 个百分点**，对数损失低 **0.01024**。这些回顾性估计由冻结预测重新计算得到。

[优化报告](docs/OPTIMIZATION_REPORT.md) · [冻结基线](docs/provenance/frozen-baseline.md)

## 可复现性

在仓库根目录运行 `python scripts/verify_baseline.py`，可对照迁移和优化记录核验 Git 跟踪文件。新实验的运行记录可保存输入和输出哈希、配置、Python 环境和 Git 提交。

`papers/` 记录论文信息，`docs/` 保存实验协议和验证证据，`scripts/` 提供完整性检查、性能基准和运行记录工具。

[复现说明](docs/reproducibility.md) · [研究概览](docs/research-overview.md)

## 数据与引用

完整复现所需的科研数据和逐场输出保存在本地研究副本。访问方式、许可和署名要求见数据指南。引用时请使用论文索引所载题名和出版信息，并为使用代码所得的结果注明对应 Git 提交。

[数据可用性](docs/data-availability.md) · [第三方许可与署名](THIRD_PARTY_NOTICES.md) · [引用信息](papers/README.md)
