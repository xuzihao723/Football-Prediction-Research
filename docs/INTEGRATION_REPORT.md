# Monorepo 迁移验收报告

验收日期为 2026-11-22。目标为 `Football-Prediction-Research` 本地仓库，分支为 `refactor/monorepo-integration`。本阶段交付两个研究项目的可运行目录、三个隔离验证环境、冻结基线、基础测试和发布前文档。

## 目录与来源

~~~text
Football-Prediction-Research/
├── .github/workflows/research-checks.yml
├── projects/
│   ├── europe-big-five/
│   │   ├── 5-Top league Data（Original Version)/  历史 ETL
│   │   ├── FootBallPredictionOnBig5/              比赛级 PRE / HT
│   │   └── FootBallRealPre/
│   │       └── FootBallRealPre/FootBallRealTimePrediction/  事件级动量
│   └── gulf-market-anchored/
│       ├── src/  configs/  tools/  tests/
│       └── official_data/  results/             数据及结果主要本地保留
├── papers/README.md
├── scripts/verify_baseline.py
├── docs/
│   ├── provenance/
│   ├── research-overview.md
│   ├── paper-code-mapping.md
│   ├── reproducibility.md
│   ├── data-availability.md
│   └── INTEGRATION_REPORT.md
├── README.md
├── AGENTS.md
└── THIRD_PARTY_NOTICES.md
~~~

| 源快照 | 固定 SHA | 总数 | 进入本地 Git | Git 外本地保留 |
| --- | --- | ---: | ---: | ---: |
| [Europe](https://github.com/Corey30/FootBallPredictionOnBig5) | `b326af27c7df6ee802e50a2f5ab20ff5ec4ea69d` | 2,045 | 82 | 1,963 |
| [Gulf](https://github.com/Corey30/Market-AnchoredFootballForecastingOnAsia) | `e6a83c01547d4c8f9ed9b70d788ac5ba4f2147ff` | 267 | 67 | 200 |
| 合计 | | 2,312 | 149 | 2,163 |

[逐文件清单](provenance/migration-manifest.tsv) 列出上述每个文件的来源仓库、迁移前后路径、Git blob、字节数、源和目标 SHA-256、Git 归属与发布审查状态。[源记录](provenance/sources.json) 固定仓库地址与 SHA。2,303 个继承文件与源字节相同。独立复核将清单对照两份 `git ls-tree`，并核验 149 个继承的 Git 索引 blob。两份源工作树为 clean，HEAD 与 origin 各自保持原记录。

## 代码改动

以下九个继承文件均位于 `projects/europe-big-five/FootBallPredictionOnBig5/`。改动限于路径设置，Gulf 和欧洲事件级继承源码保持逐字节一致。

| 文件 | 迁移调整 |
| --- | --- |
| `data/analyze_data.py` | 分析输入与输出相对项目目录。 |
| `merged_data/feature_engineer1(assembly).py` | 读取迁入的西班牙原始表。两处原表的 SHA-256 一致。 |
| `merged_data/feature_engineer3(historical Performance and Team Strength).py` | 以脚本目录定位该阶段输入和输出。 |
| `merged_data/feature_engineer4(momentum).py` | 以脚本目录定位该阶段输入和输出。 |
| `merged_data/feature_engineer5(rest).py` | 以脚本目录定位该阶段输入和输出。 |
| `scripts/analyze_feature_importance.py` | 默认输入由项目配置解析。 |
| `scripts/tune_hyperparameters.py` | 默认输入指向迁入的特征目录。 |
| `src/momentum/config.py` | 原候选路径保留顺序，添加仓库现有 Parquet 路径。该文件内容与事件组件的归档 v3 Parquet 相同。 |
| `src/training/trainer.py` | 接收已有 CLI 传入的 `model_dir`，供保存和加载使用。已有调用的默认目录保持有效。 |

新增代码为 `scripts/verify_baseline.py` 和三个组件各自的 `tests/test_migration_basics.py`。测试工作流在 `.github/workflows/research-checks.yml`。继承的通用 `src` 命名和文件入口沿用独立目录与环境，保持脚本级运行约定。

## 环境与实际测试

本机使用 Windows x64 CPython 3.11.9。隔离环境在根目录的 `.venvs/{match,event,gulf}`，三个环境的包依赖检查均通过。所有原始 `requirements.txt` 原样保留，新增各组件的验证依赖与 3.11 约束。欧洲两套为 NumPy 2.1.3，Gulf 为 NumPy 2.2.6。比赛级和 Gulf 分别验证 XGBoost CPU 2.1.4 和 3.0.5，事件和 Gulf 验证 CatBoost 1.2.8。完整安装与 Windows/Linux 运行方式见 [Reproducibility](reproducibility.md)。

| 实际检查 | 结果 |
| --- | --- |
| 三组件 Syntax/compileall，加 116 个继承 Python 文件逐个编译 | 通过 |
| Europe 比赛级 `python -m pytest -q tests` | 6 通过，0 失败，0 跳过。含 CLI help、合成 schema、特征与时间划分、指标，以及 PRE/HT 各 36 行合成数据的 LR 调用和指定目录落盘 |
| Europe 事件级 `python -m pytest -q tests` | 2 通过，0 失败，0 跳过。事件 CLI 在临时目录验证 period 1 过滤和特征输出，并核对 60/20/20 排序 |
| Gulf 完整 `python -m pytest -q tests` | 75 通过，0 失败，0 跳过。scikit-learn 产生一条 `l1_ratio` 仅用于 elastic-net 的警告 |
| Gulf `python -m src.preflight` | 通过，输出 `READY_FOR_USER_AUDIT` |
| Gulf `python -m src.run_experiments --exp A --mode smoke` | 通过，开发分区产生 15 行预测，覆盖五个方法。结果在内存验证 |
| 从 Git 索引导出的无研究数据 Windows 检查副本，执行工作流同一批命令 | 三组件依次 6、2、40 项通过。无数据基线检查通过 |
| `python scripts/verify_baseline.py --include-local --source-root ..` | 2,312 项，0 差异 |
| Ubuntu WSL 的 `python3 scripts/verify_baseline.py` | 149 项跟踪文件与 2,163 项本地排除，0 差异。WSL 的预装解释器是 3.14.4，仅用于这个标准库检查 |

早期红测复现了欧洲调参的机器专用目录，以及比赛级入口传递 `model_dir` 时的构造错误。最终测试通过，同时明确指定子进程 UTF-8 以避免 Windows 本地编码差异。正式外测、全量训练、TensorFlow/GPU 分支、Gulf B–E 的命令行 smoke 和全套历史 ETL 重跑超出本阶段验证范围。GitHub Actions 没有在远端触发。工作流明确选择使用合成数据和已跟踪配置的测试，Gulf 完整套件的其余 35 项留给本地研究环境。窗口信息与执行约束见 [Reproducibility](reproducibility.md)。

## 冻结实验和数据检查

- [冻结基线](provenance/frozen-baseline.md) 绑定特征、标签、时间规则、模型设置、HPO、随机种子和评估。原预测、配置与研究结果均有文件哈希，并已复核副本字节。
- 本阶段使用 Gulf 定稿 Tied-Zero 与 Zero 的 2,789 个配对预测重新计算 C030 点估计。按 Tied-Zero 减 Zero，Accuracy 为 −0.014700609537，Log Loss 为 +0.010244322402，与冻结表一致。置信区间、Holm 结果和实验文件保持原值。
- Europe 修订稿的三折加独立 2024/25 与公开历史产物的协议差异仍待追溯。Gulf 各代结果和结构模型分支分别保留。 [论文与代码映射](paper-code-mapping.md)
- 本地排除资产的逻辑字节数为 2,179,318,006，包含已展开的 898,156,418 字节事件 LFS 对象。Git 索引中没有 CSV、Parquet、论文 PDF、训练权重、虚拟环境或 Gulf `results/` 文件。集成提交中最大 Git blob 为 1,067,681 字节的逐文件清单。
- 保留复制时的源许可说明和数据使用边界。历史 Windows 路径超过 260 字符，复制和验证使用长路径，本地仓库设定 `core.longpaths=true`。发布前须核对代码及数据授权。 [数据与许可审计](DATA_AND_LICENSE_AUDIT.md)

## Git 工作状态

- 本地集成提交 `f6b0741181a52eefcd504ebd23c1afefcf9b8e3a`，提交主题为 `Integrate preserved football research projects`。本报告记录集成提交，并通过同一分支的验收文档提交保存。
- 独立仓库未配置远程。未执行 Fork、PR、push 或源远程更改。没有已公开的发布包。Git 忽略本地环境及数据和结果。

## 待解决问题与后续顺序

1. 确认 Europe 论文版本对应的三折代码、最终季产物及事件试点的 374 个测试键。另立版本处理旧调参折、组内排名索引、动量 OOF 回填与预处理风险。[基线审计](BASELINE_AUDIT.md)
2. 核实 Gulf 定稿回放的缺失中间结果、SHAP 和绘图输入。先用冻结概率和分折清单验证对应关系。
3. 取得代码、赔率、数据集与预测的分发许可，补齐作者确认的 ICPRAM 名单及两份论文的最终书目信息。
4. 批准下一阶段后再处理科学修正、深度重构和全量重放。维持历史适配入口和新协议结果的独立归档。
