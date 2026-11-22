# 原始实验基线审计

审计日期为 2026-11-22。范围是两个本地浅克隆的已跟踪快照、可定位的冻结产物，以及用于比对的本地稿件。检查了 Git 树、Python 语法树和关键源码，读取了 CSV、JSON 和稿件文本，并直接从部分冻结预测重算指标。仓库中的历史代码和数据作为审计证据。本轮阶段为分析与规划，未运行训练、实验入口或会拟合模型的测试。

| 来源 | 审计提交 | 快照规模 |
|---|---|---|
| 欧洲五大联赛，`source-europe` | [`b326af27c7df6ee802e50a2f5ab20ff5ec4ea69d`](https://github.com/Corey30/FootBallPredictionOnBig5/commit/b326af27c7df6ee802e50a2f5ab20ff5ec4ea69d) | 2,045 个跟踪文件，61 个 Python 文件 |
| 海湾联赛，`source-gulf` | [`e6a83c01547d4c8f9ed9b70d788ac5ba4f2147ff`](https://github.com/Corey30/Market-AnchoredFootballForecastingOnAsia/commit/e6a83c01547d4c8f9ed9b70d788ac5ba4f2147ff) | 267 个跟踪文件，55 个 Python 文件，其中 15 个测试文件 |

两个快照的 116 个 Python 文件均可用 Python AST 解析。该检查覆盖语法，运行时依赖、环境兼容性和模型复现仍需后续验证。

## 欧洲五大联赛

以下路径相对于欧洲仓库。用 `M` 指代 `FootBallPredictionOnBig5/`，用 `E` 指代 `FootBallRealPre/FootBallRealPre/FootBallRealTimePrediction/`。

| 目录 | 实际责任与调用关系 |
|---|---|
| `5-Top league Data（Original Version)/` | 十赛季联赛数据和历史 ETL 归档。外层 `merge_leagues.py`，内层 `merged_data/` 有组装、赔率、历史、动量、休息及旅行阶段和多个中间表版本。与 M 的 `merged_data/` 有同源脚本和字节相同的表。 |
| `M/merged_data/` | 组装脚本产生比赛级标签和列，赔率脚本接合球队别名、日期和 Football-Data 开盘/收盘赔率，历史脚本构造 Elo、积分排名、主客场历史、交锋和近期状态，后续脚本产生休息和旅行列。含按联赛的 `feature_engineer5/`、曲线特征、配对桥和事件表。 |
| `M/data/`、`M/src/data/` | `src/data/loader.py` 依赖项目级 `data/data_cleaning.py`，读取整张 CSV 并尝试多个编码。清洗过滤未完成比赛、已知泄漏列、无效时间/标签、高缺失列及重复记录。 |
| `M/src/{config,features,training,evaluation,utils}/` | 配置集中在 `src/config/config.py`，定义输入/输出、列、标签、参数和种子。任务脚本调用数据加载和拆分，再调用赛前/半场特征、选择与缺失补齐、训练或集成、指标和可视化。 |
| `M/scripts/` | `run.py` 支持联赛、赛前/半场、模型和输出参数。另有旧独立入口 `run_prematch.py`、`run_halftime.py`，以及调参、重要性分析、绘图、next-goal 和单赛季曲线比较。调参搜索分类器、采样和缩放，默认按 accuracy 重拟合。 |
| `M/src/momentum/` | 事件级公式、强度、因果前缀、曲线聚合和半场加载。与 E 的事件数据通过 Parquet、比赛 ID 桥和特征 CSV 对接。 |
| `E/` | `src/Fecth_Match.py`、`src/Fetch_Event.py` 调 StatsBomb，`src/form_table.py` 和 `src/patch.py` 构造/补丁事件状态。 `build_momentum_features.py` 读取所需列，筛 `period == 1`，按比赛插值并提取曲线特征。 `run_momentum_prediction.py` 对齐球队名称，保留配对样本，按联赛 60/20/20 划分，验证集选 CatBoost/Optuna 参数，然后合并训练和验证重拟。 |
| 根目录与结果 | 有 `all_results.json`、`search_spaces.json`，无已跟踪的 `M/outputs/` 或 E 的 `momentum_prediction_results/`。两个组件各有 requirements，E 外层和内层另有重复版本。未发现已跟踪 pytest 测试、包构建配置或 CI。 |

合并表 `M/merged_data/all_leagues_2015-2016.csv` 的文件名只写了一个赛季，实际包含 2015/16–2024/25 的 18,012 行、217 列。各联赛逐表共 18,012 行，西班牙表 203 列，其余四表 217 列。E 的五张 2015/16 比赛输入总计 1,826 行，动量特征和 M 中两个相同的桥文件各 1,823 行。稿件所报事件试点测试集 374 场仍需按正式 fixture 清单重建。

### 已定位的风险

| 级别 | 证据 | 影响与处理 |
|---|---|---|
| P0 | `M/scripts/tune_hyperparameters.py:258–366` 先清洗整体，再对训练赛季集合一次性补齐，最后把验证折编号传入 `PredefinedSplit`。后者以当前验证折的补集为训练集。 | 其他验证折和较晚赛季会进入当前内折训练，日志中列出的早期训练赛季并未被实际消费。补齐与高缺失列筛选也在内折之前使用了分布信息。训练折显式指定索引，预处理和采样进入各折 pipeline。历史 JSON 与脚本关系待核，风险只确认到该入口。 |
| P0 | `M/merged_data/feature_engineer3(historical Performance and Team Strength).py:647–708` 对 league/season 子表重置行号后，用 `row.name` 向外表写赛前排名和积分。 | 多组会覆盖外表相同的前部行，其他行可能保留默认 0。读写须以原行索引或 fixture key 对齐。先追查冻结面板来源，再单独产出修正结果。 |
| P0 | `M/scripts/run_halftime_curve_compare_1516.py:362–390` 用末折模型补早期未覆盖比赛。E 特征表的第一半场复合状态含比分和红牌，M 另有名为 prematch 的曲线比较入口。 | 补出的曲线带有后期拟合或样本内信息，不能标为严格时序 OOF。当前比赛的上半场曲线只能用于半场及以后。测试冷启动隔离、前缀不变性、跨比赛时序和 PRE/HT 列可用性。 |
| P1 | M 清洗约定屏蔽已知全场列，PRE/HT 特征选择仍依赖排除列表，主要入口在切分前清洗，高缺失列阈值为 0.6，切分后各集合分别 `process_*_features`。 | 新增或重命名列可能漏入，验证/测试批次均值、中位数会影响预测。历史中很多统计用 `shift(1)`，仍需验证同时间比赛、赛后状态更新时间和赔率抓取时点。按预测时点定义允许列并仅在训练集拟合。 |
| P1 | `M/src/data/splitter.py` 的固定季节拆分与无 season 时的按行比例拆分，`M/src/training/{trainer,ensemble}.py` 中的整数 CV 校准/堆叠，若干曲线入口逐切分 LabelEncoder。 | 固定季节名单之外的记录会被跳过，比例边界可能切开同时间组，整数 CV 缺少时间约束，缺类时概率列可能错位。保留历史行为，后续显式锁定概率列、fixture 级时间边界和缺类策略。 |
| P1 | `E/run_momentum_prediction.py:194–252` 用标准化主客队键，动量重复时保留首条。它在 train/val/test 分别剔除常量列，然后取交集。事件表旧验证摘要还记录了非单调时间以及各 8 行负的 away 近期射门/xG。 | 防止错配和使用留出集决定维度。确认原始事件 order、半场补时与重复时间事件的排序。把旧验证告警按源数据和已使用特征逐一核实。 |
| P2 | `scripts/tune_hyperparameters.py:76`、历史组装脚本含本机盘符，存在 `sys.path` 注入、导入时读数据的 `tune_england_pre_momentum.py`、多处 print 和宽泛异常处理，运行产物与源码相邻。 | 安装与导入易受 cwd 影响。冻结的配置与依赖未锁定，TensorFlow 仅某支路所需，RNN 输入形状为 `(样本,1,特征)` 且该实现未找到显式 TensorFlow seed。沉淀环境/线程和输出契约。有了产物一致性测试后，再优化读列和缓存。 |

## 海湾联赛

以下路径相对于海湾仓库。

| 目录 | 实际责任与调用关系 |
|---|---|
| `official_data/` | Qatar、Saudi、UAE 原始入口，加 `canonical/`、`recovery/`、`modeling/`。建模程序直接读取 harmonized CSV。两张规范表各有 5,499 个唯一 fixture key。规范元数据记录剔除 49 场取消赛、补录 115 场 date-only 比赛。 |
| `configs/` 与 `src/{config,feature_registry,build_feature_registry,preflight}.py` | 7 个 JSON 管理时间协议、赔率、列来源及可用性、特征分组、冻结特征选择和 HPO。五折的冻结历史表示均为 Elo + Standings。注册表检查列存在和组重叠。预检和完整/开发入口用途不同。 |
| `src/experiments/` 与 `src/run_experiments.py` | A–E 有统一 smoke/definitive 入口，smoke 用开发集伪外折。 `experiments/common.py` 贯通数据、注册表、拆分、HPO、输出。另有普通基线、锚定连续体和 PRE 线性补做路径。 |
| `src/{pooled_split,preprocessing,hpo}.py` | 2020/21–2024/25 逐季外测。内外层按实际时刻截断训练，结构模型同时间批量更新，某日存在 date-only 则该日整组处理。训练中位数与可选标准化，验证只 transform。内折以加权 Log Loss 调参，随机入口多采用协议 seed 42。 |
| `src/models/` | `market.py` 比例去水、以 Draw 为基准的 market logits 和联赛偏置再校准。 `structural.py` 做 Elo/Poisson/DC 等结构基线。 `anchoring.py` 实现 Zero、Free、Soft、Fixed、Tied 控制。 `xgb_v4.py` 有原生 margin、Fixed、Market-Context、KL 等。较早的历史、拼接、残差模型仍在。 |
| `src/{metrics,evaluation,outputs,statistics,tables}.py` | 概率与 fixture 对齐契约、Log Loss、Accuracy、multiclass Brier、Macro-F1、写预测和统计表。类顺序为 A/D/H。 `tools/analyze_xgb_v4.py` 生成分层指标、校准分箱、配对推断与 SHAP 摘要。 |
| `tools/`、`tests/`、`results/` | 结构 XGB 执行/验证、定稿 direct/sequential/robustness 执行和部分总表逻辑，15 个测试文件。旧 A–E、普通基线、XGB v4 与定稿 OOS 并存。README 提及的 `data_engineering/` 和 `docs/` 未在该 SHA 找到。 |

### 已定位的风险

| 级别 | 证据 | 影响与处理 |
|---|---|---|
| P0 | `tools/final_design_execution.py:196–224` 从 `results/anchor_continuum/canonical_prematch_oos.csv` 和 `results/final_canonical/prematch/tied_zero_lr_oos.csv` 复用结果，快照中的相关预测位于 `results/final_results/`，行标识仍需逐一核对。 `tools/analyze_xgb_v4.py` 还依赖缺失的逐场 SHAP 等输入。 | 当前 checkout 有指标和可读预测，但无法按原路径完整端到端重放。建立带生成者、输入哈希、method/representation 和主体人群的产物清单，核验等价再替换路径。将无训练复算与可选 SHAP、完整训练分开。 |
| P0 | `src/config.py` 的双条件锁由统一 runner 调用。 `tools/run_xgb_v4.py` 和 `tools/final_design_execution.py` 的直接执行入口未见同一锁调用。后者 `freeze()` 扫 `ROOT.parent.rglob('*')` 做哈希。 | 迁入 monorepo 后可能在长耗时工作中扫描同级项目和本地数据，并通过旁路启动正式评估。先限制扫描范围和输出路径，再保留所有相应运行入口的授权语义。 |
| P1 | `src/metrics.py:59` 默认概率行和误差阈值为 1e-8。冻结 B、XGB v4、七学习器预测分别有 13,077/44,624、10,340/39,046、4,355/39,046 行超过该阈值，最大约 1.46e-7。 | 本轮检查未发现这三份文件中的非有限值或越界概率，误差均小于 1e-6，符合常见 float32/CSV 舍入量级。保留原文件，在复算层记录阈值与受控归一化，单测拒绝真正的非法概率。 |
| P1 | `results/xgb_v4/reports/model_family_status.json` 记录 learnable-A 与要求所有行有历史 OOF 的 sequential 分支停止。定稿工具另外定义了丢开 warm-up 的 sequential 与 `direct_matched` 对照。 | 区分不同协议阶段；拒绝将 early-origin 回填标为合法 OOF，报告拟合覆盖与比较分母。外折是反复研究的历史样本；临场赔率时间和供应商身份仍不完整。 |
| P1 | 只在冻结 harmonized 表验证运行契约，公开树缺从来源/补录生成该表的完整 recipe。最终对比文件还保留以 Log Loss 为 primary 的字段，稿件把 Accuracy 定为首要最终评价。 | 需要数据血缘与已知时点约束。分别记录开发选择指标、旧快照的推断字段、最终报告层级。验证 pooled 与等折平均、Brier 公式、ECE 分箱、差值方向和 Holm 家族。 |
| P2 | 使用通用 `src` 命名，依赖仅为下限且缺 pytest 开发依赖。 `temporal_batches` 对每组重新扫描全键列表，重复整表/配置 IO，多套近似评估接口，测试含真实数据和模型拟合。 | 包命名隔离，分别锁运行和测试依赖。基准冻结后优化批处理和表读取。拆出可离线运行的纯协议测试与授权后运行的小拟合/真数据套件。 |

## Paper-to-Code Mapping

欧洲参考工作区已解包的 `tmp/IJOCTA_submission_check_20261121/main.tex`，SHA-256 `ec8fef2150a2e880c96312a1ada26062f44f74d4ba886eb9bca3f92b171c16b4`。海湾参考本机下载的 `ICPRAM_2027_76.pdf`，SHA-256 `5028e5eddcdc7c48245708f265ce37cef6a86146bd09e1b05c118adaabb3bcfe`，并用工作区 `output/icpram_format_20260929/manuscript_icpram.tex` 辅助定位。版本和投稿状态由作者确认。下表是所读稿件与两个 SHA 的章节、代码和产物对照，尚未完成逐表逐图的严格一致性和完整运行复现。

| 稿件内容 | 源码与产物 | 审计判断 |
|---|---|---|
| IJOCTA 多联赛 PRE/HT，历史、赔率、采样和多模型 | M 的 `merged_data/feature_engineer*`、`src/features/`、`src/training/`、`scripts/run*.py`、`scripts/tune_hyperparameters.py`、根 `all_results.json` 和 `search_spaces.json` | 研究主题和候选方法匹配。公开 runner 固定训练 2015/16–2020/21、验证 2021/22–2022/23、测试 2023/24–2024/25。稿件则是每联赛每任务三折，测试 2021/22、2022/23、2023/24，另有 2024/25 单独测试。应取得最终外折生成器、训练/验证/测试 fixture、选择/重拟日志和预测。 |
| IJOCTA 主实验结果和稳健性 | `all_results.json` 覆盖 5 联赛 × 2 任务，有三折、LR/XGB/RF/FNN/LGBM/CatB/VanillaRNN/Voting、参数、混淆矩阵和汇总。例如 England PRE LR 平均 accuracy 0.489035，fold 0 记录两个 `to-2022/to-2023` 测试标签和 760 行。 `search_spaces.json` 的 source 字段指向另一份用户 ZIP，明确未校对当前 HEAD。 | 冻结 JSON 作为历史基线保存。正式稿件每折单赛季、季后赛筛除、选优、无市场消融、校准、时序区块推断和分组 LOGO 的完整入口/逐场产物未定位，不能用 legacy 汇总证明论文数字。 |
| IJOCTA 2015/16 半场动量试点 | E 的 `build_momentum_features.py`、`run_momentum_prediction.py`、`tune_england_pre_momentum.py`。M 有 `src/momentum/`、`merged_data/momentum_merge.py` 与曲线比较、next-goal 脚本。 | E 的单季、第一半场复合特征和 CatBoost/Optuna 可对应稿件方法。独立定位源码中生成的结果目录未入 Git。M 的 next-goal 是另一个任务，不能充作全场 H/D/A 结果。正式事件消融、逐场预测和同一 374 试点测试集需确认。 |
| IJOCTA 图 | M 的 `scripts/{plot_momentum_curves,plot_tuning_summary,analyze_feature_importance}.py` 和 `src/evaluation/visualization.py` | 有曲线、性能和特征图工具。修订稿的流程图、数据分布图和原版动量图的确切生成命令、输入哈希仍待配对。 |
| ICPRAM 2027 数据、历史和市场基线 | `official_data/`、`pooled_split.py`、`models/{structural,market}.py`、`experiment_a.py`、`experiment_b.py`、`standard_baselines.py`。对应 `results/raw/`、`results/standard_prematch_baselines/`、`results/final_results/final_probability_diagnostics.csv` | 5,499 数据行、2,902 无市场 OOS、2,789 匹配市场 OOS，五折与分母匹配。Market PRE 储存 accuracy 0.55181068，Log Loss 0.95492335。未见可证明盘口在某个具体分钟获取的证据。 |
| ICPRAM 2027 线性锚定与非线性结构 | `models/anchoring.py`、`experiments/{anchor_continuum,prematch_lr_completion}.py`、`models/xgb_v4.py`、`tools/{run_xgb_v4,validate_xgb_v4,final_design_execution}.py`。产物有 `canonical_prematch_oos.csv`、`zero_oos.csv`、`tied_zero_lr_oos.csv`、XGB 分折 CSV/JSON 和 `xgb_metrics.csv`。 | 能定位四种线性机制与 Tied 对照，以及 Fixed、MC、KL 结果。冻结 C030 按 Tied−Zero：Accuracy −0.01470061，Holm p 0.01609839，Log Loss +0.01024432，与稿件以 Zero 改善方向表述一致。停掉的 v4 路径按状态文件标记。 |
| ICPRAM 2027 半场、校准、图表 | `tools/final_design_execution.py` 的比分三列和 Smart Voting 训练期 OOF，`src/tables.py`、`src/statistics.py`、`tools/{analyze_xgb_v4,summarize_final_design}.py`。冻结 `final_score_only_seven_learner_{predictions,metrics}.csv`、`final_probability_diagnostics.csv`、`final_paired_contrasts.csv`、`final_holm_family_registry.csv`。 | 七学习器同样本 PRE→HT accuracy 提升约 7.19–9.18 个百分点。最终 Accuracy 优先，HPO 以 Log Loss 选择。旧 A–E 包含其他路线，不能与定稿面板混用。图 1 架构、图 2 配对效应、图 3 PRE→HT 可定位数据依据，公开树未找到这些最终图的绘制脚本或最终表全套生成链。 |

### 无训练核算记录

- 海湾建模与规范表各有 5,499 条唯一键。读取 `experiment_a_oos.csv` 得到 2,902 个不同键；B、最终 PRE 线性和已验证 XGB OOS 都覆盖 2,789 个市场键。对 B、最终 PRE 线性和 XGB OOS，按 method/representation 检查键集合在各方法间相同，无重复 method/representation/fold/fixture 行。验证了这些文件各自内部的配对。
- 从定稿 PRE canonical Fixed-LR 的 2,789 行概率重算 accuracy 0.551093581929、Log Loss 0.955157328227、Brier 0.565337327968，与 `final_probability_diagnostics.csv` 对齐。从 XGB v4 HT Fixed 的 2,789 行得到 0.628899247042、0.820993019660、0.479490909762。使用按行类别概率及平方误差和直接聚合，未拟合参数。
- 用 Tied-Zero 与 Zero 定稿 PRE 预测的 2,789 个共同键和相同真实标签独立复算 C030 点估计，Accuracy 差 -0.014700609537、Log Loss 差 +0.010244322402，与冻结对照表相符。置信区间及 Holm p 来自冻结表。
- 迁移时分别追溯数据版本、代码版本和产物哈希。资源保存与分发规则见 [数据与许可审计](DATA_AND_LICENSE_AUDIT.md)，后续验收见 [迁移计划](MIGRATION_PLAN.md)。
