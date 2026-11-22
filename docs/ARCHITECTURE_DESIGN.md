# 科研 Monorepo 架构设计

仓库目标为独立的 `Football-Prediction-Research`。当前交付是设计，目录和命令描述后续迁移目标。研究边界是欧洲比赛级、欧洲赛事级动量和海湾市场锚定。仓库级共用能力包括文档、来源清单和 CI，各研究组件保有自己的模型、配置、环境、数据血缘与指标口径。

## 目标目录

```text
Football-Prediction-Research/
├── .github/workflows/
│   ├── europe-match.yml
│   ├── europe-event.yml
│   └── gulf.yml
├── projects/
│   ├── europe-big-five/
│   │   ├── README.md
│   │   ├── FootBallPredictionOnBig5/
│   │   │   ├── pyproject.toml
│   │   │   ├── constraints/
│   │   │   ├── src/europe_big_five/
│   │   │   ├── configs/
│   │   │   ├── tests/
│   │   │   └── results/frozen/
│   │   └── FootBallRealPre/
│   │       ├── pyproject.toml
│   │       ├── constraints/
│   │       ├── src/europe_event_momentum/
│   │       ├── configs/
│   │       ├── tests/
│   │       └── results/frozen/
│   └── gulf-market-anchored/
│       ├── pyproject.toml
│       ├── constraints/
│       ├── src/gulf_market_anchored/
│       ├── configs/
│       ├── tests/
│       ├── official_data/README.md
│       ├── results/frozen/
│       └── README.md
├── papers/README.md
├── docs/
│   ├── BASELINE_AUDIT.md
│   ├── MIGRATION_PLAN.md
│   ├── ARCHITECTURE_DESIGN.md
│   ├── DATA_AND_LICENSE_AUDIT.md
│   ├── research-overview.md
│   ├── paper-code-mapping.md
│   ├── reproducibility.md
│   ├── data-availability.md
│   └── provenance/
├── scripts/
├── README.md
├── AGENTS.md
├── .gitignore
└── THIRD_PARTY_NOTICES.md
```

三个独立的 `pyproject.toml` 提供安装、包发现、入口和测试依赖。以 Python 3.11 作为待验证的公共起点，按各组件的实际兼容矩阵锁 Python 小版本、库和平台约束。比赛级的 TensorFlow 与事件级的 StatsBomb 下载依赖单列为可选项。根目录不要求安装所有 ML 后端，共同的工作流涉及语法、测试、源/依赖清单和打包检查。欧洲保留两个组件目录名，事件包内部去除重复的三层目录，路径映射保留在 provenance。Gulf 的 `tools/` 和实验子模块先保持职责边界，再整理为各自包内 CLI。

## 组件接口和数据流

| 组件 | 运行输入 | 独立运行和交接 |
|---|---|---|
| 欧洲比赛级 | 经授权的联赛比赛级 CSV 或验证过的等价 Parquet，明确市场时点和 PRE/HT 允许特征。 | 数据校验 → 按联赛和任务定义时间折 → 训练折 pipeline → 逐场预测 → 指标和图。先支持无事件输入的主入口。需要动量时，通过显式配置引用 E 的有版本产物。 |
| 欧洲事件级 | StatsBomb 事件及比赛索引、半场分析使用的数据表和 fixture 映射。 | ingest → event state → period 1 曲线 → fixture 级动量 → CAT/Optuna 试点和按需导出。公式曲线、预测型曲线、next-goal 的标签、协议和产物分开登记。 |
| 海湾市场锚定 | 规范/建模 CSV、可用性注册表、冻结外折/HPO 配置、按分析阶段选用的 OOS 预测。 | canonical/建模输入验证 → actual-time split → 市场 prior/历史 → LR 或 XGB → 配对评估和校准。对 A–E、v4、定稿 direct/score-only 使用不同的协议 ID 和复算入口。 |

最初的可运行迁移保留各入口签名和基线设置，在项目自己的环境可从任意 cwd 调用，逐步将 `src.*` 和运行脚本注入改为具名包。导入只加载定义。各入口明确传递 `--config`、`--data-root`、`--output-dir` 和 `--seed`，默认路径锚定项目配置或显式运行目录。输入不存在、schema 错、键重复、协议不支持和策略性缺失使用可诊断错误。训练、外部采集和输出文件只由入口触发。正式海湾外测保持现有双条件语义，覆盖有同类行为的 tools 入口。

事件交接使用 fixture 级表和 schema 版本，记录 provider ID、标准 fixture ID、league/season/time、period、最终纳入事件时间和特征版本。胜平负预测输出使用明确的 `p_A,p_D,p_H` 或带标签的列映射。保留每个原始产物的列名和顺序，转换视图显式记录来源与规则。各组件自行定义 Brier 求和、ECE 分箱、时间分组等计算细节；公共表结构采用同名字段时附上指标定义。预测行的自然键至少包含 experiment、origin、fold、method、representation、fixture key。

## 基线、发布和可维护性

- `results/frozen/` 用于获许可、去重过的原始冻结输入和结果；`runs/<run-id>/` 或用户指定目录用于复算/训练产物。冻结文件只读消费。新运行用独立目录、临时文件后原子落盘、完成标记及清单，记录命令、环境、候选空间、种子、线程、操作系统、input/output SHA-256、类序与时间窗口。
- provenance 记录仓库 URL、完整 SHA、源路径到目标路径、许可证/授权记录、原生成入口和版本关联。新结果区分 `legacy`、`paper-frozen`、`corrected` 协议。历史摘要可被引用；若数据或入口缺失，复现状态记录为只可复算、需授权输入、需补运行记录或已重放。
- 代码来源清单支持保留作者、归属和必要 notices。论文页链接已获准公开的题录、状态和永久链接，逐表逐图映射来自作者确认。选定数据/模型/预测按授权清单发布。
- 每个项目 README 提供安装、无需数据的自测、基于授权数据的复算/训练命令、输出解释和来源。根 README 以研究问题、路线、论文、快速入口和数据可用性为主。根 AGENTS 记录源仓库只读及各项目测试入口。
- 日常 CI 按路径触发并在隔离环境运行包构建、导入、静态检查、合成数据测试和许可允许的小冻结产物复算。无外网数据抓取。完整复现作为单独显式本地流程。依赖和数据改变时，先用准确率、预测数、类序及分折级差异定位，再决定是否更新基线。
- 大事件表优先测试 Parquet 列投影、按 match/season 分区和逐块处理。Gulf 批分组优先测试一次分桶替代重复扫描，预处理器和缓存以协议、折、列、输入哈希为键。先做数据与结果差分，再比较峰值内存和运行时间。

## 选型依据

保留独立包便于各自接受第三方依赖和原实验协议，事件产物靠表契约互通。三个源码树直接复制进同一 Python import 根会保留原 `src` 命名与 cwd 耦合，适合临时归档。抽取通用建模库会增加类序、分割和指标语义的迁移成本，暂将共享限定于文档和发布工具。

开展顺序与验收见 [迁移计划](MIGRATION_PLAN.md)。现有源码与稿件差异见 [基线审计](BASELINE_AUDIT.md)。
