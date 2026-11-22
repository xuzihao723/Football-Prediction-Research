# 数据、体积与许可审计

审计边界是两个固定 SHA 的 Git 跟踪树、Europe Git LFS 工作树对象及可访问的公开许可声明。下文区分 Git 树 blob 字节数、克隆包大小和工作树逻辑大小。体积采用十进制 MB；列有 MiB 时按 1,048,576 字节计算。许可表登记本轮查到的声明与发布所需证据。

## 当前快照与体积

| 口径 | Europe | Gulf |
|---|---:|---:|
| 跟踪文件 | 2,045 | 267 |
| HEAD blob 逻辑总字节 | 1,177,239,792 | 103,583,870 |
| LFS | 一个有效的工作树对象，898,156,418 字节，Git 树仅跟踪指针 | 未发现 |
| 本地浅克隆 pack | 605.68 MiB | 21.16 MiB |
| 数据占比 | 1,825 个 Parquet 共 853,679,093 字节；135 个 CSV 的 Git blob 共 322,127,609 字节（含 LFS 指针） | 114 个 CSV 共 82,110,486 字节；89 个 JSON 共 21,175,426 字节（包括配置、诊断和 HPO 轨迹） |
| 字节相同的非空 blob | 27 组，额外逻辑内容 87,377,638 字节 | 0 组 |

Europe 工作树按已跟踪逻辑内容约 2.075 GB。最大文件是事件状态 CSV 的 LFS 对象；Git 普通 blob 最大为 `FootBallPredictionOnBig5/merged_data/all_leagues_2015-2016.csv`（31.57 MB）。两个状态 Parquet 各 26.33 MB 且是同一 Git blob，另有原始与运行目录间重复表。卷、磁盘压缩、Git 对象自动去重与输出目录会改变实际占用。Gulf 最大的旧预测 `results/raw/experiment_e_oos.csv` 12.83 MB，建模输入 9.38 MB，`results/xgb_v4/reports/fold_diagnostics.json` 7.18 MB，七学习器逐场预测 7.10 MB。两仓快照的普通 blob 都在 50 MiB 以下；LFS 大对象应以指针、经授权资产或可重建流程管理。 [GitHub 文件体积说明](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github)

Europe 与 Gulf 均是浅克隆，各自本地只有一个可遍历提交。审计日 GitHub API 的公开 `main` 历史响应为 Europe 三个提交（HEAD，`12027bd234a92a5fcc1fa36e4db4b007402b46a4`、`390d084c224591ba18af8349a8abe392ae6a3e46`），Gulf 一个。HEAD 分别是 [Europe](https://github.com/Corey30/FootBallPredictionOnBig5/commit/b326af27c7df6ee802e50a2f5ab20ff5ec4ea69d) 和 [Gulf](https://github.com/Corey30/Market-AnchoredFootballForecastingOnAsia/commit/e6a83c01547d4c8f9ed9b70d788ac5ba4f2147ff)。早期提交的完整 blob、删除对象、其他 refs 或私有历史不在本轮扫描范围。新仓通过逐项准入导入并保留来源 SHA，以独立历史发布。

## 迁移与公开分发准入

| 类别 | 快照中的资料 | 建议处置与理由 |
|---|---|---|
| 可迁移代码候选 | Europe 两套运行组件和必要 ETL，Gulf `src/`、`tools/`、`tests/`、`configs/`。 | 审核代码归属和授权后按 manifest 迁入。保留许可、原作者和依赖归属。重复 ETL 先比较配置和输出后选入口，原路径和文件指纹记录在来源表。 |
| 公共仓库优先候选 | 研究协议、列契约、能自主授权的合成 fixture、聚合表与论文允许公开的图/题录。 | 先验证无逐场可逆标识或受限赔率，确认作者及提供方权益再发布。样例无需第三方比赛数据。两快照本轮没有资料可仅因公开可下载而登记为无条件再分发。 |
| 需要进一步核实授权 | Europe 原始比赛统计、各阶段历史特征表、开盘/收盘补充、动量特征、bridge 和逐事件数据；Gulf 原始、恢复、canonical、harmonized 和逐场 OOS/分折诊断。 | 按数据提供方和派生类型分开审查。记录获取方式、原始数据条款、适用版本、能否公开派生物、归属方式。模型预测也含键、联赛、赛季、真实结果，诊断 JSON 还可能列出训练与测试键。 |
| 本地复现输入 | 在权限厘清前保留 Europe 的联赛表、事件匹配、LFS/Parquet 和 Gulf `official_data/` 的三层表及恢复档。 | 使用项目外或本地忽略路径。公共仓库可提供不含数据内容的 schema、输入名称和校验值，以及经许可的获取步骤。数据原件及旧归档保存在现有位置。 |
| 生成、诊断与缓存 | 欧洲 `outputs/models`、事件预测输出和传统临时阶段表；Gulf 临时 SHAP、cache、诊断、`results/standard_prematch_baselines/tests.xml`。 | 进运行目录或受控档案。XML 可作历史验证记录；无需要时可排除在公共运行包之外。冻结诊断和 HPO 表确认复现价值后优先保留，逐场键按同一授权门槛。 |
| 权重和二进制 | 当前跟踪树未发现常见 joblib、pickle、UBJ、Keras 权重，Europe 的 Parquet 与 LFS 事件 CSV 占主导；Gulf 的大 JSON 多为分折诊断、HPO 或配置。 | 保留可许可的模型配方、参数、种子和产物哈希。将来确有权重发布需求时核对模型来源、许可证、信息披露和加载格式。 |

### 保留顺序与哈希锚点

优先原样保留已核对且可分发的配置、预测、统计摘要与来源链。缺少授权的资料留在受控归档，公共文档仅引用合规的元数据。下表列的是本轮验证过的内容锚点，发布前另生成覆盖全部准入资料的清单。

| 资产 | SHA-256 | 用途 |
|---|---|---|
| Europe `all_results.json` | `e10586ac3fcba8a68b72545430d4e6fb2e6ebbd09bb7d86e279704c43b6b07e8` | 十个联赛/任务组合的历史汇总。 |
| Europe `search_spaces.json` | `021a1dd678a2b7df83ae308791a3b4adcba66a0f15128e9d199564d9361faa55` | 历史搜索空间，另有来源 ZIP 的说明。 |
| Europe LFS `real_time_data/event_state_table_big5_2015_16.csv` | `1e28a020f0b6eba0c9e1905de44cec44813b2c4bcf1147fde699a61a488a12e3` | 事件状态，可在许可允许时优先选择经 schema、行键和指标回归核验的压缩表示。 |
| Gulf `configs/protocol_final.json` | `2b76d1e4d625ed4e090eadba3a0f45403f46723d4bbd2327b9f0078a0f1d390e` | 五折、类别与种子。 |
| Gulf `official_data/modeling/ModelingData_harmonized.csv` | `641171ab19cecc89a9afa241d5cde121369fdc2dcfa5bd52f46491e37f16db0a` | 5,499 场完整运行快照。 |
| Gulf `results/final_results/canonical_prematch_oos.csv` | `81b90022863c0858dd56f314d66700e72c6798bdc2dce17e49bfa4086f2534a3` | 匹配 PRE，三组方法各 2,789 场。 |
| Gulf `results/final_results/final_probability_diagnostics.csv` | `e4e9a77b36967afb92d51fb590cf652bcd0c1b76bee4511e2a04a62247602eb5` | 56 行最终指标与分箱校准汇总。 |
| Gulf `results/xgb_v4/reports/validated_oos_predictions.csv` | `7dc6228fa462ff186b60c274234c5db5341e199f7e1521f7b544ea5c88eb5319` | 结构 XGB v4 及参照的逐场复算。 |

Europe 还应保留作者确认的外折/最终季和事件试点输入清单、参数、逐场预测和原图生成数据。Gulf 同时保留 `final_paired_contrasts.csv`、`final_holm_family_registry.csv`、`final_score_only_seven_learner_{metrics,predictions}.csv`、`final_league_specific_results.csv`、Tied/Zero/SmartVoting OOS、必要的校准 bins、v4 冻结协议/状态/分折元数据、普通基线与有复现价值的 A–E OOS。配对复算需要共享同一标识和分折，去标识化时需验证键可连接且授权覆盖。受控结果与公开结果都记录每个输入数据版本、哈希、运行路径、授权和数据血缘。

## 许可与来源证据

| 资料 | 证据 | 发布时要核对的项目 |
|---|---|---|
| Europe 原创代码与文档 | HEAD 无独立软件 license，README 也声明尚未选择。 | 版权方书面允许复制、修改和发布的范围，以及仓库采用的许可证。 [GitHub 仓库许可说明](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/licensing-a-repository) |
| Gulf 原创代码及派生成果 | HEAD 的 `LICENSE` 是自定义学术研究/署名文本，保留权利并要求原始数据许可允许再分发。 | 由权利人解释独立公开代码及结果的许可，原许可保留原文。 [Gulf 固定版本许可](https://github.com/Corey30/Market-AnchoredFootballForecastingOnAsia/blob/e6a83c01547d4c8f9ed9b70d788ac5ba4f2147ff/LICENSE) |
| FootyStats | 两份稿件的数据源，原始表和模型特征可能包含其统计、历史与赔率。官网站点与 API 条款应按实际获取类型核对。 | 获取时适用的条款、覆盖赛季、是否有研究许可及能否发原表、衍生行、预测和图。 [站点条款](https://footystats.org/terms-and-conditions)、[API 数据条款](https://footystats.org/api/documentations/terms-of-use-and-legal) |
| Football-Data.co.uk | Europe `feature_enginner2(fetch_odd).py` 中有下载来源、开盘/收盘提取和匹配。 | 采集方式、在论文实际用到的字段、历史研究使用条件、原始及拼接后逐场行的再分发。 [官方使用说明](https://football-data.co.uk/contact.php) |
| StatsBomb Open Data | Europe E 的 `statsbombpy` 获取流程、Parquet 事件、状态表和曲线。 | 固定 provider 版本及比赛清单，验证研究用途、数据/衍生表发布范围及署名和标识要求。 [官方仓库](https://github.com/hudl/open-data)、[官方用户协议](https://github.com/hudl/open-data/blob/master/LICENSE.pdf) |
| Gulf 补录与赔率 | `canonical_metadata.json` 记录 115 场补录和 date-only；现有公开资料没有完整逐列来源和赔率时点。 | 恢复行链接与来源条款，以作者记录补齐；注明可支持的数据时间声称。 |
| 依赖、稿件和图 | 两仓 imports/requirements 列出 NumPy、pandas、scikit-learn、XGBoost，以及按需的 LightGBM、CatBoost、TensorFlow、Optuna、statsbombpy 等。 | 固定版本后记录实际依赖和许可证，发布自行移植或打包的第三方代码时保留通知。确认作者认可的稿件公开版本、机构/会议权利及图中素材。 |

## 敏感信息与体积控制

- 对快照中的常见文本后缀做了秘密模式筛查，仅输出路径，并查找 `.env`、私钥及 credential 类文件名。唯一文本命中是 Gulf 的 `tools/final_design_execution.py:104` 缓存键变量 `token`，用于本地缓存文件名。此轮没有确认的 API key 或私钥，扫描未覆盖完整历史、二进制内容或无明显格式的值。
- 迁移前对拟导出文件和目标索引做专门 secret scan，对命中使用仅路径和类型的报告。数据清单另查可识别的用户、设施或路径信息。保留 source 的历史与本地数据，审查范围以拟公开内容为准。
- 发布使用 manifest 白名单、按组件粒度的忽略和体积检查。将重复表通过源路径＋哈希引用一次。更新大型研究数据时记录访问规则和校验值，以授权允许的数据获取脚本、私有存储或带许可的归档连接复现入口。
