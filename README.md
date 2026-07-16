# 玄武质火山岩构造环境智能判别

项目题目：**基于多元素地球化学与可解释机器学习的玄武质火山岩构造环境判别**

英文题目：**Tectonic Discrimination of Basaltic Volcanic Rocks Using Multi-Element Geochemistry and Interpretable Machine Learning**

本仓库保存从原始数据、质量控制、特征工程、模型验证、地球化学解释到论文写作的可复现流程。

## 当前状态

- GEOROC 的 9 个 BASALT 原始分卷已按官方 MD5 固定：109,882 条原始记录；ARC–CIB–OIB 宽特征建模队列 54,012 条。
- GEOROC 随机森林 macro-F1 从随机分层的 0.861 降至文献分组的 0.786、保守引文重叠分组的 0.747和地区留出的 0.654，证明随机拆分明显乐观。
- PetDB 已按同一 `Whole Rock + Basalt + 全部变量/元数据` 合同取得 MORB、ARC、OIB、CIB 四类原始导出，并固定查询、哈希和包内条款。
- EarthChem 曾为同时提交的 OIB/CIB 任务生成同一 S3 键，导致覆盖且 OIB CSV 缺表头。异常包被保留但排除；OIB 和 CIB 已分别重新导出并通过哈希、表头、行数及行级链接检查。
- PetDB 四类经完全重复删除、FeO 当量统一和 Sample URL 中位数聚合后共有 13,718 个样品；高置信构造标签 13,566 个。宽特征可建模 6,148 个：MORB 3,650、OIB 1,281、ARC 754、CIB 463。
- OIB 导出中 `Th` 整列缺失，因此四分类主验证排除这一结构性缺列，仅使用在四类中均至少有观测的 21 个宽特征；22 特征结果只作偏倚敏感性对照。
- 规范化并排序地理描述后，PetDB 四分类随机森林在保守引文重叠分组下 macro-F1 为 0.760、平衡准确率 0.755；地区留出为 0.648/0.650，随机分层为 0.915/0.926。CIB 地区留出召回率仅 0.127，是当前最主要的泛化短板。
- 缺失模式本身仍携带类别信号：全部 22 特征的保守引文分组平衡准确率为 0.535；排除结构性缺列后降至 0.417。论文必须并列报告该诊断，不能把随机拆分分数当作地质普适性。
- 当前四分类已打破“PetDB=MORB”的完全来源–类别对应，可作为 PetDB 内部四类验证；共有三类的双向跨库诊断和一组统一四类外部转移均已完成，但每类仍需多个独立省域才能估计全球泛化。
- 共有 ARC/CIB/OIB 的跨库转移诊断已完成：宽义 CIB 下，GEOROC→PetDB 随机森林 macro-F1 为 0.804，而 PetDB→GEOROC 为 0.484。将 GEOROC CIB 限定为裂谷火山岩后，反向 macro-F1 提高到 0.619，但正向为 0.788，说明训练覆盖与标签本体共同控制迁移表现。
- 逐样品误判审计发现东非裂谷是 CIB 地区失稳的主体。347 个数据库标注 CIB 中，地区留出时 304 个被判为 OIB。将整个东非裂谷从训练中删除后，随机森林仍给出平均 OIB 概率 0.609、CIB 概率 0.075，OIB/CIB 过渡指数 0.886（文献组聚类 95% 区间 0.870–0.899）。论文将其解释为裂谷—地幔柱—陆洋转换的地球化学连续体，而非直接重标。
- 已审计三个 PANGAEA MORB 候选源；对应论文已进入 PetDB 的 Reykjanes Ridge 数据被排除。其余两个 CC BY 数据集得到 18 个主量完整、未检出样品名或论文重叠的 MORB 玻璃样品。PetDB 10 主量随机森林在该试验性外测上的 MORB 召回率为 0.944；这不是四分类外测 macro-F1，仍需扩充地区、介质与样本量。
- 已从 Figshare `10.6084/m9.figshare.25295671.v1` 固定 44 个 MORB 全岩样品及工作簿/API 元数据；8 个东太平洋海隆样品明确转载自 Li et al. (2019)，仅作敏感性分析。其余 36 个西南印度洋脊样品无 PetDB 规范化样品名重叠，暂列“仓储原创、论文归属待确认”外测队列。18 元素随机森林和逻辑回归的 MORB 召回率分别为 1.000 和 0.944，平均 MORB 概率为 0.918 和 0.867；因为仍是单研究、单类别，不能作为完整外部四分类性能。
- 已取得并审计出版独立的 ARC、CIB 与 OIB 队列：新赫布里底弧前 12 个、Rio Grande Rift/Jemez Lineament 28 个、Mauna Loa 2022 喷发 16 个；论文 DOI、样品名、材料、SiO₂、主量总和、Fe 统一和 22 元素覆盖均已逐项检查。Vate Trough 的 21 个初始弧后裂谷样品单列为不计分过渡队列。
- 统一 18 元素四类外测共 92 个样品。随机森林的准确率、平衡准确率和 macro-F1 为 0.837、0.866 和 0.807；ARC、MORB、OIB 召回率均为 1.000，CIB 为 0.464。CIB 错误主要指向 ARC，符合 Rio Grande Rift 对古 Farallon 浅俯冲改造和大陆岩石圈作用的继承；Vate Trough 则有 0.762 被判为 MORB。每类只有一个研究/省域代理，因此不报告伪精确的逐样品自助法区间。
- 四分类解释性分析使用保守引文重叠分组的折外置换重要度，并辅以全队列 SHAP 描述。主要组合为 TiO₂、Sr、K₂O、Nb 和 Y；TiO₂ 与 Sr 的折间波动较大，解释必须视为关联而非因果。
- 保守引文重叠分组随机森林的多分类 Brier 分数为 0.236，top-label ECE 为 0.075；折外概率已绘制可靠性曲线，但尚未进行后校准。

## 目录

- `data/raw/`：不可修改的来源文件与官方元数据
- `data/processed/`：版本化样品级建模数据
- `src/`：数据处理、建模与验证程序
- `notebooks/`：可执行、可审阅的分析记录
- `reports/`：质量报告、结果表和阶段总结
- `paper/`：论文正文与参考文献
- `references/`：来源、许可和贡献文献台账
- `config/`：标签、导出和分析参数

## 主要复现入口

```powershell
python src/profile_georoc.py
python src/build_model_dataset.py
python src/run_baseline_models.py
python src/run_feature_bias_sensitivity.py
python src/run_interpretability.py
python src/profile_petdb_morb.py
python src/build_petdb_morb_dataset.py
python src/run_petdb_source_confounding_diagnostic.py
python src/build_petdb_primary4_dataset.py
python src/run_petdb_primary4_validation.py
python src/run_petdb_primary4_missingness_diagnostic.py
python src/run_cross_database_validation.py
D:\AI地学\.venv\Scripts\python.exe src/run_petdb_primary4_interpretability.py
D:\AI地学\.venv\Scripts\python.exe src/build_petdb_primary4_figures.py
D:\AI地学\.venv\Scripts\python.exe src/audit_pangaea_morb_candidates.py
D:\AI地学\.venv\Scripts\python.exe src/run_pangaea_morb_external_validation.py
D:\AI地学\.venv\Scripts\python.exe src/build_pangaea_morb_external_figure.py
python src/run_petdb_primary4_error_audit.py
python src/run_east_african_rift_geochemical_audit.py
python src/run_rift_transition_stress_test.py
python src/build_petdb_error_audit_figure.py
python src/build_rift_transition_figure.py
python src/audit_figshare_morb_candidate.py
python src/run_figshare_morb_external_validation.py
python src/build_figshare_morb_external_figure.py
python src/audit_arc_cib_external_candidates.py
python src/audit_mauna_loa_oib_external.py
python src/run_arc_cib_external_validation.py
python src/run_multiclass_external_validation.py
python src/build_multiclass_external_figure.py
python src/validate_manuscript_claims.py
python src/test_data_contract.py
```

主要数据与结果：

- `data/processed/petdb_primary4_v0_1.parquet`
- `reports/data_quality/petdb_primary4_processing_profile.json`
- `reports/modeling/petdb_primary4_overall_metrics.csv`
- `reports/modeling/petdb_primary4_missingness_summary.csv`
- `reports/modeling/cross_database_overall_metrics.csv`
- `reports/modeling/petdb_primary4_permutation_importance_summary.csv`
- `reports/modeling/petdb_primary4_calibration_summary.csv`
- `reports/data_quality/pangaea_morb_candidate_audit.csv`
- `reports/modeling/pangaea_morb_external_summary.csv`
- `reports/modeling/petdb_primary4_error_audit_samples.csv.gz`
- `reports/modeling/east_african_rift_stress_summary.csv`
- `reports/data_quality/figshare_25295671_morb_candidate_audit.csv`
- `reports/modeling/figshare_morb_external_summary.csv`
- `reports/data_quality/arc_cib_external_candidate_audit.csv`
- `reports/data_quality/mauna_loa_oib_external_audit.json`
- `reports/modeling/multiclass_external_metrics.csv`
- `reports/modeling/multiclass_external_per_class.csv`
- `reports/modeling/multiclass_external_backarc_summary.csv`
- `figures/petdb_primary4_validation_performance.pdf`
- `figures/petdb_primary4_grouped_confusion.pdf`
- `figures/petdb_primary4_calibration.pdf`
- `figures/cross_database_transfer.pdf`
- `figures/pangaea_morb_external_probabilities.pdf`
- `figures/petdb_primary4_error_audit.pdf`
- `figures/east_african_rift_transition.pdf`
- `figures/figshare_morb_external_validation.pdf`
- `figures/multiclass_external_validation.pdf`
- `reports/data_quality_and_baseline_validation.md`
- `paper/manuscript_draft.md`

`reports/phase1_validation_report.html` 是早期 GEOROC 三分类阶段快照，不代表当前 PetDB 四分类结果。
