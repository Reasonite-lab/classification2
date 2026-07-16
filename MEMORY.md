# Codex 项目继承记忆

> **使用方法：** 后续任务开始时，先完整读取本文件，再读取 `PROJECT_STATUS.md` 和 `FINAL_HANDOFF.md`。不要重新下载已冻结数据，不要重跑已通过的实验，除非手稿数值或数据合同发生变化。

## 1. 项目目标与科学边界

- 中文项目：**基于多元素地球化学与可解释机器学习的玄武质火山岩构造环境判别**。
- 研究对象固定为全岩玄武质火山岩，不扩展到所有火成岩。
- 目标是达到当代地学前沿水平，允许适度复杂算法，但不把论文做成计算机方法论论文。
- 已决定的主线：机器学习的价值不是追求随机拆分高分，而是检验多元素地球化学信号能否跨文献、省域和数据库迁移，并将系统误判解释为地幔源区、俯冲交代、地壳混染、岩浆分异和构造过渡的记录。
- 论文核心结论：玄武岩分类器能识别端元构造环境，但在大陆裂谷和弧后过渡环境中，继承的俯冲或地幔柱信号可使岩石化学与现今构造位置解耦。因此输出应是带不确定性的地球化学亲和性诊断，不是普遍硬标签。

## 2. 当前手稿冻结版

- 英文题目：**Geochemical memory limits machine-learning discrimination of basalt tectonic settings**（85 字符）。
- 英文 Markdown：`paper/nsr_manuscript_v1.md`
- 中文 Markdown：`paper/nsr_manuscript_v1_zh.md`
- 英文 Word：`paper/NSR_manuscript_v1_en.docx`
- 中文 Word：`paper/NSR_manuscript_v1_zh.docx`
- 英文 PDF：`reports/render/NSR_manuscript_v1_en.pdf`
- 中文 PDF：`reports/render/NSR_manuscript_v1_zh.pdf`
- 英文摘要 149 词，主文 2,122 词，方法 267 词，6 个关键词，25 篇参考文献，5 幅图 + 1 个表。
- 结构按 NSR Research Article 要求整理：单独标题页，方法位于结论之后，含数据可用性、CRediT、利益冲突、图例和 alt text。
- 中英文 Word 均由 Microsoft Word 后台导出为 14 页 PDF，已逐页检查。
- 可访问性审计：high=0，medium=0，low=0；表头已标记，5 幅图均为内联图并有替代文本。

## 3. 冻结数据规模

- GEOROC Rock Types 2026-06：109,882 条 BASALT 原始记录；54,012 条宽特征模型记录。
- GEOROC 宽类：ARC 18,083，宽义 CIB 30,086，OIB 5,843。
- PetDB 四类全岩玄武岩：6,148 件样品；MORB 3,650，ARC 754，OIB 1,281，CIB 463。
- PetDB 主模型使用四类均观测的 21 个化学变量。Th 因 PetDB OIB 整列缺失被从主分析删除。
- 外部测试使用 18 个所有严格队列共有的正值变量：SiO2、TiO2、Al2O3、FeOT、CaO、MgO、MnO、K2O、Na2O、P2O5、Rb、Sr、Y、Zr、Nb、La、Ce、Nd。
- 主要处理后数据位于 `data/processed/`；数据审计位于 `reports/data_quality/`。

## 4. 外部独立测试

严格外部评分集共 231 件样品，每类 2 个省域代理：

- ARC：南新赫布里底弧前 12；哥斯达黎加火山前缘 18。
- CIB：Rio Grande Rift/Jemez Lineament 28；Big Pine Volcanic Field 70。
- MORB：西南印度洋中脊 36；南大西洋中脊 18.0–20.6°S 12。
- OIB：Mauna Loa 2022 喷发 16；La Palma 2021 喷发 39。
- 外部队列与 PetDB 训练表无文献重叠或精确样品名重叠。
- 哥斯达黎加来自 GEOROC 2JETOA，严格限于 IRAZU、MIRAVALLES 和 TURRIALBA 中 common18 完整的 18 件样品。
- 南大西洋中脊来自 Zhong et al. (2019) Table S1，DOI `10.3390/min9110659`，12/12 件 common18 完整。
- 西南印度洋中脊队列尚未确定关联期刊论文，已在手稿表 1 中明确标为 provisional；投稿前应建立正式论文关联或替换该队列。

## 5. 不得随意改写的关键结果

- GEOROC 随机森林 macro-F1：随机 0.861；引文重叠分组 0.747；地点分组 0.654。
- PetDB 随机森林 macro-F1：随机 0.915；引文重叠分组 0.760；地点分组 0.648。
- PetDB 逻辑回归 macro-F1：随机 0.822；引文重叠 0.718；地点 0.646。
- GEOROC 缺失性单独预测 macro-F1：引文分组 0.289；地点分组 0.258；显式缺失指示未改善主模型。
- PetDB 保守分组随机森林：Brier 0.236，top-label ECE 0.075。概率只能解释为对训练类的相对亲和性，不是字面地质后验概率。
- 跨数据库随机森林 macro-F1：宽义本体 GEOROC→PetDB 0.804，PetDB→GEOROC 0.484；裂谷对齐本体为 0.788 和 0.619。
- 外部逻辑回归：accuracy 0.442，balanced accuracy 0.576，macro-F1 0.486。
- 外部随机森林：accuracy 0.745，balanced accuracy 0.822，macro-F1 0.764，log loss 0.755。
- 外部随机森林类别召回率：ARC 29/30 = 0.967；CIB 51/98 = 0.520；MORB 48/48 = 1.000；OIB 44/55 = 0.800。
- 省域召回率：Costa Rica 0.944；South New Hebrides 1.000；Big Pine 0.543；Rio Grande Rift 0.464；SWIR 1.000；SMAR 1.000；La Palma 0.718；Mauna Loa 1.000。
- 保守置换重要性主要变量：TiO2 0.0918，Sr 0.0633，K2O 0.0305，Zr 0.0253，MnO 0.0212，Nb 0.0211，Y 0.0205。这些是模型依赖，不能宣称为唯一因果示踪剂。
- 东非裂谷 347 件 CIB 完整省域留出：平均 p(OIB)=0.609，平均 p(CIB)=0.075，307/347 预测为 OIB。不得将其解释为数据库标签错误。
- Vate Trough 21 件未评分样品：随机森林预测 MORB 76.2%，ARC 14.3%，CIB 4.8%，OIB 4.8%。不得将它们强行计为 ARC 或 MORB 准确率。
- La Palma OIB 召回率 0.718；第 1–20 天分异较强 tephrite 为 0.421；20 天后原始 basanite 为 1.000。

## 6. 关键脚本与报告入口

- 多类外部验证：`src/run_multiclass_external_validation.py`
- 南大西洋 MORB 构建：`src/build_smar_morb_external.py`
- 哥斯达黎加 ARC 构建：`src/build_georoc_costa_rica_arc_external.py`
- GEOROC ARC 省域审计：`src/audit_georoc_arc_provinces.py`
- NSR 合规检查：`src/validate_nsr_manuscript.py`
- Word 生成：`src/build_nsr_docx.py`
- Word 后台导出：`src/render_docx_with_word.ps1`
- 数据与模型合同测试：`src/test_data_contract.py`
- NSR 合规报告：`reports/manuscript/nsr_compliance_v1.json`
- 外部预测：`reports/modeling/multiclass_external_predictions.csv`
- 外部总指标：`reports/modeling/multiclass_external_metrics.csv`
- 外部类别指标：`reports/modeling/multiclass_external_per_class.csv`
- 外部主图：`figures/multiclass_external_validation.png` 和 `.pdf`
- NSR 当前官方样式清单：`references/nsr_style_checklist_2026.pdf`
- 补充数据索引：`paper/SUPPLEMENTARY_DATA_INDEX.md`
- 补充文件机读清单：`reports/manuscript/supplementary_inventory.csv`（39 个冻结证据文件，含字节数与 SHA-256）。
- 重建补充清单：`python src/build_supplementary_inventory.py`。

## 7. 环境与运行记忆

- 工作区：`D:\AI地学`。
- 用户允许安装所需软件，尽量安装在 D 盘。
- Anaconda Python 命令 `python` 已具备 pandas、pyarrow、scikit-learn 和 PyYAML；运行 `python src/test_data_contract.py` 可通过，但会出现 numexpr 版本警告，警告不影响测试通过。
- 文档生成使用 Codex 捆绑 Python：`C:\Users\ASUS\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe`。
- 项目本地依赖位于 `.deps/`；已安装 `xlrd==2.0.1` 和 `PyYAML==6.0.2`。
- LibreOffice 26.2.4 官方 MSI 已下载至 `software/LibreOffice_26.2.4_Win_x86-64.msi`，但安装进程卡住后已终止；不必继续安装。已改用本机 Microsoft Word 后台导出 PDF，可靠可用。
- 如仅修改文本：先改 Markdown，再运行 `src/build_nsr_docx.py`，然后用 `src/render_docx_with_word.ps1` 导出 PDF，最后重跑 `src/validate_nsr_manuscript.py`。
- 如修改数据或模型：必须同时重跑外部验证、手稿数值检查、数据合同测试，并更新本文件。

## 8. 投稿前仍需用户提供

- 已知通讯邮箱：`xubangkun439@gmail.com`。
- 需确认英文作者姓名、作者顺序和是否只有一位作者。手稿暂用 `Bangkun Xu`，中文暂用“徐邦坤”，两者都必须由用户最终确认。
- 需院系/单位、城市、邮编、国家、通讯电话。
- 需基金名称与项目号；若无基金，明确写无。
- 需致谢和其他作者 CRediT 贡献。
- 需为复现包申请公开存档 DOI。在没有上述信息和 DOI 时，当前文件只是可供导师/合作者审阅的完整初稿，不应直接投稿。

## 9. 后续任务优先级

1. **最高优先级：** 收到作者和单位信息后，替换手稿中所有确认占位符，重新生成和渲染中英文 Word/PDF。
2. 组织补充数据和复现包，申请 DOI。
3. 在不增加算法复杂度的前提下，补充每类第 3 个完全独立省域，优先 CIB 和 OIB。
4. 完成后生成投稿附件：cover letter、highlights/推广文字、supplementary data index 和可复现性声明。
5. 不要在没有省域留出改善证据时追加深度学习、堆叠集成或大规模超参数搜索。

## 10. 最后状态

- `python src/test_data_contract.py` 最后结果：**GEOROC/PetDB data, model, sensitivity, and interpretability contracts passed.**
- `src/validate_nsr_manuscript.py` 最后结果：标题、摘要、主文、方法、关键词、参考文献、图表数、引文范围全部通过。
- 当前工作已可停止并安全继承。若没有新的作者信息或投稿决定，不需要继续消耗资源。
- 2026-07-15 收尾审计已将 GEOROC Sample Metadata、Haase 2020、Day 2022 La Palma 和 Zhong 2019 SMAR 补入 `references/data_sources.csv`；该表现有 14 个唯一来源记录。
- 轻量交接包：`handoff/NSR_basalt_project_handoff_v1.zip`，SHA-256 `4D3C1D6EA29872D9E4D3A0300C62EF35B712083A7204118E896CB962132940F2`。生成时中文 Word 正被 Microsoft Word 占用，因此压缩包包含中文 Markdown 和 PDF，不包含中文 DOCX；独立中文 DOCX 仍保存于 `paper/NSR_manuscript_v1_zh.docx`。
- **当前推荐交接包：** `handoff/NSR_basalt_project_handoff_v2.zip`，包含更新后的 39 文件补充索引、SWIR 期刊关联审计和 14 项来源登记表；SHA-256 `8B0F0ED70A50D40F14CEE2E3D4A706290833D365360530F5FE866FD930D91D7D`。中文 DOCX 因生成时被 Word 占用而未纳入压缩包，但独立文件完整保留。
- SWIR 期刊关联复审记录：`reports/data_quality/swir_publication_link_audit_2026-07-15.md`。官方 Mendeley 记录仅给出数据集 DOI，精确题名检索未找到匹配期刊 DOI，因此仍保留 provisional 状态，不与其他 SWIR 论文强行关联。
