# 基于多元素地球化学与可解释机器学习的玄武质火山岩构造环境判别

**Tectonic Discrimination of Basaltic Volcanic Rocks Using Multi-Element Geochemistry and Interpretable Machine Learning**

> 稿件状态：v0.9 高级草稿（2026-07-15）。GEOROC 三分类、PetDB 四分类、双向跨库诊断、解释与校准、东非裂谷整省留出、MORB 介质扩展，以及 ARC–CIB–MORB–OIB 的统一四类外部转移均已完成。统一外测含 92 个样品，但每类仅有一个研究/省域代理，且 MORB 队列的论文 DOI 尚待确认；因此结果已达到论文主线验证阶段，但仍不是最终投稿版。

## 摘要

玄武质火山岩的主量和微量元素组合记录了地幔源区、部分熔融、俯冲组分输入、分离结晶与地壳混染的综合影响，但不同构造环境的地球化学范围广泛重叠，传统二元或三元判别图难以系统利用高维信息。本研究建立一条从原始数据固化、岩性与标签审计、文献/地区分组验证，到偏倚负对照、特征解释和省域机制审计的可复现流程。基于 GEOROC Rock Types 2026-06 版中的 109,882 条 BASALT 记录，严格筛选得到 54,012 条 ARC–CIB–OIB 宽特征建模记录。平衡随机森林的 macro-F1 从随机分层的 0.861 降至保守引用重叠组件分组的 0.747 和地区留出的 0.654。进一步从 PetDB 以相同合同获得 6,148 个 MORB–ARC–OIB–CIB 四分类样品。为避免 OIB 中 Th 整列缺失造成结构性泄漏，主验证使用四类共同有观测的 21 个特征；随机森林 macro-F1 从随机分层的 0.915 降至保守引文分组的 0.760 和地区留出的 0.648，地区留出时 CIB 召回率仅为 0.127。东非裂谷贡献了这一失稳的主体：整省删除训练后，其数据库标注 CIB 的样品仍具有平均 0.609 的 OIB 概率和 0.886 的 OIB/CIB 过渡指数。共有三类的 GEOROC→PetDB 与 PetDB→GEOROC 转移 macro-F1 分别为 0.804 和 0.484。保守引文分组置换重要度与 SHAP 共同指向 TiO₂、Sr、K₂O、Nb 和 Y；折外 Brier 分数为 0.236、top-label ECE 为 0.075。在统一的 18 元素模型上，进一步建立 92 个样品的四类外测：新赫布里底弧前 ARC 12 个、Rio Grande Rift/Jemez Lineament CIB 28 个、西南印度洋脊 MORB 36 个和 Mauna Loa 2022 OIB 16 个。随机森林外测准确率、平衡准确率和 macro-F1 分别为 0.837、0.866 和 0.807；ARC、MORB、OIB 召回率分别为 1.000、1.000 和 1.000，而 CIB 仅为 0.464。21 个 Vate Trough 初始弧后裂谷样品不作硬标签计分，其中 76.2% 被随机森林判为 MORB，显示早期弧后减压熔融与弱板片组分的连续性。结果表明，多元素模型可以提取可解释的构造信号，但最有信息量的失败集中在继承俯冲改造的大陆裂谷和构造过渡域。随机拆分、地区外推、标签本体、测量缺失、概率校准和省域独立性必须作为结论边界。

**关键词：** 玄武岩；构造环境；地球化学；机器学习；分组交叉验证；SHAP；GEOROC

## 1 引言

利用玄武质火山岩全岩地球化学判别构造环境，是岩石地球化学和古构造重建中的经典问题。Ti–V、Ti–Zr–Y、Zr–Y–Nb 和 Th–Ta–Hf 等判别图将对部分熔融、流体/熔体交代和源区富集敏感的元素组合压缩为低维图解。然而，许多经典边界并非由严格的外部验证推导，地球化学数据的封闭效应也会影响线性判别分析（Vermeesch, 2006a）。基于分类树和其他机器学习方法的研究进一步说明，多维主量、微量元素与同位素信息可以支持定量构造分类（Vermeesch, 2006b; Petrelli & Perugini, 2016; Ueki et al., 2018）。

高维模型也带来新的可信度问题。全球地球化学数据库由大量已发表研究汇编而成，同一文献、地区、岩套或分析批次内的样品并非统计独立。若随机地将逐样品记录分配到训练集与测试集，模型可能借助文献或区域相关结构获得过于乐观的性能。此外，不同年代和研究目标下的元素测量覆盖并不相同，模型可能学到“哪些元素被测量”而不是岩石成分本身。

本研究聚焦于玄武质火山岩，提出一个以数据追溯、标签保守性、防泄漏验证和岩石学可解释性为核心的工作流。当前阶段回答四个问题：（1）ARC、宽义大陆板内玄武岩（CIB）、OIB 和 MORB 是否具有可量化的多元素差异；（2）随机、文献分组和地区留出验证对性能估计有多大影响；（3）缺失模式与结构性缺列携带多少类别信号；（4）模型依赖的特征是否与已知地球化学过程相容。GEOROC 用于大样本三分类与解释基线，PetDB 同源四类数据用于避免数据库来源与 MORB 标签完全对应。

## 2 数据与方法

### 2.1 数据源与版本固化

主数据源为 GEOROC Compilation: Rock Types，DOI `10.25625/2JETOA`，使用 2026-06-01 生成的 9 个 BASALT 分卷。原始文件合计 109,882 条记录和 171 个字段，每个文件均以官方 MD5 固定。样品书目信息来自 GEOROC Compilation: Sample Metadata，DOI `10.25625/4EZ7ID`，引用表为 2024-12-01 快照。GEOROC 与 PetDB 的关系型数据架构用于整合样品元数据、分析结果和文献来源（Lehnert et al., 2000）。

PetDB 2.0 四类固定导出均限定 Whole Rock、Basalt、全部元数据和全部输出变量，构造标签分别为 `SPREADING CENTER`、`VOLCANIC ARC`、`OCEAN ISLAND` 和 `CONTINENTAL RIFT`。交付主表分别包含 22,680、4,632、6,555 和 1,707 条分析记录，对应 7,668、1,694、3,304 和 1,052 个唯一 Sample URL。每份归档均固定 SHA-256，完整清单见 `config/petdb_primary4_exports.yaml`。因构造条件是包含式检索，只在各主表与样品元数据表的规范化样品名和坐标均达到 100% 行级一致后才接受逐行标签关联；仅具有目标标签的 Sample URL 被视为高置信，混合/次级标签排除。

EarthChem 初次同时生成 OIB 与 CIB 时为两个任务分配了同一 S3 键，后完成任务覆盖先完成任务，且该 OIB 包的两张 CSV 缺少表头。异常包被保留为审计证据但不进入分析；OIB 和 CIB 随后分别重新导出，并重新校验查询、哈希、表头、行数和行级链接。

样品级队列以 Sample URL 为粒度。先使用导出表的全部列识别完全重复分析，再对每个样品的非空测定取中位数，同时保留方法、引用、测定数与冲突标志。四类分别删除 32、2、5 和 10 条完全重复分析。铁以 FeO 当量统一，优先使用直接报告 FeOT，其次为 `0.8998 × Fe2O3T`，再次为 `FeO + 0.8998 × Fe2O3`。严格候选要求目标构造标签唯一、SiO₂ 40–55 wt%、完整 10 主量、85–105 wt% 计算总量且引用可追溯。最终宽特征可建模样品为 MORB 3,650、ARC 754、OIB 1,281、CIB 463，共 6,148 个；不活动特征队列共 4,005 个。

### 2.2 岩性筛选与标签本体

入选记录必须为全岩火山岩分析，岩石名称包含 basalt，SiO₂ 介于 40–55 wt%，且可计算的主量总量介于 85–105 wt%。当前高置信三分类将 `CONVERGENT MARGIN` 映射为 ARC，`OCEAN ISLAND` 映射为 OIB，并将 `INTRAPLATE VOLCANICS`、`CONTINENTAL FLOOD BASALT` 与 `RIFT VOLCANICS` 合并为宽义 CIB。所有原始细标签同时保留。GEOROC `SUBMARINE RIDGE` 包含 Ninetyeast Ridge、Walvis Ridge 和 Broken Ridge 等非典型洋中脊地点，因此未被作为 MORB 代理标签。

一个 Citation-ID 集合若同时对应多个主标签，则从宽特征建模队列排除。同时根据 Citation-ID 集合之间的重叠关系构建连通组件，用于极保守的文献相关性压力测试。

地区分组优先采用明确的 spreading-center 名称；其余样品将全部一级地理描述规范化为大写、去重并按词典序排序后组合。该步骤使 `COUNTRY | BRAZIL, LARGE_IGNEOUS_PROVINCE | CAMP` 与顺序相反的同义记录落入同一组，避免同一省域因字段顺序不同跨越训练折和测试折。

### 2.3 特征工程

宽特征集包含 SiO₂、TiO₂、Al₂O₃、FeOᵀ、CaO、MgO、MnO、K₂O、Na₂O、P₂O₅、V、Cr、Ni、Rb、Sr、Y、Zr、Nb、La、Ce、Nd 和 Th，共22 个候选特征。不活动特征集包含 TiO₂、P₂O₅、V、Cr、Ni、Y、Zr、Nb、La、Ce、Nd、Sm、Eu、Yb、Lu、Hf、Ta、Th 和 U，共19 个候选特征。微量元素中小于等于零的数值作为缺失处理。宽特征基线要求至少观测到 22 个候选特征中的 12 个。PetDB OIB 导出完全缺少 Th，因此四分类主验证排除 Th，仅使用在四类中均至少有观测的 21 个宽特征；22 特征结果保留为结构性缺失偏倚对照。所有正值特征进行 log10 变换，缺失值在每个训练折内以中位数填补。位置、引用、样品名、岩石名、文件名与任何标签字段被明确禁止作为模型输入。

### 2.4 模型与交叉验证

比较模型包括类别先验虚拟分类器、类别平衡逻辑回归和类别平衡随机森林。GEOROC 基线随机森林含 120 棵树，PetDB 四分类含 160 棵树。四种五折验证策略分别为：（1）逐记录随机分层；（2）Citation-ID 集合分组；（3）一级地区分组；（4）Citation-ID 重叠连通组件分组。主指标为 macro-F1，并报告准确率、平衡准确率、weighted-F1、对数损失和分类召回率。

### 2.5 偏倚负对照与可解释性

为分离元素值与测量覆盖模式的信号，在同一个同时满足宽特征和不活动特征条件的 43,005 条 GEOROC 队列上，比较宽特征值、宽特征值+显式缺失指示、不活动特征值和仅宽特征缺失指示四种输入。GEOROC 置换重要性在五折 Citation-ID 集合分组的留出数据上计算，每折最多使用 4,000 条留出记录并重复置换 4 次；SHAP 使用全队列拟合的 100 棵树描述性随机森林和 500 条分层抽样。

PetDB 四分类的置换重要性使用五折 Citation-ID 重叠组件留出，每折对全部留出样品重复置换 6 次，并以平衡准确率下降为指标。描述性 SHAP 使用 160 棵树的全队列模型和 800 个分层样品。SHAP 方向均以 log10 元素值与该类别 SHAP 值的 Spearman 相关系数表示。校准从折外概率独立计算多分类 Brier 分数与 10 个等宽置信度箱的 top-label expected calibration error（ECE），不进行测试集后校准。

### 2.6 PetDB 四分类与缺失偏倚门控

PetDB 四类采用完全相同的样品聚合、FeO 当量和质量控制合同。四分类主验证排除在任一类别整列无观测的特征，并以 Citation-ID 重叠组件分组作为保守主证据、地区分组作为空间外推压力测试。另用类别平衡逻辑回归分别输入仅缺失指示和无显式缺失指示的 log10 元素值；该诊断同时在全部 22 个候选宽特征和去除结构性缺列后的 21 个共同特征上进行。此前 GEOROC–PetDB 来源诊断仍保留，用于说明“PetDB 仅有 MORB”阶段的设计为何不可识别，但不再作为当前 PetDB 内部四分类的主验证。

### 2.7 跨数据库转移诊断

在两库共有的 ARC、CIB 和 OIB 上进行双向外部测试。所有模型均使用相同的 21 个共同宽特征，训练库的中位数只用于填补测试库缺失值。一个方向以 54,012 条 GEOROC 记录训练、2,498 个 PetDB 样品测试，另一方向反之。该设计不通过目标库调参。需要强调，GEOROC CIB 合并 `INTRAPLATE VOLCANICS`、`CONTINENTAL FLOOD BASALT` 和 `RIFT VOLCANICS`，PetDB CIB 则限定 `CONTINENTAL RIFT`，因此双向结果同时包含数据库转移和标签本体转移。

### 2.8 PANGAEA MORB 试验性外测

从 PANGAEA 筛选三个许可明确的 MORB 候选数据集。Reykjanes Ridge 数据（PANGAEA.956077）对应论文已出现在 PetDB MORB 引文中，因此排除。保留 Mid-Atlantic Ridge near Ascension Island（PANGAEA.727391，CC BY 3.0）与 DSDP Hole 24-238（PANGAEA.707361，CC BY 3.0）；规范化样品名与论文标题均未在 PetDB MORB 导出中检出完全匹配。对分析点取样品中位数，要求 10 个共同主量元素完整、SiO₂ 40–55 wt% 和主量总和 85–105 wt%，得到 18 个玻璃或玻璃包裹体样品。为匹配外测特征覆盖，另以 PetDB 6,148 个四分类样品和 10 个共同主量元素训练类别平衡模型；内部参照仍使用保守引文重叠分组。外测只包含 MORB，因此仅报告 MORB 召回率和 MORB 概率，不计算四分类 macro-F1。

### 2.9 误判审计与东非裂谷整省留出压力测试

将 PetDB 随机森林的逐样品折外预测按 `record_id` 回连样品、引文和地理元数据，分别统计引文重叠组件分组与地区分组的定向混淆、高风险地区和高置信误判。东非裂谷队列定义为规范化一级地区 `EAST AFRICAN RIFT`，共 361 个宽特征样品，来自 34 个引文组；其中数据库标签为 CIB 347 个、MORB 14 个。

为避免五折划分中其他地区构成对结果的偶然影响，另进行整省留出压力测试：从训练集中完全删除上述 361 个样品，以其余 5,787 个 PetDB 样品和相同 21 个特征训练类别平衡逻辑回归与 500 棵树的类别平衡随机森林，再对东非裂谷计算四类概率。定义 OIB/CIB 过渡指数为 `p(OIB) / [p(OIB) + p(CIB)]`。概率只解释为相对于训练类别的多元素地球化学亲和性，不作为重新标注依据。均值和 OIB 概率高于 CIB 概率的样品比例之 95% 区间，通过对引文组而非单个样品进行 2,000 次聚类自助抽样获得，以降低同一研究内样品相关造成的伪精确。最后在 log10 尺度比较东非裂谷 CIB、其他 CIB 与 OIB 的逐元素中位数；这些差异用于提出待检验机制，而不用于唯一成因反演。

### 2.10 Figshare–Mendeley 西南印度洋脊全岩 MORB 外测

Figshare 数据集 `10.6084/m9.figshare.25295671.v1`（CC BY 4.0）包含东太平洋海隆与西南印度洋脊 MORB 的主量和微量元素。原始工作簿、API 元数据、MD5 与 SHA-256 均固定；只抽取 Table S1 的 44 个自然样品，不纳入 Tables S2–S6 的原始熔体、结晶或混合模型值。后续 Mendeley Data 记录 `10.17632/fntfnf92tg.1` 补充了研究题名，但截至审计日仍未关联期刊论文 DOI。工作簿注明 8 个东太平洋海隆样品来自 Li et al. (2019)，故只作敏感性分析；其余 36 个西南印度洋脊样品无二手来源脚注，且规范化样品名与 PetDB MORB 无完全匹配，因此作为“仓储原创、论文归属待确认”的暂定严格队列，而不表述为完全出版独立。

为利用其完整微量元素覆盖，从 PetDB 四类样品中选择 10 个主量元素和 Rb、Sr、Y、Zr、Nb、La、Ce、Nd 共 18 个共同特征，使用与主分析相同的 log10 变换、中位数填补、类别平衡逻辑回归和随机森林。内部参照采用保守引文重叠组件五折分组；外测报告 MORB 召回率和 MORB 概率。另逐样品统计每个已观测特征是否超出 PetDB 训练集 log10 尺度的 1%–99% 范围，以区分分类错误与明显训练域外推。该外测仍是单类、单研究队列，不能估计四分类外部 macro-F1。

### 2.11 ARC、CIB 与 OIB 出版独立外测队列

ARC 候选来自 PANGAEA.922011（CC BY），对应 Haase et al. (2020) 的南新赫布里底岛弧与 Vate Trough 数据。只保留全岩分析，并优先采用 ICP-MS 微量元素、XRF 主量元素。Erromango 11 个样品和 Vulcan Seamount 1 个样品定义为弧前 ARC；Vate Trough 21 个全岩样品依据原论文定义保留为 `BACK_ARC_TRANSITION`，不强制计为 ARC 或 MORB。四个 `E15/E16/E38/E70` 短样品名在 PetDB 中发生字符串碰撞，但 PetDB 坐标和引文分别指向 Galápagos 或东非，故记为歧义短名而非真实重复；源论文 DOI 与 PetDB 引文均无重叠。

CIB 候选来自 Rowe et al. (2015) 的出版商补充表 S1，包括 Rio Grande Rift 和 Jemez Lineament 的 29 个整岩样品。工作簿已给出无挥发分归一化主量元素和独立的分析总量；模型使用归一化值，分析总量仅用于质量控制。按本研究 SiO₂ 40–55 wt% 合同排除 EB06-7B（55.59 wt%），保留 28 个样品。该论文明确将样品置于大陆裂谷背景，同时讨论 Farallon 浅俯冲遗留的岩石圈地幔交代、流体活动元素富集和地壳混染，因此该队列既是严格构造位置外测，也是对“成分记忆是否覆盖构造位置”的机制检验。

OIB 候选来自 Rhoads et al. (2025) 的 Mauna Loa 2022 喷发开放补充表 S1（CC BY 4.0）。保留 16 个 2022 喷发整岩样品；源表总铁以 Fe₂O₃ᵀ 报告，按 `FeOᵀ = 0.8998 × Fe₂O₃ᵀ` 转换。22 个候选宽特征均完整，论文 DOI 和规范化样品名均未在 PetDB 中检出。由于 16 个样品属于同一喷发时间序列，统计单位仍是一个火山/喷发队列。

### 2.12 统一四类外部转移与弧后机制压力测试

为避免各外部队列使用不同特征合同，从 PetDB 选择所有四个队列共同覆盖的 18 个元素：10 个主量元素以及 Rb、Sr、Y、Zr、Nb、La、Ce 和 Nd。严格四类外测包含 ARC 12、CIB 28、MORB 36 和 OIB 16 个样品，共 92 个；训练阶段不查看外测标签调参。报告准确率、平衡准确率、macro-F1、逐类召回率、对数损失和真实类别概率。逐样品域检查统计 log10 元素值超出 PetDB 训练集 1%–99% 包络的数量。Vate Trough 21 个样品使用同一模型预测，但作为不计分的初始弧后裂谷压力测试单独报告。

92 个样品只代表每类一个研究/省域代理，因此不进行会把同研究样品误当独立重复的逐样品自助法置信区间。当前外测点估计回答的是“模型能否转移到这四个特定、出版独立或仓储独立的队列”，不是全球四类省域泛化误差；后者需要每类多个独立省域后采用省域聚类或分层重采样。

## 3 结果

### 3.1 数据处理与队列构成

原始 109,882 条记录中，65,146 条通过严格岩性、材料、SiO₂ 和三分类标签范围筛选。3,033 条因主量分析不完整或总量不合理而排除，得到 62,113 条处理记录。44 个跨主标签的模糊 Citation-ID 集合共影响 748 条记录。最终宽特征建模队列为 54,012 条，不活动特征队列为 43,005 条。宽特征建模队列包含 ARC 18,561 条、CIB 30,084 条和 OIB 5,367 条。

**表1  处理队列流程**

| 阶段 | 记录数 |
|---|---:|
| GEOROC BASALT 原始记录 | 109,882 |
| 严格岩性与三分类范围 | 65,146 |
| 主量总量质量控制后 | 62,113 |
| 宽特征建模队列 | 54,012 |
| 不活动特征公共队列 | 43,005 |

### 3.2 验证策略决定了性能估计的乐观程度

平衡随机森林在随机分层验证中达到 0.874 的准确率和 0.861 的 macro-F1（表2）。按 Citation-ID 集合分组后，macro-F1 降为 0.786；按引用重叠连通组件分组时为 0.747；按一级地区分组时进一步降为 0.654。这一单调下降表明，文献和地区相关性是随机逐样品分割的主要泄漏风险。

**表2  平衡随机森林在不同五折验证中的表现**

| 验证策略 | 准确率 | 平衡准确率 | Macro-F1 | 对数损失 |
|---|---:|---:|---:|---:|
| 随机分层 | 0.874 | 0.855 | 0.861 | 0.380 |
| Citation-ID 集合分组 | 0.815 | 0.777 | 0.786 | 0.484 |
| 引用重叠连通组件 | 0.797 | 0.728 | 0.747 | 0.515 |
| 一级地区分组 | 0.739 | 0.635 | 0.654 | 0.629 |

在 Citation-ID 集合分组验证中，ARC、CIB 和 OIB 的召回率分别为 0.777、0.859 和 0.694。在地区留出验证中，对应值降为 0.711、0.821 和 0.373，其中 OIB 的降幅最大。

### 3.3 缺失模式不是宽特征模型的主要信号来源

在43,005条公共队列上，宽特征值模型在文献分组与地区分组验证中的 macro-F1 分别为 0.783 和 0.640（表3）。加入显式缺失指示后分别为 0.782 和 0.639，未显示改善。仅使用缺失指示时，macro-F1 仅为 0.289 和 0.258。不活动特征模型分别为 0.730 和 0.583，较宽特征值低约 0.053 和 0.057。

**表3  公共队列的特征与缺失偏倚敏感性（macro-F1 折均值）**

| 输入方案 | 文献分组 | 地区分组 |
|---|---:|---:|
| 宽特征值 | 0.783 | 0.640 |
| 宽特征值 + 缺失指示 | 0.782 | 0.639 |
| 不活动特征值 | 0.730 | 0.583 |
| 仅宽特征缺失指示 | 0.289 | 0.258 |

### 3.4 TiO₂–Nb 主导的多元素组合具有最强留出判别力

Citation-ID 集合分组留出集上的置换重要性从高到低主要为 TiO₂（0.159 ± 0.015）、Nb（0.102 ± 0.007）、Al₂O₃（0.078 ± 0.007）、CaO（0.071 ± 0.010）、Sr（0.058 ± 0.009）、MgO（0.052 ± 0.012）、SiO₂（0.050 ± 0.006）和 Na₂O（0.046 ± 0.007）。全局平均绝对 SHAP 的前五位为 TiO₂、Al₂O₃、Nb、CaO 和 SiO₂，与留出置换结果大体一致。

SHAP 方向显示，低 TiO₂、低 Nb、高 Al₂O₃ 和高 SiO₂ 与 ARC 预测正相关，而高 TiO₂、高 Nb 和高 Zr 与 OIB 预测正相关；高 Sr 与 ARC 预测正相关。这些方向与弧岩中流体相容元素输入和高场强元素相对亏损，以及 OIB 相对富 Ti–Nb–Zr 的宽泛认识相容。然而，K₂O 等变量的方向可能受地区、分异与广义标签影响，不应作为单独成因证据。

### 3.5 PetDB 四分类在文献与地区留出下显著降分

四类导出经重复删除和 Sample URL 中位数聚合后共得到 13,718 个样品，其中 13,566 个具有唯一目标构造标签。严格候选 7,540 个；宽特征可建模 6,148 个，不活动特征可建模 4,005 个。宽特征队列包含 MORB 3,650、OIB 1,281、ARC 754 和 CIB 463 个样品。

在排除结构性缺列 Th 后，类别平衡随机森林的 macro-F1 从随机分层的 0.915 降至 Citation-ID 集合分组的 0.769、引用重叠组件分组的 0.760 和地区留出的 0.648；相应平衡准确率为 0.926、0.761、0.755 和 0.650。保守引文重叠分组中 ARC、CIB、MORB 和 OIB 的召回率分别为 0.871、0.501、0.937 和 0.712；地区留出时分别为 0.873、0.127、0.940 和 0.660。CIB 是最不稳定类别。地区分组结果较前一版进一步降低，是因为地理描述经规范化排序后，字段顺序不同但语义相同的省域不再跨折。

仅用缺失模式时，全部 22 个候选特征在引文重叠分组下的平均平衡准确率为 0.535、macro-F1 为 0.480；排除 Th 后仍为 0.417 和 0.350。使用 21 个共同特征且不显式加入缺失指示的逻辑回归，在引文重叠分组下平均平衡准确率为 0.703、macro-F1 为 0.687。这说明元素值具有独立判别力，但文献报告习惯仍携带不可忽略的类别信号。

### 3.6 跨数据库转移具有明显方向不对称

使用宽义 CIB 时，GEOROC 训练、PetDB 测试的随机森林准确率为 0.813、平衡准确率为 0.848、macro-F1 为 0.804；ARC、CIB 和 OIB 召回率分别为 0.934、0.901 和 0.710。反向用 PetDB 训练、GEOROC 测试时，准确率降至 0.507、平衡准确率为 0.648、macro-F1 为 0.484；对应召回率为 0.814、0.251 和 0.880。

把 GEOROC CIB 限定为 `RIFT_VOLCANICS` 后，GEOROC 队列从 54,012 条减为 27,349 条。GEOROC→PetDB 的 macro-F1 为 0.788，CIB 召回率从 0.901 降至 0.536；PetDB→GEOROC 的 macro-F1 则升至 0.619，CIB 召回率为 0.276。更严格的本体对齐改善了较窄训练域向 GEOROC 裂谷子集的转移，但宽义 GEOROC CIB 因覆盖范围更广，反而更容易包容 PetDB CIB。该不对称支持训练支持域与标签本体共同控制外测表现。

### 3.7 四分类解释以 TiO₂–Sr–K₂O–Nb–Y 组合为主

保守引文重叠留出上的置换重要性前五位为 TiO₂（0.092 ± 0.051）、Sr（0.063 ± 0.030）、K₂O（0.031 ± 0.021）、Zr（0.025 ± 0.019）和 MnO（0.021 ± 0.021）；Nb（0.021 ± 0.022）与 Y（0.021 ± 0.010）紧随其后。描述性全队列 SHAP 的前五位为 Sr、TiO₂、K₂O、Nb 和 Y，与留出置换结果大体一致。

SHAP 方向显示，高 TiO₂ 和高 Nb 倾向提高 OIB 与 CIB 预测、降低 ARC 预测；高 Sr、K₂O 和 Rb 强烈降低 MORB 预测，而较高 Y 和 CaO 倾向提高 MORB 预测。OIB 同时表现为较高 TiO₂、Nb、La、P₂O₅ 和 FeOᵀ 以及较低 SiO₂ 和 Y 的模型关联。上述方向与弧岩高场强元素相对亏损、洋岛玄武岩富集端元及 MORB 相对低不相容元素的宽泛认识相容，但仍混合了熔融程度、分异、源区与文献选择效应。

保守引文分组随机森林的多分类 Brier 分数为 0.236，top-label ECE 为 0.075，平均最大类别概率为 0.774。可靠性曲线显示不同类别存在方向不一的局部偏差，尤其 ARC 在中高概率段偏保守，而 CIB 在部分中等概率段偏过度自信；因此论文报告折外概率与校准图，但不把未经后校准的概率解释为严格后验概率。

### 3.8 独立 PANGAEA MORB 小样本外测支持但不能定论

10 主量元素模型在 PetDB 内部保守引文重叠分组中的随机森林 macro-F1 为 0.761、平衡准确率为 0.766。全 PetDB 拟合后，对 18 个 PANGAEA MORB 玻璃样品的随机森林召回率为 0.944，平均 MORB 概率为 0.809；逻辑回归召回率为 0.889，平均 MORB 概率为 0.748。随机森林唯一误判样品 191DS2 被判为 CIB。该结果表明模型在数据仓库和分析介质变化下仍可识别多数 MORB，但样本只来自两个数据集、仅覆盖一个真实类别，不能转换为四分类外部性能或证明全球泛化。

### 3.9 东非裂谷呈现稳定的 OIB 亲和性而非普通 CIB 外推行为

地区分组折外预测中，东非裂谷 361 个样品仅 6 个与数据库标签一致。347 个 CIB 样品中，304 个被判为 OIB（比例 0.876）、19 个为 MORB、19 个为 ARC，仅 5 个保留为 CIB，对应该省域 CIB 召回率 0.014。错误并非由单篇文献控制：Kenya 的 Cancel Vazquez et al. (2024) 队列 59 个 CIB 全部误判，Ethiopia 的 Rooney et al. (2023) 队列 49 个全部判为 OIB，另有多篇独立研究表现出相同方向。

整省删除后重新训练得到更严格且方向一致的结果。随机森林将 347 个 CIB 中 307 个判为 OIB，仅 4 个判为 CIB；平均 OIB 概率为 0.609（文献组聚类 95% 区间 0.566–0.646），平均 CIB 概率为 0.075（0.065–0.089），OIB/CIB 过渡指数为 0.886（0.870–0.899），样品中 `p(OIB) > p(CIB)` 的比例为 0.988（95% 区间 0.968–1.000）。逻辑回归同样给出平均 OIB 概率 0.705、CIB 概率 0.174 和过渡指数 0.811，说明这一方向不依赖树模型。

逐元素中位数显示东非裂谷 CIB 在多项关键变量上更接近 OIB 而非其他 CIB。例如 SiO₂ 中位数分别为 48.10、48.34 和 46.32 wt%（东非裂谷 CIB、OIB、其他 CIB），TiO₂ 为 2.69、2.80 和 3.17 wt%，Nb 为 26.92、24.97 和 84.98 ppm，Sr 为 442、392 和 611 ppm，FeOᵀ 为 12.35、11.35 和 9.89 wt%。Al₂O₃、Zr、La、Ce 和 V 也呈不同程度的 OIB 方向接近。该组合说明东非裂谷不是普通 CIB 训练域的随机离群点，而是具有省域一致性的化学域偏移；但中位数仍混合源区、熔融程度、分异、地壳作用、改造和文献选择，不能唯一反演某一种地幔端元。

### 3.10 西南印度洋脊全岩外测扩展了 MORB 的介质与元素覆盖

18 特征模型在 PetDB 内部保守引文重叠分组中的随机森林 macro-F1 为 0.756、平衡准确率为 0.754；逻辑回归分别为 0.685 和 0.702。对 36 个暂定严格西南印度洋脊样品，随机森林全部判为 MORB，召回率 1.000，平均和中位 MORB 概率分别为 0.918 和 0.959；逻辑回归召回率为 0.944，平均和中位 MORB 概率为 0.867 和 0.920。只有样品 30III-TVG18 有 1 个特征落在 PetDB 训练集 1%–99% 范围外，其余 35 个样品全部位于逐特征经验范围内。作为敏感性分析，8 个二手来源东太平洋海隆样品被两种模型全部判为 MORB，但不计入严格证据。

逻辑回归把 20VI-TVG9 和 43IV-S27TVG15-1 判为 ARC，而随机森林对二者的 MORB 概率分别为 0.767 和 0.785。两者具有低 Nb（0.65、0.84 ppm）、低 La–Ce–Nd 和较低 TiO₂，容易在线性模型中与弧玄武岩的高场强元素亏损重叠；但其 Sr/Y 分别约 6.43 和 4.78，且 K₂O 仅 0.15 和 0.25 wt%、MgO 为 9.48 和 7.99 wt%，整体更符合亏损 MORB 组合。该误差说明，Nb 亏损不能单独等同于俯冲环境，多元素非线性组合能够部分缓解亏损 MORB 与弧玄武岩的判别歧义。

### 3.11 四类外部转移的主失效集中在大陆裂谷

统一 18 元素随机森林在 92 个严格外测样品上的准确率为 0.837、平衡准确率为 0.866、macro-F1 为 0.807、对数损失为 0.435；逻辑回归对应为 0.772、0.805、0.734 和 0.610。随机森林将 12 个弧前 ARC、36 个 MORB 和 16 个 OIB 全部正确识别，真实类别平均概率分别为 0.803、0.918 和 0.725。逻辑回归对三类的召回率分别为 0.917、0.944 和 1.000。

两种模型的主要失效均为 CIB。随机森林只正确识别 28 个 Rio Grande Rift/Jemez Lineament 样品中的 13 个，CIB 召回率为 0.464；12 个被判为 ARC、2 个为 OIB、1 个为 MORB。逻辑回归只正确 10 个，14 个判为 ARC、4 个判为 OIB，CIB 召回率为 0.357。错误具有显著地区结构：随机森林对 Springerville 的 8 个样品识别 7 个 CIB，对 Zuni Bandera 识别 3/4，但 Central RGR 为 0/5，Raton–Clayton 为 1/6，Southern RGR 为 2/5。只有 17.9% 的 CIB 样品有任一共同特征超出训练集 1%–99% 包络，说明大多数错误发生在训练变量范围内，而不是简单的数值域外推。

Vate Trough 初始弧后裂谷队列呈现与弧前截然不同的预测构成。随机森林将 21 个样品中的 16 个（0.762）判为 MORB、3 个判为 ARC、1 个为 CIB、1 个为 OIB，平均 MORB 概率为 0.551；逻辑回归将 17 个（0.810）判为 MORB、4 个为 ARC，平均 MORB 概率为 0.653。该结果与原研究提出的早期裂谷减压熔融、弱俯冲组分和与弧前分离的熔融区一致，但模型概率不能单独量化板片贡献。

## 4 讨论

### 4.1 多元素判别的地球化学意义

模型对 TiO₂、Nb、Zr、Sr 与主量元素组合的共同依赖，支持“构造环境信号是多过程叠加的高维特征”而非单一判别边界。Ueki et al. (2018) 同样指出，不同构造环境的有效表征需要多个元素与同位素组合，反映部分熔融、结晶分异、混合和多源组分的联合影响。GEOROC 的 Ti–Nb 主轴与 ARC/OIB 经验差异一致；PetDB 四分类进一步显示 Sr、K₂O、Y 和 CaO 对 MORB 与富集型端元的分离作用。主量元素与过渡元素的重要性说明，分离结晶、熔融程度、源区和地壳过程也与数据库中的构造标签共变。

### 4.2 防泄漏验证比算法间的小幅性能差更重要

随机分层与地区留出之间的 macro-F1 差在 GEOROC 为 0.207，在 PetDB 为 0.267，显著大于常见模型调参可能带来的边际改善。这意味着，对文献汇编型地球化学数据库，测试集如何定义是第一级方法学问题。Citation-ID 分组估计模型对新文献的泛化，地区留出更接近对新区域的转移，两者都比随机逐样品分割更符合论文的应用声称。

### 4.3 OIB 地区外推失败揭示了类内异质性

OIB 召回率在文献分组与地区留出之间从 0.694 降至 0.373，显示该类的文献内可分性不能直接转化为区域外推。可能的地质原因包括地幔端元组合、岩石圈交互、部分熔融程度和火山建筑演化的地区差异；数据原因包括区域样本数不平衡、分析年代和元素覆盖差异。后续应结合 leave-one-island/region-out 分析、地点级权重和独立 OIB 数据来源区分这些解释。

### 4.4 可解释性不等于成因因果识别

置换重要性和 SHAP 使模型依赖可以被审查，但它们无法单独分离源区成分、部分熔融、结晶分异、后期改造和数据采集结构的作用。特征相关还会导致重要性在元素之间分摊或遮蔽。PetDB 的 TiO₂ 和 Sr 在折间重要性标准差较大，且部分折中的 Zr、MnO、FeOᵀ 等重要性接近或低于零，说明地理/文献组成会改变特征排序。因此，本研究将 SHAP 方向表述为“与某地球化学机制相容的模型关联”，而不是机器学习对地质过程的因果证明。

### 4.5 同源四类数据解除完全来源混杂，但没有消除报告偏倚

PetDB 现在同时包含 MORB、ARC、OIB 和 CIB，因而类别不再与数据库来源一一对应，四分类内部验证比“PetDB MORB + GEOROC 其他三类”的直接池化设计更可识别。然而，OIB 的 Th 整列缺失以及共同特征上的缺失模式分类能力表明，查询结果的变量覆盖仍可与类别共变。排除结构性缺列使保守引文分组随机森林 macro-F1 从 0.809 降至 0.760，说明偏倚控制会实质改变性能估计。PetDB 内部验证仍不能替代 GEOROC↔PetDB 的跨数据库外部测试。

### 4.6 外部验证的不对称揭示标签本体与覆盖范围问题

GEOROC→PetDB 的三分类转移明显优于 PetDB→GEOROC。较大的 GEOROC 训练集覆盖更广的地区和岩套，可能更容易包容 PetDB 子集；反向训练则不能覆盖 GEOROC 的宽义 CIB。尤其 CIB 召回率从 0.901 变为 0.251，不能简单解释为某一数据库质量较差，而应首先视为标签范围和训练支持域不对称。将 GEOROC CIB 限定到裂谷子类后，反向 CIB 召回率仅小幅提高到 0.276，而正向 CIB 召回率下降到 0.536，说明本体对齐并不能消除训练支持域和省域组成差异。

### 4.7 东非裂谷误判揭示构造标签与岩浆过程连续体的冲突

东非裂谷的 CIB→OIB 定向误判具有三个不同于普通分类错误的特征：跨国家和文献重复、在整省删除训练后仍由逻辑回归与随机森林共同复现、并伴随 TiO₂–Nb–Sr–FeOᵀ 等多元素中位数向 OIB 域移动。经典综合已经指出，东非裂谷玄武岩由次大陆岩石圈地幔、HIMU 型和 OIB 类似的地幔柱组分共同贡献，且不同伸展阶段的岩石圈与软流圈贡献会变化（Furman, 2007）。因此，数据库的 `CONTINENTAL RIFT` 是构造位置标签，而模型看到的是源区混合、熔融和演化共同形成的组成状态；二者并不要求一一对应。

近期研究进一步支持把该现象放在裂谷演化而非简单二分法中理解。Turkana 的上新世玄武质脉冲可由地幔柱影响的上地幔在岩石圈减薄期间减压熔融产生，随后盾火山阶段又加入富集的交代岩石圈组分（Cancel Vazquez et al., 2024）。Afar 陆—洋转换研究表明，伸展由断层和岩石圈拉伸向岩浆侵入与新生洋壳形成转换，交代富集岩石圈地幔的再熔融可参与大体积玄武岩形成（Rooney et al., 2023）。挥发分—氧化还原研究则显示 Afar 地幔柱具有中等水含量和热异常，约 1%–29% 的部分熔融及其产生的熔体本身可能解释显著地球物理异常，而不需要诉诸异常高氧逸度（Brounce et al., 2024）。这些约束与高 OIB 亲和性相容，但不证明全部东非裂谷样品来自单一地幔柱端元。

由此，本研究建议把“构造环境判别”从互斥硬标签改写为分层问题：先识别稳定板块边界、洋中脊和板内端元，再对大陆裂谷报告裂谷阶段、地幔源区亲和性和超出训练支持域的程度。东非裂谷可作为 `TRANSITIONAL_RIFT` 压力队列保留，而不应为提高准确率而直接重标为 OIB。该地质驱动的软标签/层级框架比继续增加模型复杂度更能回答古构造应用中的真实不确定性。

### 4.8 亏损 MORB 的弧亲和误判界定了算法复杂度的合理用途

西南印度洋脊队列说明，增加微量元素覆盖和有限的非线性表达能力可以改善真正的地球化学边界，而不需要把研究转向复杂模型竞赛。逻辑回归的两个 ARC 误判集中在极低 Nb/Y 的亏损 MORB 端元，反映了弧岩浆与高程度亏损 MORB 都可能表现不相容元素贫化；随机森林利用 Sr/Y、K₂O、MgO 与稀土元素的联合状态恢复了 MORB 判断。这种改进的价值不在于 2 个样品的分数变化，而在于它暴露了传统“Nb 亏损即俯冲”简化规则的失效条件。

同时，随机森林的 36/36 不能被解释为全球 MORB 已完成验证。样品来自同一仓储研究，超慢速扩张的西南印度洋脊本身包含强烈的分段、熔融程度与地幔非均一性；样品间也不等同于独立省域重复。合理的下一步是扩大到多个洋脊、扩张速率和分析实验室，并保留逻辑模型作为可解释基线、随机森林作为适度非线性主模型，而不是继续叠加深度网络或大规模集成模型。

### 4.9 四类外测把“准确率问题”转化为继承信号与构造位置的冲突

弧前、洋中脊和 Mauna Loa 队列的高召回率表明，18 元素组合能够跨实验室和数据仓储识别三个端元。但 Rio Grande Rift/Jemez Lineament 的低 CIB 召回率说明，大陆裂谷不是一个只由当前伸展位置定义的均一成分域。Rowe et al. (2015) 将该区的挥发分和流体活动元素变化与 Laramide 期 Farallon 浅俯冲造成的岩石圈地幔水化/交代联系起来，并指出大陆火山岩中的 Ba、K、Sr 等传统俯冲指标还会受地壳混染影响。因此，模型把部分样品判为 ARC 并非与地质认识无关的随机错误，而是捕捉到裂谷岩浆对古俯冲改造和大陆岩石圈作用的成分记忆。

这与东非裂谷的 CIB→OIB 失效构成互补：两个裂谷都具有 CIB 构造位置，但前者主要向 ARC 域偏移，后者主要向 OIB 域偏移。由此可见，单一 `CIB` 标签内部至少包含“古俯冲改造岩石圈控制”“地幔柱/软流圈控制”和不同程度的混合端元。未来更合理的输出不是用更复杂算法强迫所有裂谷样品回到 CIB，而是同时报告构造位置、源区亲和性、裂谷阶段和域外程度。Vate Trough 的 MORB 主导预测进一步说明，弧后早期裂谷也会沿 ARC–MORB 连续体移动，支持将过渡环境作为独立机制层而非硬标签异常。

外测结果也限定了算法复杂度。随机森林相对逻辑回归的 macro-F1 提升为 0.073，主要来自对亏损 MORB、CIB 子域和非线性元素组合的更好处理；这一幅度足以保留一个适度非线性主模型，但不足以证明需要深度网络或大规模集成。下一阶段若增加复杂度，应只检验有地球化学含义的交互，例如 Nb–Th–Sr 与主量分异指标的联合、同位素—微量元素源区混合和裂谷阶段层级，而不是追求脱离省域验证的排行榜性能。

## 5 局限性与下一阶段

第一，统一四类外测虽包含 92 个样品，但每类只有一个研究/省域代理；样品数不能替代省域数，因而 macro-F1=0.807 是特定四队列的点估计，不是全球外推精度。第二，MORB 队列虽有仓储许可、样品追踪和独立样品名，但仍缺少关联期刊论文 DOI，严格出版独立性弱于 ARC、CIB 和 OIB 队列。第三，OIB 的 Th 整列缺失及共同特征的缺失模式信号说明报告偏倚没有被完全消除。第四，CIB 地区留出和 Rio Grande Rift 外测召回率分别仅为 0.127 和 0.464，且两个裂谷分别向 OIB 与 ARC 域偏移；在更多裂谷省验证前，不能把任何一种偏移视为所有大陆裂谷的统一规律。第五，折外概率未经后校准，ECE 0.075 表明不能把输出概率直接解释为严格后验概率。第六，当前主模型没有系统整合同位素、年龄和定量分异/改造指标，无法唯一拆分源区、熔融程度、岩石圈交互和地壳作用。第七，GEOROC 引用表比岩石数据快照早 18 个月，当前有 86 个已使用 Citation-ID 未在本地书目表中解析，虽不影响 ID 分组，但需在最终补充材料中补齐。

下一阶段优先为每类补充至少三个独立省域/研究，尤其扩展 Baikal、Rhine Graben、其他大陆裂谷、多个岛弧段、热点岛链和不同扩张速率洋脊，并补齐西南印度洋脊数据的论文关联。随后开展 leave-one-province-out 与省域聚类不确定性分析。机制层面只在数据覆盖允许时引入 Sr–Nd–Pb–Hf 同位素、年龄、分异与蚀变控制，构建“构造位置—裂谷/弧后阶段—源区亲和性”的层级输出。最终补充材料将发布样品级纳排理由、引文组、折分配、折外概率、省域误判表和所有外部源文件哈希。

## 6 阶段性结论

GEOROC 三分类与 PetDB 四分类共同表明，ARC、宽义 CIB、OIB 和 MORB 之间存在可由多元素机器学习量化的地球化学差异，但性能强烈依赖验证设计。GEOROC 随机森林 macro-F1 从随机分层的 0.861 降至地区留出的 0.654；PetDB 四分类从 0.915 降至保守引文分组的 0.760 和地区留出的 0.648。统一 18 元素四类外测中，随机森林对 92 个样品取得 0.837 的准确率、0.866 的平衡准确率和 0.807 的 macro-F1；ARC、MORB 和 OIB 召回率均为 1.000，但 CIB 仅为 0.464。结合东非裂谷的稳定 OIB 亲和性、Rio Grande Rift 的 ARC 亲和性和 Vate Trough 的 MORB 主导预测，系统性误判揭示了构造位置标签与源区继承、古俯冲改造、地幔柱作用和裂谷/弧后阶段之间的连续体。TiO₂、Sr、K₂O、Nb 和 Y 构成主要判别组合；逻辑回归提供可审查基线，随机森林提供必要而有限的非线性。可信应用必须同时控制文献、地区、结构性缺列、概率校准、数据库转移、标签本体和省域独立性，并把大陆裂谷与弧后盆地作为层级机制问题而非强制硬分类。

## 图件清单

- 图1  不同交叉验证策略的基线性能：`figures/baseline_cv_performance.pdf`
- 图2  Citation-ID 集合分组随机森林混淆矩阵：`figures/baseline_confusion_citation.pdf`
- 图3  特征值与缺失模式敏感性：`figures/feature_bias_sensitivity.pdf`
- 图4  Citation-ID 集合分组留出置换重要性：`figures/permutation_importance.pdf`
- 图5  分类全局 SHAP 重要性：`figures/shap_global_heatmap.pdf`
- 图6  log10 元素值与分类 SHAP 的 Spearman 方向：`figures/shap_direction_heatmap.pdf`
- 图7  PetDB 四分类不同验证策略性能：`figures/petdb_primary4_validation_performance.pdf`
- 图8  PetDB 引文与地区分组混淆矩阵：`figures/petdb_primary4_grouped_confusion.pdf`
- 图9  PetDB 保守引文分组概率可靠性曲线：`figures/petdb_primary4_calibration.pdf`
- 图10 PetDB 保守引文分组置换重要性：`figures/petdb_primary4_permutation_importance.pdf`
- 图11 PetDB 四分类全局 SHAP 重要性：`figures/petdb_primary4_shap_global_heatmap.pdf`
- 图12 PetDB 四分类 SHAP 方向：`figures/petdb_primary4_shap_direction_heatmap.pdf`
- 图13 GEOROC↔PetDB 三分类转移：`figures/cross_database_transfer.pdf`
- 图14 PANGAEA MORB 试验性外测逐样品概率：`figures/pangaea_morb_external_probabilities.pdf`
- 图15 PetDB 四分类地点风险与定向误判审计：`figures/petdb_primary4_error_audit.pdf`
- 图16 东非裂谷整省留出概率亲和性与元素域偏移：`figures/east_african_rift_transition.pdf`
- 图17 西南印度洋脊全岩 MORB 外测与亏损 MORB–弧判别歧义：`figures/figshare_morb_external_validation.pdf`
- 图18 四类外部转移混淆矩阵、逐类召回率与 Vate Trough 弧后压力测试：`figures/multiclass_external_validation.pdf`

## 数据与代码可用性

原始 GEOROC 分卷、PetDB 四类导出压缩包与条款文本、官方/重算哈希、服务端异常证据、处理数据、标签映射、程序、折外预测、模型指标、解释性结果和可执行笔记本均保存于项目工作区。数据契约程序校验原始哈希、行级链接、队列计数、结构性缺列处理和折外指标。PetDB 条款文本与浏览器许可标签曾存在不一致，项目保守按 CC BY-SA 4.0 处理，并将在补充材料中展开所有贡献文献。

Figshare–Mendeley MORB 扩展队列保留原始 XLSX、API 元数据、双哈希、工作表—行号映射、逐样品来源层级、PetDB 样品名重叠审计、18 元素处理表和逐模型概率。东太平洋海隆二手数据与西南印度洋脊暂定原创数据在所有结果表中保持分层，不合并为严格独立证据。

ARC/CIB/OIB 外测分别保留 PANGAEA.922011 原始 TSV、Rio Grande Rift 出版商 Table S1 XLSX 和 Mauna Loa 2022 Springer Table S1 XLSX，均固定 SHA-256。处理表保存材料筛选、SiO₂ 门控、Fe 价态统一、论文/样品去重、逐样品域检查和四类概率；Vate Trough 过渡队列始终与计分外测分离。

## 参考文献（当前核心条目）

Breiman, L. (2001). Random forests. *Machine Learning*, 45, 5–32. https://doi.org/10.1023/A:1010933404324

Brounce, M., Scoggins, S., Fischer, T. P., Ford, H., & Byrnes, J. (2024). Volatiles and redox along the East African Rift. *Geochemistry, Geophysics, Geosystems*, 25, e2024GC011657. https://doi.org/10.1029/2024GC011657

Cancel Vazquez, S. M., Rooney, T. O., Brown, E. L., Bollinger, A., Bastow, I. D., Steiner, R. A., & Kappelman, J. (2024). Basaltic pulses and lithospheric thinning—Plio-Pleistocene magmatism and rifting in the Turkana Depression (East African Rift System). *Journal of Geophysical Research: Solid Earth*, 129, e2024JB029166. https://doi.org/10.1029/2024JB029166

Furman, T. (2007). Geochemistry of East African Rift basalts: An overview. *Journal of African Earth Sciences*, 48, 147–160. https://doi.org/10.1016/j.jafrearsci.2006.06.009

Almeev, R., Holtz, F., Koepke, J., Haase, K. M., & Devey, C. W. (2009). Geochemical composition of basaltic glasses and glass inclusions from the Mid Atlantic Ridge near Ascension Island [Data set]. PANGAEA. https://doi.org/10.1594/PANGAEA.727391

Hertogen, J. G. H., Janssens, M.-J., & Palme, H. (1980). Major and minor element compositions of natural MORB glasses from DSDP Hole 24-238 [Data set]. PANGAEA. https://doi.org/10.1594/PANGAEA.707361

Haase, K. M., Gress, M. U., Lima, S. M., Regelous, M., Beier, C., Romer, R. L., & Bellon, H. (2020). Evolution of magmatism in the New Hebrides Island Arc and in initial back-arc rifting, SW Pacific. *Geochemistry, Geophysics, Geosystems*, 21, e2020GC008946. https://doi.org/10.1029/2020GC008946

Haase, K. M., et al. (2020). Elemental composition from XRF and ICP-MS measurements of volcanic rocks of the Vate Trough and New Hebrides Island arc [Data set]. PANGAEA. https://doi.org/10.1594/PANGAEA.922011

Zhu, C. (2024). Supplementary Tables S1 to S6.xlsx [Data set]. Figshare. https://doi.org/10.6084/m9.figshare.25295671.v1

Zhu, C. (2025). Cadmium isotope compositions of basalts from the East Pacific Rise and Southwest Indian Ridge: Implications for magma differentiation and mantle heterogeneity [Data set]. Mendeley Data. https://doi.org/10.17632/fntfnf92tg.1

Lehnert, K. A., Su, Y., Langmuir, C. H., Sarbas, B., & Nohl, U. (2000). A global geochemical database structure for rocks. *Geochemistry, Geophysics, Geosystems*, 1(5). https://doi.org/10.1029/1999GC000026

Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting model predictions. *Advances in Neural Information Processing Systems*, 30.

Petrelli, M., & Perugini, D. (2016). Solving petrological problems through machine learning: the study case of tectonic discrimination using geochemical and isotopic data. *Contributions to Mineralogy and Petrology*, 171, 81. https://doi.org/10.1007/s00410-016-1292-2

Petrelli, M., Caricchi, L., & Perugini, D. (2024). Machine learning in petrology: state-of-the-art and future perspectives. *Journal of Petrology*, 65(5), egae036. https://doi.org/10.1093/petrology/egae036

Rooney, T. O., Brown, E. L., Bastow, I. D., Arrowsmith, J. R., & Campisano, C. J. (2023). Magmatism during the continent–ocean transition. *Earth and Planetary Science Letters*, 614, 118189. https://doi.org/10.1016/j.epsl.2023.118189

Rowe, M. C., Lassiter, J. C., & Goff, K. (2015). Basalt volatile fluctuations during continental rifting: An example from the Rio Grande Rift, USA. *Geochemistry, Geophysics, Geosystems*, 16, 1254–1273. https://doi.org/10.1002/2014GC005649

Rhoads, E. A., Kutyrev, A., Bindeman, I. N., et al. (2025). Rhenium-osmium and oxygen isotope homogeneity during the 2022 Mauna Loa eruption and implications for basaltic magma storage. *Bulletin of Volcanology*, 87, 38. https://doi.org/10.1007/s00445-025-01825-0

Ueki, K., Hino, H., & Kuwatani, T. (2018). Geochemical discrimination and characteristics of magmatic tectonic settings: a machine-learning-based approach. *Geochemistry, Geophysics, Geosystems*, 19. https://doi.org/10.1029/2017GC007401

Vermeesch, P. (2006a). Tectonic discrimination diagrams revisited. *Geochemistry, Geophysics, Geosystems*, 7, Q06017. https://doi.org/10.1029/2005GC001092

Vermeesch, P. (2006b). Tectonic discrimination of basalts with classification trees. *Geochimica et Cosmochimica Acta*, 70, 1839–1848. https://doi.org/10.1016/j.gca.2005.12.016

GEOROC Compilation: Rock Types. DOI: https://doi.org/10.25625/2JETOA

GEOROC Compilation: Sample Metadata. DOI: https://doi.org/10.25625/4EZ7ID
