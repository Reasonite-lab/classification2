# 研究论文（中文审阅翻译版）

# 地球化学记忆限制了机器学习对玄武岩构造环境的判别

**短标题：** 玄武岩判别中的地球化学记忆

**徐邦坤（英文署名及作者排序待确认）**^1,*

^1 **[院系、单位、城市、邮编和国家待确认]**

*通讯作者。电子邮箱：xubangkun439@gmail.com；电话及传真：**[待确认]**。

**文章类型：** Research Article

**推广短句（约30词英文对应内容）：** 本研究表明，玄武岩分类器可以跨越端元构造环境迁移，但当继承的俯冲或地幔柱信号使岩石化学与现今构造位置解耦时，会发生系统性失效。

> 本文件是英文 NSR 投稿初稿的逐节中文翻译，供科学内容审阅使用；正式投稿应使用英文版。

## 摘要

玄武岩成分既记录构造环境，也保留地幔和地壳过程的继承信息，因此构造判别的关键是外推而非插值。我们依据可审计的岩性、铁价态和来源合同，汇编了54 012件GEOROC和6148件PetDB全岩玄武岩。随机森林在随机拆分下的macro-F1为0.915；当完整留出引文重叠组分和地点时，分别降至0.760和0.648。使用18个共同元素建立的模型，在四类各含2个省域代理、共231件独立来源样品的外部测试中获得0.822的平衡准确率和0.764的macro-F1。MORB和ARC召回率分别为1.000和0.967，而CIB仅为0.520。误判具有明确的地质结构：继承的俯冲信号使Big Pine裂谷玄武岩偏向ARC，东非裂谷玄武岩则偏向OIB。因此，可信的构造判别应是对地球化学亲和性和记忆的含不确定性诊断，而不是普适的硬标签。

**关键词：** 玄武岩；构造环境判别；地球化学；机器学习；外部验证；地幔源区

## 引言

全岩玄武岩地球化学长期被用于推断构造环境。经典判别图把特定主量和微量元素压缩到低维边界中，以区分洋中脊、火山弧和板内岩浆作用[1,2]。其地质依据十分清楚：不相容元素富集、流体活动元素转移、部分熔融和分离结晶会形成相互关联的多元素信号。但这些图解也存在重要的统计局限。由有限参考数据推导的边界在新地区可能失效，而且地球化学数据具有成分数据特征，分析覆盖不均，并按文献和省域强烈聚集[3]。分类树及后来的多变量模型利用了更多信息，却没有消除这些采样和迁移问题[4]。

全球数据库使更严格的检验成为可能。GEOROC和PetDB整合了来自不同实验室、年代和研究目的的测量结果[5]。它们的规模足以支持机器学习，但同一篇论文、一次喷发或同一省域内的样品并不统计独立。若逐记录随机拆分，近缘岩石可能同时进入训练集和测试集，模型得到的是在熟悉研究内部插值的奖励，而不是向新地质系统迁移的能力。测量覆盖也是潜在捷径：若某一类别系统性缺少某个元素，模型可能学会“哪些元素被测量”，而不是岩石成分本身。

已有研究表明，多元素和同位素数据可以用于机器学习构造判别[6,7]；近期综述则强调了可解释模型在岩石学中的作用[8]。尚未解决的地质问题不是一个灵活分类器能否拟合汇编数据库，而是判别依据能否经受文献、省域和数据库转移，以及失效是否对应可识别的岩石成因过程。因此，本研究采用类别平衡逻辑回归作为可审计的线性基线，以类别平衡随机森林提供有限的非线性扩展[9]。置换重要性与SHAP值用于互补描述模型依赖[10]，而不被解释为某个元素对预测的唯一因果作用。

本研究建立一套以来源审计为起点的全岩玄武岩构造判别流程。我们定量评估随机拆分的乐观偏差，检验结构性缺失，比较GEOROC与PetDB之间的双向迁移，并构建包含ARC、CIB、MORB和OIB四类、每类2个省域代理的外部测试。随后，我们把系统性误判当作地质观测本身来分析。核心结果是：端元构造环境的迁移表现良好，而大陆裂谷和弧后过渡环境暴露了现今构造位置与继承地球化学记忆之间的不一致。

## 结果与讨论

### 数据合同把样品化学与数据库结构分离

GEOROC 2026年6月Rock Types版本包含109 882条标记为BASALT的记录[11]。经过岩性、材料、SiO2、主量总和、引文和标签一致性筛选后，保留54 012条宽特征记录，其中ARC 18 083条、宽义CIB 30 086条、OIB 5843条。这里的宽义CIB合并了板内火山岩、大陆溢流玄武岩和裂谷火山岩；原始细标签仍被保留，使这一标签本体可以被审计而不是隐藏。

我们分别从PetDB导出扩张中心、火山弧、洋岛和大陆裂谷四组全岩玄武岩[12]。删除重复分析、按样品中位数聚合、统一铁价态并保守关联标签后，得到6148件样品：MORB 3650件、ARC 754件、OIB 1281件、CIB 463件。PetDB主模型使用在四类中均被观测的21个化学变量。由于PetDB OIB导出的Th整列缺失，主分析删除Th。22变量模型仅作为诊断，用于证明结构性缺失可能形成误导性的类别线索。地理名称、样品名、文献、文件名、分析方法和所有标签字段均禁止进入模型。

### 随机拆分显著高估迁移性能

在GEOROC上，类别平衡随机森林在分层随机拆分下的macro-F1为0.861；完整留出相互重叠的Citation-ID连通组分时为0.747；留出一级地点时降至0.654。PetDB四分类中的下降更明显：随机拆分为0.915，保守的引文重叠分组为0.760，地点分组为0.648（图1）。类别平衡逻辑回归相应从0.822降至0.718和0.646。因此，在空间外推条件下，随机森林相对于逻辑回归的大部分表观优势消失。额外的算法灵活性有助于熟悉域内的拟合，却不能解决省域转移。

[此处插入图1]

CIB控制了相当一部分性能下降。在PetDB地点留出测试中，CIB召回率仅为0.127，尽管其随机拆分表现很高。这一结果不能仅用类别不平衡解释。大陆裂谷玄武岩跨越岩石圈地幔、软流圈和地幔柱贡献，具有不同伸展程度、继承的俯冲改造和地壳相互作用。因此，一个构造位置标签覆盖了多种成分状态。

结构性缺失可以被模型识别，但不是分组性能的主要来源。在共同包含43 005条记录的GEOROC子集中，仅使用缺失指示的随机森林在引文分组下macro-F1为0.289，在地点分组下为0.258；把显式缺失指示加入元素值并未改善结果。相比之下，PetDB全22变量诊断把OIB中整列缺失的Th当作强类别信息，随后将一个元素完整的外部OIB时间序列推向CIB。最终外部合同因此删除Th、V、Cr和Ni，并要求18个共同元素均为正值且完整。

### 模型重要性具有地质一致性，但不是唯一解释

[此处插入图2]

在保守引文重叠验证下，置换重要性把TiO2排在首位（平衡准确率平均下降0.0918），随后为Sr（0.0633）、K2O（0.0305）、Zr（0.0253）、MnO（0.0212）、Nb（0.0211）和Y（0.0205；图2）。平均绝对SHAP重要性也把Sr、TiO2、K2O、Nb和Y列为主要变量。两种方法的一致性具有意义，因为它们回答不同问题：置换重要性衡量在留出组上的预测损失，SHAP则分配已拟合预测的贡献。

这些元素构成对过程敏感的组合，而不是彼此独立的构造示踪剂。TiO2、Nb、Zr和Y对源区富集与熔融程度敏感；Sr和K2O还记录板片输入、斜长石行为、岩浆分异和地壳相互作用[13]。它们的联合重要性与岩石学机制一致，但仅凭全岩化学无法唯一拆分这些过程。因此，模型解释应被视为关于源区和分异的待检验假说，并结合同位素、矿物和年龄信息验证。

概率质量同样依赖验证设计。在保守PetDB拆分中，随机森林多分类Brier分数为0.236，最高概率类别的期望校准误差为0.075[14]。这些值足以比较相对亲和性，却不足以把概率直接当作严格后验概率。因此，0.8的模型概率应表述为与某训练类别高度相似，而不是80%的地质真值。

### 跨数据库迁移揭示标签本体与采样的不对称

即使两个模型都使用相同的21个变量，且目标数据库从不参与拟合，GEOROC与PetDB之间的ARC-CIB-OIB三分类迁移仍明显不对称（图3）。在宽义CIB本体下，GEOROC→PetDB的macro-F1为0.804，而PetDB→GEOROC仅为0.484。把GEOROC CIB限制为裂谷火山岩后，后者提高到0.619，但不对称仍然存在；相应的GEOROC→PetDB值为0.788。

[此处插入图3]

这种方向性与数据覆盖相符。GEOROC提供更大、更宽的源域，而PetDB CIB仅来自大陆裂谷检索。标签本体对齐删除了部分溢流玄武岩和一般板内岩浆记录，因而改善迁移；但残余差距表明，相同类别名称和相同特征列并不保证地质总体可交换。

### 对称外部测试区分端元环境与大陆裂谷记忆

严格外部测试包含231件样品，每一类均有2个省域代理（表1）。外部文献和完全匹配的样品名均未出现在PetDB训练表中。18元素类别平衡随机森林获得0.745的准确率、0.822的平衡准确率和0.764的macro-F1；逻辑回归基线分别为0.442、0.576和0.486（图4）。这一非线性增益足以支持随机森林作为主模型，但模型复杂度仍显著低于深度学习或大型集成，并保留直接的置换和SHAP审计能力。

[此处插入表1]

| 类别 | 省域代理 | 来源 | n | 随机森林召回率 |
|---|---|---|---:|---:|
| ARC | 南新赫布里底弧前缘 | PANGAEA.922011 | 12 | 1.000 |
| ARC | 哥斯达黎加火山前缘 | GEOROC 2JETOA | 18 | 0.944 |
| CIB | Rio Grande Rift/Jemez Lineament | Rowe等Table S1 | 28 | 0.464 |
| CIB | Big Pine火山场 | GEOROC 2JETOA | 70 | 0.543 |
| MORB | 西南印度洋中脊 | Figshare 25295671 | 36 | 1.000 |
| MORB | 南大西洋中脊18.0-20.6°S | Zhong等Table S1 | 12 | 1.000 |
| OIB | Mauna Loa 2022年喷发 | Rhoads等Table S1 | 16 | 1.000 |
| OIB | La Palma 2021年喷发 | Day等Table S1 | 39 | 0.718 |

**表1.** 严格外部全岩玄武岩队列。每类2个省域代理只达到最小对称设计，仍不足以估计稳定的省域聚类置信区间。西南印度洋中脊仓储队列仍属暂定，因为尚未确定关联期刊论文；第二个MORB队列则来自出版商原始补充表并有明确论文关联。

[此处插入图4]

两个洋中脊省域的MORB召回率均为1.000；南新赫布里底和哥斯达黎加的ARC召回率分别为1.000和0.944。OIB迁移受喷发状态影响：Mauna Loa为1.000，La Palma为0.718。在La Palma，随机森林对第1-20天分异粗面玄武岩的召回率为0.421，而第20天之后较原始的碱玄岩为1.000。喷发期间构造标签并未改变；性能变化说明分异和补给可以使样品在分类器亲和空间内移动[21]。

CIB始终是最弱类别。Rio Grande Rift召回率为0.464，Big Pine为0.543；Big Pine的大多数错误预测为ARC。这是地质失效，而不仅是数值失效。Rio Grande Rift岩浆作用保留了与早期Farallon板片俯冲有关的水化和交代岩石圈地幔信号[18]。Big Pine玄武岩需要非均一地幔源区，在单次喷发尺度上还受到不同程度的地壳混染和晶体载荷影响[19,20]。以现今构造类别训练的分类器因此会捕捉继承的板片或地壳信号，并把样品分配到ARC成分域。

### 过渡省域应报告亲和性，而不是强制硬标签

预测之前，我们把整个东非裂谷省从训练中删除。347件数据库标签为CIB的样品中，随机森林给出平均OIB概率0.609、平均CIB概率0.075，并把307件预测为OIB。这一明显偏移与大陆裂解过程中地幔柱影响软流圈和富集岩石圈的不同贡献相容[15-17]，但不能证明数据库标签错误。它说明构造位置标签与源区亲和信号可以合理地分离（图5）。

[此处插入图5]

第二个过渡测试使用Vate Trough初始弧后裂谷的21件样品。这些样品被有意排除在ARC或MORB计分之外。随机森林把76.2%预测为MORB、14.3%为ARC，CIB和OIB各占4.8%。MORB占优与弧后初始张开期间减压熔融增强、板片贡献减弱相容[22]。若把这些样品当作普通分类错误，就会丢失使其具有科学价值的过程信息。

这些外部失效共同定义了一个可解释的连续体。一部分大陆裂谷因岩石圈地幔保留俯冲记忆而偏向ARC；另一部分因地幔柱或富集软流圈贡献占优而偏向OIB。随着减压熔融增强，早期弧后盆地则偏向MORB。这一结构支持分层输出：现今构造位置、多元素源区亲和性、过渡阶段和超出训练域的程度。只有当额外算法复杂度能在省域留出检验中改善这些地质组成时，才值得引入。

## 结论

多元素机器学习能够提取可迁移的玄武岩信号，但其可信度更多由地质验证决定，而不是由随机拆分准确率决定。在PetDB中，随机森林macro-F1从随机拆分的0.915降至引文重叠分组的0.760和地点分组的0.648。在四类各含2个省域代理、共231件样品的严格外部测试中，18元素随机森林达到0.822的平衡准确率和0.764的macro-F1。它对MORB和ARC端元几乎完全迁移，但CIB召回率仅为0.520。

误判揭示了原因。玄武岩成分不仅记录现今构造环境，还记录源区富集、继承交代、熔融和分异。Big Pine和Rio Grande裂谷玄武岩可以保留ARC式记忆，东非裂谷玄武岩可以呈现OIB亲和性，Vate Trough弧后玄武岩可以接近MORB。TiO2、Sr、K2O、Nb和Y构成稳健的多元素判别基础，但任何重要性方法都不能使其成为唯一因果示踪剂。因此，合理产品不是普适构造标签，而是同时报告类别亲和性、域转移和地质过渡的可审计、不确定性约束诊断；逻辑回归提供透明基线，随机森林提供必要但有限的非线性。

## 材料与方法

我们固定了GEOROC Rock Types v2026-06[11]和四个PetDB 2.0全岩玄武岩导出包[12]，并保存源查询和校验哈希。入选记录须由材料和岩石名称字段确定为全岩火山玄武岩，SiO2为40-55 wt%，统一后的主量总和为85-105 wt%（严格外部队列为90-105 wt%），且构造标签唯一。总铁优先采用报告的FeOT，其次按0.8998 × Fe2O3T或FeO + 0.8998 × Fe2O3计算。删除重复分析后，其余分析按样品取中位数。

正浓度进行log10变换，缺失值仅在各训练折内用中位数填补。PetDB内部模型使用四类均有观测的21个变量。外部验证使用所有严格队列共同具备的18个变量：SiO2、TiO2、Al2O3、FeOT、CaO、MgO、MnO、K2O、Na2O、P2O5、Rb、Sr、Y、Zr、Nb、La、Ce和Nd。主模型不输入元数据或缺失指示。

类别平衡逻辑回归和类别平衡随机森林分别接受五折分层随机、引文集合分组、引文重叠连通组分分组和地点分组验证。所有指标由折外预测计算。置换重要性在引文重叠留出折上计算；描述性SHAP值来自拟合的160棵树随机森林。校准使用多分类Brier分数和10箱最高类别期望校准误差，不对测试集进行后校准。

外部队列在预测前确定，要求18个共同元素完整且为正值，并与PetDB的文献和完全匹配样品名进行重叠审计。231件计分样品每类包含2个省域代理。东非裂谷和Vate Trough作为整省或机制压力测试单独分析。项目档案保留代码、折分配、预测表、哈希及详细纳入/排除审计。

## 数据与软件可用性

GEOROC源数据见DOI 10.25625/2JETOA，样品元数据见DOI 10.25625/4EZ7ID。PetDB源查询和EarthChem导出归档均保存校验哈希。每个外部队列的出版商或仓储来源列于表1和补充数据。所有处理程序、标签映射、折分配、折外概率、外部预测、清单和文件哈希均已固定在项目工作区中。投稿前将补充可复现包的公共归档DOI；本稿任何结论均不依赖不可获得的拟合模型对象。

## 补充数据

补充数据将包括完整数据字典、来源清单、排除账本、标签本体、分类型缺失矩阵、分组折分配、全部指标、校准表、置换与SHAP输出、外部样品审计、省域预测表及所有源文件校验哈希。

## 致谢

感谢向GEOROC、PetDB、PANGAEA、Figshare、Mendeley Data及所引出版商补充材料贡献测量和开展数据整理的研究人员。**[其他个人或机构致谢待确认。]**

## 经费

**[投稿前确认经费声明和项目编号。]**

## 作者贡献

徐邦坤：研究构思、方法、调查、数据整理、正式分析、可视化、初稿撰写、审阅与修改。**[如有其他作者，姓名及CRediT角色须在投稿前确认。]**

## 利益冲突

利益冲突声明：无。

## 参考文献

1. Pearce JA and Cann JR. Tectonic setting of basic volcanic rocks determined using trace element analyses. *Earth Planet Sci Lett* 1973; 19: 290-300.

2. Wood DA. The application of a Th-Hf-Ta diagram to problems of tectonomagmatic classification and to establishing the nature of crustal contamination of basaltic lavas. *Earth Planet Sci Lett* 1980; 50: 11-30.

3. Vermeesch P. Tectonic discrimination diagrams revisited. *Geochem Geophys Geosyst* 2006; 7: Q06017.

4. Vermeesch P. Tectonic discrimination of basalts with classification trees. *Geochim Cosmochim Acta* 2006; 70: 1839-48.

5. Lehnert KA, Su Y and Langmuir CH et al. A global geochemical database structure for rocks. *Geochem Geophys Geosyst* 2000; 1: 1012.

6. Petrelli M and Perugini D. Solving petrological problems through machine learning: the study case of tectonic discrimination using geochemical and isotopic data. *Contrib Mineral Petrol* 2016; 171: 81.

7. Ueki K, Hino H and Kuwatani T. Geochemical discrimination and characteristics of magmatic tectonic settings: a machine-learning-based approach. *Geochem Geophys Geosyst* 2018; 19: 1327-47.

8. Petrelli M, Caricchi L and Perugini D. Machine learning in petrology: state-of-the-art and future perspectives. *J Petrol* 2024; 65: egae036.

9. Breiman L. Random forests. *Mach Learn* 2001; 45: 5-32.

10. Lundberg SM and Lee S-I. A unified approach to interpreting model predictions. *Adv Neural Inf Process Syst* 2017; 30: 4765-74.

11. GEOROC Data Group. GEOROC compilation: Rock Types, version 2026-06. 2026, doi: 10.25625/2JETOA.

12. EarthChem. PetDB 2.0 whole-rock basalt query exports for spreading centre, volcanic arc, ocean island and continental rift settings. 2026. https://www.earthchem.org/petdb (14 July 2026, date last accessed).

13. Sun S-S and McDonough WF. Chemical and isotopic systematics of oceanic basalts: implications for mantle composition and processes. In: Saunders AD and Norry MJ (eds). *Magmatism in the Ocean Basins*. London: Geological Society, 1989, 313-45.

14. Brier GW. Verification of forecasts expressed in terms of probability. *Mon Weather Rev* 1950; 78: 1-3.

15. Furman T. Geochemistry of East African Rift basalts: an overview. *J Afr Earth Sci* 2007; 48: 147-60.

16. Rooney TO, Brown EL and Bastow ID et al. Magmatism during the continent-ocean transition. *Earth Planet Sci Lett* 2023; 614: 118189.

17. Brounce M, Scoggins S and Fischer TP et al. Volatiles and redox along the East African Rift. *Geochem Geophys Geosyst* 2024; 25: e2024GC011657.

18. Rowe MC, Lassiter JC and Goff K. Basalt volatile fluctuations during continental rifting: an example from the Rio Grande Rift, USA. *Geochem Geophys Geosyst* 2015; 16: 1254-73.

19. Blondes MS, Reiners PW and Ducea MN et al. Temporal-compositional trends over short and long time-scales in basalts of the Big Pine Volcanic Field, California. *Earth Planet Sci Lett* 2008; 269: 140-54.

20. Gao R, Lassiter JC and Ramirez G. Origin of temporal compositional trends in monogenetic vent eruptions: insights from the crystal cargo in the Papoose Canyon sequence, Big Pine Volcanic Field, CA. *Earth Planet Sci Lett* 2017; 457: 227-37.

21. Day JMD, Troll VR and Aulinas M et al. Mantle source characteristics and magmatic processes during the 2021 La Palma eruption. *Earth Planet Sci Lett* 2022; 597: 117793.

22. Haase KM, Gress MU and Lima SM et al. Evolution of magmatism in the New Hebrides Island Arc and in initial back-arc rifting, SW Pacific. *Geochem Geophys Geosyst* 2020; 21: e2020GC008946.

23. Zhong Y, Liu W and Sun Z et al. Geochemistry and mineralogy of basalts from the South Mid-Atlantic Ridge (18.0-20.6°S): evidence of a heterogeneous mantle source. *Minerals* 2019; 9: 659.

24. Rhoads EA, Kutyrev A and Bindeman IN et al. Rhenium-osmium and oxygen isotope homogeneity during the 2022 Mauna Loa eruption and implications for basaltic magma storage. *Bull Volcanol* 2025; 87: 38.

25. Zhu C. Supplementary Tables S1 to S6.xlsx. Figshare. 2024, doi: 10.6084/m9.figshare.25295671.v1.

## 图注

**图1. PetDB四分类任务的分组验证。** 五折折外预测下，比较随机、引文集合、引文重叠连通组分和一级地点拆分的macro-F1与平衡准确率。21变量面板仅包含PetDB四类均有观测的元素。

**替代文本：** 成对柱状图显示随机拆分下随机森林和逻辑回归性能最高，引文分组后降低，地点留出时最低。

**图2. 保守置换重要性。** 在每个引文重叠留出折中置换变量后，计算平衡准确率的平均下降；误差线为折间标准差。重要性描述模型依赖，不代表唯一因果归因。

**替代文本：** 水平柱状图把TiO2和Sr列为最强变量，随后为K2O、Zr、MnO、Nb和Y，并显示五个留出折之间的不确定性。

**图3. 跨数据库双向迁移。** 使用相同21变量在GEOROC与PetDB间开展ARC-CIB-OIB三分类迁移。目标数据库从不参与拟合，并比较宽义和裂谷对齐的CIB本体。

**替代文本：** 成对柱状图显示GEOROC→PetDB强于PetDB→GEOROC；把大陆板内类别对齐为裂谷火山岩可以改善反向迁移，但不能消除差距。

**图4. 四类外部迁移与弧后压力测试。** 类别平衡逻辑回归和随机森林的行归一化混淆矩阵、逐类召回率及不计分Vate Trough预测比例。计分测试含231件样品，每类2个省域代理。

**替代文本：** 混淆矩阵和召回率柱状图显示ARC与MORB接近完全正确，OIB较强而CIB较弱；Vate Trough预测以MORB亲和性为主。

**图5. 东非裂谷作为裂谷—地幔柱地球化学过渡。** 训练时完整删除东非裂谷。A图给出各国四类平均概率；B图显示使数据库CIB队列向OIB训练域移动的元素中位数。

**替代文本：** 各国堆叠柱以OIB概率为主；水平排序显示SiO2、FeOT和Nb使东非裂谷CIB相对于其他CIB更接近OIB中位数。
