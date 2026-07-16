from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "02_label_qc_and_baseline.ipynb"

nb = nbf.v4.new_notebook()
nb["metadata"]["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
nb["metadata"]["language_info"] = {"name": "python", "version": "3.12"}
nb["cells"] = [
    nbf.v4.new_markdown_cell(
        """# 标签质量控制与防泄漏基线

## tl;dr

GEOROC 高置信三分类数据经岩性、材料、主量总量、引用冲突和特征覆盖筛选后，有 **54,012** 条记录可进入宽特征基线。随机森林在随机分层验证中的准确率为 **87.4%**，按 Citation-ID 集合分组后为 **81.5%**，按一级地区外推为 **73.9%**。OIB 的地区外推召回率仅 **37.3%**，说明随机逐样品切分明显高估跨地区泛化能力。"""
    ),
    nbf.v4.new_markdown_cell(
        """## Context & Methods

本 notebook 记录标签本体、处理队列和首个未调参基线。模型不是最终结论；其作用是验证数据管线、量化验证策略差异，并为后续特征解释与外部验证确定主协议。

### Key Assumptions

- GEOROC `CONVERGENT MARGIN` 暂作为宽泛 ARC，而不宣称全部为岛弧玄武岩。
- `INTRAPLATE VOLCANICS`、`CONTINENTAL FLOOD BASALT` 和 `RIFT VOLCANICS` 合并为宽泛 CIB，同时保留细标签。
- `SUBMARINE RIDGE` 不等于 MORB；最终四分类需由 PetDB 或作者基准数据提供明确 MORB。
- 论文主性能不能采用随机逐记录切分；文献分组和地区外推是更可信的估计。"""
    ),
    nbf.v4.new_code_cell(
        """from pathlib import Path
import json
import pandas as pd
from IPython.display import Image, display

cwd = Path.cwd().resolve()
ROOT = cwd if (cwd / 'src' / 'build_model_dataset.py').exists() else cwd.parent
assert (ROOT / 'data' / 'processed' / 'georoc_primary3_v0_1.parquet').exists()
ROOT"""
    ),
    nbf.v4.new_markdown_cell("## Data\n\n读取已通过固定输入哈希和数据契约检查的处理数据与模型输出。"),
    nbf.v4.new_code_cell(
        """profile = json.loads((ROOT / 'reports' / 'data_quality' / 'primary3_processing_profile.json').read_text(encoding='utf-8'))
class_counts = pd.read_csv(ROOT / 'reports' / 'data_quality' / 'primary3_class_counts.csv')
coverage = pd.read_csv(ROOT / 'reports' / 'data_quality' / 'primary3_feature_coverage.csv')
overall = pd.read_csv(ROOT / 'reports' / 'modeling' / 'baseline_overall_metrics.csv')
per_class = pd.read_csv(ROOT / 'reports' / 'modeling' / 'baseline_per_class_metrics.csv')
pd.Series(profile, name='value').to_frame()"""
    ),
    nbf.v4.new_code_cell("class_counts"),
    nbf.v4.new_markdown_cell("## Results"),
    nbf.v4.new_code_cell(
        """overall.sort_values(['cv_strategy', 'macro_f1'], ascending=[True, False]).round(3)"""
    ),
    nbf.v4.new_code_cell(
        """display(Image(filename=str(ROOT / 'figures' / 'baseline_cv_performance.png')))"""
    ),
    nbf.v4.new_code_cell(
        """citation_rf = per_class[(per_class.cv_strategy == 'citation_set_grouped') & (per_class.model == 'random_forest_balanced')]
location_rf = per_class[(per_class.cv_strategy == 'location_root_grouped') & (per_class.model == 'random_forest_balanced')]
pd.concat([citation_rf, location_rf]).round(3)"""
    ),
    nbf.v4.new_code_cell(
        """display(Image(filename=str(ROOT / 'figures' / 'baseline_confusion_citation.png')))"""
    ),
    nbf.v4.new_markdown_cell(
        """## Takeaways

1. 随机森林明显优于逻辑回归，但优势会随验证难度增加而缩小；当前不能据此宣称已解决跨地区识别。
2. 随机分层比文献分组高约6个百分点，比地区外推高约13.5个百分点，证实空间与文献相关性是主要泄漏风险。
3. CIB 最容易识别；OIB 的地区外推最不稳定，可能反映 OIB 端元多样性、地区偏差和与部分板内玄武岩的连续过渡。
4. `citation_overlap_component` 是极保守压力测试；一个重叠引用组件包含近1.2万条记录，说明 GEOROC 多来源合并会形成复杂依赖。
5. 下一步应比较宽特征与不活动元素模型，并在 PetDB MORB 数据加入后重新进行四分类分组验证。"""
    ),
]

OUT.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, OUT)
print(OUT)

