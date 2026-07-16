from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "01_georoc_initial_data_quality.ipynb"

nb = nbf.v4.new_notebook()
nb["metadata"]["kernelspec"] = {
    "display_name": "Python 3",
    "language": "python",
    "name": "python3",
}
nb["metadata"]["language_info"] = {"name": "python", "version": "3.12"}

nb["cells"] = [
    nbf.v4.new_markdown_cell(
        """# GEOROC 玄武岩原始数据首轮质量画像

## tl;dr

GEOROC 2026-06 的 9 个 `BASALT` 分卷包含 **109,882** 条记录。按“全岩＋火山岩＋名称含玄武岩＋SiO₂ 40–55 wt%”筛选后有 **72,180** 条严格候选。构造标签严重不均衡，且洋中脊覆盖不足，因此 GEOROC 适合作为汇聚边缘、板内和洪流玄武岩的主来源，但必须用 PetDB 补足 MORB，并采用按文献与地区分组的外推验证。"""
    ),
    nbf.v4.new_markdown_cell(
        """## Context & Methods

本 notebook 是原始数据质量审计的可执行伴随记录。它重新运行项目画像程序，检查固定版本文件，并读取生成的高信号汇总。

### Key Assumptions

- 分析单位是一条 GEOROC 全岩分析记录，而不是一个独立岩浆事件。
- `TECTONIC SETTING` 仅作为待审计的来源标签，尚未直接等同于最终论文类别。
- SiO₂ 40–55 wt% 是保守候选规则，不是最终岩石学分类的唯一依据。
- 缺失值和零值在后续阶段必须区分；本阶段不做统计填补。"""
    ),
    nbf.v4.new_code_cell(
        """from pathlib import Path
import json
import runpy
import pandas as pd
import matplotlib.pyplot as plt

cwd = Path.cwd().resolve()
ROOT = cwd if (cwd / 'src' / 'profile_georoc.py').exists() else cwd.parent
assert (ROOT / 'src' / 'profile_georoc.py').exists(), ROOT
ROOT"""
    ),
    nbf.v4.new_markdown_cell("## Data\n\n重新生成首轮画像，确保结果可由固定的 9 个官方分卷复现。"),
    nbf.v4.new_code_cell(
        """runpy.run_path(str(ROOT / 'src' / 'profile_georoc.py'), run_name='__main__')"""
    ),
    nbf.v4.new_code_cell(
        """quality_dir = ROOT / 'reports' / 'data_quality'
profile = json.loads((quality_dir / 'initial_profile.json').read_text(encoding='utf-8'))
manifest = pd.read_csv(quality_dir / 'raw_file_manifest.csv')
class_counts = pd.read_csv(quality_dir / 'tectonic_class_counts_raw.csv')
strict_counts = pd.read_csv(quality_dir / 'tectonic_class_counts_strict_candidate.csv')
coverage = pd.read_csv(quality_dir / 'field_coverage.csv')
pd.Series(profile, name='value').to_frame()"""
    ),
    nbf.v4.new_markdown_cell("## Results"),
    nbf.v4.new_code_cell(
        """assert manifest['rows'].sum() == profile['rows']
assert manifest['md5'].nunique() == 9
manifest"""
    ),
    nbf.v4.new_code_cell(
        """strict_counts.head(12)"""
    ),
    nbf.v4.new_code_cell(
        """plot_data = strict_counts.head(10).sort_values('n')
fig, ax = plt.subplots(figsize=(9, 5.5))
ax.barh(plot_data['tectonic_setting_raw'], plot_data['n'], color='#35618f')
ax.set_title('GEOROC strict basalt candidates by raw tectonic label')
ax.set_xlabel('Records')
ax.set_ylabel('')
ax.spines[['top', 'right']].set_visible(False)
fig.tight_layout()
plt.show()"""
    ),
    nbf.v4.new_code_cell(
        """core_fields = [
    'SIO2(WT%)','TIO2(WT%)','AL2O3(WT%)','FEOT(WT%)','CAO(WT%)','MGO(WT%)',
    'NB(PPM)','ZR(PPM)','Y(PPM)','TH(PPM)','HF(PPM)','TA(PPM)',
    'LA(PPM)','CE(PPM)','ND(PPM)','SM(PPM)','YB(PPM)','LU(PPM)'
]
coverage.loc[coverage['field'].isin(core_fields)].sort_values('non_null_pct', ascending=False)"""
    ),
    nbf.v4.new_markdown_cell(
        """## Takeaways

1. 原始文件完整且可追溯，9 个分卷行数与汇总一致。
2. 严格候选队列规模足以支持分组外推验证，但类别不平衡显著，不能只报告总体准确率。
3. GEOROC 的原始构造标签没有直接提供充分的 MORB 类；PetDB 补充是主数据设计的一部分。
4. 约一成记录缺少构造标签；这些记录不能进入监督训练，但可在模型定稿后作为未标记预测或异常检测样本。
5. 后续必须先审查重复、缺失机制、检出限与标签映射，再开展模型训练。"""
    ),
]

OUT.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, OUT)
print(OUT)

