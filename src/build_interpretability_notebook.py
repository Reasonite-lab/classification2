from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "03_feature_bias_and_interpretability.ipynb"

nb = nbf.v4.new_notebook()
nb["metadata"]["kernelspec"] = {
    "display_name": "Python 3",
    "language": "python",
    "name": "python3",
}
nb["metadata"]["language_info"] = {"name": "python", "version": "3.12"}
nb["cells"] = [
    nbf.v4.new_markdown_cell(
        """# 特征偏倚敏感性与可解释性分析

## tl;dr

在完全相同的 **43,005** 条样品上，仅使用“元素是否被测量”的缺失模式无法获得强判别力；在宽特征值模型中显式加入缺失指示也没有改善结果。因此，当前性能不是主要由文献测试覆盖模式“作弊”获得的。

在按 Citation-ID 集合分组的留出数据上，置换重要性最高的特征是 TiO₂、Nb、Al₂O₃、CaO 和 Sr。这些特征的 SHAP 方向与弧岩富流体相容元素、低 HFSE，以及 OIB 相对富 Ti–Nb 的经验特征基本一致。所有 SHAP 方向都只是模型内关联，不是成因因果证据。"""
    ),
    nbf.v4.new_markdown_cell(
        """## Context & Methods

本笔记本回答两个问题：

1. 模型是否可能主要学到不同文献或类别的元素测试覆盖差异？
2. 在防止同一文献跨训练/验证后，哪些地球化学特征对判别最稳定？

偏倚敏感性比较宽特征值、宽特征值+缺失指示、不活动元素值和仅缺失指示四种输入。置换重要性在五折 Citation-ID 集合分组验证的留出集上计算；SHAP 用全队列描述性随机森林和分层抽样估计。

### Interpretation boundary

- 置换重要性会在相关元素之间分摊或遮蔽信号。
- SHAP 的正负方向描述模型关联，不能单独证明地幔源区、俯冲流体或地壳混染机制。
- 当前三分类不包含 MORB，因此结论只适用于宽义 ARC–CIB–OIB 区分。"""
    ),
    nbf.v4.new_code_cell(
        """from pathlib import Path
import json
import pandas as pd
from IPython.display import Image, display

cwd = Path.cwd().resolve()
ROOT = cwd if (cwd / 'reports' / 'modeling').exists() else cwd.parent
required = [
    ROOT / 'reports' / 'modeling' / 'feature_bias_sensitivity_summary.csv',
    ROOT / 'reports' / 'modeling' / 'permutation_importance_summary.csv',
    ROOT / 'reports' / 'modeling' / 'shap_global_overall.csv',
    ROOT / 'reports' / 'modeling' / 'shap_direction_spearman.csv',
]
for path in required:
    assert path.exists(), path
ROOT"""
    ),
    nbf.v4.new_markdown_cell("## Data"),
    nbf.v4.new_code_cell(
        """sensitivity = pd.read_csv(required[0])
permutation = pd.read_csv(required[1])
shap_overall = pd.read_csv(required[2])
shap_direction = pd.read_csv(required[3])
sens_manifest = json.loads((ROOT / 'reports' / 'modeling' / 'feature_bias_sensitivity_manifest.json').read_text(encoding='utf-8'))
interp_manifest = json.loads((ROOT / 'reports' / 'modeling' / 'interpretability_manifest.json').read_text(encoding='utf-8'))
pd.DataFrame({
    'artifact': ['bias sensitivity', 'interpretability'],
    'cohort_rows': [sens_manifest['common_cohort_rows'], interp_manifest['cohort_rows']],
    'scope': ['same complete immobile-ready cohort', interp_manifest['permutation_validation']],
})"""
    ),
    nbf.v4.new_markdown_cell("## Results"),
    nbf.v4.new_code_cell(
        """sensitivity.pivot(index='input_spec', columns='cv_strategy', values='macro_f1_mean').round(3)"""
    ),
    nbf.v4.new_code_cell(
        """display(Image(filename=str(ROOT / 'figures' / 'feature_bias_sensitivity.png')))"""
    ),
    nbf.v4.new_code_cell(
        """permutation.head(12).assign(
    importance_mean=lambda x: x.importance_mean.round(3),
    importance_sd_between_folds=lambda x: x.importance_sd_between_folds.round(3),
)"""
    ),
    nbf.v4.new_code_cell(
        """display(Image(filename=str(ROOT / 'figures' / 'permutation_importance.png')))"""
    ),
    nbf.v4.new_code_cell(
        """shap_overall.head(12).assign(
    mean_abs_shap_over_classes=lambda x: x.mean_abs_shap_over_classes.round(3)
)"""
    ),
    nbf.v4.new_code_cell(
        """display(Image(filename=str(ROOT / 'figures' / 'shap_global_heatmap.png')))
display(Image(filename=str(ROOT / 'figures' / 'shap_direction_heatmap.png')))"""
    ),
    nbf.v4.new_markdown_cell(
        """## Takeaways

1. **缺失模式是可见但非主导的偏倚。** 仅缺失模式在文献分组和地区分组验证中的 macro-F1 分别约为 0.289 和 0.258；将缺失指示加入宽特征值没有带来改善。
2. **宽特征比仅不活动元素更强，但更容易受分异、改造和分析差异影响。** 在文献分组/地区分组中，不活动元素模型的 macro-F1 分别低约 0.053/0.057。论文应同时报告两组结果。
3. **TiO₂ 和 Nb 是最稳定的区分轴。** 它们同时位于留出集置换重要性和全局 SHAP 的前列，与弧岩相对亏损 HFSE、OIB 相对富集的宽泛地球化学认识相容。
4. **Al₂O₃、CaO、MgO、SiO₂ 等主量元素贡献不可忽略。** 它们可能包含分离结晶、部分熔融程度、地壳混染与区域数据结构的混合信息，不应被简化为单一构造机制。
5. **当前结果可作为中期证据，不是最终论文结论。** 仍需引入具有明确洋中脊产状的 MORB，并用独立数据集做外部验证。"""
    ),
]

OUT.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, OUT)
print(OUT)
