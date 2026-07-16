from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANUSCRIPT = ROOT / "paper" / "nsr_manuscript_v1.md"
OUT = ROOT / "reports" / "manuscript" / "nsr_compliance_v1.json"


def word_count(value: str) -> int:
    return len(re.findall(r"\b[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*\b", value))


def between(text: str, start: str, end: str) -> str:
    return text.split(start, 1)[1].split(end, 1)[0]


def main() -> None:
    text = MANUSCRIPT.read_text(encoding="utf-8")
    title = re.search(
        r"^# (Geochemical memory[^\n]+)$", text, flags=re.MULTILINE
    ).group(1)
    abstract = between(text, "## ABSTRACT\n\n", "\n\n**Keywords")
    introduction_to_conclusion = between(
        text, "## INTRODUCTION\n\n", "\n\n## MATERIALS AND METHODS"
    )
    methods = between(
        text,
        "## MATERIALS AND METHODS\n\n",
        "\n\n## DATA AND SOFTWARE AVAILABILITY",
    )
    references = between(
        text, "## REFERENCES\n\n", "\n\n## FIGURE LEGENDS"
    )
    reference_numbers = [
        int(value) for value in re.findall(r"^(\d+)\.", references, re.MULTILINE)
    ]
    keyword_line = re.search(r"\*\*Keywords:\*\* ([^\n]+)", text).group(1)
    keywords = [item.strip() for item in keyword_line.split(";")]
    figure_cues = re.findall(r"\[INSERT FIGURE (\d+) HERE\]", text)
    table_cues = re.findall(r"\[INSERT TABLE (\d+) HERE\]", text)
    citation_numbers = [
        int(value)
        for marker in re.findall(r"\[([0-9, -]+)\]", introduction_to_conclusion)
        for value in re.findall(r"\d+", marker)
    ]
    report = {
        "manuscript": str(MANUSCRIPT.relative_to(ROOT)),
        "title": title,
        "title_characters": len(title),
        "abstract_words": word_count(abstract),
        "main_text_words_introduction_through_conclusion": word_count(
            introduction_to_conclusion
        ),
        "methods_words": word_count(methods),
        "keyword_count": len(keywords),
        "reference_count": len(reference_numbers),
        "reference_number_sequence_valid": reference_numbers
        == list(range(1, len(reference_numbers) + 1)),
        "citation_numbers_within_reference_list": bool(citation_numbers)
        and max(citation_numbers) <= len(reference_numbers),
        "figure_cues": figure_cues,
        "table_cues": table_cues,
        "display_item_count": len(set(figure_cues)) + len(set(table_cues)),
        "placeholder_count": text.count("[to be confirmed]")
        + text.count("[TO BE CONFIRMED]"),
        "limits": {
            "title_characters_max": 100,
            "abstract_words_max": 150,
            "main_text_words_max": 5000,
            "methods_words_max": 500,
            "keywords_max": 6,
            "references_max": 50,
            "display_items_max": 6,
        },
    }
    report["checks"] = {
        "title": report["title_characters"] <= 100,
        "abstract": report["abstract_words"] <= 150,
        "main_text": report[
            "main_text_words_introduction_through_conclusion"
        ]
        <= 5000,
        "methods": report["methods_words"] <= 500,
        "keywords": report["keyword_count"] <= 6,
        "references": report["reference_count"] <= 50,
        "display_items": report["display_item_count"] <= 6,
        "reference_sequence": report["reference_number_sequence_valid"],
        "citation_range": report["citation_numbers_within_reference_list"],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not all(report["checks"].values()):
        raise SystemExit("NSR manuscript compliance check failed")


if __name__ == "__main__":
    main()
