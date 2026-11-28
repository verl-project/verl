#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate an OCR rule JSON from the doc-quality repo.

Reads doc-quality/skills/01.评估标准/common/h2_registry.json and the rule
markdown files it points at, then emits a ProjectRule JSON that
`ocr review --rule <file>` consumes. The rule targets Markdown/RST docs and
injects doc-quality's 3-dimension (正确性/完整性/易理解) checklist as the
review guidance, with a severity mapping table so LLM output maps to OCR's
critical/high/medium/low vocabulary.

Usage:
    python generate_rule.py <doc-quality-repo-path> <output-rule.json>
"""

import json
import os
import sys
from pathlib import Path

DIMENSION_WEIGHTS = {
    "正确性": 0.4,
    "完整性": 0.3,
    "易理解": 0.3,
}

SEVERITY_MAP = [
    "## 严重性映射（LLM 必须遵循）",
    "",
    "doc-quality 使用 F/S/N/H 四级，映射到 OCR 的 severity 字段：",
    "",
    "| doc-quality 级别 | OCR severity | 含义 |",
    "|---|---|---|",
    "| 致命 F           | critical     | 阻断核心任务或安全/合规风险 |",
    "| 严重 S           | high         | 不阻断但明显拖累任务 |",
    "| 一般 N           | medium       | 读者可感知的质量瑕疵 |",
    "| 提示 H           | low          | 无感知的规范微调 |",
    "",
    "定级方法：依次回答 F→S→N→H 四级判定问句，第一个「是」即为级别：",
    "- F：按文档操作，任务会必然失败或引发损失吗？",
    "- S：用户必须停下来额外处理（绕路/试错/查外部资料）才能继续吗？",
    "- N：读者会把现状感知为「质量问题」吗？",
    "- H：现状仅属「改了更规范」的打磨空间吗？",
    "",
    "所有发现必须设置 severity 为 critical/high/medium/low 之一，category 用 documentation。",
    "",
]


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def load_registry(repo: Path) -> dict:
    p = repo / "skills" / "01.评估标准" / "common" / "h2_registry.json"
    if not p.exists():
        raise FileNotFoundError(f"h2_registry.json not found at {p}")
    return json.loads(read_text(p))


def build_rule_text(repo: Path, registry: dict) -> str:
    rules_root = repo / "skills" / "01.评估标准"
    common_dir = rules_root / "common"

    parts = []

    parts.append("# 文档质量检测规则（来源：doc-quality 项目 H2 检测项）\n"
                 "\n"
                 "你是文档质量审查员。对当前 PR 中变更的文档逐条执行下列\n"
                 "检测规则，覆盖正确性、完整性、易理解三个维度。每个维度包含若干 H2\n"
                 "检测项，请对照其规则文件内容逐项检查，命中即报。\n"
                 "\n"
                 "## 检测范围\n"
                 "\n"
                 "- Markdown/RST 文档正文（docs/zh/**/*.md, *.md 等）\n"
                 "- gettext 翻译文件（docs/locale/en/LC_MESSAGES/**/*.po）\n"
                 "- 代码块（非注释部分）、行内代码、URL、文件路径、命令行内容按各规则文件的豁免区处理\n"
                 "- frontmatter 中的配置性字段不检测\n"
                 "\n"
                 "## .po 文件检测说明\n"
                 "\n"
                 "- 检查 msgstr 中的英文翻译是否准确、完整、符合技术文档规范\n"
                 "- 检查翻译是否有错别字、语法错误、术语不一致\n"
                 "- 检查 msgid（中文原文）与 msgstr（英文翻译）是否语义对应\n"
                 "- 跳过 .po 文件头部 metadata（Project-Id-Version、Content-Type 等）\n"
                 "- 跳过空 msgstr（尚未翻译的条目不报错）\n")

    parts.extend(SEVERITY_MAP)

    scoring = common_dir / "_文档评分规则.md"
    if scoring.exists():
        parts.append("## 评分体系（供定级参考，脚本计算由 doc-quality 工具完成）\n")
        parts.append(read_text(scoring).strip())
        parts.append("")
    else:
        print(f"warning: scoring rules file missing: {scoring}", file=sys.stderr)

    common_spec = common_dir / "_公共检测规范.md"
    if common_spec.exists():
        parts.append("## 公共检测规范\n")
        parts.append(read_text(common_spec).strip())
        parts.append("")
    else:
        print(f"warning: common spec missing: {common_spec}", file=sys.stderr)

    for dim in ("正确性", "完整性", "易理解"):
        h2_list = registry.get(dim)
        if not h2_list:
            print(f"warning: dimension {dim} has no H2 entries", file=sys.stderr)
            continue

        weight = DIMENSION_WEIGHTS.get(dim, 0.3)
        parts.append(f"## 维度：{dim}（权重 {weight}）\n")
        parts.append(f"共 {len(h2_list)} 个 H2 检测项，逐项执行。\n")

        for entry in h2_list:
            seq = entry.get("seq", "?")
            h2_name = entry.get("h2", "")
            rule_file = entry.get("rule_file", "")
            script_rule_file = entry.get("script_rule_file")

            rf_path = rules_root / rule_file if rule_file else None
            if rf_path and not rf_path.exists():
                print(
                    f"warning: rule_file missing for {dim} seq {seq}: {rf_path}",
                    file=sys.stderr,
                )
                parts.append(f"### {seq}. {h2_name}\n")
                parts.append(f"（规则文件缺失：{rule_file}）\n")
                continue

            parts.append(f"### {dim} - {seq}. {h2_name}\n")
            parts.append(read_text(rf_path).strip())
            parts.append("")

            if script_rule_file:
                srf_path = rules_root / script_rule_file
                if srf_path.exists():
                    parts.append(f"#### 补充：脚本检测规则（{script_rule_file}）\n")
                    parts.append(read_text(srf_path).strip())
                    parts.append("")
                else:
                    print(
                        f"warning: script_rule_file missing for {dim} seq {seq}: {srf_path}",
                        file=sys.stderr,
                    )

    parts.append("## 输出要求\n")
    parts.append("- 每条 finding 的 severity 必须按上面的映射表填写 critical/high/medium/low\n"
                 "- category 固定填 documentation\n"
                 "- 仅报告命中的问题，不要报告通过项\n"
                 "- 引用具体行号和原文片段\n"
                 "- 优先给出可执行的修复建议\n")

    return "\n".join(parts)


def main():
    if len(sys.argv) != 3:
        print(f"usage: {sys.argv[0]} <doc-quality-repo-path> <output-rule.json>")
        sys.exit(2)

    repo = Path(sys.argv[1]).resolve()
    out = Path(sys.argv[2]).resolve()

    if not repo.is_dir():
        print(f"error: repo path not a directory: {repo}", file=sys.stderr)
        sys.exit(1)

    registry = load_registry(repo)
    total = sum(len(v) for v in registry.values())
    print(f"loaded h2_registry.json: {total} H2 entries across "
          f"{len(registry)} dimensions", file=sys.stderr)

    rule_text = build_rule_text(repo, registry)
    rule_json = {
        "rules": [
            {
                "path": "**/*.md",
                "rule": rule_text,
                "merge_system_rule": False,
            },
            {
                "path": "**/*.rst",
                "rule": rule_text,
                "merge_system_rule": False,
            },
            {
                "path": "**/*.po",
                "rule": rule_text,
                "merge_system_rule": False,
            },
        ],
        "include": [
            "**/*.md",
            "**/*.rst",
            "**/*.po",
        ],
        # include is force-include (additive), not an allowlist — non-matching
        # files still fall through to OCR's built-in extension allowlist. To
        # achieve a strict "docs-only" whitelist, exclude all code/config/binary
        # extensions that OCR would otherwise review with default rules.
        "exclude": [
            "**/*.{go,py,pyi,ipynb,java,kt,scala,rs,c,cpp,cc,cxx,h,hpp,hxx,cs,rb,php,swift,ts,tsx,js,jsx,mjs,cjs,vue,svelte,lua,pl,sh,bash,zsh,fish,ps1,bat,cmd,sql,json,jsonc,yaml,yml,toml,ini,cfg,conf,xml,html,htm,css,scss,sass,less,mo,pot,lock,sum,mod,work,gradle,cmake,make,Dockerfile,Makefile}",
        ],
    }

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(rule_json, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"rule JSON written: {out} ({out.stat().st_size} bytes)", file=sys.stderr)


if __name__ == "__main__":
    main()
