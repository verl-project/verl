#!/usr/bin/env python3
#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the verl project.
#
"""
Direct doc translation workflow: docs/ascend_tutorial/zh -> docs/ascend_tutorial/en.

Translates Chinese docs under docs/ascend_tutorial/zh/ (Markdown and
reStructuredText) into English docs of the SAME format written to
docs/ascend_tutorial/en/, mirroring the source structure:

    docs/ascend_tutorial/zh/get_start/install_guidance.rst
        -> docs/ascend_tutorial/en/get_start/install_guidance.rst

There is no Sphinx front-end rendering gettext catalogs any more, so each run
RENDERS the translated content directly into plain Markdown/RST English
documents under docs/ascend_tutorial/en/. The per-block Chinese -> English
translations are cached in Sphinx gettext .po format under
docs/ascend_tutorial/locale/en/LC_MESSAGES/ (one .po per source document, one
msgid/msgstr pair per translatable block) so incremental runs reuse cached
translations and only call the API for changed blocks.

Each source document is split into blocks at blank-line boundaries (fenced
code blocks stay intact even when they contain blank lines). Blocks containing
Chinese are translated via the DeepSeek API; pure-ASCII/structural blocks
(code, markup, separators) pass through untouched. Blocks are re-assembled in
the original order, preserving the exact document structure.

Excluded files (never translated):
    docs/ascend_tutorial/zh/index.rst   (Sphinx toctree index)

Usage:
    python translate_md.py --first-time     # full translation of all docs
    python translate_md.py --all            # incremental (changed docs/blocks)
    python translate_md.py --files <path>   # specific source files

Translation skill:
    Every API call injects the translation skill document
    (.agent/skills/translate-ascend/skill.md) as part of the system prompt.
    The skill REQUIRES the translator to read the translation standard
    (https://developers.google.com/style) BEFORE translating anything.
    Use --skill-doc (or TRANSLATION_SKILL_DOC env var) to override the path.
"""

import argparse
import asyncio
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Optional

from openai import AsyncOpenAI, AuthenticationError

# ---------------------------------------------------------------------------
# Directory layout (relative to the repository root)
# ---------------------------------------------------------------------------
# Source:     docs/ascend_tutorial/zh/          Chinese .md / .rst files
# Output:     docs/ascend_tutorial/en/          English .md / .rst files
# Translation memory (.po cache): docs/ascend_tutorial/locale/en/LC_MESSAGES/

ZH_DIR = Path("docs/ascend_tutorial/zh")
EN_DIR = Path("docs/ascend_tutorial/en")
LOCALE_DIR = Path("docs/ascend_tutorial/locale")
PO_DIR = LOCALE_DIR / "en" / "LC_MESSAGES"

# The Sphinx toctree index of the tutorial is maintained manually and must
# NOT be translated.
EXCLUDED_SOURCE_REL = ("index.rst",)

# .po timestamps use Beijing time (UTC+8) so headers read as local time
# regardless of the CI runner's timezone (GitHub Actions runs on UTC).
_BEIJING_TZ = timezone(timedelta(hours=8))

# ---------------------------------------------------------------------------
# LLM provider configuration
# ---------------------------------------------------------------------------
# The translation engine uses the OpenAI-compatible chat-completions API, so
# any provider with an OpenAI-compatible endpoint can be plugged in by
# overriding the base URL and model name. The default endpoint below points
# to a Volcano Engine (Volcengine) API Gateway proxy of an OpenAI-compatible
# service; the default model is glm-5.2 (Zhipu AI) exposed through that
# gateway. Both can be overridden via LLM_API_BASE / LLM_MODEL / --api-base /
# --model.
# These are also exposed as --api-base / --model CLI arguments and as the
# workflow_dispatch inputs `api_base` / `model` in the GitHub Actions workflow.
DEFAULT_API_BASE = "https://st8tp3ajl0df3n8b8l8qu.apigateway-cn-beijing.volceapi.com/v1"
DEFAULT_MODEL = "glm-5.2"

# ---------------------------------------------------------------------------
# Translation skill document
# ---------------------------------------------------------------------------
SKILL_DOC_PATH = Path(".agent/skills/translate-ascend/skill.md")

_skill_doc_cache: Optional[str] = None


def load_skill_doc(path: Optional[Path] = None) -> str:
    """Load the translation skill document content.

    Returns the raw skill Markdown content, or an empty string when the
    document is missing / unreadable (the pipeline then falls back to the
    plain system prompt).

    The result is cached module-wide so concurrent calls in the same run
    only read the file once. A non-None ``path`` (from --skill-doc) clears
    the cache so a caller-specified document is always honored.
    """
    global _skill_doc_cache
    if path is not None:
        _skill_doc_cache = None
        skill_path = path
    else:
        skill_path = SKILL_DOC_PATH
    if _skill_doc_cache is not None:
        return _skill_doc_cache
    try:
        if not skill_path.exists():
            print(f"  Skill doc not found: {skill_path} (translating without skill)", flush=True)
            _skill_doc_cache = ""
            return _skill_doc_cache
        _skill_doc_cache = skill_path.read_text(encoding="utf-8").strip()
        print(f"  Loaded translation skill doc: {skill_path} ({len(_skill_doc_cache)} chars)", flush=True)
    except OSError as e:
        print(f"  Warning: failed to read skill doc {skill_path}: {e} (translating without skill)", flush=True)
        _skill_doc_cache = ""
    return _skill_doc_cache


def _extract_skill_version(skill_doc: str) -> str:
    """Read the ``version`` field from the skill doc front-matter."""
    m = re.search(r"^version:\s*([0-9A-Za-z.+-]+)", skill_doc, re.MULTILINE)
    return m.group(1).strip() if m else ""

# ---------------------------------------------------------------------------
# Translation prompts
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a professional technical documentation translation expert, "
    "proficient in Chinese-to-English technical document translation. "
    "Before translating anything you MUST read and follow the translation "
    "standard embedded in the skill document below (the key points of the "
    "Google Developer Documentation Style Guide, https://developers.google.com/style). "
    "Skipping the translation standard produces non-compliant translations "
    "that must be redone, so the standard is mandatory for every request."
)

BLOCK_TRANSLATION_PROMPT = """Translate the following Chinese text block into English.

Rules:
1. Return ONLY the translated text, no explanations, no markdown fences.
2. Preserve the EXACT original structure: markdown/RST heading markers
   (#, ==, --, ~~), list prefixes (-, 1., 4.1), table separators (|, ----),
   inline markup (`, **, *), links, and code fences. Do NOT renumber,
   reorder, merge, or split lines, paragraphs, list items, table rows, or
   code blocks.
2b. Keep the leading whitespace (spaces/tabs) of EVERY line exactly as in
   the source, especially inside code blocks and indented RST blocks.
3. Follow the translation standard (Google Developer Documentation Style
   Guide): prefer active voice (imperative for instructions), use simple
   present tense, use second person ("you"), write complete sentences with
   explicit subjects and verbs, put articles (a/an/the) before singular
   countable nouns, keep sentences under ~25 words, and use parallel
   structures.
4. Do NOT use foreign words (etc., e.g., i.e., via) or contractions
   (can't, it's, don't) - use "and so on", "for example", "that is",
   "through/by/using", "cannot", "it is", "do not" instead.
5. Proper nouns (product names, environment variables, API identifiers,
   repository names) stay exactly as-is.
6. In code blocks or inline code, translate ONLY the Chinese comments and
   string literals; leave all code syntax, variable names, and keywords
   unchanged.
7. If any sentence is too ambiguous to translate faithfully, keep the
   original Chinese as-is; never guess.

Text to translate:
{content}"""


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------


def _get_source_commit(source: Path) -> str:
    """Return a stable content fingerprint of a source document at HEAD.

    Uses the file's blob object id: `git rev-parse HEAD:<repo-rel-path>`.
    GitHub Actions checkouts are shallow (fetch-depth: 1), so `git log`
    usually returns nothing there; the blob id changes only when the file's
    CONTENT changes and it works on shallow clones.
    """
    if not source.exists():
        return ""
    try:
        cwd = Path.cwd().resolve()
        try:
            rel = source.resolve().relative_to(cwd)
        except ValueError:
            return ""
        res = subprocess.run(
            ["git", "rev-parse", f"HEAD:{rel.as_posix()}"],
            capture_output=True,
            text=True,
            check=True,
        )
        return res.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def _get_po_source_commit(po_path: Path) -> str:
    """Read the X-Source-Commit header field recorded in a .po cache file."""
    try:
        raw = po_path.read_text(encoding="utf-8")
    except OSError:
        return ""
    m = re.search(r"X-Source-Commit:\s*([0-9a-fA-F]{7,40})", raw)
    return m.group(1) if m else ""


def _get_po_skill_version(po_path: Path) -> str:
    """Read the X-Skill-Version header field recorded in a .po cache file."""
    try:
        raw = po_path.read_text(encoding="utf-8")
    except OSError:
        return ""
    m = re.search(r"X-Skill-Version:\s*([0-9A-Za-z.+-]+)", raw)
    return m.group(1) if m else ""



def _po_cache_path(src: Path) -> Path:
    """Translation-memory .po path that mirrors a source document."""
    return PO_DIR / src.relative_to(ZH_DIR).with_suffix(".po")


def find_source_files() -> List[Path]:
    """Find all translatable source documents under docs/ascend_tutorial/zh/.

    Excluded files (currently only docs/ascend_tutorial/zh/index.rst) are
    skipped.
    """
    result = []
    for ext in ("*.md", "*.rst"):
        for p in sorted(ZH_DIR.rglob(ext)):
            rel = p.relative_to(ZH_DIR).as_posix()
            if rel in EXCLUDED_SOURCE_REL:
                continue
            result.append(p)
    return result


def _source_needs_update(src: Path, cache_po: Path, skill_version: str) -> bool:
    """True when the source content or the skill version changed since the
    cached .po was written."""
    if not cache_po.exists():
        return True
    cur_commit = _get_source_commit(src)
    po_commit = _get_po_source_commit(cache_po)
    po_skill = _get_po_skill_version(cache_po)
    if (cur_commit and po_commit and cur_commit == po_commit
            and skill_version and po_skill and skill_version == po_skill):
        return False
    return True


def find_changed_source_files(skill_version: str) -> List[Path]:
    """Find source documents that need (re)translation vs their .po cache."""
    changed = []
    for src in find_source_files():
        if _source_needs_update(src, _po_cache_path(src), skill_version):
            changed.append(src)
    return changed


# ---------------------------------------------------------------------------
# PO / POT file parsing
# ---------------------------------------------------------------------------


def _unescape_po(s: str) -> str:
    """Decode gettext string escapes: \\n, \\t, \\r, \\\\, \\\"."""
    out = []
    i = 0
    n = len(s)
    while i < n:
        ch = s[i]
        if ch == "\\" and i + 1 < n:
            nxt = s[i + 1]
            if nxt == "n":
                out.append("\n")
            elif nxt == "t":
                out.append("\t")
            elif nxt == "r":
                out.append("\r")
            elif nxt == "\\":
                out.append("\\")
            elif nxt == '"':
                out.append('"')
            else:
                out.append(nxt)
            i += 2
        else:
            out.append(ch)
            i += 1
    return "".join(out)


def _escape_po(s: str) -> str:
    """Escape a string for a single gettext quoted string."""
    s = s.replace("\\", "\\\\")
    s = s.replace('"', '\\"')
    return s

def _extract_po_value(block: str, field: str) -> Optional[str]:
    """Extract the value of a msgid/msgstr field from a PO block.

    Supports both single-line (`msgid "..."`) and multi-line
    (`msgid ""` + `"line\\n"` continuation) forms and round-trips embedded
    newlines exactly.
    """
    lines = block.split("\n")
    # Multi-line form: `field ""` followed by quoted continuation lines.
    for i, line in enumerate(lines):
        if line.strip() == f'{field} ""':
            parts = []
            for cont in lines[i + 1:]:
                m = re.match(r'\s*"((?:[^"\\]|\\.)*)"', cont)
                if m:
                    parts.append(_unescape_po(m.group(1)))
                else:
                    break
            if parts:
                return "".join(parts)
            return ""
    m = re.search(rf'{field}\s+"((?:[^"\\]|\\.)*)"', block)
    if m:
        return _unescape_po(m.group(1))
    return None


def parse_pot_file(filepath: Path) -> dict:
    """Parse a .po / .pot file and return entries dict.

    Returns dict mapping msgid -> entry dict:
    {
        "msgid": str,
        "msgstr": str,
        "translated": bool,
    }
    The gettext header block (msgid "" / msgstr "" with metadata) is skipped.
    """
    entries = {}
    if not filepath.exists():
        return entries

    raw = filepath.read_text(encoding="utf-8")
    blocks = raw.split("\n\n")
    for block in blocks:
        block = block.strip()
        if not block:
            continue

        msgid = _extract_po_value(block, "msgid")
        msgstr = _extract_po_value(block, "msgstr")
        # The gettext header block has an empty msgid (with metadata fields);
        # real entries always have a non-empty msgid.
        if msgid is None or msgid == "":
            continue
        entries[msgid] = {
            "msgid": msgid,
            "msgstr": msgstr or "",
            "translated": bool(msgstr and msgstr.strip()),
        }
    return entries


def _append_po_field(lines: List[str], field: str, value: str) -> None:
    """Append a gettext field, splitting multi-line values into continuation
    lines so the written file round-trips exactly through _extract_po_value."""
    if "\n" in value:
        lines.append(f'{field} ""')
        parts = value.split("\n")
        for i, part in enumerate(parts):
            esc = _escape_po(part)
            if i < len(parts) - 1:
                lines.append(f'"{esc}\\n"')
            else:
                lines.append(f'"{esc}"')
    else:
        lines.append(f'{field} "{_escape_po(value)}"')


def write_po_file(filepath: Path, entries: dict, changed: bool = True,
                  source_commit: str = "", skill_version: str = "") -> None:
    """Write entries dict to a .po translation-memory file.

    One msgid/msgstr pair is stored per translatable block of the source
    document. ``changed`` controls whether the POT/PO creation timestamps are
    stamped:
    - True (content changed): write the current POT-Creation-Date /
      PO-Revision-Date so the diff clearly marks the file as updated.
    - False (content identical): write a stable header WITHOUT the timestamp
      fields, so an unchanged cache file is rewritten byte-identically and
      produces no git diff.

    ``source_commit`` records the git blob id of the source document from
    which the translations were produced; incremental runs compare it with the
    source's current blob id to decide whether to re-translate. ``skill_version``
    records the skill-document version in effect when the translations were
    written; when the skill changes, affected files are re-processed so
    terminology edits propagate.
    """
    filepath.parent.mkdir(parents=True, exist_ok=True)

    now_str = datetime.now(_BEIJING_TZ).strftime("%Y-%m-%d %H:%M%z")
    lines = []

    # Header.
    lines.append('# English translations for verl ascend_tutorial docs.\n'
                 '# Copyright (c) 2025 Huawei Technologies Co., Ltd.\n'
                 '#\n'
                 'msgid ""\n'
                 'msgstr ""\n'
                 '"Project-Id-Version: verl-docs\\n"\n')
    if changed:
        lines.append(f'"POT-Creation-Date: {now_str}\\n"\n'
                     f'"PO-Revision-Date: {now_str}\\n"\n')
    if source_commit:
        lines.append(f'"X-Source-Commit: {source_commit}\\n"\n')
    if skill_version:
        lines.append(f'"X-Skill-Version: {skill_version}\\n"\n')
    lines.append('"Last-Translator: Auto Translation (DeepSeek)\\n"\n'
                 '"Language-Team: English\\n"\n'
                 '"Language: en\\n"\n'
                 '"MIME-Version: 1.0\\n"\n'
                 '"Content-Type: text/plain; charset=UTF-8\\n"\n'
                 '"Content-Transfer-Encoding: 8bit\\n"\n'
                 '"Plural-Forms: nplurals=2; plural=(n != 1);\\n"\n')

    for entry in entries.values():
        msgid = entry.get("msgid", "")
        if not msgid:
            continue
        msgstr = entry.get("msgstr", "")
        _append_po_field(lines, "msgid", msgid)
        _append_po_field(lines, "msgstr", msgstr)
        lines.append("")

    filepath.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Block splitting / rendering
# ---------------------------------------------------------------------------


_CJK_RE = re.compile(r"[\u4e00-\u9fff]")


def _contains_cjk(s: str) -> bool:
    """True when the text contains CJK (Chinese) characters."""
    return bool(_CJK_RE.search(s))


def split_source_into_blocks(content: str) -> List[tuple]:
    """Split source content into (text, is_translatable) pieces.

    Pieces are separated at blank-line boundaries so paragraphs, list items,
    tables, and code blocks stay intact. Blank-line separators are preserved
    verbatim as separate (non-translatable) pieces. Fenced code blocks
    (``` ... ```) are kept as a single piece even when they contain blank
    lines. A piece is translatable when it contains CJK characters;
    pure-ASCII / structural pieces pass through untouched.

    Re-joining the pieces in order reproduces the original content exactly.
    """
    pieces: List[tuple] = []
    buffer: List[str] = []
    buf_cjk = False
    sep = ""
    in_fence = False
    ends_with_nl = content.endswith("\n")
    lines = content.split("\n")
    # ``split`` yields a trailing '' when the content ends with '\n'; that is
    # the line terminator of the last line, not a blank line.
    if lines and lines[-1] == "":
        lines.pop()
    n_lines = len(lines)

    def emit_block():
        nonlocal buffer, buf_cjk
        text = "".join(buffer)
        if text:
            pieces.append((text, buf_cjk))
        buffer = []
        buf_cjk = False

    def emit_sep():
        nonlocal sep
        if sep:
            pieces.append((sep, False))
            sep = ""

    for idx, line in enumerate(lines):
        is_last = idx == n_lines - 1
        line_ending = "\n" if (not is_last or ends_with_nl) else ""
        stripped = line.strip()
        is_fence_delim = stripped.startswith("```")

        if not in_fence and is_fence_delim:
            # A fence starts; flush any pending separator before it so the
            # blank line stays between the previous block and the fence.
            if sep:
                emit_sep()
            in_fence = True
            buffer.append(line + line_ending)
            if _contains_cjk(line):
                buf_cjk = True
            continue
        if in_fence and is_fence_delim:
            in_fence = False
            buffer.append(line + line_ending)
            continue
        if not in_fence and stripped == "":
            emit_block()
            sep += "\n"
            continue
        if not in_fence and sep:
            emit_sep()
        buffer.append(line + line_ending)
        if _contains_cjk(line):
            buf_cjk = True

    emit_block()
    emit_sep()
    return pieces


def _restore_enumeration_prefix(msgid: str, msgstr: str) -> str:
    """Ensure the translated text keeps the 'N. ' enumeration prefix of its msgid.

    The DeepSeek model sometimes drops the list-number prefix when translating
    Chinese headings, e.g.

        msgid  "1. 安装与环境配置"
        msgstr "Installation and Environment Configuration"

    loses the leading "1. ". Since the msgid prefix is structural (list
    numbering), we re-attach it before the text is written to the output.

    Only the first non-blank line is considered; sub-sequences (e.g.
    "1.1 ") or other leading numbers already present in the translation are
    left untouched.
    """
    if not msgid or not msgstr:
        return msgstr

    # List-number prefixes come in two shapes: "1. " (top-level item) and
    # "4.1 " / "1.1.2 " (nested item, no trailing dot). Match either, and
    # reuse the source's exact prefix form (dot included only when present).
    m = re.match(r"^\s*(\d+(?:\.\d+)*\.?)\s+", msgid)
    if not m:
        return msgstr

    prefix = m.group(1) + " "

    stripped = msgstr.lstrip("\n")
    leading_ws = msgstr[:len(msgstr) - len(stripped)]

    # Already has the prefix -> leave it as-is.
    if re.match(r"^\s*\d+(?:\.\d+)*\.?\s+", stripped):
        return msgstr

    return leading_ws + prefix + stripped


def _restore_leading_ws(src: str, trans: str) -> str:
    """Restore the exact leading whitespace of each source line on the
    translated text.

    The LLM sometimes drops the indentation of lines inside code blocks when
    translating Chinese comments (e.g. a bash comment ``  # ...`` becomes
    ``# ...``). In RST this breaks literal-block indentation and produces
    docutils errors such as "Unexpected indentation".

    Only applied when both texts have the same number of lines (otherwise the
    model merged/split lines and we cannot align them safely).
    """
    if not src or not trans:
        return trans
    src_lines = src.split("\n")
    tr_lines = trans.split("\n")
    if len(src_lines) != len(tr_lines):
        return trans
    out = []
    for s, t in zip(src_lines, tr_lines):
        ws = s[:len(s) - len(s.lstrip(" \t"))]
        out.append(ws + t.lstrip(" \t"))
    return "\n".join(out)


def _fix_rst_underlines(text: str) -> str:
    """Extend RST section-title underlines so they are never shorter than the
    title text.

    RST requires the underline to be at least as long as the title; after
    translation the English title is often longer than the original Chinese
    one, but the underline (a separate block that contains no Chinese and is
    kept verbatim) stays short, which makes docutils emit
    "Title underline too short" warnings/errors.

    A line followed by a line of only ``= - ~ ^ " # * +`` characters and that
    is not a list item is treated as a section title and its underline is
    padded to the title length.
    """
    lines = text.split("\n")
    n = len(lines)
    for i in range(n - 1):
        title = lines[i]
        underline = lines[i + 1]
        if not title or title[0] in " \t":
            continue
        if re.match(r"^(?:[-*] |\d+\. |[#*] )", title):
            continue
        if re.fullmatch(r"[=\-~^\"#*+]{2,}", underline):
            if len(underline) < len(title.rstrip()):
                lines[i + 1] = underline[0] * len(title.rstrip())
    return "\n".join(lines)


def _strip_rst_fences(text: str) -> str:
    """Remove stray Markdown code fences (````` ``` `````) from reStructuredText.

    RST literal blocks are indented text and must NOT contain Markdown
    ````` ``` ````` fences. The LLM sometimes emits fences around RST code
    blocks; the indented content that RST actually needs is kept verbatim, so
    dropping the fence lines restores a valid RST literal block.

    Only meaningful for .rst output documents.
    """
    out = []
    for line in text.split("\n"):
        if re.fullmatch(r"`{3,}[\w+\- ]*", line.strip()):
            continue
        out.append(line)
    return "\n".join(out)


# ---------------------------------------------------------------------------
# Glossary enforcement
# ---------------------------------------------------------------------------


_EN_TERM_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9\s\-_().,/'+]*$")


def _looks_like_english_term(s: str) -> bool:
    """True if ``s`` looks like a plain English glossary value."""
    return bool(s) and bool(_EN_TERM_PATTERN.match(s)) and not _CJK_RE.search(s)


def _lower_first_word(w: str) -> str:
    """Lowercase the first letter of a word unless it is an acronym (all caps)."""
    if w.isupper() or len(w) <= 1:
        return w
    return w[:1].lower() + w[1:]


def _casing_variants(authoritative: str) -> List[str]:
    """Generate plausible casing variants of an authoritative term.

    E.g. "Ascend Platform" -> ["Ascend platform", "ascend platform"].
    Acronyms (e.g. "NPU") keep their caps. The authoritative spelling itself
    is excluded from the returned variants.
    """
    words = authoritative.split()
    if len(words) < 2:
        return []
    variants = set()
    # Only the first word keeps its case; the rest get lowercase first letters.
    variants.add(" ".join([words[0]] + [_lower_first_word(w) for w in words[1:]]))
    # Every word lowercased.
    variants.add(" ".join(_lower_first_word(w) for w in words))
    # Fully-lowercase fallback.
    variants.add(" ".join(w.lower() for w in words))
    variants.discard(authoritative)
    return list(variants)


def _build_glossary_rules(skill_doc: str) -> List[tuple]:
    """Extract deterministic (variant -> authoritative) replacement pairs from
    the skill document's glossary tables.

    Every markdown table row ``| 中文术语 | English (authoritative) | Notes |``
    in the skill document is scanned. For every English form in the second
    column we generate variants that the LLM tends to output instead of the
    authoritative spelling:

    - separator variants: hyphen/underscore collapsed to spaces;
    - casing variants: same word sequence with a different letter case.

    ``apply_glossary_rules()`` then deterministically enforces the
    authoritative spellings on every translated string, so the pipeline no
    longer depends on the LLM choosing to follow the glossary.

    Only plain-English values are accepted; explanatory cells (e.g. "正确用法",
    "and so on") and quoted examples are ignored.
    """
    rules: List[tuple] = []
    if not skill_doc:
        return rules
    for line in skill_doc.splitlines():
        stripped = line.strip()
        if not stripped.startswith("|"):
            continue
        cells = [c.strip() for c in stripped.strip("|").split("|")]
        if len(cells) < 2:
            continue
        en_cell = cells[1]
        if not en_cell or set(en_cell) <= {"-", " ", ":"}:
            continue
        for authoritative in en_cell.split("/"):
            authoritative = authoritative.strip().strip("`").strip()
            if not _looks_like_english_term(authoritative):
                continue
            # Separator variants: hyphen/underscore -> spaces.
            normalized = re.sub(r"[-_]", " ", authoritative)
            normalized = re.sub(r"\s+", " ", normalized).strip()
            if normalized and normalized != authoritative:
                rules.append((normalized, authoritative))
            # Casing variants: same words, different letter case.
            for variant in _casing_variants(authoritative):
                rules.append((variant, authoritative))
    # Longer variants first so e.g. "Atlas A3 training products" wins over any
    # shorter overlapping variant.
    rules.sort(key=lambda r: len(r[0]), reverse=True)
    return rules


def apply_glossary_rules(text: str, rules: List[tuple]) -> str:
    """Replace glossary variants in ``text`` with their authoritative spellings.

    Inline code spans and fenced code blocks are left untouched so identifiers
    and string literals are never rewritten.
    """
    if not text or not rules:
        return text
    protected: List[str] = []

    def _protect(m):
        protected.append(m.group(0))
        return f"\x00{len(protected) - 1}\x00"

    text = re.sub(r"```.*?```", _protect, text, flags=re.DOTALL)
    text = re.sub(r"`[^`\n]+`", _protect, text)
    for variant, authoritative in rules:
        text = re.sub(r"(?<![\w])" + re.escape(variant) + r"(?![\w])", authoritative, text)
    return re.sub(r"\x00(\d+)\x00", lambda m: protected[int(m.group(1))], text)


# ---------------------------------------------------------------------------
# Translation engine
# ---------------------------------------------------------------------------


class DocTranslator:
    """Translate zh source documents into English md/rst documents under
    docs/ascend_tutorial/en/, using .po files as translation memory."""

    def __init__(self, api_key: str, skill_doc: Optional[str] = None,
                 api_base: str = "", model: str = ""):
        self.api_base = api_base or DEFAULT_API_BASE
        self.model = model or DEFAULT_MODEL
        # Per-request timeout and retry budget (env-tunable). Without a timeout
        # a hung request (slow gateway, stalled connection) would block the
        # whole sequential translation for minutes per block. When no API key
        # is provided the client stays None so cache-only re-rendering works
        # without network access; blocks needing fresh translation will fail
        # closed with a clear message.
        self.client = (AsyncOpenAI(
            api_key=api_key,
            base_url=self.api_base,
            timeout=float(os.getenv("LLM_TIMEOUT", "120")),
            max_retries=int(os.getenv("LLM_MAX_RETRIES", "1")),
        ) if api_key else None)
        # Number of blocks translated concurrently per document. Raising this
        # shortens total runtime but may hit provider rate limits (429).
        self._concurrency = max(1, int(os.getenv("LLM_CONCURRENCY", "4")))
        # Extra per-block retries for transient failures (timeout / 429 / net
        # jitter). A block is only considered failed after all attempts.
        self._block_retries = max(0, int(os.getenv("LLM_BLOCK_RETRIES", "2")))
        # The skill document (.agent/skills/translate-ascend/skill.md) defines
        # the mandatory translation standard and the authoritative glossary.
        # It is injected into every request's system prompt so the standard is
        # always read before translating.
        self.skill_doc = skill_doc or ""
        self._glossary_rules: List[tuple] = _build_glossary_rules(self.skill_doc)
        self._skill_version = _extract_skill_version(self.skill_doc)

    def _system_prompt(self, context: str = "") -> str:
        """Build the system prompt, injecting the skill document when available."""
        parts = [SYSTEM_PROMPT]
        if self.skill_doc:
            parts.append(
                "Follow the skill document below for the mandatory translation "
                "standard (Google Developer Documentation Style Guide key points), "
                "terminology, and structure rules. The standard MUST be read and "
                "followed before translating anything; its glossary is "
                "authoritative. Use it for every translation.\n\n"
                "===== BEGIN TRANSLATION SKILL DOCUMENT =====\n"
                f"{self.skill_doc}\n"
                "===== END TRANSLATION SKILL DOCUMENT =====")
        if context:
            parts.append(f"(File: {context})")
        return "\n\n".join(parts)

    async def _translate_single(self, content: str, context: str = "") -> Optional[str]:
        """Translate a single text block via the configured LLM API."""
        if self.client is None:
            print(f"  No API key configured - block for '{context}' needs fresh translation, skipped",
                  flush=True)
            return None
        prompt = BLOCK_TRANSLATION_PROMPT.replace("{content}", content)
        system = self._system_prompt(context)

        trailing_nl = content.endswith("\n")
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=8192,
                temperature=0.3,
            )
            text = response.choices[0].message.content
            if not text:
                return None
            text = text.strip()
            if trailing_nl and not text.endswith("\n"):
                text += "\n"
            return text
        except AuthenticationError:
            # Invalid API key / authentication failure. Let it propagate so the
            # caller can fail the whole document immediately instead of wasting
            # API calls on every remaining block.
            raise
        except Exception as e:
            print(f"API error translating '{context}': {e}")
            return None


    async def translate_file(self, src: Path) -> bool:
        """Translate one source document and render the English md/rst output.

        Returns True when the file was fully translated and written (English
        doc and/or .po cache), False when there is nothing to do OR when the
        translation API failed.

        Fail-closed behavior: when the API fails (e.g. an invalid API key
        returning 401, or a per-block error), failed blocks are NEVER persisted
        as if they were translated. The English output document and the .po
        cache are left untouched and the file is reported as failed, so the
        workflow never commits documents full of untranslated Chinese text.
        """
        rel = src.relative_to(ZH_DIR)
        dst = EN_DIR / rel
        cache_po = _po_cache_path(src)
        name = rel.as_posix()

        source_commit = _get_source_commit(src)
        cached_commit = _get_po_source_commit(cache_po)
        cached_skill = _get_po_skill_version(cache_po)
        source_changed = not (source_commit and cached_commit
                              and source_commit == cached_commit
                              and cached_skill and cached_skill == self._skill_version)

        content = src.read_text(encoding="utf-8")
        pieces = split_source_into_blocks(content)
        po_entries = parse_pot_file(cache_po)

        new_entries = {}
        en_parts: List[Optional[str]] = [None] * len(pieces)
        translated_new = 0
        reused = 0
        failed = 0
        pending: List[int] = []

        # First pass: fill structural blocks and cache hits, collect the blocks
        # that still need API translation.
        for idx, (text, translatable) in enumerate(pieces):
            if not translatable:
                en_parts[idx] = text
                continue
            cached = po_entries.get(text)
            if cached and cached.get("msgstr"):
                restored = _restore_leading_ws(text, cached["msgstr"])
                new_entries[text] = {"msgid": text, "msgstr": restored, "translated": True}
                en_parts[idx] = restored
                reused += 1
                continue
            pending.append(idx)

        # Translate the pending blocks concurrently (bounded by _concurrency)
        # and collect results in document order via the original block index.
        sem = asyncio.Semaphore(self._concurrency)

        async def translate_one(idx: int) -> tuple:
            text = pieces[idx][0]
            async with sem:
                translation = None
                for attempt in range(self._block_retries + 1):
                    try:
                        translation = await self._translate_single(text, name)
                    except AuthenticationError:
                        return idx, None, "AUTH", ""
                    if translation is not None:
                        break
                    # Transient failure (timeout / 429 / net jitter): back off
                    # and retry before giving up on this block.
                    if attempt < self._block_retries:
                        await asyncio.sleep(1.0 * (attempt + 1))
                if translation is None:
                    return idx, None, None, text
                translation = _restore_enumeration_prefix(text, translation)
                translation = apply_glossary_rules(translation, self._glossary_rules)
                translation = _restore_leading_ws(text, translation)
                return idx, translation, "OK", ""

        auth_failed = False
        failed_previews: List[str] = []
        results = await asyncio.gather(*(translate_one(idx) for idx in pending))
        for idx, translation, status, failed_text in results:
            text = pieces[idx][0]
            if status == "AUTH":
                auth_failed = True
            elif status is None:
                failed += 1
                failed_previews.append(failed_text)
                new_entries[text] = {"msgid": text, "msgstr": "", "translated": False}
                en_parts[idx] = text
            else:
                new_entries[text] = {"msgid": text, "msgstr": translation, "translated": True}
                en_parts[idx] = translation
                translated_new += 1

        if auth_failed:
            # Invalid API key / authentication failure: fail the document
            # closed (no output, no cache, nothing to commit).
            print(f"  AUTH FAIL: {name} (invalid API key) - output NOT written", flush=True)
            return False

        if failed > 0:
            # Fail closed: do not write the English doc nor the .po cache, so
            # the workflow never commits a document containing untranslated
            # Chinese (and never reuses fake translations in later runs).
            preview = " | ".join(repr(t[:100]) for t in failed_previews[:3])
            print(f"  FAIL: {name}: {failed} block(s) failed to translate - output NOT written", flush=True)
            print(f"        failed block preview: {preview}", flush=True)
            return False

        en_doc = "".join(p for p in en_parts if p is not None)
        # RST post-processing so sphinx/docutils do not emit errors:
        # - section-title underlines are often shorter than the translated title;
        # - stray Markdown ``` fences are invalid in RST and must be removed.
        en_doc = _fix_rst_underlines(en_doc)
        if src.suffix == ".rst":
            en_doc = _strip_rst_fences(en_doc)

        # Nothing changed and the rendered English doc already exists: skip.
        if (not source_changed and translated_new == 0 and dst.exists()
                and dst.read_text(encoding="utf-8") == en_doc):
            print(f"  Skip: {name} (unchanged)", flush=True)
            return False

        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(en_doc, encoding="utf-8")

        # Refresh the .po cache when the source changed (new X-Source-Commit /
        # X-Skill-Version stamps) or when new translations were added.
        if source_changed or translated_new > 0 or not cache_po.exists():
            write_po_file(cache_po, new_entries, changed=(translated_new > 0),
                          source_commit=source_commit, skill_version=self._skill_version)

        print(f"  {name}: {reused} reused, {translated_new} new -> {rel}", flush=True)
        return True

    async def translate_files(self, src_list: List[Path], output_json: str) -> int:
        """Translate a list of source files sequentially and save results JSON."""
        print(f"Translating {len(src_list)} source file(s)", flush=True)

        ok_files = []
        success_files = []
        for src in src_list:
            ok = await self.translate_file(src)
            if ok:
                rel = src.relative_to(ZH_DIR)
                ok_files.append(str(src))
                success_files.append(str(EN_DIR / rel))
                success_files.append(str(_po_cache_path(src)))

        total = len(src_list)
        ok_count = len(ok_files)
        print(f"\nResult: {ok_count}/{total} translated", flush=True)

        # Report failed documents explicitly so a green step never hides
        # documents that failed to translate.
        failed_files = [str(src) for src in src_list if str(src) not in ok_files]
        if failed_files:
            print(f"FAILED ({len(failed_files)} document(s)):", flush=True)
            for f in failed_files:
                print(f"  - {f}", flush=True)

        report = {
            "success_files": success_files,
            "failed_files": failed_files,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_files": total,
            "success_count": ok_count,
            "failed_count": len(failed_files),
        }
        if output_json:
            out = Path(output_json)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
            print(f"Results written to {output_json}", flush=True)
        elif success_files:
            # No results JSON requested: still print which files were produced
            # so the user / CI can see / stage them.
            print("Translated files:", flush=True)
            for f in success_files:
                print(f"  - {f}", flush=True)

        return 0 if ok_count > 0 else 1


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def write_empty_json(output_json: str, reason: str = ""):
    if not output_json:
        print(f"No source files processed: {reason}", flush=True)
        return
    report = {
        "success_files": [],
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_files": 0,
        "success_count": 0,
        "note": reason,
    }
    out = Path(output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Empty result written to {output_json} (reason: {reason})", flush=True)


async def async_main():
    parser = argparse.ArgumentParser(
        description="Doc translation: docs/ascend_tutorial/zh -> docs/ascend_tutorial/en (md/rst)")
    parser.add_argument("--first-time", action="store_true",
                        help="Full translation: translate ALL source documents")
    parser.add_argument("--all", action="store_true",
                        help="Incremental: translate only changed documents/blocks")
    parser.add_argument("--files", help="Comma-separated source file paths under docs/ascend_tutorial/zh")
    parser.add_argument("--output-json", default=os.getenv("OUTPUT_JSON", ""),
                        help="Optional path to a results JSON file (default: no file is written)")
    parser.add_argument(
        "--api-key",
        default=os.getenv("TRANSLATION_ASCEND",
                          os.getenv("LLM_API_KEY", os.getenv("DEEPSEEK_API_KEY", ""))),
        help="LLM API key (env: TRANSLATION_ASCEND, fallback LLM_API_KEY / DEEPSEEK_API_KEY)",
    )
    parser.add_argument(
        "--api-base",
        default=os.getenv("LLM_API_BASE", ""),
        help=("OpenAI-compatible API base URL "
              f"(default: {DEFAULT_API_BASE}, e.g. Zhipu https://open.bigmodel.cn/api/paas/v4)"),
    )
    parser.add_argument("--model", default=os.getenv("LLM_MODEL", ""),
                        help=f"Model name (default: {DEFAULT_MODEL}, e.g. Zhipu glm-4-plus)")
    parser.add_argument(
        "--skill-doc",
        default=os.getenv("TRANSLATION_SKILL_DOC", ""),
        help="Path to the translation skill document (default: .agent/skills/translate-ascend/skill.md)",
    )
    args = parser.parse_args()

    output_json = args.output_json

    api_key = (
        args.api_key
        or os.getenv("TRANSLATION_ASCEND")
        or os.getenv("LLM_API_KEY")
        or os.getenv("DEEPSEEK_API_KEY")
    )
    if not api_key:
        print("Warning: no LLM API key set (TRANSLATION_ASCEND / LLM_API_KEY / DEEPSEEK_API_KEY). "
              "Documents whose blocks are fully cached will still be re-rendered; blocks that "
              "need a fresh translation will fail (fail-closed).", flush=True)

    api_base = args.api_base or DEFAULT_API_BASE
    model = args.model or DEFAULT_MODEL
    print(f"LLM provider: {api_base} | model: {model}", flush=True)

    # Load the translation skill document (mandatory translation standard and
    # authoritative glossary). It is injected into every translation request.
    skill_doc_path = Path(args.skill_doc) if args.skill_doc else None
    skill_doc = load_skill_doc(skill_doc_path)

    # Step 1: Determine which source files to translate
    src_list = []
    if args.files:
        seen = set()
        for f in args.files.split(","):
            p = Path(f.strip())
            if not p.exists():
                p = ZH_DIR / f.strip()
            if p.exists() and p.suffix in (".md", ".rst"):
                try:
                    p.resolve().relative_to(ZH_DIR.resolve())
                except ValueError:
                    continue
                if p not in seen:
                    seen.add(p)
                    src_list.append(p)
    elif args.first_time:
        src_list = find_source_files()
    elif args.all:
        src_list = find_changed_source_files(_extract_skill_version(skill_doc))
    else:
        msg = "specify --first-time, --all, or --files"
        print(f"Error: {msg}", flush=True)
        write_empty_json(output_json, msg)
        return 1

    if src_list:
        print(f"Found {len(src_list)} source file(s) to translate", flush=True)
        for p in src_list:
            print(f"  - {p}", flush=True)
    else:
        # No files to translate - write empty result so the workflow doesn't
        # get stuck.
        reason = "excluded" if args.first_time else "no changes"
        print(f"No source files to translate ({reason})", flush=True)
        write_empty_json(output_json, f"no source files to translate ({reason})")
        return 0

    translator = DocTranslator(api_key=api_key, skill_doc=skill_doc,
                               api_base=args.api_base, model=args.model)
    return await translator.translate_files(src_list, output_json)


if __name__ == "__main__":
    sys.exit(asyncio.run(async_main()))
