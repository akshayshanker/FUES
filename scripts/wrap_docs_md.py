#!/usr/bin/env python3
"""Hard-wrap FUES docs markdown prose to 78 columns."""

from __future__ import annotations

import re
import sys
import textwrap
from pathlib import Path

WIDTH = 78

FENCE_RE = re.compile(r"^(\s*)(```|~~~)")
TABLE_RE = re.compile(r"^\s*\|")
TABLE_SEP_RE = re.compile(r"^\s*\|[-:| ]+\|\s*$")
HEADING_RE = re.compile(r"^#{1,6}\s")
ADMON_RE = re.compile(r"^!!!\s")
TAB_RE = re.compile(r'^===\s+"')
HTML_RE = re.compile(r"^<\w+")
DISPLAY_MATH_START = re.compile(r"^\s*\\\[")
DISPLAY_MATH_END = re.compile(r"^\s*\\\]")
LIST_RE = re.compile(r"^(\s*)([-*+]|\d+\.)\s+(.*)$")


def wrap_plain(prefix: str, body: str, cont_indent: str) -> list[str]:
    body = " ".join(body.split())
    if not body:
        return [prefix.rstrip()]
    first_width = max(1, WIDTH - len(prefix))
    cont_width = max(1, WIDTH - len(cont_indent))
    lines = textwrap.wrap(
        body,
        width=first_width,
        break_long_words=False,
        break_on_hyphens=False,
    )
    if not lines:
        return [prefix.rstrip()]
    out = [prefix + lines[0]]
    for part in lines[1:]:
        for w in textwrap.wrap(
            part,
            width=cont_width,
            break_long_words=False,
            break_on_hyphens=False,
        ):
            out.append(cont_indent + w)
    return out


def wrap_line(line: str) -> list[str]:
    stripped = line.strip()
    if not stripped:
        return [""]

    m = LIST_RE.match(line)
    if m:
        indent, marker, body = m.group(1), m.group(2), m.group(3)
        prefix = f"{indent}{marker} "
        cont = indent + " " * (len(marker) + 1)
        return wrap_plain(prefix, body, cont)

    m = re.match(r"^(\s*>\s*)(.*)$", line)
    if m:
        prefix, body = m.group(1), m.group(2)
        cont = " " * len(prefix)
        return wrap_plain(prefix, body, cont)

    indent = line[: len(line) - len(line.lstrip(" "))]
    body = line[len(indent) :]
    return wrap_plain(indent, body, indent)


def wrap_table_row(line: str) -> str | list[str]:
    if not TABLE_RE.match(line) or TABLE_SEP_RE.match(line):
        return line

    cells = [c.strip() for c in line.strip().strip("|").split("|")]
    row = "| " + " | ".join(cells) + " |"
    if len(row) <= WIDTH:
        return line

    idx = max(range(len(cells)), key=lambda i: len(cells[i]))
    fixed = cells[:idx] + cells[idx + 1 :]
    prefix = "| " + " | ".join(fixed[:idx] + [""] + fixed[idx:]) + " |"
    avail = WIDTH - len(prefix)
    if avail < 16:
        return line

    parts = textwrap.wrap(
        cells[idx],
        width=avail,
        break_long_words=False,
        break_on_hyphens=False,
    )
    if len(parts) <= 1:
        return line

    rows: list[str] = []
    for i, part in enumerate(parts):
        if i == 0:
            row_cells = cells[:idx] + [part] + cells[idx + 1 :]
        else:
            row_cells = [""] * len(cells)
            row_cells[idx] = part
        candidate = "| " + " | ".join(row_cells) + " |"
        if len(candidate) > WIDTH:
            return line
        rows.append(candidate)
    return rows


def transform_file(text: str) -> str:
    lines = text.splitlines()
    out: list[str] = []
    i = 0
    in_fence = False
    in_front_matter = False
    front_matter_active = False
    in_display_math = False
    para_buf: list[str] = []

    def flush_para() -> None:
        nonlocal para_buf
        if not para_buf:
            return
        merged = " ".join(ln.strip() for ln in para_buf if ln.strip())
        indent = para_buf[0][: len(para_buf[0]) - len(para_buf[0].lstrip(" "))]
        out.extend(wrap_plain(indent, merged, indent))
        para_buf = []

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        if stripped == "---" and i == 0:
            flush_para()
            front_matter_active = True
            in_front_matter = True
            out.append(line)
            i += 1
            continue

        if in_front_matter:
            out.append(line)
            if stripped == "---" and front_matter_active:
                in_front_matter = False
                front_matter_active = False
            i += 1
            continue

        if stripped == "---":
            flush_para()
            out.append(line)
            i += 1
            continue

        if FENCE_RE.match(line):
            flush_para()
            in_fence = not in_fence
            out.append(line)
            i += 1
            continue

        if in_fence:
            out.append(line)
            i += 1
            continue

        if DISPLAY_MATH_START.match(line):
            flush_para()
            in_display_math = True
            out.append(line)
            i += 1
            continue

        if in_display_math:
            out.append(line)
            if DISPLAY_MATH_END.match(line):
                in_display_math = False
            i += 1
            continue

        if not stripped:
            flush_para()
            out.append("")
            i += 1
            continue

        if HEADING_RE.match(line) or TAB_RE.match(line) or HTML_RE.match(stripped):
            flush_para()
            out.append(line)
            i += 1
            continue

        if TABLE_RE.match(line):
            flush_para()
            wrapped = wrap_table_row(line)
            if isinstance(wrapped, list):
                out.extend(wrapped)
            else:
                out.append(wrapped)
            i += 1
            continue

        if ADMON_RE.match(stripped):
            flush_para()
            out.extend(wrap_line(line))
            i += 1
            continue

        if LIST_RE.match(line) or re.match(r"^\s*>\s+", line):
            flush_para()
            out.extend(wrap_line(line))
            i += 1
            continue

        if re.match(r"^\s{4,}\S", line) and not LIST_RE.match(line):
            flush_para()
            out.extend(wrap_line(line))
            i += 1
            continue

        if para_buf and (line.startswith(" ") or not stripped.startswith("#")):
            para_buf.append(line)
            i += 1
            continue

        flush_para()
        para_buf = [line]
        i += 1

    flush_para()
    return "\n".join(out) + ("\n" if text.endswith("\n") else "")


def tighten_remaining(text: str) -> str:
    lines = text.splitlines()
    out: list[str] = []
    in_fence = False
    in_front_matter = False
    in_display_math = False

    for i, line in enumerate(lines):
        stripped = line.strip()

        if stripped == "---" and i == 0:
            in_front_matter = True
            out.append(line)
            continue
        if in_front_matter:
            out.append(line)
            if stripped == "---":
                in_front_matter = False
            continue

        if FENCE_RE.match(line):
            in_fence = not in_fence
            out.append(line)
            continue
        if in_fence:
            out.append(line)
            continue

        if DISPLAY_MATH_START.match(line):
            in_display_math = True
            out.append(line)
            continue
        if in_display_math:
            out.append(line)
            if DISPLAY_MATH_END.match(line):
                in_display_math = False
            continue

        if HTML_RE.match(stripped) or HEADING_RE.match(line):
            out.append(line)
            continue

        if TABLE_RE.match(line):
            out.append(line)
            continue

        if len(line) > WIDTH:
            out.extend(wrap_line(line))
            continue

        out.append(line)

    return "\n".join(out) + ("\n" if text.endswith("\n") else "")


def process_file(text: str) -> str:
    updated = transform_file(text)
    for _ in range(3):
        nxt = tighten_remaining(updated)
        if nxt == updated:
            break
        updated = nxt
    return updated


def main(paths: list[str]) -> int:
    for path_str in paths:
        path = Path(path_str)
        original = path.read_text(encoding="utf-8")
        updated = process_file(original)
        if updated != original:
            path.write_text(updated, encoding="utf-8")
            print(f"wrapped: {path}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        docs = Path(__file__).resolve().parents[1] / "docs"
        targets = sorted(docs.rglob("*.md"))
    else:
        targets = [Path(p) for p in sys.argv[1:]]
    raise SystemExit(main([str(p) for p in targets]))
