#!/usr/bin/env python3
"""
Sync model.md into the notebook as a markdown cell.

model.md is the source of truth for the model description.
This script replaces the corresponding markdown cell in the
notebook with the contents of model.md — needed as a pre-build
step for the static site (mkdocs-jupyter with execute: false).

Also fixes any display_data outputs missing the required
'metadata' field (nbformat validation).

Usage:
    python sync_md_to_notebook.py          # preview (dry-run)
    python sync_md_to_notebook.py --apply  # write changes
"""

from __future__ import annotations
import argparse, json, sys
from pathlib import Path

NB_PATH = Path(__file__).parent / 'retirement_fues.ipynb'
MD_PATH = Path(__file__).parent / 'model.md'

ANCHOR = '## 1. Model'


def source_to_list(text: str) -> list[str]:
    if not text:
        return []
    lines = text.split('\n')
    result = [line + '\n' for line in lines[:-1]]
    if lines[-1]:
        result.append(lines[-1])
    return result


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--apply', action='store_true',
                        help='Write changes to the notebook (default: dry-run)')
    args = parser.parse_args()

    if not MD_PATH.exists():
        print(f'ERROR: {MD_PATH} not found', file=sys.stderr)
        sys.exit(1)

    with open(NB_PATH) as f:
        nb = json.load(f)

    md_content = MD_PATH.read_text().rstrip()
    changes = 0

    # Find the cell that starts with the anchor
    for i, cell in enumerate(nb['cells']):
        src = ''.join(cell['source']).strip()
        if not src.startswith(ANCHOR):
            continue

        if cell['cell_type'] == 'code':
            print(f'  CONVERT cell {i}: code → markdown')
            if args.apply:
                cell['cell_type'] = 'markdown'
                cell['source'] = source_to_list(md_content)
                cell.pop('execution_count', None)
                cell.pop('outputs', None)
            changes += 1
        elif cell['cell_type'] == 'markdown':
            old = ''.join(cell['source']).rstrip()
            if old != md_content:
                print(f'  UPDATE cell {i}: markdown content changed')
                if args.apply:
                    cell['source'] = source_to_list(md_content)
                changes += 1
            else:
                print(f'  cell {i}: already up to date')
        break
    else:
        print(f'WARNING: no cell starting with "{ANCHOR}" found', file=sys.stderr)

    # Fix invalid outputs (nbformat validation)
    out_fixes = 0
    for cell in nb['cells']:
        for out in cell.get('outputs', []):
            if out.get('output_type') in ('display_data', 'execute_result'):
                if 'metadata' not in out:
                    out['metadata'] = {}
                    out_fixes += 1
            if out.get('output_type') == 'stream':
                if 'name' not in out:
                    out['name'] = 'stdout'
                    out_fixes += 1

    if out_fixes:
        print(f'  FIX {out_fixes} output(s) with missing required fields')
        changes += out_fixes

    if changes == 0:
        print('Nothing to do.')
        return

    if not args.apply:
        print(f'\n{changes} change(s) pending — use --apply to write.')
        return

    with open(NB_PATH, 'w') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
        f.write('\n')
    print(f'Wrote {NB_PATH.name}')


if __name__ == '__main__':
    main()
