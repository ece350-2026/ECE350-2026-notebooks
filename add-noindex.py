#!/usr/bin/env python3
"""
Add noindex/nofollow meta tags to all index.html under docs/ if missing.
Run after marimo WASM exports so re-exports don't re-enable search indexing.

Usage (from repo root):
  python docs/add-noindex.py
"""
from pathlib import Path
import re

DOCS = Path(__file__).resolve().parent


def add_noindex_if_missing(filepath: Path) -> bool:
    """Add noindex meta tags after charset if missing. Returns True if file was modified."""
    text = filepath.read_text(encoding="utf-8")
    if "noindex" in text:
        return False
    # Insert after first <meta charset...> line; handle various formats
    match = re.search(r"^(\s*)(<meta\s+charset=[^>]+>\s*\n)", text, re.IGNORECASE | re.MULTILINE)
    if not match:
        return False
    indent = match.group(1)
    charset_line = match.group(2)
    insert = f"{indent}<meta name=\"robots\" content=\"noindex, nofollow\">\n{indent}<meta name=\"googlebot\" content=\"noindex, nofollow\">\n"
    new_text = text.replace(charset_line, charset_line + insert, 1)
    if new_text == text:
        return False
    filepath.write_text(new_text, encoding="utf-8")
    return True

def main():
    modified = []
    for path in sorted(DOCS.rglob("index.html")):
        if path.is_file() and path.parent.name != "layouts":
            if add_noindex_if_missing(path):
                modified.append(path.relative_to(DOCS))
    if modified:
        print("Added noindex meta tags to:")
        for p in modified:
            print(f"  {p}")
    else:
        print("All index.html files already have noindex. No changes.")

if __name__ == "__main__":
    main()
