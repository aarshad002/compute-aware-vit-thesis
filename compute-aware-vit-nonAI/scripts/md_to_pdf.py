"""Convert a Markdown file to PDF (pure-Python: markdown + xhtml2pdf).

Usage: python scripts/md_to_pdf.py <input.md> [output.pdf]
"""
import sys
from pathlib import Path

import markdown
from xhtml2pdf import pisa

# characters that reportlab's base fonts can't render -> ASCII fallbacks
REPLACEMENTS = {
    "→": "->", "←": "<-",
    "✅ ": "", "❌ ": "", "✅": "", "❌": "", "✓": "", "✗": "",
    "≥": ">=", "≤": "<=", "≈": "~", "≠": "!=",
    "₀": "0", "₁": "1", "₂": "2", "₃": "3", "₄": "4",
    "₅": "5", "₆": "6", "₇": "7", "₈": "8", "₉": "9",
    "‑": "-", "–": "-", "—": "-", "‘": "'", "’": "'",
    "“": '"', "”": '"', "…": "...", "×": "x", "⚠": "[!]",
}

CSS = """
@page { size: A4; margin: 1.6cm 1.5cm; }
body { font-family: Helvetica, Arial, sans-serif; font-size: 9.5pt; color: #1a1a1a; line-height: 1.4; }
h1 { font-size: 17pt; border-bottom: 2px solid #333; padding-bottom: 4px; }
h2 { font-size: 13pt; margin-top: 16px; border-bottom: 1px solid #bbb; padding-bottom: 2px; }
h3 { font-size: 11pt; }
table { border: 1px solid #888; border-collapse: collapse; width: 100%; margin: 8px 0; }
th { background-color: #e8e8e8; border: 1px solid #888; padding: 4px 6px; text-align: left; font-size: 9pt; }
td { border: 1px solid #888; padding: 4px 6px; font-size: 9pt; }
code { background-color: #f0f0f0; font-family: Courier, monospace; font-size: 8.5pt; }
pre { background-color: #f5f5f5; padding: 6px; border: 1px solid #ddd; }
hr { border: none; border-top: 1px solid #ccc; }
"""


def convert(md_path: Path, pdf_path: Path):
    text = md_path.read_text(encoding="utf-8")
    for a, b in REPLACEMENTS.items():
        text = text.replace(a, b)
    body = markdown.markdown(text, extensions=["tables", "fenced_code", "sane_lists"])
    html = f"<html><head><meta charset='utf-8'><style>{CSS}</style></head><body>{body}</body></html>"
    with open(pdf_path, "wb") as f:
        result = pisa.CreatePDF(html, dest=f)
    if result.err:
        raise RuntimeError(f"PDF generation had {result.err} error(s)")
    print(f"wrote {pdf_path}")


if __name__ == "__main__":
    inp = Path(sys.argv[1])
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else inp.with_suffix(".pdf")
    convert(inp, out)
