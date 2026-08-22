"""Convert Pacify markdown sources to PDF with page numbers and footers."""
import os, sys
from pathlib import Path
# project-relative: <root>/data
DATA_ROOT = str(Path(__file__).resolve().parents[2] / "data")

import re, os, glob
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                                TableStyle, KeepTogether)

SRC = f"{DATA_ROOT}/_source"
OUT = f"{DATA_ROOT}/documents"

MANUALS = {"manual_probook14", "manual_phonex", "manual_vision27"}

styles = getSampleStyleSheet()
S = {
    "h1": ParagraphStyle("h1", parent=styles["Heading1"], fontSize=17, leading=21,
                         spaceAfter=6, spaceBefore=2, textColor=colors.HexColor("#111111")),
    "h2": ParagraphStyle("h2", parent=styles["Heading2"], fontSize=13, leading=16,
                         spaceAfter=4, spaceBefore=10, textColor=colors.HexColor("#222222")),
    "h3": ParagraphStyle("h3", parent=styles["Heading3"], fontSize=11.5, leading=14,
                         spaceAfter=3, spaceBefore=8),
    "body": ParagraphStyle("body", parent=styles["BodyText"], fontSize=9.5, leading=13.5,
                           spaceAfter=4, alignment=0),
    "code": ParagraphStyle("code", parent=styles["BodyText"], fontName="Courier",
                           fontSize=8.5, leading=11, leftIndent=8, spaceAfter=5,
                           backColor=colors.HexColor("#f4f4f4")),
    "cell": ParagraphStyle("cell", parent=styles["BodyText"], fontSize=8, leading=10),
    "cellh": ParagraphStyle("cellh", parent=styles["BodyText"], fontSize=8, leading=10,
                            fontName="Helvetica-Bold"),
}


def inline(t: str) -> str:
    t = t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    t = re.sub(r"\*\*\*(.+?)\*\*\*", r"<b><i>\1</i></b>", t)
    t = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", t)
    t = re.sub(r"(?<!\*)\*([^*]+?)\*(?!\*)", r"<i>\1</i>", t)
    t = re.sub(r"`(.+?)`", r"<font face='Courier'>\1</font>", t)
    return t


def parse_table(lines, i):
    rows = []
    while i < len(lines) and lines[i].strip().startswith("|"):
        raw = lines[i].strip().strip("|")
        cells = [c.strip() for c in raw.split("|")]
        if not all(re.fullmatch(r":?-{2,}:?", c) for c in cells if c):
            rows.append(cells)
        i += 1
    return rows, i


def build_table(rows):
    if not rows:
        return None
    ncol = max(len(r) for r in rows)
    rows = [r + [""] * (ncol - len(r)) for r in rows]
    data = []
    for ri, r in enumerate(rows):
        st = S["cellh"] if ri == 0 else S["cell"]
        data.append([Paragraph(inline(c), st) for c in r])
    avail = 170 * mm
    w = avail / ncol
    t = Table(data, colWidths=[w] * ncol, repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8e8e8")),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#999999")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.white, colors.HexColor("#fafafa")]),
    ]))
    return t


def md_to_story(path):
    lines = open(path, encoding="utf-8").read().split("\n")
    story, i = [], 0
    while i < len(lines):
        ln = lines[i]
        s = ln.strip()
        if not s:
            i += 1
            continue
        if s.startswith("```"):
            i += 1
            buf = []
            while i < len(lines) and not lines[i].strip().startswith("```"):
                buf.append(lines[i].replace(" ", "&nbsp;"))
                i += 1
            i += 1
            story.append(Paragraph("<br/>".join(buf), S["code"]))
            continue
        if s.startswith("|"):
            rows, i = parse_table(lines, i)
            t = build_table(rows)
            if t:
                story.append(Spacer(1, 3))
                story.append(t)
                story.append(Spacer(1, 6))
            continue
        if s.startswith("---"):
            story.append(Spacer(1, 5))
            i += 1
            continue
        if s.startswith("### "):
            story.append(Paragraph(inline(s[4:]), S["h3"]))
        elif s.startswith("## "):
            story.append(Paragraph(inline(s[3:]), S["h2"]))
        elif s.startswith("# "):
            story.append(Paragraph(inline(s[2:]), S["h1"]))
        elif s.startswith("- "):
            story.append(Paragraph("&bull; " + inline(s[2:]), S["body"]))
        else:
            story.append(Paragraph(inline(s), S["body"]))
        i += 1
    return story


def make_footer(doc_ref):
    def footer(canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 7)
        canvas.setFillColor(colors.HexColor("#666666"))
        canvas.line(20 * mm, 15 * mm, 190 * mm, 15 * mm)
        canvas.drawString(20 * mm, 11 * mm, doc_ref)
        canvas.drawRightString(190 * mm, 11 * mm, f"Page {doc.page}")
        canvas.drawCentredString(105 * mm, 11 * mm, "Pacify Electronics Pvt. Ltd.")
        canvas.restoreState()
    return footer


def extract_ref(path):
    txt = open(path, encoding="utf-8").read()
    m = re.search(r"\*\*Document(?: reference)?:\*\*\s*([A-Z0-9\-]+)", txt)
    ref = m.group(1) if m else os.path.basename(path).replace(".md", "").upper()
    m2 = re.search(r"\*\*Effective from:\*\*\s*(.+)", txt)
    eff = m2.group(1).strip() if m2 else "15 January 2026"
    return f"{ref}  |  Effective {eff}"


os.makedirs(OUT, exist_ok=True)
os.makedirs(f"{OUT}/manuals", exist_ok=True)

results = []
for src in sorted(glob.glob(f"{SRC}/*.md")):
    name = os.path.basename(src).replace(".md", "")
    sub = "manuals/" if name in MANUALS else ""
    dst = f"{OUT}/{sub}{name}.pdf"
    doc = SimpleDocTemplate(dst, pagesize=A4,
                            leftMargin=20 * mm, rightMargin=20 * mm,
                            topMargin=18 * mm, bottomMargin=22 * mm,
                            title=name, author="Pacify Electronics Pvt. Ltd.")
    f = make_footer(extract_ref(src))
    doc.build(md_to_story(src), onFirstPage=f, onLaterPages=f)
    results.append((f"{sub}{name}.pdf", doc.page))

print(f"{'document':45s} pages")
print("-" * 55)
total = 0
for n, p in results:
    print(f"{n:45s} {p:5d}")
    total += p
print("-" * 55)
print(f"{'TOTAL':45s} {total:5d}")
