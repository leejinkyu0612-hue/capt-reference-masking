# -*- coding: utf-8 -*-
"""Build an IC4e (IEEE 2-column, US letter) manuscript from Markdown using the official IC4e Word template
(converted from Strict to Transitional OOXML by Word), then export PDF with MS Word and report the page count.

Markdown conventions
  # Title
  Abstract—...                      -> 'Abstract' style
  Index Terms—...                   -> 'Keywords' style
  ## Heading (auto-numbered I., II.) ; ## Acknowledgment / ## References -> unnumbered 'Heading 5'
  ### Subheading (auto A., B.)
  ![Fig. 1. caption](path){width_cm}
  TABLE I. Caption   (line before a pipe table) ; 'Note:' line after table -> 'table footnote'
  - bullet
  [[FORM]] -> append the template's 'Authors' background' page (not part of the paper, not published)
Usage: python build_ieee.py manuscript.md out.docx [--form]"""
import re, sys, os, copy, subprocess
from docx import Document
from docx.shared import Pt, Cm, Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

HERE = os.path.dirname(os.path.abspath(__file__))
TEMPLATE = os.path.join(HERE, 'ic4e_template_tr.docx')
AUTHOR = ['Jinkyu Lee', 'HappyAI', 'Seoul, Republic of Korea', 'leejinkyu0612@naver.com', 'ORCID 0000-0002-0886-1572']
FORM_ROW = ['Jinkyu Lee', 'CEO (Ph.D.), HappyAI', 'AI for education; speech and language technology', 'https://happyaidata.kr']


def add_runs(p, text, size=None, bold=None, italic=None):
    parts = re.split(r'(\*\*[^*]+\*\*|\*[^*]+\*|\^[^^]+\^)', text)
    for part in parts:
        if not part:
            continue
        if part.startswith('**') and part.endswith('**'):
            r = p.add_run(part[2:-2]); r.bold = True
        elif part.startswith('*') and part.endswith('*') and len(part) > 2:
            r = p.add_run(part[1:-1]); r.italic = True
        elif part.startswith('^') and part.endswith('^'):
            r = p.add_run(part[1:-1]); r.font.superscript = True
        else:
            r = p.add_run(part)
        if size: r.font.size = Pt(size)
        if bold is not None and not part.startswith('**'): r.bold = bold if r.bold is None else r.bold
        if italic is not None and not (part.startswith('*') and not part.startswith('**')): r.italic = italic
    return p


def set_cell_border(cell, **kw):
    tcPr = cell._tc.get_or_add_tcPr()
    b = tcPr.find(qn('w:tcBorders'))
    if b is None:
        b = OxmlElement('w:tcBorders'); tcPr.append(b)
    for edge, val in kw.items():
        el = b.find(qn('w:' + edge))
        if el is None:
            el = OxmlElement('w:' + edge); b.append(el)
        for k, v in val.items():
            el.set(qn('w:' + k), str(v))


def make_sectpr(base, cols, space, sect_type='continuous'):
    s = copy.deepcopy(base)
    for el in s.findall(qn('w:cols')):
        s.remove(el)
    c = OxmlElement('w:cols'); c.set(qn('w:space'), str(space))
    if cols > 1: c.set(qn('w:num'), str(cols))
    # insert cols before docGrid if present
    dg = s.find(qn('w:docGrid'))
    if dg is not None: dg.addprevious(c)
    else: s.append(c)
    t = s.find(qn('w:type'))
    if t is None:
        t = OxmlElement('w:type'); s.insert(0, t)
    t.set(qn('w:val'), sect_type)
    for ref in s.findall(qn('w:footerReference')) + s.findall(qn('w:headerReference')):
        s.remove(ref)
    pg = s.find(qn('w:titlePg'))
    if pg is not None: s.remove(pg)
    return s


def build(md_path, out_docx, base_dir=None, form=False):
    base_dir = base_dir or os.path.dirname(os.path.abspath(md_path))
    doc = Document(TEMPLATE)
    body = doc.element.body
    sects = body.findall('.//' + qn('w:sectPr'))
    two_col = [s for s in sects if (s.find(qn('w:cols')) is not None and s.find(qn('w:cols')).get(qn('w:num')) == '2')]
    base_sect = copy.deepcopy(two_col[0] if two_col else sects[-1])
    # keep the authors' background form (last table + following paragraphs) for optional re-use
    form_tbl = copy.deepcopy(doc.tables[-1]._tbl)
    for el in list(body):
        if el.tag != qn('w:sectPr'):
            body.remove(el)
    final = body.find(qn('w:sectPr'))
    body.remove(final)
    body.append(make_sectpr(base_sect, 2, 360))

    lines = open(md_path, encoding='utf-8').read().split('\n')
    # title + author (one-column section)
    title = next(l[2:].strip() for l in lines if l.startswith('# '))
    doc.add_paragraph(title, style='paper title')
    p = doc.add_paragraph(style='Author'); p.add_run(AUTHOR[0])
    for k, a in enumerate(AUTHOR[1:]):
        q = doc.add_paragraph(style='Affiliation'); r = q.add_run(a); r.font.size = Pt(10)
        if k == 0: r.italic = True
        q.paragraph_format.space_after = Pt(0)
    last = doc.paragraphs[-1]; last.paragraph_format.space_after = Pt(12)
    pPr = last._p.get_or_add_pPr(); pPr.append(make_sectpr(base_sect, 1, 720, 'nextPage'))

    i = 0; in_refs = False; started = False
    while i < len(lines):
        ln = lines[i].rstrip()
        if ln.startswith('# ') or not ln.strip():
            i += 1; continue
        if ln.strip() == '[[FORM]]':
            form = True; i += 1; continue
        if ln.startswith('Abstract—'):
            p = doc.add_paragraph(style='Abstract'); r = p.add_run('Abstract—'); r.italic = True; r.bold = True
            add_runs(p, ln[len('Abstract—'):], bold=True); i += 1; continue
        if ln.startswith('Index Terms—') or ln.startswith('Keywords—'):
            key, rest = ln.split('—', 1)
            p = doc.add_paragraph(style='Keywords'); r = p.add_run(key + '—'); r.italic = True; r.bold = True
            add_runs(p, rest, italic=True, bold=True); i += 1; continue
        m = re.match(r'^(#{2,3}) (.*)', ln)
        if m:
            txt = m.group(2).strip()
            if len(m.group(1)) == 2 and txt.lower() in ('acknowledgment', 'references'):
                h = doc.add_paragraph(txt, style='Heading 5'); in_refs = txt.lower() == 'references'
            else:
                doc.add_paragraph(txt, style='Heading 1' if len(m.group(1)) == 2 else 'Heading 2')
            i += 1; continue
        fm = re.match(r'^!\[(.*?)\]\((.*)\)(?:\{([\d.]+)\})?\s*$', ln)
        if fm:
            cap, pth, w = fm.group(1), fm.group(2), float(fm.group(3) or 8.4)
            p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.space_before = Pt(4); p.paragraph_format.space_after = Pt(0); p.paragraph_format.first_line_indent = 0
            p.paragraph_format.keep_with_next = True
            p.add_run().add_picture(pth if os.path.isabs(pth) else os.path.join(base_dir, pth), width=Cm(w))
            c = doc.add_paragraph(style='figure caption'); add_runs(c, re.sub(r'^Fig\. \d+\.\s*', '', cap))  # style auto-numbers 'Fig. n.'
            i += 1; continue
        if re.match(r'^TABLE [IVX]+\.', ln.strip()):
            h = doc.add_paragraph(style='table head'); add_runs(h, re.sub(r'^TABLE [IVX]+\.\s*', '', ln.strip()))  # style auto-numbers 'TABLE n.'
            h.paragraph_format.keep_with_next = True
            i += 1; continue
        if ln.startswith('|'):
            rows = []
            while i < len(lines) and lines[i].startswith('|'):
                cells = [c.strip() for c in lines[i].strip().strip('|').split('|')]
                if not all(re.match(r'^:?-{2,}:?$', c) for c in cells):
                    rows.append(cells)
                i += 1
            ncol = max(len(r) for r in rows)
            t = doc.add_table(rows=len(rows), cols=ncol); t.alignment = WD_TABLE_ALIGNMENT.CENTER
            grp = [ri for ri, row in enumerate(rows) if ri > 0 and row[0].startswith('**') and all(not x for x in row[1:])]
            for ri, row in enumerate(rows):
                for ci in range(ncol):
                    cell = t.cell(ri, ci); cell.text = ''
                    para = cell.paragraphs[0]; para.style = doc.styles['table copy']
                    para.paragraph_format.space_after = Pt(0); para.paragraph_format.first_line_indent = 0
                    txt = row[ci] if ci < len(row) else ''
                    add_runs(para, txt, size=7.5)
                    if ri == 0:
                        for rr in para.runs: rr.bold = True
                    para.alignment = WD_ALIGN_PARAGRAPH.LEFT if ci == 0 else WD_ALIGN_PARAGRAPH.CENTER
                    top = {'val': 'single', 'sz': 8, 'color': '000000'} if ri == 0 else {'val': 'nil'}
                    bottom = {'val': 'single', 'sz': 4 if ri == 0 else 8, 'color': '000000'} if ri in (0, len(rows) - 1) else {'val': 'nil'}
                    set_cell_border(cell, top=top, bottom=bottom, left={'val': 'nil'}, right={'val': 'nil'})
                    tcPr = cell._tc.get_or_add_tcPr(); mar = OxmlElement('w:tcMar')
                    for side in ('left', 'right'):
                        e = OxmlElement('w:' + side); e.set(qn('w:w'), '40'); e.set(qn('w:type'), 'dxa'); mar.append(e)
                    tcPr.append(mar)
            for ri in grp:
                mg = t.cell(ri, 0).merge(t.cell(ri, ncol - 1))
                for para in mg.paragraphs[1:]:
                    para._element.getparent().remove(para._element)
                mg.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.LEFT
                set_cell_border(mg, top={'val': 'single', 'sz': 2, 'color': '808080'}, bottom={'val': 'nil'}, left={'val': 'nil'}, right={'val': 'nil'})
            # column widths proportional to content, total = column width (8.6 cm)
            body_rows = [r for k, r in enumerate(rows) if k not in grp]
            req = []
            for ci_ in range(ncol):
                cells = [re.sub(r'[*^]', '', r[ci_]) if ci_ < len(r) else '' for r in body_rows]
                L = max([len(x) for x in cells[1:]] + [max((len(w) for w in cells[0].split()), default=3), 3])
                req.append(min(L, 34) + 2)
            full = 8.6; tot = sum(req)
            t.autofit = False
            for ri in range(len(rows)):
                for ci_ in range(ncol):
                    t.cell(ri, ci_).width = Cm(full * req[ci_] / tot)
            continue
        if ln.startswith('Note:') or ln.startswith('ᵃ'):
            p = doc.add_paragraph(style='Body Text'); add_runs(p, ln.strip(), size=7); p.paragraph_format.first_line_indent = 0; p.alignment = WD_ALIGN_PARAGRAPH.LEFT
            p.paragraph_format.space_after = Pt(6); i += 1; continue
        if ln.startswith('- '):
            p = doc.add_paragraph(style='bullet list'); add_runs(p, ln[2:].strip()); i += 1; continue
        if in_refs:
            p = doc.add_paragraph(style='references'); add_runs(p, ln.strip()); i += 1; continue
        p = doc.add_paragraph(style='Body Text'); add_runs(p, ln.strip())
        i += 1
    if form:
        # authors' background form on a new page (template requirement; 'the form itself will not be published')
        last = doc.paragraphs[-1]
        pPr = last._p.get_or_add_pPr(); pPr.append(make_sectpr(base_sect, 2, 360, 'continuous'))
        h = doc.add_paragraph(); h.paragraph_format.page_break_before = True; h.alignment = WD_ALIGN_PARAGRAPH.LEFT
        r = h.add_run("Authors' background"); r.bold = True; r.font.size = Pt(16); r.font.name = 'Arial'
        body.find(qn('w:sectPr')).addprevious(form_tbl)
        tbl = doc.tables[-1]
        for ci, v in enumerate(FORM_ROW):
            cell = tbl.rows[1].cells[ci]; cell.text = ''
            rr = cell.paragraphs[0].add_run(v); rr.font.size = Pt(9); rr.font.name = 'Arial'
        n1 = doc.add_paragraph(); r = n1.add_run('*This form helps us to understand your paper better, the form itself will not be published.'); r.bold = True; r.font.size = Pt(9); r.font.name = 'Arial'
        n2 = doc.add_paragraph(); r = n2.add_run('*Title can be chosen from: master student, Phd candidate, assistant professor, lecture, senior lecture, associate professor, full professor'); r.bold = True; r.font.size = Pt(9); r.font.name = 'Arial'
        fs = body.find(qn('w:sectPr'))
        for el in fs.findall(qn('w:cols')):
            el.attrib.pop(qn('w:num'), None)
        n1.alignment = n2.alignment = WD_ALIGN_PARAGRAPH.LEFT
    # document properties
    cp = doc.core_properties; cp.author = 'Jinkyu Lee'; cp.title = title; cp.comments = ''; cp.last_modified_by = 'Jinkyu Lee'
    doc.save(out_docx)
    return out_docx


def to_pdf(docx_path, pdf_path):
    ps = os.path.join(HERE, 'docx2pdf.ps1')
    r = subprocess.run(['pwsh', '-NoProfile', '-File', ps, '-Docx', os.path.abspath(docx_path), '-Pdf', os.path.abspath(pdf_path)],
                       capture_output=True, text=True)
    m = re.search(r'WORDPAGES=(\d+)', r.stdout)
    if not m:
        raise RuntimeError('word export failed: ' + r.stdout[-500:] + r.stderr[-500:])
    return int(m.group(1))


if __name__ == '__main__':
    md, out = sys.argv[1], sys.argv[2]
    build(md, out, form='--form' in sys.argv)
    pdf = os.path.splitext(out)[0] + '.pdf'
    n = to_pdf(out, pdf)
    import fitz
    print('pages_word', n, 'pages_pdf', fitz.open(pdf).page_count, pdf)

