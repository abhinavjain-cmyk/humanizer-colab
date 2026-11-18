"""
hz_doc.py - document pipeline for the humanizer notebook.

  parse_document(path)            -> {"front": [...], "blocks": [{"kind","text"}...]}
      kinds: set | q | h | p      (PDF in the MUJ template layout, or .docx)
  humanize_blocks(doc, rewrite, min_words, max_words, samples, log)
      -> new doc; `rewrite(text) -> str` is the model call
  build_pdf(doc, out_path)        -> A4, Times 12/16, 1 inch margins, front page, justified
  doc_to_text(doc)                -> plain text (for a detector paste)

Layout is the MUJ assignment template measured from the user's accepted PDF:
A4, 72 pt margins, Times-Roman 12 on 16 pt leading, front page with six
centred bold LABEL : VALUE lines, SET headings centred bold, questions bold
justified, sub-headings bold left, paragraphs justified with 10 pt after.
"""

import re

WORD = re.compile(r"[A-Za-z0-9][A-Za-z0-9'\-]*")
SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")
PROPER = re.compile(r"(?<![.!?]\s)(?<!^)\b[A-Z][a-z]{2,}")
NUMBER = re.compile(r"\d")
QNUM = re.compile(r"^\s*(\d{1,2})\s*[.)]\s+")


def words(s):
    return len(WORD.findall(s))


# ----------------------------------------------------------------------------
# parsing
# ----------------------------------------------------------------------------


def _pdf_lines(path):
    import pdfplumber

    out = []
    with pdfplumber.open(path) as pdf:
        for pi, pg in enumerate(pdf.pages):
            ws = pg.extract_words(extra_attrs=["fontname", "size"], use_text_flow=True)
            rows = {}
            for w in ws:
                rows.setdefault(round(w["top"], 1), []).append(w)
            for t in sorted(rows):
                L = sorted(rows[t], key=lambda w: w["x0"])
                out.append(
                    dict(
                        page=pi,
                        top=t,
                        x0=L[0]["x0"],
                        x1=L[-1]["x1"],
                        bold=all("Bold" in w["fontname"] or "bold" in w["fontname"] for w in L),
                        text=" ".join(w["text"] for w in L),
                    )
                )
    return out


def _classify_bold(t):
    if re.match(r"^SET\s*[-:]?\s*\d", t, re.I):
        return "set"
    if QNUM.match(t):
        return "q"
    return "h"


def parse_pdf(path):
    ls = _pdf_lines(path)
    if not ls:
        raise ValueError("no text layer in this PDF (scanned?)")
    # front page = page 0 if it has no question lines
    page0 = [l for l in ls if l["page"] == 0]
    has_q0 = any(l["bold"] and QNUM.match(l["text"]) for l in page0)
    if not has_q0 and len(page0) <= 10:
        front = [l["text"] for l in page0]
        body = [l for l in ls if l["page"] > 0]
    else:
        front, body = [], ls
    blocks, cur, prev = [], None, None
    for l in body:
        same_page = prev is not None and l["page"] == prev["page"]
        step = abs(l["top"] - prev["top"]) if prev is not None else 0
        cont = (
            prev is not None
            and cur is not None
            and l["bold"] == prev["bold"]
            and (
                (same_page and 12 <= step <= 20 and cur["kind"] in ("p", "q"))
                or (not same_page and prev["x1"] > 480 and cur["kind"] in ("p", "q"))
            )
        )
        # a new numbered point or a line that starts far right breaks continuation
        if cont and not l["bold"] and re.match(r"^\s*\d{1,2}[.)]\s", l["text"]) and prev["x1"] < 470:
            cont = False
        if cont:
            cur["text"] += " " + l["text"]
        else:
            kind = _classify_bold(l["text"]) if l["bold"] else "p"
            cur = dict(kind=kind, text=l["text"].strip())
            blocks.append(cur)
        prev = l
    return dict(front=front, blocks=blocks)


def parse_docx(path):
    import docx

    d = docx.Document(path)
    paras = []
    for p in d.paragraphs:
        t = p.text.strip()
        if not t:
            continue
        runs = [r for r in p.runs if r.text.strip()]
        bold = bool(runs) and all(bool(r.bold) or (p.style and "Heading" in p.style.name) for r in runs)
        centred = p.alignment == 1
        paras.append(dict(text=t, bold=bold, centred=centred, style=p.style.name if p.style else ""))
    # front = leading short lines before the first SET / question
    front, i = [], 0
    while i < len(paras) and not (paras[i]["bold"] and _classify_bold(paras[i]["text"]) in ("set", "q")) and i < 10:
        front.append(paras[i]["text"])
        i += 1
    if not any(":" in f for f in front):
        front, i = [], 0
    blocks = []
    for p in paras[i:]:
        t = p["text"]
        if p["bold"] or "Heading" in p["style"]:
            kind = _classify_bold(t)
        elif QNUM.match(t) and words(t) > 6 and t.rstrip().endswith(("?", ".")) and t[0].isdigit() and not blocks:
            kind = "q"
        else:
            kind = "p"
        blocks.append(dict(kind=kind, text=t))
    return dict(front=front, blocks=blocks)


def parse_document(path):
    low = path.lower()
    if low.endswith(".pdf"):
        return parse_pdf(path)
    if low.endswith(".docx"):
        return parse_docx(path)
    raise ValueError("give a .pdf or .docx")


# ----------------------------------------------------------------------------
# answers in / answers out
# ----------------------------------------------------------------------------


def groups(doc):
    """[(kind, [blocks])]: 'set' singletons, 'q' groups with their h/p, 'pre' for anything before the first question."""
    out, cur = [], None
    for b in doc["blocks"]:
        if b["kind"] == "set":
            out.append(("set", [b]))
            cur = None
        elif b["kind"] == "q":
            cur = [b]
            out.append(("q", cur))
        else:
            if cur is None:
                cur = []
                out.append(("pre", cur))
            cur.append(b)
    return out


def answer_text(blocks):
    """Body of one answer as the model sees it: headings on their own line, paragraphs separated by blank lines."""
    parts = []
    for b in blocks[1:]:
        parts.append(b["text"])
    return "\n\n".join(parts)


def text_to_blocks(out):
    """Model output back to h/p blocks. Short line with no final stop = heading."""
    out = out.replace("[end of text]", "")
    out = re.sub(r"(?m)^#+\s*", "", out)
    out = re.sub(r"\*\*|__", "", out)
    paras = [p.strip() for p in re.split(r"\n\s*\n", out) if p.strip()]
    if paras and QNUM.match(paras[0]) and words(paras[0]) > 5 and len(paras[0]) < 400:
        paras = paras[1:]  # the model re-stated the question
    blocks = []
    for p in paras:
        for l in [x.strip() for x in p.split("\n") if x.strip()]:
            l = re.sub(r"^\s*[-*•]\s+", "", l)
            w = words(l)
            is_head = w <= 10 and not re.match(r"^\d+[.)]", l) and (not l.endswith((".", "!", ":", ",")) or l.isupper())
            if is_head:
                l = l.rstrip("?:.")
                if l.isupper():
                    l = l.capitalize()
                blocks.append(dict(kind="h", text=l))
            else:
                blocks.append(dict(kind="p", text=l))
    return blocks


# ---- trimming (whole sentences, least specific first) ----


def _specific(text):
    return bool(PROPER.search(" " + text[1:]) or NUMBER.search(text))


GLUE = re.compile(r"^(And|But|So|Then|Which|Else|Because|That|This|It|They|These|Those|Such|Here|There|Hence|Thus|Otherwise|Also)\b")
SETUP = re.compile(r"^(If|When|Suppose|Say|Imagine|Take|Consider)\b")


def _protected(sents, si):
    return si == 0 or (si == 1 and len(sents[0].split()) < 8)


def _span(sents, si):
    idx = [si]
    if SETUP.match(sents[si]) and si + 1 < len(sents):
        idx.append(si + 1)
    j = idx[-1] + 1
    while j < len(sents) and GLUE.match(sents[j]):
        idx.append(j)
        j += 1
    return idx


def _pick(blocks, allow_specific):
    order = sorted([i for i, b in enumerate(blocks) if b["kind"] == "p"], key=lambda i: -words(blocks[i]["text"]))
    for pi in order:
        sents = SENT.split(blocks[pi]["text"].strip())
        if len(sents) < 2:
            continue
        for si in range(len(sents) - 1, -1, -1):
            if _protected(sents, si):
                continue
            group = _span(sents, si)
            if any(_protected(sents, g) for g in group):
                continue
            text = " ".join(sents[g] for g in group)
            if (_specific(text) or len(text.split()) < 8) and not allow_specific:
                continue
            return pi, group
    return None


def body_words(blocks):
    return sum(words(b["text"]) for b in blocks if b["kind"] in ("p", "h"))


def trim_blocks(blocks, max_words):
    total = body_words(blocks)
    while total > max_words:
        hit = _pick(blocks, False) or _pick(blocks, True)
        if hit is None:
            break
        pi, group = hit
        sents = SENT.split(blocks[pi]["text"].strip())
        removed = " ".join(sents[g] for g in group)
        blocks[pi] = dict(kind="p", text=" ".join(x for g, x in enumerate(sents) if g not in group))
        total -= words(removed)
    return blocks, total


# ---- numbers check ----


def numbers(s):
    return set(re.findall(r"\d[\d,.]*\d|\d", s))


def _humanize_answer(blks, rewrite, min_words, max_words, samples):
    """One answer: rewrite (best of `samples`), re-block, trim to band. Returns (blocks incl. question, report row, log lines)."""
    q = blks[0]
    src = answer_text(blks)
    src_words = words(src)
    want = numbers(src)
    best, lines = None, []
    for s in range(max(1, samples)):
        out = rewrite(src)
        cand = text_to_blocks(out)
        n = body_words(cand)
        nums_ok = want <= numbers(out)
        in_band = min_words <= n <= max_words
        score = (nums_ok, in_band, n >= min_words, -abs(n - (min_words + max_words) / 2))
        lines.append(f"   sample {s + 1}: {n} words, numbers kept: {nums_ok}")
        if best is None or score > best[0]:
            best = (score, cand, n, nums_ok)
        if nums_ok and in_band:
            break
    cand, n, nums_ok = best[1], best[2], best[3]
    if n > max_words:
        cand, n = trim_blocks(cand, max_words)
    flag = "" if min_words <= n <= max_words else ("SHORT" if n < min_words else "LONG")
    row = dict(question=q["text"][:70], words_in=src_words, words_out=n, numbers_ok=nums_ok, flag=flag)
    lines.append(f"   -> {src_words} -> {n} words {flag}{'' if nums_ok else '  CHECK NUMBERS'}")
    return [q] + cand, row, lines


def humanize_docs(docs, rewrite, min_words=400, max_words=450, samples=1, workers=6, log=print):
    """Rewrite every answer of every doc, `workers` answers at a time across all docs.
    docs: list of (name, doc). Returns list of (name, new_doc, report) in input order."""
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import time

    plan, jobs = [], []
    for di, (name, doc) in enumerate(docs):
        parts = groups(doc)
        plan.append(parts)
        for gi, (kind, blks) in enumerate(parts):
            if kind == "q":
                jobs.append((di, gi, blks))
    log(f"{len(jobs)} answers from {len(docs)} document(s), {workers} at a time")
    done, results, t0 = 0, {}, time.time()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_humanize_answer, blks, rewrite, min_words, max_words, samples): (di, gi) for di, gi, blks in jobs}
        for f in as_completed(futs):
            di, gi = futs[f]
            try:
                results[(di, gi)] = f.result()
            except Exception as e:  # keep the original answer if the model call failed
                blks = plan[di][gi][1]
                results[(di, gi)] = (blks, dict(question=blks[0]["text"][:70], words_in=body_words(blks), words_out=body_words(blks), numbers_ok=True, flag=f"ERROR {type(e).__name__}"), [f"   failed: {e}"])
            done += 1
            log(f"[{done}/{len(jobs)} {time.time() - t0:.0f}s] {docs[di][0][:40]} | {results[(di, gi)][1]['question'][:50]}")
            for line in results[(di, gi)][2]:
                log(line)
    out = []
    for di, (name, doc) in enumerate(docs):
        blocks, report = [], []
        for gi, (kind, blks) in enumerate(plan[di]):
            if kind == "q":
                nb, row, _ = results[(di, gi)]
                blocks.extend(nb)
                report.append(row)
            else:
                blocks.extend(blks)
        out.append((name, dict(front=doc["front"], blocks=blocks), report))
    return out


def humanize_blocks(doc, rewrite, min_words=400, max_words=450, samples=1, log=print):
    """Single-document, sequential (kept for compatibility)."""
    (_, new, report), = humanize_docs([("doc", doc)], rewrite, min_words, max_words, samples, workers=1, log=log)
    return new, report


# ----------------------------------------------------------------------------
# output
# ----------------------------------------------------------------------------


def doc_to_text(doc):
    return "\n\n".join(b["text"] for b in doc["blocks"])


def build_pdf(doc, out_path):
    from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer

    def esc(t):
        return t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    base = dict(fontName="Times-Roman", fontSize=12, leading=16)
    S = {
        "front": ParagraphStyle("front", fontName="Times-Bold", fontSize=12, leading=16, alignment=TA_CENTER, spaceAfter=16),
        "set": ParagraphStyle("set", fontName="Times-Bold", fontSize=12, leading=16, alignment=TA_CENTER, spaceAfter=10, keepWithNext=1),
        "q": ParagraphStyle("q", fontName="Times-Bold", fontSize=12, leading=16, alignment=TA_JUSTIFY, spaceBefore=12, spaceAfter=10),
        "h": ParagraphStyle("h", fontName="Times-Bold", fontSize=12, leading=16, alignment=TA_LEFT, spaceAfter=6, keepWithNext=1),
        "p": ParagraphStyle("p", **base, alignment=TA_JUSTIFY, spaceAfter=10),
    }
    pdf = SimpleDocTemplate(out_path, pagesize=A4, leftMargin=72, rightMargin=72, topMargin=72, bottomMargin=72)
    story = []
    if doc["front"]:
        story.append(Spacer(1, 144))
        for line in doc["front"]:
            story.append(Paragraph(esc(line), S["front"]))
        story.append(PageBreak())
    for b in doc["blocks"]:
        story.append(Paragraph(esc(b["text"]), S[b["kind"]]))
    pdf.build(story)
    return out_path


def word_report(doc):
    rows = []
    for kind, blks in groups(doc):
        if kind == "q":
            rows.append((blks[0]["text"][:60], body_words(blks)))
    return rows
