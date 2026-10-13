"""Download the primary regulatory texts and write data/corpus/ plus manifest.json.

Needs network access, requests and pymupdf (pip install -e ".[models]"). Not used by the tests.
"""
from __future__ import annotations

import datetime as dt
import re
import sys
import time
from pathlib import Path

import pymupdf
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from reg_rag.corpus import (  # noqa: E402
    ManifestEntry, clean_pdf_text, ecfr_xml_to_markdown, normalize_guidance_text, sha256_bytes, sha256_file,
    write_manifest,
)

# config
CORPUS_DIR = ROOT / "data" / "corpus"
ECFR_DATE = "2026-10-01"  # point-in-time eCFR version so the corpus is reproducible
ECFR_URL = "https://www.ecfr.gov/api/versioner/v1/full/{date}/title-12.xml?{query}"
SR117_URL = "https://www.federalreserve.gov/boarddocs/srletters/2011/sr1107a1.pdf"
FR2052A_URL = "https://www.federalreserve.gov/reportforms/forms/FR_2052a20220429_f.pdf"
ECFR_DOCS = {
    "12cfr249": ("12 CFR Part 249 Liquidity Risk Measurement, Standards, and Monitoring (Regulation WW)", ["part=249"]),
    "12cfr252-d": ("12 CFR Part 252 Subpart D Enhanced Prudential Standards for BHCs with $100 billion or more (incl. liquidity, 252.34 and 252.35)", ["part=252&subpart=D"]),
    "12cfr252-e": ("12 CFR Part 252 Subpart E Supervisory Stress Test Requirements", ["part=252&subpart=E"]),
    "12cfr217-b": ("12 CFR Part 217 Subpart B Capital Ratio Requirements and Buffers", ["part=217&subpart=B"]),
    "12cfr217-c": ("12 CFR Part 217 Subpart C Definition of Capital", ["part=217&subpart=C"]),
    "12cfr217-d-excerpt": ("12 CFR Part 217 Subpart D excerpt: 217.30, 217.31, 217.32 (standardized approach risk weights)",
                           [f"part=217&section=217.{n}" for n in (30, 31, 32)]),
}
SR117_SUBHEADINGS = [
    "Model Development and Implementation", "Model Use", "Key Elements of Comprehensive\nValidation",
    "1. Evaluation of Conceptual Soundness", "2. Ongoing Monitoring", "3. Outcomes Analysis",
    "Validation of Vendor and Other Third-Party Products", "Board of Directors and Senior Management",
    "Policies and Procedures", "Roles and Responsibilities", "Internal Audit", "External Resources",
    "Model Inventory", "Documentation",
]
PDF_DROP = [r"Page \d+( of \d+)?", r"FR 2052a Instructions"]
HEADERS = {"User-Agent": "Mozilla/5.0", "Accept-Encoding": "gzip"}
# end config


def get(url: str) -> bytes:
    for attempt in range(5):  # eCFR intermittently answers 502/503 under load
        r = requests.get(url, headers=HEADERS, timeout=120)
        if r.status_code < 500:
            break
        time.sleep(3 * (attempt + 1))
    r.raise_for_status()
    return r.content


def pdf_text(data: bytes) -> str:
    doc = pymupdf.open(stream=data, filetype="pdf")
    return "\n".join(page.get_text("text") for page in doc)


def main() -> None:
    CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    today = dt.date.today().isoformat()
    entries: list[ManifestEntry] = []

    for doc_id, (title, queries) in ECFR_DOCS.items():
        urls = [ECFR_URL.format(date=ECFR_DATE, query=q) for q in queries]
        try:
            raws = [get(u) for u in urls]
            md = "\n\n".join(ecfr_xml_to_markdown(r.decode("utf-8")) for r in raws)
            status, note = "ok", f"eCFR point-in-time version {ECFR_DATE}"
        except Exception as exc:  # record and continue
            raws, md, status, note = [], "", "failed", repr(exc)
        out = CORPUS_DIR / f"{doc_id}.md"
        if md:
            out.write_text(md, encoding="utf-8")
        entries.append(ManifestEntry(doc_id, title, out.name, urls, today, [sha256_bytes(r) for r in raws],
                                     sha256_file(out) if md else "", len(md), status, note))
        print(doc_id, status, len(md))

    try:
        raw = get(SR117_URL)
        txt = normalize_guidance_text(clean_pdf_text(pdf_text(raw), PDF_DROP), "I. INTRODUCTION",
                                      r"[IVX]+\. [A-Z][A-Z ,&-]+", SR117_SUBHEADINGS)
        out = CORPUS_DIR / "sr11-7.md"
        out.write_text(txt, encoding="utf-8")
        entries.append(ManifestEntry("sr11-7", "SR 11-7 Attachment: Supervisory Guidance on Model Risk Management (April 4, 2011)",
                                     out.name, [SR117_URL], today, [sha256_bytes(raw)], sha256_file(out), len(txt),
                                     note="text extracted with PyMuPDF (markitdown reordered footnotes into the body); "
                                          "footnotes dropped, paragraphs rebuilt, contents page removed"))
        print("sr11-7 ok", len(txt))
    except Exception as exc:
        entries.append(ManifestEntry("sr11-7", "SR 11-7", "sr11-7.md", [SR117_URL], today, [], "", 0, "failed", repr(exc)))

    try:
        raw = get(FR2052A_URL)
        txt = clean_pdf_text(pdf_text(raw), PDF_DROP)
        starts = [m.start() for m in re.finditer(r"^General Instructions\s*\n\s*Purpose", txt, re.M)]
        ends = [m.start() for m in re.finditer(r"^Product Definitions\s*\n", txt, re.M)]
        body = "# FR 2052a Instructions\n" + normalize_guidance_text(
            txt[starts[-1]:ends[-1]], "General Instructions", r"\A\Z", [], infer_headings=True)
        out = CORPUS_DIR / "fr2052a-instructions.md"
        out.write_text(body, encoding="utf-8")
        entries.append(ManifestEntry("fr2052a-instructions",
                                     "FR 2052a Instructions (version dated 2022-04-29, General Instructions and Field Definitions only)",
                                     out.name, [FR2052A_URL], today, [sha256_bytes(raw)], sha256_file(out), len(body),
                                     note="product definition tables omitted; the 2022 version may not be the current form"))
        print("fr2052a ok", len(body))
    except Exception as exc:
        entries.append(ManifestEntry("fr2052a-instructions", "FR 2052a Instructions", "fr2052a-instructions.md",
                                     [FR2052A_URL], today, [], "", 0, "failed", repr(exc)))

    write_manifest(entries, CORPUS_DIR / "manifest.json")


if __name__ == "__main__":
    main()
