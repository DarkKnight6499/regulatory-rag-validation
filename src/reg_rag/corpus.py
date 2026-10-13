"""Corpus text preparation: eCFR XML to markdown, PDF text cleanup, manifest handling."""
from __future__ import annotations

import hashlib
import json
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, asdict
from pathlib import Path

SKIP_TAGS = {"CITA", "AUTH", "SOURCE", "EDNOTE", "SECAUTH", "APPRO", "HED"}
PARA_TAGS = {"P", "FP", "FP-1", "FP-2", "NOTE", "EXTRACT"}
HEADING_PREFIX = {"DIV5": "#", "DIV6": "##", "DIV7": "###", "DIV8": "###"}


def spaced_hyphens(text: str) -> str:
    """Repo style forbids em dashes, so source em dashes become spaced hyphens."""
    return re.sub(r"\s*\u2014\s*", " - ", text)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(Path(path).read_bytes())


def _flat(el: ET.Element) -> str:
    return re.sub(r"\s+", " ", "".join(el.itertext())).strip()


def _walk(el: ET.Element, out: list[str]) -> None:
    tag = el.tag
    if tag in SKIP_TAGS:
        return
    if tag in HEADING_PREFIX:
        head = el.find("HEAD")
        if head is not None:
            out.append("")
            out.append(f"{HEADING_PREFIX[tag]} {_flat(head)}")
        for child in el:
            if child.tag != "HEAD":
                _walk(child, out)
        return
    if tag in PARA_TAGS:
        text = _flat(el)
        if text:
            out.append(text)
        return
    if tag == "FTNT":
        text = re.sub(r"^\d+\s*", "", _flat(el))
        if text:
            out.append(f"[footnote] {text}")
        return
    if tag in {"TABLE", "GPOTABLE", "GPH"}:
        text = _flat(el)
        if text:
            out.append(f"[table] {text}")
        return
    if tag == "HEAD":
        return
    for child in el:
        _walk(child, out)


def ecfr_xml_to_markdown(xml_text: str) -> str:
    """Convert an eCFR versioner XML fragment into heading plus one-paragraph-per-line markdown."""
    xml_text = xml_text.replace("﻿", "")
    xml_text = re.sub(r"^\s*<\?xml[^>]*\?>", "", xml_text)
    root = ET.fromstring(xml_text)
    lines: list[str] = []
    _walk(root, lines)
    text = spaced_hyphens("\n".join(lines))
    return re.sub(r"\n{3,}", "\n\n", text).strip() + "\n"


def clean_pdf_text(raw: str, drop_line_patterns: list[str] | None = None) -> str:
    """Strip repeating headers and page markers and normalise whitespace in extracted PDF text."""
    drop = [re.compile(p) for p in (drop_line_patterns or [])]
    out: list[str] = []
    for line in raw.splitlines():
        s = line.strip()
        if any(p.fullmatch(s) for p in drop):
            continue
        out.append(re.sub(r"[ \t]+", " ", s))
    return "\n".join(out)


@dataclass
class ManifestEntry:
    doc_id: str
    title: str
    file: str
    source_urls: list[str]
    retrieved: str
    raw_sha256: list[str]
    sha256: str
    chars: int
    status: str = "ok"
    note: str = ""


def write_manifest(entries: list[ManifestEntry], path: Path) -> None:
    Path(path).write_text(json.dumps([asdict(e) for e in entries], indent=2) + "\n", encoding="utf-8")


def read_manifest(path: Path) -> list[dict]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def verify_manifest(manifest_path: Path, corpus_dir: Path) -> list[str]:
    """Return a list of problems (missing or hash-mismatched files); empty means the corpus is intact."""
    problems: list[str] = []
    for e in read_manifest(manifest_path):
        if e["status"] != "ok":
            continue
        f = Path(corpus_dir) / e["file"]
        if not f.exists():
            problems.append(f"missing {e['file']}")
        elif sha256_file(f) != e["sha256"]:
            problems.append(f"hash mismatch {e['file']}")
    return problems


FOOTNOTE_RE = re.compile(r"\[F[o]?o?tnote\s*\d+.*?End of Footnote\s*\d+\.\]", re.S)


def _looks_like_heading(ln: str) -> bool:
    """Short title-like line with no sentence punctuation; used only where no heading registry exists."""
    return (len(ln) <= 48 and ln[0].isupper() and not re.search(r"[.,;:?!]$", ln) and not ln.startswith(("-", "•"))
            and len(ln.split()) <= 7 and len(ln) >= 5 and "+" not in ln and not re.search(r"\d{3,}|\d$", ln)
            and ln.split()[-1].lower() not in {"of", "the", "and", "to", "for", "a", "in", "or"})


def normalize_guidance_text(raw: str, start_marker: str, section_re: str, subheadings: list[str],
                            end_marker: str | None = None, join_threshold: float = 0.88,
                            infer_headings: bool = False) -> str:
    """Rebuild paragraphs from line-wrapped PDF text and tag headings.

    Output uses '# ' for numbered sections, '## ' for subheadings, one paragraph per line.
    A paragraph ends when a line closes a sentence and is clearly shorter than a full line.
    """
    text = raw[raw.rindex(start_marker):]
    if end_marker and end_marker in text:
        text = text[:text.rindex(end_marker)]
    text = FOOTNOTE_RE.sub(" ", text).replace("[Page Break]", "")
    for h in subheadings:
        if "\n" in h:
            text = text.replace(h, h.replace("\n", " "))
    heads = {h.replace("\n", " ") for h in subheadings}
    sec = re.compile(section_re)
    lines = [ln.strip() for ln in text.splitlines()]
    lines = [ln for ln in lines if ln]
    widths = sorted(len(ln) for ln in lines)
    full = widths[int(len(widths) * 0.9)]
    out: list[str] = []
    cur: list[str] = []

    def flush() -> None:
        if cur:
            out.append(re.sub(r"\s+", " ", " ".join(cur)).strip())
            cur.clear()

    for ln in lines:
        if sec.fullmatch(ln):
            flush()
            out.append("# " + ln)
        elif ln in heads or (infer_headings and not cur and _looks_like_heading(ln)):
            flush()
            out.append("## " + ln)
        else:
            cur.append(ln)
            if ln[-1] in ".?:" and len(ln) < join_threshold * full:
                flush()
    flush()
    return spaced_hyphens("\n".join(out)) + "\n"
