"""Chunk regulatory text by legal structure (section, paragraph) with citation metadata."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

MAX_CHARS = 1500
MIN_CHARS = 220

ROMAN_LOWER = ["i", "ii", "iii", "iv", "v", "vi", "vii", "viii", "ix", "x", "xi", "xii", "xiii", "xiv", "xv", "xvi",
               "xvii", "xviii", "xix", "xx"]
MARKER_RE = re.compile(r"^\(([A-Za-z]{1,4}|\d{1,3})\)(?:\s+|$)")
MULTI_MARKER_RE = re.compile(r"^((?:\([A-Za-z]{1,4}\)|\(\d{1,3}\))+)(?:\s+|$)")
INLINE_CHILD_RE = re.compile(r"^(\([A-Za-z0-9]{1,4}\)\s+[^()]{0,160}?(?: - |\. ))(\((?:\d{1,3}|[A-Z]{1,3})\)\s+.*)$", re.S)


@dataclass(frozen=True)
class Passage:
    chunk_id: str
    doc_id: str
    citation: str
    section_path: str
    text: str
    covers: tuple[str, ...]
    title: str = ""

    @property
    def index_text(self) -> str:
        """Text used for retrieval: citation and section title give lexical and semantic anchors."""
        return f"{self.title}. {self.text}" if self.title else self.text


def sentences(text: str) -> list[str]:
    """Split into sentences; legal text rarely has abbreviations that break this rule badly."""
    parts = re.split(r"(?<=[.;:?!])\s+(?=[A-Z(\[])", text.strip())
    return [p for p in parts if p]


def split_long(text: str, max_chars: int = MAX_CHARS) -> list[str]:
    """Greedy sentence packing so no piece exceeds max_chars (a single long sentence stays whole)."""
    pieces: list[str] = []
    cur = ""
    for s in sentences(text):
        if cur and len(cur) + 1 + len(s) > max_chars:
            pieces.append(cur)
            cur = s
        else:
            cur = f"{cur} {s}".strip()
    if cur:
        pieces.append(cur)
    return pieces or [text]


# ---------------------------------------------------------------- CFR

def _marker_level(token: str) -> int:
    if token.isdigit():
        return 1
    if token.islower():
        return 2 if token in ROMAN_LOWER and token not in {"i", "v", "x"} and len(token) > 1 else 0
    if token.isupper():
        return 4 if token.lower() in ROMAN_LOWER and len(token) > 1 else 3
    return 0


@dataclass
class _Node:
    token: str
    level: int
    text: str
    children: list["_Node"] = field(default_factory=list)
    key: str = ""
    label: str = ""


def _letter_next(prev: str | None, tok: str) -> bool:
    return prev is not None and len(prev) == 1 and len(tok) == 1 and ord(tok) == ord(prev) + 1


def _roman_next(prev: str | None, tok: str) -> bool:
    if prev is None:
        return tok == "i"
    if prev in ROMAN_LOWER and tok in ROMAN_LOWER:
        return ROMAN_LOWER.index(tok) == ROMAN_LOWER.index(prev) + 1
    return False


def _definition_label(text: str) -> str:
    if text.startswith("[table]"):
        return "table"
    m = re.match(r"^(.{3,120}?)\s+(means|is defined|has the meaning|are|refers)\b", text)
    label = m.group(1) if m else " ".join(text.split()[:5])
    return label.strip(" ,.:")


def _parse_section(lines: list[str]) -> list[_Node]:
    """Turn the paragraph lines of one section into a tree using marker levels and sequence checks."""
    flat: list[tuple[str | None, str]] = []
    for line in lines:
        mm = MULTI_MARKER_RE.match(line)
        if not mm:
            flat.append((None, line))
            continue
        toks = re.findall(r"\(([^)]+)\)", mm.group(1))
        rest = line[mm.end():]
        inline = INLINE_CHILD_RE.match(line) if len(toks) == 1 else None
        for t in toks[:-1]:
            flat.append((t, ""))
        if inline:
            flat.append((toks[-1], inline.group(1)[mm.end():]))
            m2 = MARKER_RE.match(inline.group(2))
            flat.append((m2.group(1), inline.group(2)[m2.end():]))
        else:
            flat.append((toks[-1], rest))
    roots: list[_Node] = []
    stack: list[_Node] = []
    last_at_level: dict[int, str] = {}
    prev_text = ""
    for tok, text in flat:
        if tok is None:
            node = _Node("", -1, text, label=_definition_label(text))
            roots.append(node)
            if re.search(r"(means|is defined|has the meaning)", text[:200]) or text.endswith(":"):
                stack, last_at_level = [node], {}  # a definition or lead-in owns the markers that follow
            prev_text = text
            continue
        level = _marker_level(tok)
        if tok.isdigit() and stack:
            prev5 = last_at_level.get(5)
            if (tok == "1" and stack[-1].level in (3, 4)) or (prev5 is not None and int(tok) == int(prev5) + 1):
                level = 5  # fifth hierarchy level reuses digits under an uppercase-letter paragraph
        if tok in {"i", "v", "x"}:
            as_letter = _letter_next(last_at_level.get(0), tok)
            as_roman = _roman_next(last_at_level.get(2), tok)
            if as_letter and not as_roman:
                level = 0
            elif as_roman and not as_letter:
                level = 2
            elif tok == "i":
                level = 2 if prev_text.rstrip().endswith((":", " -")) else 0
            else:
                level = 0 if as_letter else 2
        node = _Node(tok, level, text)
        while stack and stack[-1].level >= level:
            stack.pop()
        (stack[-1].children if stack else roots).append(node)
        stack.append(node)
        last_at_level[level] = tok
        for deeper in [k for k in last_at_level if k > level]:
            del last_at_level[deeper]
        prev_text = text
    return roots


def _node_text(node: _Node) -> str:
    own = (f"({node.token}) " if node.token else "") + node.text
    return " ".join([own] + [_node_text(c) for c in node.children])


def _covers(node: _Node) -> list[str]:
    return [node.key] + [k for c in node.children for k in _covers(c)]


def _assign_keys(nodes: list[_Node], prefix: str, section_cite: str) -> None:
    for n in nodes:
        n.key = f"{prefix}({n.token})" if n.token else f"{section_cite} [{n.label}]"
        _assign_keys(n.children, n.key, section_cite)


Item = tuple[str, list[str], str]


def _emit(node: _Node, out: list[Item]) -> None:
    """Emit (citation, covers, text) items: the whole node if it fits, else its own text then its children."""
    full = _node_text(node)
    if len(full) <= MAX_CHARS:
        out.append((node.key, _covers(node), full))
        return
    own = (f"({node.token}) " if node.token else "") + node.text
    if node.children:
        out.append((node.key, [node.key], own))
        for c in node.children:
            _emit(c, out)
        return
    pieces = split_long(own)
    for i, p in enumerate(pieces, 1):
        out.append((node.key + (f" (part {i})" if len(pieces) > 1 else ""), [node.key], p))


def _join_cite(a: str, b: str, section_cite: str) -> str:
    """Citation for a merged chunk, e.g. '12 CFR 249.10(a) to (b)'."""
    if a.startswith(section_cite) and b.startswith(section_cite) and "[" not in a + b and " n." not in a + b:
        return f"{a.split(' to ')[0]} to {b[len(section_cite):]}"
    return f"{a} and {b}"


def _merge_small(items: list[Item], section_cite: str) -> list[Item]:
    """Merge adjacent undersized pieces within a section so retrieval units carry enough context."""
    merged: list[Item] = []
    for cite, cov, text in items:
        if merged:
            pc, pcov, ptext = merged[-1]
            if (len(ptext) < MIN_CHARS or len(text) < MIN_CHARS) and len(ptext) + 1 + len(text) <= MAX_CHARS:
                merged[-1] = (_join_cite(pc, cite, section_cite), pcov + cov, f"{ptext} {text}")
                continue
        merged.append((cite, cov, text))
    return merged


def chunk_cfr_markdown(doc_id: str, md: str, cite_prefix: str = "12 CFR") -> list[Passage]:
    """Chunk eCFR markdown (as produced by corpus.ecfr_xml_to_markdown) by section and paragraph."""
    passages: list[Passage] = []
    subpart = ""
    section_id = ""
    section_title = ""
    lines: list[str] = []
    footnotes: list[str] = []

    def flush() -> None:
        nonlocal lines, footnotes
        if not section_id:
            lines, footnotes = [], []
            return
        cite = f"{cite_prefix} {section_id}"
        roots = _parse_section(lines)
        _assign_keys(roots, cite, cite)
        items: list[Item] = []
        for r in roots:
            _emit(r, items)
        items = _merge_small(items, cite)
        for k, fn in enumerate(footnotes, 1):
            for j, piece in enumerate(split_long(fn), 1):
                items.append((f"{cite} n.{k}", [f"{cite} n.{k}"], f"[footnote] {piece}"))
        for i, (c, cov, text) in enumerate(items):
            covers = tuple(dict.fromkeys([cite] + cov))
            passages.append(Passage(
                chunk_id=f"{doc_id}:{section_id}:{i}", doc_id=doc_id, citation=_pretty(c, cite),
                section_path=f"{subpart} > § {section_id} {section_title}".strip(" >"),
                text=text, covers=covers, title=f"§ {section_id} {section_title}"))
        lines, footnotes = [], []

    for raw in md.splitlines():
        line = raw.rstrip()
        if not line.strip():
            continue
        m = re.match(r"^###\s+§+\s*([\d.]+)\s*(.*)$", line)
        if m:
            flush()
            section_id, section_title = m.group(1), m.group(2).strip()
            continue
        if line.startswith("#"):
            flush()
            if line.startswith("## "):
                subpart = line[3:].strip()
            section_id = ""
            continue
        if line.startswith("[footnote]"):
            footnotes.append(line[len("[footnote]"):].strip())
        else:
            lines.append(line)
    flush()
    return passages


def _pretty(cite: str, section_cite: str) -> str:
    """Render definition keys readably."""
    return cite.replace(" [", " (definition of ").replace("]", ")") if "[" in cite else cite


# ---------------------------------------------------------------- guidance text

def chunk_guidance_markdown(doc_id: str, md: str, cite_prefix: str) -> list[Passage]:
    """Chunk normalised guidance (see corpus.normalize_guidance_text): '# Section', '## Subsection', paragraph lines."""
    passages: list[Passage] = []
    section = ""
    sub = ""
    paras: list[tuple[int, str]] = []
    counter = 0

    def flush() -> None:
        nonlocal paras
        if not paras:
            return
        items: list[tuple[list[int], str]] = []
        for n, text in paras:
            pieces = split_long(text) if len(text) > MAX_CHARS else [text]
            for p in pieces:
                if items and (len(items[-1][1]) < MIN_CHARS or len(p) < MIN_CHARS) and len(items[-1][1]) + 1 + len(p) <= MAX_CHARS:
                    items[-1] = (items[-1][0] + [n], f"{items[-1][1]} {p}")
                else:
                    items.append(([n], p))
        rm = re.match(r"^([IVX]+)\.\s", section)
        sec_label = rm.group(1) if rm else section
        for i, (ns, text) in enumerate(items):
            ns = sorted(set(ns))
            span = f"para {ns[0]}" if len(ns) == 1 else f"paras {ns[0]}-{ns[-1]}"
            dup = [j for j, (other, _) in enumerate(items) if sorted(set(other)) == ns]
            if len(dup) > 1:
                span += f" (part {dup.index(i) + 1})"
            where = (f"{sec_label} ({sub})" if sub else sec_label) if rm else (sub or sec_label)
            covers = tuple(f"{cite_prefix}, {where}, para {n}" for n in ns)
            passages.append(Passage(
                chunk_id=f"{doc_id}:{sec_label}:{len(passages)}", doc_id=doc_id,
                citation=f"{cite_prefix}, {where}, {span}",
                section_path=f"{section} > {sub}" if sub else section,
                text=text, covers=covers, title=f"{section}{' / ' + sub if sub else ''}"))
        paras = []

    for raw in md.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("## "):
            flush()
            sub, counter = line[3:].strip(), 0
        elif line.startswith("# "):
            flush()
            section, sub, counter = line[2:].strip(), "", 0
        else:
            counter += 1
            paras.append((counter, line))
    flush()
    return passages
