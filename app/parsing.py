import io
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import pdfplumber

SUPPORTED_TYPES = {".md", ".markdown", ".txt", ".pdf"}

MD_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*$")
NUMBERED_HEADING = re.compile(r"^(\d+(?:\.\d+)*)[.)]?\s+([A-Z].{0,70})$")
TABLE_RULE = re.compile(r"^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?\s*$")
UNDERLINE = re.compile(r"^(=+|-+)\s*$")
LIST_ITEM = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+")
APPLIES_TO = re.compile(r"^\W*applies to\W*(.+)$", re.IGNORECASE | re.MULTILINE)


class UnreadableDocument(Exception):
    pass


@dataclass
class Block:
    kind: str
    section: str
    text: str = ""
    header: list[str] = field(default_factory=list)
    rows: list[list[str]] = field(default_factory=list)


@dataclass
class ParsedDocument:
    title: str
    blocks: list[Block]
    applies_to: str = ""


def read_document(filename: str, data: bytes) -> ParsedDocument:
    suffix = Path(filename).suffix.lower()
    if suffix == ".pdf":
        text = pdf_to_markdown(data)
    else:
        text = decode_text(data)

    fallback_title = Path(filename).stem.replace("-", " ").replace("_", " ").title()
    document = parse_text(text, fallback_title)
    if not any(block.text or block.rows for block in document.blocks):
        raise UnreadableDocument("The file has no readable text.")
    return document


def decode_text(data: bytes) -> str:
    if b"\x00" in data:
        raise UnreadableDocument("The file looks binary, not text.")
    for encoding in ("utf-8-sig", "cp1252"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise UnreadableDocument("The file is not valid text.")


def clean_inline(text: str) -> str:
    text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"(\*\*|__|`)", "", text)
    return re.sub(r"\s+", " ", text).strip()


def parse_text(text: str, fallback_title: str) -> ParsedDocument:
    lines = text.replace("\r\n", "\n").split("\n")
    uses_md_headings = any(MD_HEADING.match(line) for line in lines)

    blocks: list[Block] = []
    stack: list[tuple[int, str]] = []
    title = ""
    paragraph: list[str] = []

    def section_path() -> str:
        names = [name for level, name in stack if level > 1]
        return " > ".join(names or [name for _, name in stack])

    def flush_paragraph() -> None:
        if paragraph:
            blocks.append(Block("text", section_path(), clean_inline(" ".join(paragraph))))
            paragraph.clear()

    def open_heading(level: int, name: str) -> None:
        nonlocal title
        flush_paragraph()
        while stack and stack[-1][0] >= level:
            stack.pop()
        stack.append((level, name))
        if level == 1 and not title:
            title = name

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        next_line = lines[i + 1] if i + 1 < len(lines) else ""

        if not line:
            flush_paragraph()
            i += 1
        elif "|" in line and TABLE_RULE.match(next_line) and "|" in next_line:
            flush_paragraph()
            i = read_table(lines, i, section_path(), blocks)
        elif uses_md_headings:
            heading = MD_HEADING.match(line)
            if heading:
                open_heading(len(heading.group(1)), clean_inline(heading.group(2)))
            else:
                add_line(line, paragraph, flush_paragraph)
            i += 1
        else:
            heading = plain_heading(line, next_line, first_heading=not stack)
            if heading:
                open_heading(*heading)
                i += 2 if UNDERLINE.match(next_line.strip()) else 1
            else:
                add_line(line, paragraph, flush_paragraph)
                i += 1

    flush_paragraph()
    applies_to = APPLIES_TO.search(text)
    return ParsedDocument(
        title or fallback_title,
        blocks,
        clean_inline(applies_to.group(1)) if applies_to else "",
    )


def add_line(line: str, paragraph: list[str], flush_paragraph) -> None:
    if LIST_ITEM.match(line):
        flush_paragraph()
        paragraph.append(LIST_ITEM.sub("", line))
        flush_paragraph()
    else:
        paragraph.append(line)


def plain_heading(line: str, next_line: str, first_heading: bool) -> tuple[int, str] | None:
    if len(line.split()) > 12 or line[-1] in ".:;,?":
        return None
    numbered = NUMBERED_HEADING.match(line)
    if numbered:
        return numbered.group(1).count(".") + 2, line
    if UNDERLINE.match(next_line.strip()) and next_line.strip():
        return (1 if next_line.strip()[0] == "=" else 2), line
    if line.isupper() and len(line) > 3:
        return (1 if first_heading else 2), line
    return None


def read_table(lines: list[str], start: int, section: str, blocks: list[Block]) -> int:
    header = split_row(lines[start])
    rows = []
    i = start + 2
    while i < len(lines) and "|" in lines[i]:
        cells = split_row(lines[i])
        rows.append((cells + [""] * len(header))[: len(header)])
        i += 1
    blocks.append(Block("table", section, header=header, rows=rows))
    return i


def split_row(line: str) -> list[str]:
    return [clean_inline(cell) for cell in line.strip().strip("|").split("|")]


def pdf_to_markdown(data: bytes) -> str:
    try:
        with pdfplumber.open(io.BytesIO(data)) as pdf:
            pages = [read_pdf_page(page) for page in pdf.pages]
    except Exception as error:
        raise UnreadableDocument("The PDF could not be opened.") from error

    sizes = Counter(item["size"] for page in pages for item in page if item["kind"] == "line")
    if not sizes:
        raise UnreadableDocument("The PDF has no text layer. It may be a scanned image.")

    body_size = sizes.most_common(1)[0][0]
    bigger_sizes = sorted({size for size in sizes if size > body_size * 1.12}, reverse=True)

    output = []
    previous_bottom = None
    for page in pages:
        for item in page:
            if item["kind"] == "table":
                output.append("\n" + item["text"] + "\n")
                previous_bottom = None
                continue

            text = item["text"]
            is_short = len(text.split()) <= 12 and text[-1] not in ".:;,"
            if item["size"] in bigger_sizes and is_short:
                output.append("\n" + "#" * (bigger_sizes.index(item["size"]) + 1) + " " + text + "\n")
            elif item["bold"] and is_short:
                output.append("\n" + "#" * (len(bigger_sizes) + 1) + " " + text + "\n")
            else:
                gap_before = previous_bottom is not None and item["top"] - previous_bottom > item["size"] * 0.8
                output.append(("\n" if gap_before else "") + text)
            previous_bottom = item["bottom"]
        previous_bottom = None
    return "\n".join(output)


def has_clean_header(table) -> bool:
    rows = table.extract()
    return len(rows) >= 2 and all(cell and cell.strip() for cell in rows[0])


def read_pdf_page(page) -> list[dict]:
    tables = [table for table in page.find_tables() if has_clean_header(table)]
    boxes = [table.bbox for table in tables]
    items = []

    for table in tables:
        rows = [[clean_inline(cell or "") for cell in row] for row in table.extract()]
        lines = ["| " + " | ".join(rows[0]) + " |", "|" + "---|" * len(rows[0])]
        lines += ["| " + " | ".join(row) + " |" for row in rows[1:]]
        items.append({"kind": "table", "top": table.bbox[1], "text": "\n".join(lines)})

    for line in page.extract_text_lines(return_chars=True):
        if not line["text"].strip():
            continue
        middle = (line["top"] + line["bottom"]) / 2
        if any(box[1] <= middle <= box[3] and box[0] <= line["x0"] <= box[2] for box in boxes):
            continue
        chars = line["chars"]
        items.append({
            "kind": "line",
            "top": line["top"],
            "bottom": line["bottom"],
            "text": line["text"].strip(),
            "size": round(chars[0]["size"], 1),
            "bold": all("bold" in char["fontname"].lower() for char in chars),
        })
    return sorted(items, key=lambda item: item["top"])
