import re
from dataclasses import dataclass
from itertools import groupby

from app import config
from app.parsing import Block, ParsedDocument

SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9₹(\"'])")


@dataclass
class Chunk:
    section: str
    kind: str
    text: str
    search_text: str


def chunk_document(document: ParsedDocument) -> list[Chunk]:
    chunks = []

    def add(section: str, kind: str, text: str) -> None:
        path = " > ".join(part for part in (document.title, section) if part)
        chunks.append(Chunk(section, kind, text, f"{path}\n{text}"))

    for (section, kind), group in groupby(document.blocks, key=lambda b: (b.section, b.kind)):
        blocks = list(group)
        if kind == "table":
            for block in blocks:
                for row_text in table_rows(block):
                    add(section, "table", row_text)
        else:
            for text in split_text(block.text for block in blocks):
                add(section, "text", text)
    return chunks


def table_rows(block: Block) -> list[str]:
    return [
        " | ".join(f"{name}: {value}" for name, value in zip(block.header, row) if value)
        for row in block.rows
        if any(row)
    ]


def split_sentences(text: str) -> list[str]:
    return [sentence.strip() for sentence in SENTENCE_END.split(text) if sentence.strip()]


def split_text(paragraphs) -> list[str]:
    sentences = [s for paragraph in paragraphs for s in split_sentences(paragraph)]
    chunks: list[str] = []
    current: list[str] = []
    for sentence in sentences:
        if current and word_count(current) + len(sentence.split()) > config.MAX_CHUNK_WORDS:
            chunks.append(" ".join(current))
            current = current[-1:]
        current.append(sentence)
    if current:
        chunks.append(" ".join(current))
    return chunks


def word_count(sentences: list[str]) -> int:
    return sum(len(sentence.split()) for sentence in sentences)
