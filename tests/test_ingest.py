import io

import pytest
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Table, TableStyle

from app.chunking import chunk_document
from app.parsing import UnreadableDocument, read_document

MARKDOWN = """# Sample Policy

## 1. Benefits

Everyone gets lunch. **Twice** a week.

| Tier | Dental |
| --- | --- |
| Gold | Covered |
| Basic | Not covered |

## 2. Rules

### 2.1 Dress code

1. Wear shoes.
2. Wear a badge.
"""


def test_markdown_sections_are_nested_and_clean():
    chunks = chunk_document(read_document("sample.md", MARKDOWN.encode()))
    first = chunks[0]
    assert first.section == "1. Benefits"
    assert first.text == "Everyone gets lunch. Twice a week."
    assert first.search_text.startswith("Sample Policy > 1. Benefits")
    assert chunks[-1].section == "2. Rules > 2.1 Dress code"


def test_numbered_list_in_markdown_is_not_a_heading():
    sections = {c.section for c in chunk_document(read_document("sample.md", MARKDOWN.encode()))}
    assert not any("Wear shoes" in section for section in sections)


def test_each_table_row_keeps_its_column_names():
    chunks = chunk_document(read_document("sample.md", MARKDOWN.encode()))
    rows = [chunk.text for chunk in chunks if chunk.kind == "table"]
    assert rows == ["Tier: Gold | Dental: Covered", "Tier: Basic | Dental: Not covered"]


def test_plain_text_headings_are_detected():
    text = "TRAVEL RULES\n\n1. Booking\nBook early.\n\n2. Claims\n2.1 Deadline\nFile within 30 days.\n"
    document = read_document("rules.txt", text.encode())
    chunks = chunk_document(document)
    assert document.title == "TRAVEL RULES"
    assert [chunk.section for chunk in chunks] == ["1. Booking", "2. Claims > 2.1 Deadline"]


def test_applies_to_is_read_from_the_header():
    text = "# Leave\n\n**Applies to:** Full-time employees in India\n\n## Rules\n\nThis applies to everyone.\n"
    assert read_document("leave.md", text.encode()).applies_to == "Full-time employees in India"


def test_title_falls_back_to_the_file_name():
    document = read_document("travel-rules.txt", b"1. Booking\nBook early.\n")
    assert document.title == "Travel Rules"
    assert chunk_document(document)[0].section == "1. Booking"


def test_text_with_no_headings_still_works():
    chunks = chunk_document(read_document("note.txt", b"Offices close at 6 pm on Fridays."))
    assert chunks[0].section == ""
    assert "Fridays" in chunks[0].text


def test_long_section_is_split_with_one_sentence_of_overlap():
    sentences = " ".join(f"Rule number {i} applies to every employee in the company today." for i in range(40))
    chunks = chunk_document(read_document("long.md", f"# Long\n\n## Rules\n\n{sentences}".encode()))
    assert len(chunks) > 1
    assert all(len(chunk.text.split()) <= 120 for chunk in chunks)
    last_sentence = chunks[0].text.split(". ")[-1]
    assert last_sentence.rstrip(".") in chunks[1].text


def make_pdf() -> bytes:
    styles = getSampleStyleSheet()
    table = Table([["Tier", "Dental"], ["Gold", "Covered"], ["Basic", "Not covered"]])
    table.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.5, colors.black)]))
    buffer = io.BytesIO()
    SimpleDocTemplate(buffer, pagesize=A4).build([
        Paragraph("Pay Policy", styles["Title"]),
        Paragraph("Dental plans", styles["Heading2"]),
        Paragraph("Each employee gets one plan.", styles["BodyText"]),
        Paragraph("A second paragraph explains that plans renew every year.", styles["BodyText"]),
        table,
    ])
    return buffer.getvalue()


def test_pdf_keeps_headings_and_table_rows():
    document = read_document("pay.pdf", make_pdf())
    chunks = chunk_document(document)
    assert document.title == "Pay Policy"
    assert "Tier: Basic | Dental: Not covered" in [c.text for c in chunks if c.kind == "table"]
    assert all(chunk.section == "Dental plans" for chunk in chunks)


@pytest.mark.parametrize("name, data", [
    ("bad.md", b"\x00\x01\x02binary"),
    ("empty.txt", b"   \n  "),
    ("broken.pdf", b"this is not a pdf"),
])
def test_unreadable_files_raise_a_clear_error(name, data):
    with pytest.raises(UnreadableDocument):
        read_document(name, data)
