"""PDF text extraction with Unicode normalization."""

import re
from pathlib import Path

import pymupdf


# Common Unicode ligatures found in academic PDFs
_LIGATURES = {
    "\ufb00": "ff",
    "\ufb01": "fi",
    "\ufb02": "fl",
    "\ufb03": "ffi",
    "\ufb04": "ffl",
}


def _normalize(text: str) -> str:
    """Normalize Unicode ligatures and collapse excess whitespace."""
    for old, new in _LIGATURES.items():
        text = text.replace(old, new)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


def extract_pdf_pages(pdf_path: Path, start: int, end: int) -> str:
    """Extract text from a range of pages (1-indexed, inclusive).

    Args:
        pdf_path: Path to the PDF file.
        start: First page number (1-indexed).
        end: Last page number (1-indexed, inclusive).

    Returns:
        Cleaned, concatenated text from the specified pages.
    """
    doc = pymupdf.open(pdf_path)
    parts: list[str] = []
    for i in range(start - 1, min(end, doc.page_count)):
        parts.append(doc[i].get_text())
    return _normalize("\n".join(parts))


def extract_pdf_with_page_map(
    pdf_path: Path,
) -> tuple[str, list[int]]:
    """Extract all text from a PDF with per-character page number mapping.

    Returns:
        Tuple of (full_text, page_numbers) where page_numbers[i] is the
        1-indexed page number for character i in full_text.
    """
    doc = pymupdf.open(pdf_path)
    parts: list[str] = []
    page_map: list[int] = []

    for page_idx in range(doc.page_count):
        page_num = page_idx + 1  # 1-indexed
        raw_text = doc[page_idx].get_text()
        normalized = _normalize(raw_text)
        parts.append(normalized)
        page_map.extend([page_num] * len(normalized))

    # Join with newlines (which also need page attribution)
    full_parts: list[str] = []
    full_map: list[int] = []
    for i, (part, page_num) in enumerate(
        zip(parts, [p + 1 for p in range(doc.page_count)])
    ):
        if i > 0:
            full_parts.append("\n")
            full_map.append(page_num)
        full_parts.append(part)
        full_map.extend([page_num] * len(part))

    full_text = "".join(full_parts)
    # Final normalization pass may change length, so rebuild map
    normalized_text = _normalize(full_text)
    if len(normalized_text) == len(full_text):
        return normalized_text, full_map

    # If normalization changed length (collapsed newlines), rebuild map
    # by walking both strings in parallel
    new_map: list[int] = []
    old_idx = 0
    for char in normalized_text:
        if old_idx < len(full_map):
            new_map.append(full_map[old_idx])
        else:
            new_map.append(full_map[-1] if full_map else 1)
        # Advance old_idx to match
        if old_idx < len(full_text) and full_text[old_idx] == char:
            old_idx += 1
        else:
            # Normalization removed chars; skip ahead in original
            while old_idx < len(full_text) and full_text[old_idx] != char:
                old_idx += 1
            if old_idx < len(full_text):
                old_idx += 1

    return normalized_text, new_map


def extract_pdf_chapter(pdf_path: Path, chapter: int) -> str:
    """Extract a chapter using the PDF's table of contents.

    Finds the chapter-level TOC entry matching the given chapter number
    and extracts all pages from that chapter's start to the next chapter's
    start (exclusive).

    Args:
        pdf_path: Path to the PDF file.
        chapter: Chapter number to extract (e.g. 1, 2, 3).

    Returns:
        Cleaned text for the entire chapter.

    Raises:
        ValueError: If the chapter is not found in the TOC.
    """
    doc = pymupdf.open(pdf_path)
    toc = doc.get_toc()

    # Find chapter-level entries (level 1 in TOC, title starts with digit)
    chapter_entries: list[tuple[str, int]] = []
    for level, title, page in toc:
        if level == 1 and re.match(rf"^{chapter}\b", title.strip()):
            start_page = page
        if level == 1 and re.match(r"^\d+\b", title.strip()):
            chapter_entries.append((title.strip(), page))

    # Find start and end pages
    start_page = None
    end_page = None
    for idx, (title, page) in enumerate(chapter_entries):
        if title.startswith(str(chapter)):
            start_page = page
            if idx + 1 < len(chapter_entries):
                end_page = chapter_entries[idx + 1][1] - 1
            else:
                end_page = doc.page_count
            break

    if start_page is None:
        raise ValueError(
            f"Chapter {chapter} not found in TOC. "
            f"Available: {[t for t, _ in chapter_entries]}"
        )

    return extract_pdf_pages(pdf_path, start_page, end_page)


def prepare_pdf_fixture(
    pdf_path: Path, pages: tuple[int, int], output_dir: Path
) -> Path:
    """Extract pages from a PDF and write to output_dir as .txt.

    Args:
        pdf_path: Source PDF file.
        pages: (start, end) page range, 1-indexed inclusive.
        output_dir: Directory to write the extracted text.

    Returns:
        Path to the written .txt file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = pdf_path.stem
    start, end = pages
    out_path = output_dir / f"{stem}_p{start}-{end}.txt"
    text = extract_pdf_pages(pdf_path, start, end)
    out_path.write_text(text, encoding="utf-8")
    return out_path
