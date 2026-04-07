"""Recursive character text splitting with overlap and page tracking."""

import re

from yoke.ingestion.models import Chunk


def chunk_text(
    text: str,
    source_file: str,
    *,
    target_size: int = 2000,
    overlap: int = 200,
    page_numbers: list[int] | None = None,
) -> list[Chunk]:
    """Split text into overlapping chunks with metadata.

    Uses a greedy approach: accumulate text until target_size, then backtrack
    to the nearest natural boundary (paragraph > sentence > word).

    Args:
        text: Full document text.
        source_file: Filename for metadata.
        target_size: Target chunk size in characters (~512 tokens).
        overlap: Overlap size in characters (~50 tokens).
        page_numbers: Optional per-character page number mapping.

    Returns:
        List of Chunk objects.
    """
    if not text.strip():
        return []

    if page_numbers is not None and len(page_numbers) != len(text):
        raise ValueError(
            f"page_numbers length ({len(page_numbers)}) must match "
            f"text length ({len(text)})"
        )

    # Step 1: Find split points (non-overlapping segment boundaries)
    splits = _find_splits(text, target_size)

    # Step 2: Create chunks with overlap
    chunks: list[Chunk] = []
    for i, (seg_start, seg_end) in enumerate(splits):
        if i > 0 and overlap > 0:
            overlap_start = max(0, seg_start - overlap)
            chunk_str = text[overlap_start:seg_end]
            char_start = overlap_start
        else:
            chunk_str = text[seg_start:seg_end]
            char_start = seg_start

        # Determine page numbers
        if page_numbers is not None:
            ps = max(0, min(char_start, len(page_numbers) - 1))
            pe = min(seg_end, len(page_numbers))
            chunk_pages = sorted(set(page_numbers[ps:pe])) or [1]
        else:
            chunk_pages = [1]

        chunks.append(Chunk(
            chunk_index=i,
            text=chunk_str,
            page_numbers=chunk_pages,
            source_file=source_file,
        ))

    return chunks


def _find_splits(text: str, target_size: int) -> list[tuple[int, int]]:
    """Find non-overlapping segment boundaries in the text.

    Returns list of (start, end) offset pairs.
    """
    n = len(text)
    splits: list[tuple[int, int]] = []
    pos = 0

    while pos < n:
        # If remaining text fits in one segment, take it all
        if pos + target_size >= n:
            splits.append((pos, n))
            break

        # Find the best split point near target_size
        window_end = pos + target_size
        split_at = _find_boundary(text, window_end)

        # Ensure we make forward progress
        if split_at <= pos:
            split_at = window_end

        splits.append((pos, split_at))
        pos = split_at

    return splits


def _find_boundary(text: str, ideal_pos: int) -> int:
    """Find the nearest natural boundary near ideal_pos.

    Searches a window of ±10% around ideal_pos for the best break point.
    Prefers paragraph breaks > sentence breaks > word breaks, choosing the
    one closest to ideal_pos within each tier.
    """
    n = len(text)
    ideal_pos = min(ideal_pos, n)

    margin = 200  # ±200 chars (~50 tokens) around ideal position
    search_start = max(0, ideal_pos - margin)
    search_end = min(n, ideal_pos + margin)
    search_text = text[search_start:search_end]

    def _closest_to_ideal(matches: list[re.Match]) -> int | None:
        if not matches:
            return None
        # Return the match end position closest to ideal_pos
        best = min(matches, key=lambda m: abs((search_start + m.end()) - ideal_pos))
        return search_start + best.end()

    # Prefer paragraph boundary (\n\n)
    pos = _closest_to_ideal(list(re.finditer(r"\n\n+", search_text)))
    if pos is not None:
        return pos

    # Then sentence boundary
    pos = _closest_to_ideal(list(re.finditer(r"[.!?]\s+", search_text)))
    if pos is not None:
        return pos

    # Then word boundary
    pos = _closest_to_ideal(list(re.finditer(r"\s+", search_text)))
    if pos is not None:
        return pos

    return ideal_pos
