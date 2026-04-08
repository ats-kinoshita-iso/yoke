"""CLI entry point for query: python -m yoke.query "what is X?" --db ./data/yoke.db

Runs hybrid retrieval, prints diagnostics, optionally generates an answer.
"""

import argparse
import io
import sys
from pathlib import Path

from dotenv import load_dotenv

from yoke.pipeline import GENERATION_MODEL, query
from yoke.retrieval import RetrievalResult, RetrievalTimings, retrieve_with_timings


def _print_diagnostics(
    results: list[RetrievalResult],
    timings: RetrievalTimings,
) -> None:
    """Print retrieval diagnostics to stderr."""
    print("\n--- Retrieval Diagnostics ---", file=sys.stderr)
    print(
        f"  Latency: {timings.total_ms:.0f}ms total "
        f"(embed={timings.embedding_ms:.0f}ms "
        f"dense={timings.dense_ms:.0f}ms "
        f"sparse={timings.sparse_ms:.0f}ms "
        f"rrf={timings.rrf_ms:.0f}ms)",
        file=sys.stderr,
    )
    print(f"  Results: {len(results)}", file=sys.stderr)
    print(file=sys.stderr)

    for i, r in enumerate(results, 1):
        source_tag = []
        if r.dense_rank is not None:
            source_tag.append(f"dense#{r.dense_rank}")
        if r.sparse_rank is not None:
            source_tag.append(f"sparse#{r.sparse_rank}")
        source_str = " + ".join(source_tag) if source_tag else "none"

        preview = r.chunk_text[:120].replace("\n", " ")
        print(
            f"  [{i}] rrf={r.rrf_score:.4f} ({source_str})  "
            f"[{r.source_file} p.{','.join(str(p) for p in r.page_numbers)}]",
            file=sys.stderr,
        )
        print(f"      {preview}...", file=sys.stderr)

    print("---", file=sys.stderr)


def main() -> None:
    load_dotenv(override=True)

    # Fix Windows console encoding
    if sys.stdout.encoding != "utf-8":
        sys.stdout = io.TextIOWrapper(
            sys.stdout.buffer, encoding="utf-8", errors="replace"
        )
    if sys.stderr.encoding != "utf-8":
        sys.stderr = io.TextIOWrapper(
            sys.stderr.buffer, encoding="utf-8", errors="replace"
        )

    parser = argparse.ArgumentParser(
        description="Query the Yoke knowledge base with hybrid retrieval."
    )
    parser.add_argument("question", help="Natural language query.")
    parser.add_argument(
        "--db",
        type=Path,
        default=Path("data/yoke.db"),
        help="Path to the SQLite database (default: data/yoke.db).",
    )
    parser.add_argument(
        "--bm25",
        type=Path,
        default=None,
        help="Path to the BM25 index (default: <db>.bm25.json).",
    )
    parser.add_argument(
        "-k",
        type=int,
        default=10,
        help="Number of chunks to retrieve (default: 10).",
    )
    parser.add_argument(
        "--no-generate",
        action="store_true",
        help="Skip LLM generation — only show retrieved chunks.",
    )
    parser.add_argument(
        "--model",
        default=GENERATION_MODEL,
        help=f"Generation model (default: {GENERATION_MODEL}).",
    )
    args = parser.parse_args()

    bm25_path = args.bm25 or args.db.with_suffix(".bm25.json")

    if not args.db.exists():
        print(f"Error: database not found at {args.db}", file=sys.stderr)
        sys.exit(1)
    if not bm25_path.exists():
        print(f"Error: BM25 index not found at {bm25_path}", file=sys.stderr)
        sys.exit(1)

    if args.no_generate:
        # Retrieval only — use retrieve directly
        results, timings = retrieve_with_timings(
            args.question, args.db, bm25_path, k=args.k
        )
        _print_diagnostics(results, timings)
        return

    # Full pipeline: retrieve + generate
    result = query(
        args.question,
        args.db,
        bm25_path,
        k=args.k,
        model=args.model,
    )
    _print_diagnostics(result.sources, result.retrieval_timings)

    print(
        f"\n--- Generation ({result.generation_ms:.0f}ms, "
        f"total={result.total_ms:.0f}ms) ---",
        file=sys.stderr,
    )
    print(result.answer)


if __name__ == "__main__":
    main()
