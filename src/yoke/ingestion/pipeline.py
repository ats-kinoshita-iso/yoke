"""End-to-end ingestion pipeline with CLI entry point."""

import asyncio
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from tqdm import tqdm

from yoke.config import YokeSettings, parse_model_spec
from yoke.extract import extract_pdf_pages, extract_pdf_with_page_map
from yoke.ingestion.chunking import chunk_text
from yoke.ingestion.embedding import embed_texts_async
from yoke.ingestion.enrichment import enrich_chunks
from yoke.ingestion.models import IngestResult
from yoke.ingestion.store import build_bm25_index, init_db, store_document
from yoke.models import get_model_client
from yoke.tracing import flush_tracing, init_tracing

logger = logging.getLogger(__name__)


async def ingest_directory(
    source_dir: Path,
    db_path: Path,
    *,
    summary_model: str | None = None,
    embedding_model: str | None = None,
) -> IngestResult:
    """Ingest all PDFs and text files in a directory.

    Args:
        source_dir: Directory containing .pdf and/or .txt files.
        db_path: Path for the SQLite database output.
        summary_model: Model spec for enrichment (e.g. "ollama/gemma4:e2b").
        embedding_model: OpenAI embedding model name.

    Returns:
        IngestResult with stats and any errors.
    """
    settings = YokeSettings()
    summary_model = summary_model or settings.summary_model
    embedding_model = embedding_model or settings.embedding_model

    langfuse = init_tracing()

    provider, model_name = parse_model_spec(summary_model)
    summary_client = get_model_client(provider, model_name, langfuse=langfuse)

    # Determine concurrency based on provider
    enrichment_concurrency = 3 if provider == "ollama" else 10
    embedding_concurrency = 10

    # Find input files
    pdf_files = sorted(source_dir.glob("*.pdf"))
    txt_files = sorted(source_dir.glob("*.txt"))
    all_files = pdf_files + txt_files

    if not all_files:
        return IngestResult(
            documents_processed=0,
            total_chunks=0,
            db_path=str(db_path),
            bm25_path=str(db_path.with_suffix(".bm25.json")),
            errors=[f"No .pdf or .txt files found in {source_dir}"],
        )

    conn = init_db(db_path)
    errors: list[str] = []
    total_chunks = 0
    docs_processed = 0

    file_bar = tqdm(all_files, desc="Documents", unit="doc")
    for file_path in file_bar:
        file_bar.set_postfix_str(file_path.name)
        try:
            # Step 1: Extract text
            page_map: list[int] | None = None
            if file_path.suffix == ".pdf":
                full_text, page_map = extract_pdf_with_page_map(file_path)
            else:
                full_text = file_path.read_text(encoding="utf-8")

            if not full_text.strip():
                errors.append(f"{file_path.name}: empty after extraction")
                continue

            # Step 2: Chunk (pass page map for accurate page attribution)
            chunks = chunk_text(
                full_text,
                source_file=file_path.name,
                page_numbers=page_map,
            )
            logger.info("%s: %d chunks", file_path.name, len(chunks))

            # Step 3: Enrich (parallel with semaphore)
            enrich_bar = tqdm(
                total=len(chunks),
                desc=f"  Enriching {file_path.name}",
                unit="chunk",
                leave=False,
            )
            enriched = await enrich_chunks(
                full_text,
                chunks,
                summary_client,
                max_concurrent=enrichment_concurrency,
                on_chunk_complete=lambda: enrich_bar.update(1),
            )
            enrich_bar.close()

            # Step 4: Embed (parallel with batching)
            texts_to_embed = [ec.enriched_text for ec in enriched]
            embeddings = await embed_texts_async(
                texts_to_embed,
                model=embedding_model,
                max_concurrent=embedding_concurrency,
            )

            # Step 5: Store
            store_document(conn, file_path.name, full_text, enriched, embeddings)
            total_chunks += len(chunks)
            docs_processed += 1

        except Exception as e:
            msg = f"{file_path.name}: {type(e).__name__}: {e}"
            logger.error(msg)
            errors.append(msg)

    # Build BM25 index over all chunks
    bm25_path = db_path.with_suffix(".bm25.json")
    if total_chunks > 0:
        build_bm25_index(conn, bm25_path)
        logger.info("BM25 index saved to %s", bm25_path)

    conn.close()

    # Write manifest
    manifest = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "documents_processed": docs_processed,
        "total_chunks": total_chunks,
        "summary_model": summary_model,
        "embedding_model": embedding_model,
        "db_path": str(db_path),
        "bm25_path": str(bm25_path),
        "errors": errors,
    }
    manifest_path = db_path.with_name("manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    logger.info("Manifest written to %s", manifest_path)

    flush_tracing(langfuse)

    return IngestResult(
        documents_processed=docs_processed,
        total_chunks=total_chunks,
        db_path=str(db_path),
        bm25_path=str(bm25_path),
        errors=errors,
    )


def main() -> None:
    """CLI entry point for yoke-ingest."""
    import argparse
    import sys

    from dotenv import load_dotenv

    load_dotenv(override=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Ingest PDF/text files into a Yoke knowledge base."
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        required=True,
        help="Directory containing .pdf and/or .txt files to ingest.",
    )
    parser.add_argument(
        "--db-path",
        type=Path,
        default=Path("data/yoke.db"),
        help="Path for the SQLite database (default: data/yoke.db).",
    )
    parser.add_argument(
        "--summary-model",
        default=None,
        help="Model for contextual enrichment (default: from YOKE_SUMMARY_MODEL).",
    )
    parser.add_argument(
        "--embedding-model",
        default=None,
        help="OpenAI embedding model (default: from YOKE_EMBEDDING_MODEL).",
    )
    args = parser.parse_args()

    if not args.source_dir.is_dir():
        print(f"Error: {args.source_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    result = asyncio.run(ingest_directory(
        args.source_dir,
        args.db_path,
        summary_model=args.summary_model,
        embedding_model=args.embedding_model,
    ))

    print(f"\nIngestion complete:")
    print(f"  Documents processed: {result.documents_processed}")
    print(f"  Total chunks:        {result.total_chunks}")
    print(f"  Database:            {result.db_path}")
    print(f"  BM25 index:          {result.bm25_path}")
    if result.errors:
        print(f"  Errors ({len(result.errors)}):")
        for err in result.errors:
            print(f"    - {err}")


if __name__ == "__main__":
    main()
