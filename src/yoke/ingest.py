"""Allow running as: python -m yoke.ingest --source-dir ./corpus/"""
from yoke.ingestion.pipeline import main as _main

if __name__ == "__main__":
    _main()
