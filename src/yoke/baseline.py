import argparse
from pathlib import Path

from dotenv import load_dotenv

from yoke.config import YokeSettings, generate as llm_generate

GENERATION_MODEL = YokeSettings().generation_model

SYSTEM_PROMPT = (
    "Answer the question using only the provided context. "
    "If the context does not contain enough information to answer, "
    "say 'I don't have enough information to answer this question.' "
    "Be concise and specific."
)


def ask(question: str, docs_dir: Path, *, model: str = GENERATION_MODEL) -> str:
    """Answer a question using context-stuffing from all docs in a directory."""
    paths = sorted(
        p for p in docs_dir.iterdir()
        if p.suffix in (".md", ".txt") and p.is_file()
    )
    sections = []
    for p in paths:
        sections.append(f"## {p.name}\n{p.read_text(encoding='utf-8')}")
    context = "\n---\n".join(sections)

    return llm_generate(
        model,
        f"Context:\n{context}\n\nQuestion: {question}",
        system=SYSTEM_PROMPT,
    )


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument("--docs-dir", type=Path, required=True)
    parser.add_argument("question")
    args = parser.parse_args()
    print(ask(args.question, args.docs_dir))


if __name__ == "__main__":
    main()
