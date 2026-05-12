import argparse
from pathlib import Path

import anthropic
from dotenv import load_dotenv

from yoke.tracing import flush_tracing, init_tracing, set_current_trace_id


MODEL = "claude-sonnet-4-20250514"

SYSTEM_PROMPT = (
    "Answer the question using only the provided context. "
    "If the context does not contain enough information to answer, "
    "say 'I don't have enough information to answer this question.' "
    "Be concise and specific."
)


def ask(question: str, docs_dir: Path) -> str:
    paths = sorted(
        p for p in docs_dir.iterdir()
        if p.suffix in (".md", ".txt") and p.is_file()
    )
    sections = []
    for p in paths:
        sections.append(f"## {p.name}\n{p.read_text(encoding='utf-8')}")
    context = "\n---\n".join(sections)

    langfuse = init_tracing()
    trace = None
    if langfuse is not None:
        trace = langfuse.trace(name="baseline-ask")
        set_current_trace_id(trace.id)

    client = anthropic.Anthropic()
    user_content = f"Context:\n{context}\n\nQuestion: {question}"
    message = client.messages.create(
        model=MODEL,
        temperature=0,
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_content}],
    )
    answer = message.content[0].text

    if trace is not None:
        trace.generation(
            name="baseline-llm",
            model=MODEL,
            input={"prompt": user_content, "system": SYSTEM_PROMPT},
        ).end(
            output=answer,
            usage={
                "prompt_tokens": message.usage.input_tokens,
                "completion_tokens": message.usage.output_tokens,
            },
        )

    flush_tracing(langfuse)
    return answer


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument("--docs-dir", type=Path, required=True)
    parser.add_argument("question")
    args = parser.parse_args()
    print(ask(args.question, args.docs_dir))


if __name__ == "__main__":
    main()
