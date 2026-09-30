"""Prints what the retriever returns for a question — for debugging "I don't know" answers.

Usage: .venv/Scripts/python scripts/search.py "how many units is CSC 101"
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from buddy.config import load_settings  # noqa: E402 — needs the sys.path entry above
from buddy.knowledge import KnowledgeBase  # noqa: E402


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    question = " ".join(sys.argv[1:])
    knowledge_base = KnowledgeBase.from_directory(load_settings().knowledge_dir)
    results = knowledge_base.search(question)
    if not results:
        print("No matching chunks — none of the question's keywords appear in the handbooks.")
    for rank, chunk in enumerate(results, start=1):
        print(f"\n#{rank} — {chunk.source}\n{chunk.text[:600]}")


if __name__ == "__main__":
    main()
