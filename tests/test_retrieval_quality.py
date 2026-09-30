"""Retrieval evaluation against the real handbooks: every question must retrieve its answer.

Guards BM25 tuning (chunk size, labels, tokenizer, SEARCH_RESULTS) — ranking is sensitive to
small changes, and a miss here means the bot will answer "I don't know" to a real student question.
"""

import pytest

from buddy.config import PROJECT_ROOT
from buddy.knowledge import KnowledgeBase


def _first_line(text: str) -> str:
    return text.split("\n", 1)[0]


# (question, predicate the retrieved set must satisfy by at least one chunk)
EVALUATION_SET = [
    ("How many units is CSC 101?", lambda t: "CSC101" in t.replace(" ", "") and "unit" in t.lower()),
    # Rewritten follow-up as the model produced it, including a narrow no-break space (U+202F).
    ("Which department offers CSC\u202f101, and in which semester is it taken?", lambda t: "SEMESTER" in t and "CSC 101" in t),
    ("In which level and semester is CSC 101 taken?", lambda t: "SEMESTER" in t and "CSC 101" in t),
    ("What is the mission of the cybersecurity department?", lambda t: "MISSION" in t and "Cybersecurity" in t),
    ("When was the School of Computing created?", lambda t: "2017" in t and "School of Computing" in t),
    ("What CGPA do I need for first class?", lambda t: "4.50" in t and "First Class" in t),
    ("What are the 300 level first semester courses in Information Technology?",
     lambda t: "300 LEVEL FIRST SEMESTER" in t and "Information Technology" in _first_line(t)),
    ("What is the minimum CGPA to graduate?", lambda t: "1.50" in t),
    ("When does industrial training take place?", lambda t: "Industrial Training" in t and "400" in t),
    ("What is the philosophy of the software engineering programme?",
     lambda t: "philosophy" in t.lower() and "Software Engineering" in _first_line(t)),
    ("What courses are in 200 level second semester for computer science?",
     lambda t: "200 LEVEL SECOND SEMESTER" in t and "Computer Science" in _first_line(t)),
    ("Tell me about CYS 301", lambda t: "CYS 301" in t),
    # Regression: long course tables used to lose to short paragraphs that mention "100 level".
    ("what are the 100 level first semester courses in software engineering?",
     lambda t: "Software Engineering" in _first_line(t) and "PHY 107" in t and "BIO 101" in t),
    # The same question as rewritten by the model, with non-breaking hyphens (U+2011).
    ("What are the 100‑level first‑semester courses offered by the Software Engineering department?",
     lambda t: "Software Engineering" in _first_line(t) and "PHY 107" in t and "BIO 101" in t),
    ("300l courses in information technology", lambda t: "Information Technology" in _first_line(t) and "IFT 306" in t),
]


@pytest.fixture(scope="module")
def knowledge_base():
    return KnowledgeBase.from_directory(PROJECT_ROOT / "knowledge_base")


@pytest.mark.parametrize(("question", "contains_answer"), EVALUATION_SET, ids=[q for q, _ in EVALUATION_SET])
def test_question_retrieves_its_answer(knowledge_base, question, contains_answer):
    results = knowledge_base.retrieve(question)

    assert any(contains_answer(chunk.text) for chunk in results)
