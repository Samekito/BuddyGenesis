from pathlib import Path

import pytest

from buddy.config import PROJECT_ROOT
from buddy.knowledge import (
    DEPARTMENT_NAMES,
    MAX_CHUNK_CHARS,
    Chunk,
    KnowledgeBase,
    _is_skipped_file,
    chunk_blocks,
    tokenize,
)


def test_tokenize_treats_joined_and_spaced_course_codes_alike():
    assert tokenize("CSC101") == tokenize("csc 101") == ["csc", "101", "csc101"]


def test_tokenize_does_not_invent_course_codes_from_other_numbers():
    assert "level300" not in tokenize("300 level courses") and tokenize("year 2017") == ["year", "2017"]


def test_tokenize_drops_stopwords_and_punctuation():
    assert tokenize("What is the grading system?") == ["grading", "system"]


def test_chunk_blocks_respects_size_limit_and_labels_every_chunk():
    blocks = [f"Paragraph {i} " + "word " * 60 for i in range(30)]

    chunks = chunk_blocks(blocks, "Test Dept (TEST)")

    assert len(chunks) > 1
    assert all(chunk.text.startswith("Test Dept (TEST)") for chunk in chunks)
    # Label line may push a chunk slightly past the limit, never by more than a line.
    assert all(len(chunk.text) <= MAX_CHUNK_CHARS + 100 for chunk in chunks)


def test_chunk_blocks_carries_section_heading_into_split_tables():
    table_rows = "\n".join(f"CSC {300 + i} | C | Course {i} | 2 | 1 | 0 | 3" for i in range(80))

    chunks = chunk_blocks(["300 LEVEL FIRST SEMESTER", table_rows], "CS (CSC)")

    assert len(chunks) > 1
    assert all("300 LEVEL FIRST SEMESTER" in chunk.text for chunk in chunks)


def test_search_ranks_matching_chunk_first():
    kb = KnowledgeBase([
        Chunk("A", "Cyber Security mission statement"),
        Chunk("B", "CSC 101 Introduction to Computer Science 2 units"),
        Chunk("C", "Industrial training takes place at 400 level"),
    ])

    results = kb.search("how many units is CSC101", limit=2)

    assert results[0].source == "B"


def test_search_with_only_stopwords_returns_nothing():
    kb = KnowledgeBase([Chunk("A", "Computer Science")])

    assert kb.search("what is the") == []


def test_search_with_no_overlap_returns_nothing():
    kb = KnowledgeBase([Chunk("A", "Computer Science"), Chunk("B", "Software Engineering")])

    assert kb.search("football fixtures") == []


def test_empty_knowledge_base_is_rejected():
    with pytest.raises(ValueError):
        KnowledgeBase([])


def test_lock_files_and_copies_are_skipped():
    assert _is_skipped_file(Path("~$C HANDBOOK2.docx"))
    assert _is_skipped_file(Path("CSC HANDBOOK2copy.docx"))
    assert not _is_skipped_file(Path("CSC HANDBOOK2.docx"))


def test_real_handbooks_load_one_source_per_department_without_filenames():
    kb = KnowledgeBase.from_directory(PROJECT_ROOT / "knowledge_base")

    assert {chunk.source for chunk in kb.chunks} == set(DEPARTMENT_NAMES.values())
    assert not any("HANDBOOK" in chunk.text.split("\n")[0] for chunk in kb.chunks)


def test_course_facts_list_every_department_that_has_the_course():
    kb = KnowledgeBase([
        Chunk("Dept A", "CSC 101 Introduction to Computer Science"),
        Chunk("Dept B", "Prerequisite: CSC101"),
        Chunk("Dept C", "CYS 301 only"),
    ])

    assert kb.course_facts("which department offers csc101?") == [
        "CSC 101 appears in these programmes (complete list): Dept A; Dept B."
    ]


def test_course_facts_say_when_a_course_is_not_in_any_handbook():
    kb = KnowledgeBase([Chunk("Dept A", "CSC 101")])

    assert kb.course_facts("Tell me about CSC 999") == ["CSC 999 does not appear in any School of Computing handbook."]


def test_course_facts_are_empty_without_course_codes():
    kb = KnowledgeBase([Chunk("Dept A", "CSC 101")])

    assert kb.course_facts("what is the grading system?") == []


def test_real_handbooks_list_csc101_in_all_five_departments():
    # Regression: the bot once named only 4 departments because search returns a sample of chunks.
    kb = KnowledgeBase.from_directory(PROJECT_ROOT / "knowledge_base")

    [fact] = kb.course_facts("which department offers CSC101?")

    departments = [name for code, name in DEPARTMENT_NAMES.items() if code != "SOC"]
    assert all(name in fact for name in departments)


def test_course_facts_ignore_words_that_only_look_like_codes():
    kb = KnowledgeBase([Chunk("Dept A", "CSC 101 is taken in the 100 level")])

    assert kb.course_facts("what are the 100 level courses?") == []


LEVEL_TABLES = [
    Chunk(DEPARTMENT_NAMES["SEN"], "100 LEVEL FIRST SEMESTER\nCSC 101 | C | Intro | 2"),
    Chunk(DEPARTMENT_NAMES["SEN"], "Department of Software Engineering — 100 LEVEL FIRST SEMESTER\nPHY 107 | C | Lab | 1"),
    Chunk(DEPARTMENT_NAMES["SEN"], "100 LEVEL SECOND SEMESTER\nCSC 102 | C | Computing | 3"),
    Chunk(DEPARTMENT_NAMES["CSC"], "100 LEVEL FIRST SEMESTER\nBIO 101 | C | Biology | 3"),
    Chunk(DEPARTMENT_NAMES["SEN"], "Students register at the start of 100 level."),
]


def test_level_sections_return_every_chunk_of_the_named_table():
    kb = KnowledgeBase(LEVEL_TABLES)

    sections = kb.level_sections("what are the 100 level first semester courses in software engineering?")

    assert sections == LEVEL_TABLES[:2]


def test_level_sections_without_semester_or_department_return_both_semesters_everywhere():
    kb = KnowledgeBase(LEVEL_TABLES)

    assert kb.level_sections("100l courses") == LEVEL_TABLES[:4]


def test_level_sections_understand_harmattan_and_rain_semesters():
    kb = KnowledgeBase(LEVEL_TABLES)

    assert kb.level_sections("100 level rain semester software engineering") == [LEVEL_TABLES[2]]


def test_level_sections_are_empty_without_a_level():
    kb = KnowledgeBase(LEVEL_TABLES)

    assert kb.level_sections("what is the grading system?") == []


def test_retrieve_puts_table_sections_first_without_duplicates():
    kb = KnowledgeBase(LEVEL_TABLES)

    results = kb.retrieve("100 level first semester software engineering courses")

    assert results[:2] == LEVEL_TABLES[:2]
    assert len(results) == len(set(map(id, results)))


def test_level_sections_accept_typographic_hyphens_from_the_rewrite_model():
    # Regression: the rewrite model wrote "100‑level" (non-breaking hyphen) and the lookup missed.
    kb = KnowledgeBase(LEVEL_TABLES)

    sections = kb.level_sections("What are the 100‑level first‑semester courses in Software Engineering?")

    assert sections == LEVEL_TABLES[:2]
