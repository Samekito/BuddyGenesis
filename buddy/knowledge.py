"""Handbook knowledge: loads knowledge_base/*.docx, chunks it, and answers BM25 keyword searches.

Consumed by app.py (built once at startup) and scripts/search.py (debugging retrieval).
BM25 instead of embeddings: the free host has 512 MB RAM, too little for an embedding model.
"""

import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

from docx import Document
from docx.table import Table
from rank_bm25 import BM25Okapi

# Filename prefix -> full name. The full name is prepended to every chunk so that
# "software engineering courses" matches SEN chunks even when the text only says "SEN".
DEPARTMENT_NAMES = {
    "CSC": "Department of Computer Science",
    "CYS": "Department of Cyber Security",
    "IFS": "Department of Information Systems",
    "IFT": "Department of Information Technology",
    "SEN": "Department of Software Engineering",
    "SOC": "School of Computing",
}

# ~1500 chars matches the chunk size the original Pinecone index used, and keeps
# five retrieved chunks well inside the model's context and Groq's free-tier token limits.
MAX_CHUNK_CHARS = 1500

# Short standalone paragraphs are treated as section headings (e.g. "200 LEVEL FIRST SEMESTER")
# and repeated at the top of each chunk, so a course table split across chunks keeps its level.
MAX_HEADING_CHARS = 80

# 8, not 5: shared 100-level course tables appear in all five handbooks and compete with
# prerequisite mentions; at 5 the retrieval eval (tests/test_retrieval_quality.py) missed 2 of 12.
# 8 chunks is ~3K tokens per request, well inside Groq's free-tier limits.
SEARCH_RESULTS = 8

# \s, not " ": LLM-written text often puts a narrow no-break space (U+202F) inside "CSC 101".
COURSE_CODE = re.compile(r"\b([a-z]{3})\s*(\d{3})\b")
HANDBOOK_COURSE_CODE = re.compile(r"\b([A-Z]{3})\s*(\d{3})\b")

# "100 level", "100-level", "100l" in a (lower-cased) question.
LEVEL_IN_QUERY = re.compile(r"\b([1-5]00)\s*-?\s*l(?:evel|vl)?\b")
# Handbook table headings say FIRST/SECOND SEMESTER; students also say harmattan/rain semester.
SEMESTER_WORDS = {"FIRST": ("first", "1st", "harmattan"), "SECOND": ("second", "2nd", "rain")}
DEPARTMENT_ALIASES = {
    DEPARTMENT_NAMES["CSC"]: ("computer science",),
    DEPARTMENT_NAMES["CYS"]: ("cyber security", "cybersecurity"),
    DEPARTMENT_NAMES["IFS"]: ("information system",),
    DEPARTMENT_NAMES["IFT"]: ("information technology",),
    DEPARTMENT_NAMES["SEN"]: ("software engineering",),
}

STOPWORDS = frozenset(
    "a an and are as at be by can do does for from how i in is it me my of on or "
    "please tell that the their there this to was what when where which who why will with you your".split()
)


@dataclass(frozen=True)
class Chunk:
    source: str
    text: str


class KnowledgeBase:
    def __init__(self, chunks: list[Chunk]):
        if not chunks:
            raise ValueError("Knowledge base is empty — no .docx handbooks were found.")
        self.chunks = chunks
        self._index = BM25Okapi([tokenize(chunk.text) for chunk in chunks])
        self._course_sources = _index_course_sources(chunks)

    @classmethod
    def from_directory(cls, directory: Path) -> "KnowledgeBase":
        chunks = []
        for path in sorted(directory.glob("*.docx")):
            if _is_skipped_file(path):
                continue
            chunks.extend(chunk_blocks(_read_blocks(path), _describe_source(path)))
        return cls(chunks)

    def search(self, query: str, limit: int = SEARCH_RESULTS) -> list[Chunk]:
        query_tokens = tokenize(query)
        if not query_tokens:
            return []
        scores = self._index.get_scores(query_tokens)
        ranked = sorted(range(len(self.chunks)), key=lambda i: scores[i], reverse=True)
        return [self.chunks[i] for i in ranked[:limit] if scores[i] > 0]

    def retrieve(self, query: str) -> list[Chunk]:
        """What the model is shown: every course-table section the question names, then the best search hits."""
        sections = self.level_sections(query)
        return sections + [chunk for chunk in self.search(query) if chunk not in sections]

    def level_sections(self, query: str) -> list[Chunk]:
        """All chunks of the "N00 LEVEL FIRST/SECOND SEMESTER" tables a question asks about.

        Course tables are long and full of numbers, so BM25 ranks them below short paragraphs
        that merely mention "100 level". A list question needs the whole table, so look it up directly.
        """
        text = _plain(query).lower()
        level = LEVEL_IN_QUERY.search(text)
        if level is None:
            return []
        semesters = [
            semester for semester, words in SEMESTER_WORDS.items() if any(re.search(rf"\b{word}\b", text) for word in words)
        ] or list(SEMESTER_WORDS)
        departments = {source for source, aliases in DEPARTMENT_ALIASES.items() if any(alias in text for alias in aliases)}
        headings = [re.compile(rf"\b{level.group(1)}\s*LEVEL\s+{semester}\s+SEMESTER\b") for semester in semesters]
        return [
            chunk for chunk in self.chunks
            if (not departments or chunk.source in departments) and any(heading.search(chunk.text) for heading in headings)
        ]

    def course_facts(self, query: str) -> list[str]:
        """Exact, complete statements of which handbooks list each course code in the query.

        Search returns only the top chunks — a sample — so it cannot answer "which departments
        offer CSC 101?" reliably. This index covers every handbook, so the answer is exhaustive.
        """
        facts = []
        known_prefixes = {code[:3] for code in self._course_sources}
        for code in _course_codes(query):
            # "the 100 level courses" looks like a code; only real department prefixes count.
            if code[:3] not in known_prefixes:
                continue
            sources = sorted(self._course_sources.get(code, ()))
            letters, digits = code[:3].upper(), code[3:]
            if sources:
                facts.append(f"{letters} {digits} appears in these programmes (complete list): {'; '.join(sources)}.")
            else:
                facts.append(f"{letters} {digits} does not appear in any School of Computing handbook.")
        return facts


def tokenize(text: str) -> list[str]:
    words = [token for token in re.findall(r"[a-z0-9]+", _space_letters_and_digits(text)) if token not in STOPWORDS]
    # Each course code also becomes one rare token ("csc101"). Its high IDF lets the course
    # outweigh generic words like "semester" or "department" that appear in every handbook.
    return words + _course_codes(text)


def chunk_blocks(blocks: list[str], source: str) -> list[Chunk]:
    chunks: list[Chunk] = []
    current: list[str] = []
    current_heading = ""

    def flush():
        if current:
            chunks.append(Chunk(source=source, text="\n".join(current)))
            current.clear()

    for block in blocks:
        if _is_section_heading(block):
            # A chunk never spans two sections, so its label always names the section it holds.
            # Without this, course descriptions inherited "500 LEVEL FIRST SEMESTER" from the last table.
            flush()
            current_heading = block
            continue
        for piece in _split_oversized(block):
            size_so_far = sum(len(line) + 1 for line in current)
            if current and size_so_far + len(piece) > MAX_CHUNK_CHARS:
                flush()
            if not current:
                current.append(source if not current_heading else f"{source} — {current_heading}")
            current.append(piece)
        if _is_heading(block):
            current_heading = block
    flush()
    return chunks


def _is_heading(block: str) -> bool:
    # Course title lines ("BIO 103 General Biology (1 Unit)") are short too, but label one course, not a section.
    return len(block) <= MAX_HEADING_CHARS and "|" not in block and not HANDBOOK_COURSE_CODE.match(block)


def _is_section_heading(block: str) -> bool:
    """All-caps headings ("200 LEVEL FIRST SEMESTER", "COURSE SYNOPSIS") open a new section of the handbook."""
    return _is_heading(block) and block.isupper()


def _plain(text: str) -> str:
    """Maps typographic dashes and spaces to ASCII "-" and " ".

    The rewrite model writes "100‑level" (U+2011) and "CSC 101" (U+202F); without this,
    patterns written with "-" and " " silently stop matching.
    """
    return "".join(
        "-" if unicodedata.category(char) == "Pd" else " " if unicodedata.category(char) == "Zs" else char
        for char in text
    )


def _space_letters_and_digits(text: str) -> str:
    # Split letter/digit boundaries so "CSC101" and "CSC 101" read the same.
    return re.sub(r"(?<=[a-zA-Z])(?=\d)|(?<=\d)(?=[a-zA-Z])", " ", _plain(text).lower())


def _course_codes(text: str) -> list[str]:
    """Course codes in normalised form: "CSC 101", "csc101" and "CSC 101" all give "csc101"."""
    return [f"{letters}{digits}" for letters, digits in COURSE_CODE.findall(_space_letters_and_digits(text))]


def _index_course_sources(chunks: list[Chunk]) -> dict[str, set[str]]:
    # Handbooks always write codes in capitals ("CSC 101"), so matching case-sensitively keeps
    # prose like "the 100 level" or "AND 200" out of the index.
    course_sources: dict[str, set[str]] = {}
    for chunk in chunks:
        for letters, digits in HANDBOOK_COURSE_CODE.findall(chunk.text):
            course_sources.setdefault(f"{letters.lower()}{digits}", set()).add(chunk.source)
    return course_sources


def _is_skipped_file(path: Path) -> bool:
    # "~$" files are Word lock files; "copy" files duplicate a handbook and would double its weight.
    return path.name.startswith("~$") or "copy" in path.stem.lower()


def _describe_source(path: Path) -> str:
    # Department name only: a filename like "SEN HANDBOOK1" in the text invites the model
    # to say "according to the handbook", which the system prompt forbids.
    code = path.stem.split()[0].upper()
    return DEPARTMENT_NAMES.get(code, code)


def _read_blocks(path: Path) -> list[str]:
    blocks = []
    for item in Document(str(path)).iter_inner_content():
        text = _table_to_text(item) if isinstance(item, Table) else item.text.strip()
        if text:
            blocks.append(text)
    return blocks


def _table_to_text(table: Table) -> str:
    rows = []
    for row in table.rows:
        cells: list[str] = []
        for cell in row.cells:
            text = " ".join(cell.text.split())
            # Merged cells are returned once per grid column; keep one copy.
            if text and (not cells or cells[-1] != text):
                cells.append(text)
        if cells:
            rows.append(" | ".join(cells))
    return "\n".join(rows)


def _split_oversized(block: str) -> list[str]:
    if len(block) <= MAX_CHUNK_CHARS:
        return [block]
    pieces, current = [], ""
    # Tables split on rows; long paragraphs fall back to sentence boundaries.
    for part in re.split(r"(?<=\n)|(?<=\. )", block):
        if current and len(current) + len(part) > MAX_CHUNK_CHARS:
            pieces.append(current.strip())
            current = ""
        current += part
    if current.strip():
        pieces.append(current.strip())
    return pieces
