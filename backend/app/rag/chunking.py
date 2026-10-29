"""
Parent-child chunking for the DR AI Platform RAG system.

Pipeline:
    PDF documents
        -> ingestion.py
        -> parent-child chunking
        -> child embeddings
        -> MMR retrieval
        -> reranking
        -> parent context expansion
        -> Groq answer generation

Features:
- Parent chunks with approximately 1,500 word-based tokens.
- Child chunks with approximately 300 word-based tokens.
- Configurable overlap for parents and children.
- Source document and page tracking.
- Stable document, parent, and child IDs.
- Metadata preservation.
- JSON export for inspection and testing.
- No additional dependencies beyond ingestion.py and pypdf.

IMPORTANT:
The default tokenizer is a word-based approximation, not a model-specific
tokenizer. Replace it with the embedding model's tokenizer if exact token
limits are required.
"""

from __future__ import annotations

import json
import logging
import re

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

from app.rag.ingestion import (
    IngestedPage,
    IngestionReport,
    ingest_documents,
)


# ---------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[3]

DOCUMENTS_DIR = PROJECT_ROOT / "data" / "rag_documents"

DEFAULT_OUTPUT_FILE = (
    PROJECT_ROOT / "data" / "rag_documents" / "chunking_output.json"
)

# Parent chunk settings.
PARENT_CHUNK_SIZE = 1500
PARENT_CHUNK_OVERLAP = 150

# Child chunk settings.
CHILD_CHUNK_SIZE = 300
CHILD_CHUNK_OVERLAP = 50

# A child chunk must contain at least this many word tokens.
MIN_CHILD_TOKENS = 1

# Regex-based approximation:
# - Words and numbers are individual tokens.
# - Punctuation is also retained as a token.
#
# This preserves punctuation and common medical terms reasonably well.
TOKEN_PATTERN = re.compile(
    r"\w+(?:[./'-]\w+)*|[^\w\s]",
    flags=re.UNICODE,
)


# ---------------------------------------------------------------------
# DATA STRUCTURES
# ---------------------------------------------------------------------

@dataclass
class ParentChunk:
    """A larger text segment used to provide context to the LLM."""

    parent_id: str
    document_id: str
    source_file: str
    source_path: str
    document_title: str
    category: str

    chunk_index: int
    text: str

    token_count: int
    page_numbers: list[int]

    #Page number corresponding to each token in this parent chunk.
    #used to calculate accurate page citations for child chunks.
    token_page_numbers: list[int] = field(default_factory=list)

    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ChildChunk:
    """A smaller text segment intended for embedding and retrieval."""

    child_id: str
    parent_id: str

    document_id: str
    source_file: str
    source_path: str
    document_title: str
    category: str

    chunk_index: int
    text: str

    token_count: int
    page_numbers: list[int]

    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ChunkingReport:
    """Summary of the chunking operation."""

    input_pages: int
    documents_processed: int
    parent_chunks_created: int
    child_chunks_created: int

    parent_chunk_size: int
    parent_chunk_overlap: int

    child_chunk_size: int
    child_chunk_overlap: int

    tokenizer: str

    errors: list[dict[str, str]] = field(default_factory=list)


# ---------------------------------------------------------------------
# TOKENIZATION
# ---------------------------------------------------------------------

def tokenize_text(text: str) -> list[str]:
    """
    Split text into tokens using a lightweight regex tokenizer.

    These tokens are approximate and should not be treated as exact
    LLM or embedding-model tokens.
    """

    if not text:
        return []

    return TOKEN_PATTERN.findall(text)


def count_tokens(text: str) -> int:
    """Return the approximate token count for a string."""

    return len(tokenize_text(text))


def detokenize(tokens: list[str]) -> str:
    """
    Reconstruct readable text from regex tokens.

    Avoid inserting spaces before common punctuation marks.
    """

    if not tokens:
        return ""

    text = " ".join(tokens)

    # Remove spaces before punctuation.
    text = re.sub(r"\s+([,.;:!?%)\]}])", r"\1", text)

    # Remove spaces immediately after opening brackets.
    text = re.sub(r"([(\[{])\s+", r"\1", text)

    # Normalize whitespace.
    text = re.sub(r"[ \t]+", " ", text)

    return text.strip()


def split_token_windows(
    tokens: list[str],
    chunk_size: int,
    overlap: int,
) -> list[tuple[int, int]]:
    """
    Return half-open token ranges for overlapping chunks.

    Example:
        (0, 300), (250, 550), (500, 800)

    Each range uses:
        start index inclusive
        end index exclusive
    """

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero.")

    if overlap < 0:
        raise ValueError("overlap cannot be negative.")

    if overlap >= chunk_size:
        raise ValueError(
            "overlap must be smaller than chunk_size."
        )

    if not tokens:
        return []

    step = chunk_size - overlap
    windows: list[tuple[int, int]] = []

    start = 0

    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))

        windows.append((start, end))

        if end >= len(tokens):
            break

        start += step

    return windows


# ---------------------------------------------------------------------
# PAGE AND DOCUMENT GROUPING
# ---------------------------------------------------------------------

def group_pages_by_document(
    pages: list[IngestedPage],
) -> dict[str, list[IngestedPage]]:
    """
    Group extracted pages by document ID.

    The page order is preserved using the original PDF page number.
    """

    grouped: dict[str, list[IngestedPage]] = {}

    for page in pages:
        grouped.setdefault(page.document_id, []).append(page)

    for document_id in grouped:
        grouped[document_id].sort(
            key=lambda page: page.page_number
        )

    return grouped


def build_document_token_stream(
    pages: list[IngestedPage],
) -> tuple[list[str], list[int]]:
    """
    Build a single token stream for a document while preserving the
    original page number associated with each token.

    A page transition inserts a newline boundary. Page numbers remain
    attached to tokens so chunks can identify their source pages.

    Returns:
        tokens:
            Combined token list.

        token_page_numbers:
            One source page number for each token.
    """

    ordered_pages = sorted(
        pages,
        key=lambda page: page.page_number,
    )

    all_tokens: list[str] = []
    token_page_numbers: list[int] = []

    for page in ordered_pages:
        page_tokens = tokenize_text(page.text)

        if not page_tokens:
            continue

        # Keep the token stream and page mapping exactly aligned.
        #whitespace separators are deliberately not added as tokens,
        #because tokenize_text() does not treat whitespace as tokens.
        all_tokens.extend(page_tokens)
        token_page_numbers.extend(
            [page.page_number] * len(page_tokens)
        )

    return all_tokens, token_page_numbers


def get_pages_for_token_range(
    token_page_numbers: list[int],
    start: int,
    end: int,
) -> list[int]:
    """Return ordered, unique page numbers for a token range."""

    seen: set[int] = set()
    page_numbers: list[int] = []

    for page_number in token_page_numbers[start:end]:
        if page_number not in seen:
            seen.add(page_number)
            page_numbers.append(page_number)

    return page_numbers


# ---------------------------------------------------------------------
# PARENT CHUNK CREATION
# ---------------------------------------------------------------------

def create_parent_chunks(
    pages: list[IngestedPage],
    chunk_size: int = PARENT_CHUNK_SIZE,
    overlap: int = PARENT_CHUNK_OVERLAP,
) -> list[ParentChunk]:
    """
    Create parent chunks for one document.

    Parent chunks are larger context windows. They are not the primary
    vector-search units; child chunks are intended for that purpose.
    """

    if not pages:
        return []

    document_ids = {page.document_id for page in pages}

    if len(document_ids) != 1:
        raise ValueError(
            "create_parent_chunks() expects pages from exactly "
            "one document."
        )

    ordered_pages = sorted(
        pages,
        key=lambda page: page.page_number,
    )

    first_page = ordered_pages[0]

    tokens, token_page_numbers = build_document_token_stream(
        ordered_pages
    )

    windows = split_token_windows(
        tokens,
        chunk_size=chunk_size,
        overlap=overlap,
    )

    parent_chunks: list[ParentChunk] = []

    for chunk_index, (start, end) in enumerate(windows):
        chunk_tokens = tokens[start:end]
        chunk_text = detokenize(chunk_tokens)

        if not chunk_text.strip():
            continue

        page_numbers = get_pages_for_token_range(
            token_page_numbers,
            start,
            end,
        )

        parent_id = (
            f"{first_page.document_id}__parent_{chunk_index:05d}"
        )

        metadata: dict[str, Any] = {
            "document_id": first_page.document_id,
            "source_file": first_page.source_file,
            "source_path": first_page.source_path,
            "document_title": first_page.document_title,
            "category": first_page.category,
            "chunk_type": "parent",
            "chunk_index": chunk_index,
            "page_numbers": page_numbers,
            "start_token": start,
            "end_token": end,
            "tokenizer": "regex_word_approximation",
        }

        parent_chunks.append(
            ParentChunk(
                parent_id=parent_id,
                document_id=first_page.document_id,
                source_file=first_page.source_file,
                source_path=first_page.source_path,
                document_title=first_page.document_title,
                category=first_page.category,
                chunk_index=chunk_index,
                text=chunk_text,
                token_count=len(chunk_tokens),
                page_numbers=page_numbers,
                token_page_numbers=token_page_numbers[start:end],
                metadata=metadata,
            )
        )

    return parent_chunks


# ---------------------------------------------------------------------
# CHILD CHUNK CREATION
# ---------------------------------------------------------------------

def create_child_chunks(
    parent_chunks: list[ParentChunk],
    chunk_size: int = CHILD_CHUNK_SIZE,
    overlap: int = CHILD_CHUNK_OVERLAP,
) -> list[ChildChunk]:
    """
    Create child chunks inside each parent chunk.

    Child chunks are smaller and suitable for embedding and semantic
    retrieval.

    Each child stores its parent ID so retrieval can later expand the
    selected child into the larger parent context.
    """

    child_chunks: list[ChildChunk] = []

    for parent in parent_chunks:
        tokens = tokenize_text(parent.text)

        windows = split_token_windows(
            tokens,
            chunk_size=chunk_size,
            overlap=overlap,
        )

        for child_index, (start, end) in enumerate(windows):
            child_tokens = tokens[start:end]
            child_text = detokenize(child_tokens)

            if not child_text.strip():
                continue

            # Map this child's token range to its actual source pages.
            # Parent token-page metadata should align with the parent's
            # token stream. Validate the mapping before using it.
            if len(parent.token_page_numbers) != len(tokens):
                raise ValueError(
                    f"Page mapping mismatch for parent "
                    f"{parent.parent_id}: "
                    f"{len(parent.token_page_numbers)} page mappings "
                    f"for {len(tokens)} text tokens."
                )

            child_page_numbers = get_pages_for_token_range(
                parent.token_page_numbers,
                start,
                end,
            )

            child_id = (
                f"{parent.parent_id}__child_{child_index:05d}"
            )

            metadata: dict[str, Any] = {
                "document_id": parent.document_id,
                "source_file": parent.source_file,
                "source_path": parent.source_path,
                "document_title": parent.document_title,
                "category": parent.category,
                "chunk_type": "child",
                "parent_id": parent.parent_id,
                "child_index": child_index,
                "page_numbers": child_page_numbers,
                "start_token": start,
                "end_token": end,
                "tokenizer": "regex_word_approximation",
            }

            child_chunks.append(
                ChildChunk(
                    child_id=child_id,
                    parent_id=parent.parent_id,
                    document_id=parent.document_id,
                    source_file=parent.source_file,
                    source_path=parent.source_path,
                    document_title=parent.document_title,
                    category=parent.category,
                    chunk_index=child_index,
                    text=child_text,
                    token_count=len(child_tokens),
                    page_numbers=child_page_numbers,
                    metadata=metadata,
                )
            )

    return child_chunks


# ---------------------------------------------------------------------
# FULL CHUNKING PIPELINE
# ---------------------------------------------------------------------

def chunk_documents(
    pages: list[IngestedPage],
    parent_chunk_size: int = PARENT_CHUNK_SIZE,
    parent_overlap: int = PARENT_CHUNK_OVERLAP,
    child_chunk_size: int = CHILD_CHUNK_SIZE,
    child_overlap: int = CHILD_CHUNK_OVERLAP,
) -> tuple[list[ParentChunk], list[ChildChunk], ChunkingReport]:
    """
    Convert ingested PDF pages into parent and child chunks.

    Returns:
        parent_chunks, child_chunks, report
    """

    if parent_chunk_size <= 0 or child_chunk_size <= 0:
        raise ValueError("Chunk sizes must be greater than zero.")

    if parent_overlap < 0 or child_overlap < 0:
        raise ValueError("Chunk overlaps cannot be negative.")

    if parent_overlap >= parent_chunk_size:
        raise ValueError(
            "Parent overlap must be smaller than parent chunk size."
        )

    if child_overlap >= child_chunk_size:
        raise ValueError(
            "Child overlap must be smaller than child chunk size."
        )

    if child_chunk_size > parent_chunk_size:
        logger.warning(
            "Child chunk size exceeds parent chunk size. "
            "Consider reducing child_chunk_size."
        )

    grouped = group_pages_by_document(pages)

    all_parents: list[ParentChunk] = []
    all_children: list[ChildChunk] = []
    errors: list[dict[str, str]] = []

    for document_id, document_pages in grouped.items():
        try:
            parents = create_parent_chunks(
                document_pages,
                chunk_size=parent_chunk_size,
                overlap=parent_overlap,
            )

            children = create_child_chunks(
                parents,
                chunk_size=child_chunk_size,
                overlap=child_overlap,
            )

            all_parents.extend(parents)
            all_children.extend(children)

            logger.info(
                "Chunked %s: %s parents, %s children",
                document_id,
                len(parents),
                len(children),
            )

        except Exception as exc:
            logger.exception(
                "Chunking failed for document %s",
                document_id,
            )

            errors.append(
                {
                    "document_id": document_id,
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                }
            )

    report = ChunkingReport(
        input_pages=len(pages),
        documents_processed=len(grouped) - len(errors),
        parent_chunks_created=len(all_parents),
        child_chunks_created=len(all_children),
        parent_chunk_size=parent_chunk_size,
        parent_chunk_overlap=parent_overlap,
        child_chunk_size=child_chunk_size,
        child_chunk_overlap=child_overlap,
        tokenizer="regex_word_approximation",
        errors=errors,
    )

    logger.info(
        "Chunking complete: %s parents, %s children, %s errors",
        report.parent_chunks_created,
        report.child_chunks_created,
        len(report.errors),
    )

    return all_parents, all_children, report


# ---------------------------------------------------------------------
# JSON EXPORT
# ---------------------------------------------------------------------

def save_chunks_json(
    parent_chunks: list[ParentChunk],
    child_chunks: list[ChildChunk],
    report: ChunkingReport,
    output_path: Optional[Path] = None,
) -> Path:
    """
    Save parent chunks, child chunks, and a report as JSON.

    This is intended for development and inspection. For larger
    knowledge bases, use a persistent vector store and a separate
    parent-document store rather than relying on one large JSON file.
    """

    destination = (
        Path(output_path)
        if output_path
        else DEFAULT_OUTPUT_FILE
    ).resolve()

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    payload = {
        "report": asdict(report),
        "parents": [
            asdict(parent)
            for parent in parent_chunks
        ],
        "children": [
            asdict(child)
            for child in child_chunks
        ],
    }

    with destination.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            payload,
            file,
            ensure_ascii=False,
            indent=2,
        )

    logger.info("Saved chunks to %s", destination)

    return destination


# ---------------------------------------------------------------------
# COMMAND-LINE ENTRY POINT
# ---------------------------------------------------------------------

def main() -> None:
    """
    Run ingestion and chunking together.

    Command:
        python -m app.rag.chunking

    This does not create embeddings or call the Groq API.
    """

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    try:
        print("Step 1/2: Extracting PDF pages...")

        pages, ingestion_report = ingest_documents(
            DOCUMENTS_DIR
        )

        print(
            f"Extracted text from {len(pages)} pages "
            f"across {ingestion_report.successful_pdfs} PDFs."
        )

        print("\nStep 2/2: Creating parent-child chunks...")

        parents, children, report = chunk_documents(pages)

        output_path = save_chunks_json(
            parents,
            children,
            report,
        )

    except Exception as exc:
        logger.exception("Chunking pipeline failed: %s", exc)
        raise SystemExit(1) from exc

    print("\n" + "=" * 65)
    print("DR AI PLATFORM — CHUNKING REPORT")
    print("=" * 65)

    print(f"Input pages:          {report.input_pages}")
    print(f"Documents processed:  {report.documents_processed}")
    print(f"Parent chunks:        {report.parent_chunks_created}")
    print(f"Child chunks:         {report.child_chunks_created}")

    print("\nConfiguration:")
    print(f"Parent size:          {report.parent_chunk_size}")
    print(f"Parent overlap:       {report.parent_chunk_overlap}")
    print(f"Child size:           {report.child_chunk_size}")
    print(f"Child overlap:        {report.child_chunk_overlap}")
    print(f"Tokenizer:            {report.tokenizer}")

    if report.errors:
        print("\nChunking errors:")

        for error in report.errors:
            print(
                f"- {error['document_id']}: "
                f"{error['error_type']}: {error['message']}"
            )

    print(f"\nJSON output:\n{output_path}")

    if parents:
        parent = parents[0]

        print("\nPARENT CHUNK SAMPLE")
        print("-" * 65)
        print(f"ID:       {parent.parent_id}")
        print(f"Document: {parent.source_file}")
        print(f"Pages:    {parent.page_numbers}")
        print(f"Tokens:   {parent.token_count}")
        print(parent.text[:700])

    if children:
        child = children[0]

        print("\nCHILD CHUNK SAMPLE")
        print("-" * 65)
        print(f"ID:       {child.child_id}")
        print(f"Parent:   {child.parent_id}")
        print(f"Document: {child.source_file}")
        print(f"Pages:    {child.page_numbers}")
        print(f"Tokens:   {child.token_count}")
        print(child.text[:500])

    print("\nChunking finished.")


if __name__ == "__main__":
    main()