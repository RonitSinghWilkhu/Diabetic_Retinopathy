"""
PDF ingestion module for the DR AI Platform RAG system.

Responsibilities:
- Discover PDF documents recursively.
- Extract text page by page using pypdf.
- Preserve source metadata and page numbers.
- Identify empty pages and unreadable PDFs.
- Return structured records for downstream chunking, embeddings,
  retrieval, reranking, and source citations.

Expected document directory:
    <project_root>/data/rag_documents/

Install:
    python -m pip install pypdf
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from pypdf import PdfReader


# ---------------------------------------------------------------------
# PATHS AND CONFIGURATION
# ---------------------------------------------------------------------

# ingestion.py is located at:
# <project_root>/backend/app/rag/ingestion.py
PROJECT_ROOT = Path(__file__).resolve().parents[3]

DOCUMENTS_DIR = PROJECT_ROOT / "data" / "rag_documents"

# Optional output for a JSON snapshot of extracted pages.
# The file is generated only when save_json() is explicitly called.
DEFAULT_OUTPUT_FILE = (
    PROJECT_ROOT / "data" / "rag_documents" / "ingestion_output.json"
)

SUPPORTED_EXTENSION = ".pdf"

# Logging configuration.
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------
# DATA STRUCTURES
# ---------------------------------------------------------------------

@dataclass
class IngestedPage:
    """
    Represents one extracted PDF page.

    Page numbers are 1-based, matching the page order in the PDF.
    """

    document_id: str
    source_file: str
    source_path: str
    document_title: str
    category: str
    page_number: int
    text: str
    character_count: int
    ingested_at: str
    metadata: dict[str, Any]


@dataclass
class IngestionReport:
    """Summary of a complete ingestion run."""

    documents_directory: str
    total_pdfs: int
    successful_pdfs: int
    failed_pdfs: int
    total_pages: int
    pages_with_text: int
    empty_pages: int
    total_characters: int
    errors: list[dict[str, str]]


# ---------------------------------------------------------------------
# TEXT CLEANING
# ---------------------------------------------------------------------

def clean_extracted_text(text: Optional[str]) -> str:
    """
    Clean common PDF extraction artifacts while preserving paragraph
    boundaries and meaningful textual content.

    This function does not summarize, paraphrase, or rewrite the source.
    """

    if not text:
        return ""

    # Normalize Windows and legacy line endings.
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Replace non-breaking spaces.
    text = text.replace("\u00a0", " ")

    # Remove null characters.
    text = text.replace("\x00", "")

    # Remove soft hyphens inserted by some PDF encoders.
    text = text.replace("\u00ad", "")

    # Normalize horizontal whitespace without flattening paragraphs.
    text = re.sub(r"[ \t]+", " ", text)

    # Remove spaces around line breaks.
    text = re.sub(r" *\n *", "\n", text)

    # Collapse excessive blank lines while retaining paragraph breaks.
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


# ---------------------------------------------------------------------
# DOCUMENT IDENTIFICATION AND METADATA
# ---------------------------------------------------------------------

def get_category(pdf_path: Path) -> str:
    """
    Use the immediate parent folder as the document category.

    Examples:
        ai_research/example.pdf -> ai_research
        guidelines/example.pdf  -> guidelines
    """

    try:
        relative_path = pdf_path.relative_to(DOCUMENTS_DIR)
        parent_name = relative_path.parent.as_posix()

        if parent_name in ("", "."):
            return "uncategorized"

        # Preserve nested categories if a document is in a nested folder.
        return parent_name

    except ValueError:
        return "uncategorized"


def get_document_id(pdf_path: Path) -> str:
    """
    Generate a stable ID based on the PDF's path relative to the
    knowledge-base root.

    This remains stable between ingestion runs as long as the path
    does not change.
    """

    relative_path = pdf_path.relative_to(DOCUMENTS_DIR).as_posix()

    # Keep IDs deterministic and readable.
    normalized = relative_path.lower()
    normalized = re.sub(r"[^a-z0-9._/-]+", "_", normalized)
    normalized = normalized.replace("/", "__")

    return normalized


def get_pdf_metadata(reader: PdfReader) -> dict[str, Any]:
    """Extract available PDF metadata without assuming it exists."""

    raw_metadata = reader.metadata
    metadata: dict[str, Any] = {}

    if not raw_metadata:
        return metadata

    metadata_mapping = {
        "/Title": "pdf_title",
        "/Author": "pdf_author",
        "/Subject": "pdf_subject",
        "/Creator": "pdf_creator",
        "/Producer": "pdf_producer",
        "/CreationDate": "pdf_creation_date",
        "/ModDate": "pdf_modification_date",
    }

    for pdf_key, output_key in metadata_mapping.items():
        value = raw_metadata.get(pdf_key)

        if value is not None:
            metadata[output_key] = str(value).strip()

    return metadata


def choose_document_title(
    pdf_path: Path,
    pdf_metadata: dict[str, Any],
) -> str:
    """
    Prefer the title embedded in the PDF metadata.

    Fall back to the filename if no useful title is available.
    """

    title = pdf_metadata.get("pdf_title", "")

    if title:
        title = str(title).strip()

        # Some PDFs contain placeholder metadata rather than a real title.
        if title and title.lower() not in {
            "untitled",
            "none",
            "unknown",
        }:
            return title

    return pdf_path.stem.replace("_", " ").strip()


# ---------------------------------------------------------------------
# PDF DISCOVERY
# ---------------------------------------------------------------------

def discover_pdfs(
    documents_dir: Optional[Path] = None,
) -> list[Path]:
    """
    Find PDFs recursively under the configured document directory.

    Hidden files and the generated ingestion_output.json are not PDFs
    and are therefore excluded automatically.
    """

    root = Path(documents_dir) if documents_dir else DOCUMENTS_DIR

    if not root.exists():
        raise FileNotFoundError(
            f"Knowledge-base directory does not exist: {root}\n"
            "Create data/rag_documents/ and place the PDFs inside it."
        )

    if not root.is_dir():
        raise NotADirectoryError(
            f"Expected a directory but found: {root}"
        )

    pdf_files = [
        path
        for path in root.rglob("*")
        if path.is_file()
        and path.suffix.lower() == SUPPORTED_EXTENSION
        and not any(part.startswith(".") for part in path.parts)
    ]

    return sorted(pdf_files, key=lambda path: str(path).lower())


# ---------------------------------------------------------------------
# SINGLE-PDF INGESTION
# ---------------------------------------------------------------------

def ingest_pdf(pdf_path: Path) -> list[IngestedPage]:
    """
    Extract all text-bearing pages from one PDF.

    Returns:
        A list containing one IngestedPage record per page with
        non-empty extracted text.

    Raises:
        Exception if the PDF cannot be opened or processed.

    Page numbers remain the original 1-based PDF page numbers. Empty
    pages are skipped in the returned records, but their page numbers
    are not renumbered.
    """

    pdf_path = Path(pdf_path).resolve()

    if not pdf_path.is_file():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    if pdf_path.suffix.lower() != SUPPORTED_EXTENSION:
        raise ValueError(f"Not a PDF file: {pdf_path}")

    try:
        pdf_path.relative_to(DOCUMENTS_DIR.resolve())
    except ValueError:
        raise ValueError(
            f"PDF is outside the configured knowledge-base directory: "
            f"{pdf_path}"
        )

    logger.info("Reading PDF: %s", pdf_path.name)

    # strict=False tolerates some non-critical structural issues.
    # It does not guarantee that every malformed PDF can be recovered.
    reader = PdfReader(str(pdf_path), strict=False)

    if reader.is_encrypted:
        try:
            result = reader.decrypt("")
        except Exception as exc:
            raise ValueError(
                f"Encrypted PDF cannot be opened: {pdf_path.name}"
            ) from exc

        if result == 0:
            raise ValueError(
                f"PDF requires a password: {pdf_path.name}"
            )

    pdf_metadata = get_pdf_metadata(reader)

    document_title = choose_document_title(
        pdf_path,
        pdf_metadata,
    )

    category = get_category(pdf_path)
    document_id = get_document_id(pdf_path)

    relative_path = pdf_path.relative_to(
        DOCUMENTS_DIR.resolve()
    ).as_posix()

    ingested_at = datetime.now(timezone.utc).isoformat()

    extracted_pages: list[IngestedPage] = []

    for page_index, page in enumerate(reader.pages):
        page_number = page_index + 1

        try:
            raw_text = page.extract_text()
            text = clean_extracted_text(raw_text)

        except Exception as exc:
            # Keep processing the rest of the document.
            logger.warning(
                "Could not extract page %s of %s: %s",
                page_number,
                pdf_path.name,
                exc,
            )
            continue

        if not text:
            logger.debug(
                "Skipping empty page %s of %s",
                page_number,
                pdf_path.name,
            )
            continue

        page_metadata: dict[str, Any] = {
            "document_id": document_id,
            "source_file": pdf_path.name,
            "source_path": relative_path,
            "document_title": document_title,
            "category": category,
            "page_number": page_number,
            "total_pdf_pages": len(reader.pages),
            "file_type": "pdf",
        }

        # Include the available PDF metadata.
        page_metadata.update(pdf_metadata)

        record = IngestedPage(
            document_id=document_id,
            source_file=pdf_path.name,
            source_path=relative_path,
            document_title=document_title,
            category=category,
            page_number=page_number,
            text=text,
            character_count=len(text),
            ingested_at=ingested_at,
            metadata=page_metadata,
        )

        extracted_pages.append(record)

    logger.info(
        "Extracted %s text-bearing pages from %s",
        len(extracted_pages),
        pdf_path.name,
    )

    return extracted_pages


# ---------------------------------------------------------------------
# FULL KNOWLEDGE-BASE INGESTION
# ---------------------------------------------------------------------

def ingest_documents(
    documents_dir: Optional[Path] = None,
) -> tuple[list[IngestedPage], IngestionReport]:
    """
    Ingest all PDFs in the knowledge-base directory.

    A failure in one PDF does not stop ingestion of other PDFs.

    Returns:
        (
            list_of_extracted_page_records,
            ingestion_report
        )
    """

    root = Path(documents_dir) if documents_dir else DOCUMENTS_DIR
    root = root.resolve()

    pdf_files = discover_pdfs(root)

    if not pdf_files:
        raise FileNotFoundError(
            f"No PDF files found in: {root}\n"
            "Add PDFs to data/rag_documents/ or its subfolders."
        )

    logger.info("Found %s PDF files.", len(pdf_files))

    all_pages: list[IngestedPage] = []
    errors: list[dict[str, str]] = []

    successful_pdfs = 0
    failed_pdfs = 0
    total_pages = 0
    empty_pages = 0

    for pdf_path in pdf_files:
        try:
            # Count every physical page, including empty pages.
            reader = PdfReader(str(pdf_path), strict=False)
            document_page_count = len(reader.pages)
            total_pages += document_page_count

            extracted_pages = ingest_pdf(pdf_path)

            all_pages.extend(extracted_pages)
            successful_pdfs += 1

            extracted_count = len(extracted_pages)
            empty_pages += max(
                0,
                document_page_count - extracted_count,
            )

            logger.info(
                "[OK] %s: %s/%s pages contain extracted text",
                pdf_path.name,
                extracted_count,
                document_page_count,
            )

        except Exception as exc:
            failed_pdfs += 1

            error_record = {
                "file": str(pdf_path.relative_to(root).as_posix()),
                "error_type": type(exc).__name__,
                "message": str(exc),
            }

            errors.append(error_record)

            logger.exception(
                "[FAILED] Could not ingest %s",
                pdf_path.name,
            )

    total_characters = sum(
        page.character_count for page in all_pages
    )

    report = IngestionReport(
        documents_directory=str(root),
        total_pdfs=len(pdf_files),
        successful_pdfs=successful_pdfs,
        failed_pdfs=failed_pdfs,
        total_pages=total_pages,
        pages_with_text=len(all_pages),
        empty_pages=empty_pages,
        total_characters=total_characters,
        errors=errors,
    )

    logger.info(
        "Ingestion complete: %s PDFs, %s successful, %s failed, "
        "%s extracted pages, %s characters.",
        report.total_pdfs,
        report.successful_pdfs,
        report.failed_pdfs,
        report.pages_with_text,
        report.total_characters,
    )

    return all_pages, report


# ---------------------------------------------------------------------
# OPTIONAL JSON EXPORT
# ---------------------------------------------------------------------

def save_json(
    pages: list[IngestedPage],
    report: IngestionReport,
    output_path: Optional[Path] = None,
) -> Path:
    """
    Save extracted pages and the ingestion report to JSON.

    This is useful for debugging and inspecting source metadata.
    The output JSON is not intended to replace the original PDFs.
    """

    destination = (
        Path(output_path)
        if output_path
        else DEFAULT_OUTPUT_FILE
    )

    destination = destination.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "report": asdict(report),
        "pages": [asdict(page) for page in pages],
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

    logger.info("Saved ingestion output to %s", destination)

    return destination


# ---------------------------------------------------------------------
# SIMPLE COMMAND-LINE ENTRY POINT
# ---------------------------------------------------------------------

def main() -> None:
    """
    Run this module directly to ingest the knowledge base,
    print a report, and save the extracted text to JSON.
    """

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    try:
        pages, report = ingest_documents()

    except Exception as exc:
        logger.error("Ingestion could not start: %s", exc)
        raise SystemExit(1) from exc

    print("\n" + "=" * 65)
    print("DR AI PLATFORM — PDF INGESTION REPORT")
    print("=" * 65)

    print(f"Knowledge base:       {report.documents_directory}")
    print(f"PDF files discovered: {report.total_pdfs}")
    print(f"Successful PDFs:      {report.successful_pdfs}")
    print(f"Failed PDFs:          {report.failed_pdfs}")
    print(f"Total PDF pages:      {report.total_pages}")
    print(f"Pages with text:      {report.pages_with_text}")
    print(f"Empty/unextracted:    {report.empty_pages}")
    print(f"Extracted characters: {report.total_characters:,}")

    if report.errors:
        print("\nERRORS:")

        for error in report.errors:
            print(
                f"- {error['file']}: "
                f"{error['error_type']}: {error['message']}"
            )

    output_path = save_json(pages, report)

    print(f"\nJSON output saved to:\n{output_path}")

    if pages:
        first_page = pages[0]

        print("\nFIRST EXTRACTED PAGE — SAMPLE")
        print("-" * 65)
        print(f"Document: {first_page.document_title}")
        print(f"File:     {first_page.source_file}")
        print(f"Category: {first_page.category}")
        print(f"Page:     {first_page.page_number}")
        print("\nText preview:")
        print(first_page.text[:1000])

    print("\nIngestion finished.")


if __name__ == "__main__":
    main()