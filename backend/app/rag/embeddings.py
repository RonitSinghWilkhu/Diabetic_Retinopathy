"""
Embedding generation for the DR AI Platform RAG system.

Pipeline:
    PDF documents
        -> ingestion.py
        -> chunking.py
        -> embeddings.py
        -> retrieval.py
        -> reranking
        -> Groq generation

Features:
- Generates embeddings for child chunks.
- Uses a local Sentence Transformers model.
- Normalizes vectors for cosine-similarity retrieval.
- Preserves child IDs, parent IDs, source metadata, and page numbers.
- Saves embeddings and their metadata locally.
- Supports batching to reduce memory usage.
- Provides a command-line entry point for testing.

Install:
    python -m pip install sentence-transformers numpy

Important:
The embedding model is used for text retrieval. It is separate from
the ConvNeXt-Tiny diabetic retinopathy image classification model.
"""

from __future__ import annotations

import json
import logging

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np
from sentence_transformers import SentenceTransformer

from app.rag.chunking import (
    ChildChunk,
    chunk_documents,
)
from app.rag.ingestion import ingest_documents


# ---------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[3]

DOCUMENTS_DIR = PROJECT_ROOT / "data" / "rag_documents"

# Local output directory for the embedding index.
EMBEDDINGS_DIR = PROJECT_ROOT / "artifacts" / "rag"

# Embedding output files.
DEFAULT_EMBEDDINGS_FILE = EMBEDDINGS_DIR / "child_embeddings.npy"
DEFAULT_INDEX_FILE = EMBEDDINGS_DIR / "embedding_index.json"

# Sentence Transformers model.
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Model output dimension for all-MiniLM-L6-v2.
EXPECTED_EMBEDDING_DIMENSION = 384

# Process texts in batches to manage memory.
DEFAULT_BATCH_SIZE = 32

# Normalize embeddings so their dot product equals cosine similarity.
NORMALIZE_EMBEDDINGS = True


# ---------------------------------------------------------------------
# EMBEDDING MODEL
# ---------------------------------------------------------------------

_model: Optional[SentenceTransformer] = None


def get_embedding_model() -> SentenceTransformer:
    """
    Load the embedding model once and reuse it.

    The model is downloaded on the first run if it is not already
    available in the local Hugging Face model cache.
    """

    global _model

    if _model is None:
        logger.info(
            "Loading embedding model: %s",
            EMBEDDING_MODEL_NAME,
        )

        _model = SentenceTransformer(
            EMBEDDING_MODEL_NAME,
        )

        actual_dimension = _model.get_sentence_embedding_dimension()

        if actual_dimension != EXPECTED_EMBEDDING_DIMENSION:
            raise RuntimeError(
                "Unexpected embedding dimension. "
                f"Expected {EXPECTED_EMBEDDING_DIMENSION}, "
                f"received {actual_dimension}."
            )

        logger.info(
            "Embedding model loaded. Dimension: %s",
            actual_dimension,
        )

    return _model


# ---------------------------------------------------------------------
# EMBEDDING GENERATION
# ---------------------------------------------------------------------

def generate_embeddings(
    texts: list[str],
    batch_size: int = DEFAULT_BATCH_SIZE,
    show_progress: bool = True,
) -> np.ndarray:
    """
    Convert text strings into normalized dense vectors.

    Args:
        texts:
            Text strings to embed.

        batch_size:
            Number of texts processed in one batch.

        show_progress:
            Display a progress bar during encoding.

    Returns:
        A float32 NumPy array with shape:

            (number_of_texts, 384)

    Empty input returns an empty array with shape (0, 384).
    """

    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero.")

    if not texts:
        return np.empty(
            (0, EXPECTED_EMBEDDING_DIMENSION),
            dtype=np.float32,
        )

    if any(not isinstance(text, str) for text in texts):
        raise TypeError("All embedding inputs must be strings.")

    if any(not text.strip() for text in texts):
        raise ValueError(
            "Embedding input contains empty text. "
            "Remove empty chunks before generating embeddings."
        )

    model = get_embedding_model()

    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        convert_to_numpy=True,
        normalize_embeddings=NORMALIZE_EMBEDDINGS,
    )

    embeddings = np.asarray(
        embeddings,
        dtype=np.float32,
    )

    expected_shape = (
        len(texts),
        EXPECTED_EMBEDDING_DIMENSION,
    )

    if embeddings.shape != expected_shape:
        raise RuntimeError(
            f"Unexpected embedding shape: {embeddings.shape}. "
            f"Expected: {expected_shape}."
        )

    if not np.isfinite(embeddings).all():
        raise RuntimeError(
            "Embedding output contains NaN or infinite values."
        )

    return embeddings


# ---------------------------------------------------------------------
# METADATA SERIALIZATION
# ---------------------------------------------------------------------

def child_to_metadata(child: ChildChunk) -> dict[str, Any]:
    """
    Convert a child chunk into a JSON-serializable metadata record.

    The text is retained so retrieval can return the matched passage
    without having to recreate the chunk.
    """

    return {
        "child_id": child.child_id,
        "parent_id": child.parent_id,
        "document_id": child.document_id,
        "source_file": child.source_file,
        "source_path": child.source_path,
        "document_title": child.document_title,
        "category": child.category,
        "chunk_index": child.chunk_index,
        "text": child.text,
        "token_count": child.token_count,
        "page_numbers": child.page_numbers,
        "metadata": child.metadata,
    }


# ---------------------------------------------------------------------
# SAVE EMBEDDINGS AND INDEX
# ---------------------------------------------------------------------

def save_embedding_index(
    children: list[ChildChunk],
    embeddings: np.ndarray,
    embeddings_path: Optional[Path] = None,
    index_path: Optional[Path] = None,
) -> tuple[Path, Path]:
    """
    Save the dense embedding matrix and the corresponding metadata.

    IMPORTANT:
    The row order of the NumPy array matches the order of records in
    embedding_index.json.

    Do not independently sort either file after saving.
    """

    if len(children) != len(embeddings):
        raise ValueError(
            "The number of child chunks must equal the number "
            "of embedding vectors."
        )

    if embeddings.ndim != 2:
        raise ValueError(
            "Embeddings must be a two-dimensional matrix."
        )

    if embeddings.shape[1] != EXPECTED_EMBEDDING_DIMENSION:
        raise ValueError(
            f"Expected embedding dimension "
            f"{EXPECTED_EMBEDDING_DIMENSION}, "
            f"received {embeddings.shape[1]}."
        )

    if not np.isfinite(embeddings).all():
        raise ValueError(
            "Cannot save embeddings containing NaN or infinity."
        )

    output_embeddings = (
        Path(embeddings_path)
        if embeddings_path
        else DEFAULT_EMBEDDINGS_FILE
    ).resolve()

    output_index = (
        Path(index_path)
        if index_path
        else DEFAULT_INDEX_FILE
    ).resolve()

    output_embeddings.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_index.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Save the vectors in NumPy's binary format.
    np.save(
        str(output_embeddings),
        embeddings.astype(np.float32),
        allow_pickle=False,
    )

    # Save the metadata in the matching row order.
    index_data = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "embedding_model": EMBEDDING_MODEL_NAME,
        "embedding_dimension": EXPECTED_EMBEDDING_DIMENSION,
        "normalized_embeddings": NORMALIZE_EMBEDDINGS,
        "similarity_metric": "cosine",
        "number_of_embeddings": len(children),
        "embedding_file": output_embeddings.name,
        "records": [
            child_to_metadata(child)
            for child in children
        ],
    }

    with output_index.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            index_data,
            file,
            ensure_ascii=False,
            indent=2,
        )

    logger.info(
        "Saved %s embeddings to %s",
        len(children),
        output_embeddings,
    )

    logger.info(
        "Saved embedding metadata to %s",
        output_index,
    )

    return output_embeddings, output_index


# ---------------------------------------------------------------------
# LOAD EXISTING EMBEDDING INDEX
# ---------------------------------------------------------------------

def load_embedding_index(
    embeddings_path: Optional[Path] = None,
    index_path: Optional[Path] = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Load the saved embedding matrix and metadata.

    Checks that the matrix and metadata have matching row counts.
    """

    vector_file = (
        Path(embeddings_path)
        if embeddings_path
        else DEFAULT_EMBEDDINGS_FILE
    ).resolve()

    metadata_file = (
        Path(index_path)
        if index_path
        else DEFAULT_INDEX_FILE
    ).resolve()

    if not vector_file.is_file():
        raise FileNotFoundError(
            f"Embedding file not found: {vector_file}"
        )

    if not metadata_file.is_file():
        raise FileNotFoundError(
            f"Embedding metadata not found: {metadata_file}"
        )

    embeddings = np.load(
        str(vector_file),
        allow_pickle=False,
    )

    with metadata_file.open(
        "r",
        encoding="utf-8",
    ) as file:
        index_data = json.load(file)

    if embeddings.ndim != 2:
        raise ValueError(
            "Loaded embeddings must be a two-dimensional matrix."
        )

    if embeddings.shape[1] != EXPECTED_EMBEDDING_DIMENSION:
        raise ValueError(
            f"Unexpected embedding dimension: {embeddings.shape[1]}"
        )

    if len(index_data.get("records", [])) != len(embeddings):
        raise ValueError(
            "Embedding count does not match metadata record count."
        )

    if not np.isfinite(embeddings).all():
        raise ValueError(
            "Loaded embeddings contain NaN or infinity."
        )

    return embeddings, index_data


# ---------------------------------------------------------------------
# BUILD THE COMPLETE EMBEDDING INDEX
# ---------------------------------------------------------------------

def build_embedding_index(
    batch_size: int = DEFAULT_BATCH_SIZE,
    embeddings_path: Optional[Path] = None,
    index_path: Optional[Path] = None,
) -> dict[str, Any]:
    """
    Run ingestion, parent-child chunking, embedding generation,
    and local index persistence.

    Returns a summary dictionary.

    This function rebuilds the index from the current PDFs every time
    it is called. It does not incrementally update an existing index.
    """

    logger.info("Starting RAG embedding-index build.")

    # Step 1: extract PDF text.
    logger.info("Step 1/4: Ingesting PDF documents.")

    pages, ingestion_report = ingest_documents(
        DOCUMENTS_DIR,
    )

    if not pages:
        raise RuntimeError(
            "Ingestion returned no text-bearing pages. "
            "Check your PDFs and ingestion report."
        )

    logger.info(
        "Ingested %s text-bearing pages from %s PDFs.",
        len(pages),
        ingestion_report.successful_pdfs,
    )

    if ingestion_report.failed_pdfs:
        logger.warning(
            "%s PDFs could not be ingested.",
            ingestion_report.failed_pdfs,
        )

    # Step 2: create parent and child chunks.
    logger.info("Step 2/4: Creating parent-child chunks.")

    parents, children, chunking_report = chunk_documents(
        pages,
    )

    if not children:
        raise RuntimeError(
            "Chunking returned no child chunks. "
            "Check the extracted text and chunking configuration."
        )

    logger.info(
        "Created %s parent chunks and %s child chunks.",
        len(parents),
        len(children),
    )

    if chunking_report.errors:
        logger.warning(
            "%s documents had chunking errors.",
            len(chunking_report.errors),
        )

    # Step 3: embed the child text.
    logger.info("Step 3/4: Generating child embeddings.")

    texts = [
        child.text
        for child in children
    ]

    embeddings = generate_embeddings(
        texts,
        batch_size=batch_size,
        show_progress=True,
    )

    # Step 4: persist vectors and metadata.
    logger.info("Step 4/4: Saving embedding index.")

    saved_embeddings_path, saved_index_path = (
        save_embedding_index(
            children=children,
            embeddings=embeddings,
            embeddings_path=embeddings_path,
            index_path=index_path,
        )
    )

    summary = {
        "status": "success",
        "documents_directory": str(DOCUMENTS_DIR),
        "pdfs_discovered": ingestion_report.total_pdfs,
        "pdfs_successfully_ingested": (
            ingestion_report.successful_pdfs
        ),
        "pdfs_failed": ingestion_report.failed_pdfs,
        "text_bearing_pages": len(pages),
        "parent_chunks": len(parents),
        "child_chunks": len(children),
        "embedding_count": len(embeddings),
        "embedding_dimension": int(embeddings.shape[1]),
        "embedding_model": EMBEDDING_MODEL_NAME,
        "embeddings_path": str(saved_embeddings_path),
        "metadata_path": str(saved_index_path),
        "ingestion_errors": ingestion_report.errors,
        "chunking_errors": chunking_report.errors,
    }

    logger.info("Embedding index build completed successfully.")

    return summary


# ---------------------------------------------------------------------
# COMMAND-LINE ENTRY POINT
# ---------------------------------------------------------------------

def main() -> None:
    """Build and verify the local embedding index."""

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    try:
        summary = build_embedding_index()

    except Exception as exc:
        logger.exception(
            "Embedding-index build failed: %s",
            exc,
        )

        raise SystemExit(1) from exc

    print("\n" + "=" * 65)
    print("DR AI PLATFORM — EMBEDDING INDEX REPORT")
    print("=" * 65)

    print(f"Status:               {summary['status']}")
    print(f"PDFs discovered:      {summary['pdfs_discovered']}")
    print(
        "PDFs ingested:        "
        f"{summary['pdfs_successfully_ingested']}"
    )
    print(f"PDFs failed:          {summary['pdfs_failed']}")
    print(f"Text-bearing pages:   {summary['text_bearing_pages']}")
    print(f"Parent chunks:        {summary['parent_chunks']}")
    print(f"Child chunks:         {summary['child_chunks']}")
    print(f"Embeddings:           {summary['embedding_count']}")
    print(
        "Vector dimension:     "
        f"{summary['embedding_dimension']}"
    )
    print(f"Model:                {summary['embedding_model']}")

    print("\nSaved files:")
    print(f"Embeddings: {summary['embeddings_path']}")
    print(f"Metadata:   {summary['metadata_path']}")

    # Load the files back to verify that they can be read.
    vectors, index_data = load_embedding_index(
        Path(summary["embeddings_path"]),
        Path(summary["metadata_path"]),
    )

    print("\nVerification:")
    print(f"Loaded vector shape:  {vectors.shape}")
    print(
        "Loaded metadata rows: "
        f"{len(index_data['records'])}"
    )

    if len(vectors) > 0:
        first_record = index_data["records"][0]

        print("\nFIRST EMBEDDING RECORD")
        print("-" * 65)
        print(f"Child ID:  {first_record['child_id']}")
        print(f"Parent ID: {first_record['parent_id']}")
        print(f"Source:    {first_record['source_file']}")
        print(f"Category:  {first_record['category']}")
        print(f"Pages:     {first_record['page_numbers']}")
        print(f"Text:      {first_record['text'][:300]}")

    print("\nEmbedding index is ready for retrieval development.")


if __name__ == "__main__":
    main()