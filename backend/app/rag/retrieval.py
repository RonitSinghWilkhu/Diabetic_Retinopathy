from __future__ import annotations

import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer


# ============================================================
# Configuration
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts" / "rag"

EMBEDDINGS_PATH = ARTIFACTS_DIR / "child_embeddings.npy"
METADATA_PATH = ARTIFACTS_DIR / "embedding_index.json"

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EXPECTED_EMBEDDING_DIMENSION = 384

DEFAULT_CANDIDATE_K = 20
DEFAULT_TOP_K = 5
DEFAULT_MMR_LAMBDA = 0.70

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s",
)

logger = logging.getLogger(__name__)


# ============================================================
# Data models
# ============================================================

@dataclass
class RetrievedChunk:
    """A child chunk retrieved from the RAG embedding index."""

    child_id: str
    parent_id: str
    text: str

    source_file: str
    category: str
    page_numbers: list[int]

    similarity_score: float
    mmr_score: float

    metadata: dict[str, Any]


@dataclass
class RetrievalResult:
    """The result returned by the retrieval pipeline."""

    query: str
    results: list[RetrievedChunk]
    candidate_count: int
    retrieval_method: str


# ============================================================
# Retrieval system
# ============================================================

class RAGRetriever:
    """
    Semantic retrieval over precomputed child-chunk embeddings.

    Features:
        1. Semantic search using cosine similarity.
        2. MMR selection for relevance and diversity.
        3. Parent-child grouping.
        4. Source metadata preservation.

    The embedding index is loaded once when the retriever is
    initialized, rather than reloaded for every question.
    """

    def __init__(
        self,
        embeddings_path: Path = EMBEDDINGS_PATH,
        metadata_path: Path = METADATA_PATH,
        model_name: str = EMBEDDING_MODEL_NAME,
    ) -> None:
        self.embeddings_path = Path(embeddings_path)
        self.metadata_path = Path(metadata_path)
        self.model_name = model_name

        self.embeddings: np.ndarray
        self.metadata: list[dict[str, Any]]
        self.model: SentenceTransformer | None = None

        self._load_index()

    # --------------------------------------------------------
    # Index loading
    # --------------------------------------------------------

    def _load_index(self) -> None:
        """Load and validate the saved embeddings and metadata."""

        if not self.embeddings_path.exists():
            raise FileNotFoundError(
                f"Embedding matrix not found: {self.embeddings_path}\n"
                "Build the index first with:\n"
                "python -m app.rag.embeddings"
            )

        if not self.metadata_path.exists():
            raise FileNotFoundError(
                f"Embedding metadata not found: {self.metadata_path}"
            )

        logger.info("Loading embedding matrix: %s", self.embeddings_path)

        self.embeddings = np.load(
            self.embeddings_path,
            allow_pickle=False,
        )

        logger.info("Loading embedding metadata: %s", self.metadata_path)

        with self.metadata_path.open("r", encoding="utf-8") as file:
            raw_metadata = json.load(file)

        self.metadata = self._extract_metadata_records(raw_metadata)

        if self.embeddings.ndim != 2:
            raise ValueError(
                "Embedding matrix must be two-dimensional. "
                f"Received shape: {self.embeddings.shape}"
            )

        if self.embeddings.shape[1] != EXPECTED_EMBEDDING_DIMENSION:
            raise ValueError(
                "Unexpected embedding dimension. "
                f"Expected {EXPECTED_EMBEDDING_DIMENSION}, "
                f"received {self.embeddings.shape[1]}."
            )

        if len(self.metadata) != self.embeddings.shape[0]:
            raise ValueError(
                "Embedding and metadata counts do not match. "
                f"Embeddings: {self.embeddings.shape[0]}, "
                f"metadata records: {len(self.metadata)}."
            )

        if not np.isfinite(self.embeddings).all():
            raise ValueError("Embedding matrix contains NaN or infinite values.")

        # Ensure all vectors are normalized before using dot products
        # as cosine similarity scores.
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)

        if np.any(norms == 0):
            raise ValueError("Embedding matrix contains zero-length vectors.")

        self.embeddings = self.embeddings / norms

        logger.info(
            "Index loaded successfully: %d vectors, %d dimensions.",
            self.embeddings.shape[0],
            self.embeddings.shape[1],
        )

    @staticmethod
    def _extract_metadata_records(
        raw_metadata: Any,
    ) -> list[dict[str, Any]]:
        """
        Support common JSON layouts:
            - a list of metadata records
            - {"records": [...]}
            - {"metadata": [...]}
            - {"items": [...]}
        """

        if isinstance(raw_metadata, list):
            records = raw_metadata

        elif isinstance(raw_metadata, dict):
            records = None

            for key in ("records", "metadata", "items", "chunks", "children"):
                value = raw_metadata.get(key)

                if isinstance(value, list):
                    records = value
                    break

            if records is None:
                raise ValueError(
                    "Could not find a list of metadata records in "
                    f"{METADATA_PATH.name}. "
                    "Expected a list or a dictionary containing "
                    "'records', 'metadata', 'items', 'chunks', or 'children'."
                )

        else:
            raise ValueError(
                "Embedding metadata must be a JSON list or dictionary."
            )

        validated_records: list[dict[str, Any]] = []

        for index, record in enumerate(records):
            if not isinstance(record, dict):
                raise ValueError(
                    f"Metadata record at index {index} is not a dictionary."
                )

            validated_records.append(record)

        return validated_records

    # --------------------------------------------------------
    # Embedding model
    # --------------------------------------------------------

    def _load_model(self) -> SentenceTransformer:
        """Load the question-embedding model on first use."""

        if self.model is None:
            logger.info("Loading query embedding model: %s", self.model_name)

            self.model = SentenceTransformer(self.model_name)

            dimension = self.model.get_embedding_dimension()

            if dimension != EXPECTED_EMBEDDING_DIMENSION:
                raise ValueError(
                    f"Model dimension mismatch: expected "
                    f"{EXPECTED_EMBEDDING_DIMENSION}, received {dimension}."
                )

            logger.info("Query embedding model loaded.")

        return self.model

    def embed_query(self, query: str) -> np.ndarray:
        """Convert a question into a normalized embedding vector."""

        if not isinstance(query, str) or not query.strip():
            raise ValueError("Query must be a non-empty string.")

        model = self._load_model()

        vector = model.encode(
            query.strip(),
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        vector = np.asarray(vector, dtype=np.float32)

        if vector.shape != (EXPECTED_EMBEDDING_DIMENSION,):
            raise ValueError(
                "Unexpected query embedding shape: "
                f"{vector.shape}"
            )

        if not np.isfinite(vector).all():
            raise ValueError("Query embedding contains invalid values.")

        return vector

    # --------------------------------------------------------
    # Metadata helpers
    # --------------------------------------------------------

    @staticmethod
    def _get_value(
        record: dict[str, Any],
        possible_keys: tuple[str, ...],
        default: Any = None,
    ) -> Any:
        """Return the first matching metadata field."""

        for key in possible_keys:
            if key in record and record[key] is not None:
                return record[key]

        return default

    @staticmethod
    def _normalize_pages(value: Any) -> list[int]:
        """Normalize page metadata into a list of integers."""

        if value is None:
            return []

        if isinstance(value, int):
            return [value]

        if isinstance(value, (list, tuple, set)):
            pages: list[int] = []

            for item in value:
                try:
                    pages.append(int(item))
                except (TypeError, ValueError):
                    continue

            return sorted(set(pages))

        return []

    @staticmethod
    def _is_usable_chunk(text: Any) -> bool:
        """
        Reject empty, extremely short, and clearly bibliography-only chunks.
        Keep ordinary scientific text, including text containing citations.
        """
        if not isinstance(text, str):
            return False

        cleaned = " ".join(text.split())

        # Very short chunks rarely provide sufficient standalone evidence.
        if len(cleaned) < 80:
            return False

        lowered = cleaned.lower()

        # Reject chunks that consist primarily of a section heading.
        bibliography_headings = {
            "references",
            "bibliography",
            "reference list",
            "works cited",
            "literature cited",
        }

        if lowered.strip(" :.-") in bibliography_headings:
            return False

        # Detect explicit bibliography headings at the beginning.
        # Require additional reference-like patterns to reduce false positives.
        has_bibliography_heading = (
            lowered.startswith("references\n")
            or lowered.startswith("bibliography\n")
            or lowered.startswith("reference list\n")
            or lowered.startswith("works cited\n")
        )

        doi_count = lowered.count("doi:")
        url_count = lowered.count("https://") + lowered.count("http://")

        # Bibliography sections often contain several DOI/URL entries.
        if has_bibliography_heading and (doi_count + url_count) >= 2:
            return False

        return True

    def _make_result(
        self,
        index: int,
        similarity: float,
        mmr_score: float,
    ) -> RetrievedChunk:
        """Convert an indexed record into a typed retrieval result."""

        record = self.metadata[index]

        child_id = self._get_value(
            record,
            ("child_id", "chunk_id", "id"),
            f"child_{index:05d}",
        )

        parent_id = self._get_value(
            record,
            ("parent_id", "parent_chunk_id"),
            "",
        )

        text = self._get_value(
            record,
            ("text", "chunk_text", "content", "page_content"),
            "",
        )

        source_file = self._get_value(
            record,
            ("source_file", "filename", "source", "file_name"),
            "Unknown source",
        )

        category = self._get_value(
            record,
            ("category", "document_category"),
            "uncategorized",
        )

        page_numbers = self._normalize_pages(
            self._get_value(
                record,
                ("page_numbers", "pages", "page_number", "page"),
                [],
            )
        )

        return RetrievedChunk(
            child_id=str(child_id),
            parent_id=str(parent_id),
            text=str(text),
            source_file=str(source_file),
            category=str(category),
            page_numbers=page_numbers,
            similarity_score=float(similarity),
            mmr_score=float(mmr_score),
            metadata=record,
        )

    # --------------------------------------------------------
    # Semantic search
    # --------------------------------------------------------

    def semantic_search(
        self,
        query: str,
        top_k: int = DEFAULT_CANDIDATE_K,
    ) -> list[RetrievedChunk]:
        """
        Retrieve the most semantically similar child chunks.

        Because document vectors are normalized, the dot product
        is equivalent to cosine similarity.
        """

        if top_k < 1:
            raise ValueError("top_k must be at least 1.")

        query_vector = self.embed_query(query)

        scores = self.embeddings @ query_vector

        eligible_indices = np.array(
            [
                index
                for index, record in enumerate(self.metadata)
                if self._is_usable_chunk(
                    self._get_value(
                        record,
                        ("text", "chunk_text", "content", "page_content"),
                        "",
                    )
                )
            ],
            dtype=np.int64,
        )

        if eligible_indices.size == 0:
            return []

        candidate_count = min(top_k, eligible_indices)
        eligible_scores = scores[eligible_indices]

        # Select the strongest candidates without sorting the entire
        # collection when only a small subset is required.
        local_indices = np.argpartition(
            eligible_scores,
            -candidate_count,
        )[-candidate_count:]

        candidate_indices = eligible_indices[local_indices]

        candidate_indices = candidate_indices[
            np.argsort(scores[candidate_indices])[::-1]
        ]

        results: list[RetrievedChunk] = []

        for index in candidate_indices:
            results.append(
                self._make_result(
                    index=int(index),
                    similarity=float(scores[index]),
                    mmr_score=float(scores[index]),
                )
            )

        return results

    # --------------------------------------------------------
    # Maximal Marginal Relevance
    # --------------------------------------------------------

    def mmr_search(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        candidate_k: int = DEFAULT_CANDIDATE_K,
        lambda_mult: float = DEFAULT_MMR_LAMBDA,
    ) -> RetrievalResult:
        """
        Select relevant but diverse child chunks using MMR.

        MMR score:
            lambda * relevance
            - (1 - lambda) * maximum similarity to selected chunks

        lambda_mult near 1 emphasizes relevance.
        lambda_mult near 0 emphasizes diversity.
        """

        if top_k < 1:
            raise ValueError("top_k must be at least 1.")

        if candidate_k < 1:
            raise ValueError("candidate_k must be at least 1.")

        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string.")

        query_vector = self.embed_query(query)

        # Cosine similarity between the query and every child chunk.
        relevance_scores = self.embeddings @ query_vector

        eligible_indices = np.array(
            [
                index
                for index, record in enumerate(self.metadata)
                if self._is_usable_chunk(
                    self._get_value(
                        record,
                        ("text", "chunk_text", "content", "page_content"),
                        "",
                    )
                )
            ],
            dtype=np.int64,
        )

        if eligible_indices.size == 0:
            return RetrievalResult(
                query=query,
                results=[],
                candidate_count=0,
                retrieval_method="semantic_search_with_mmr",
            )

        candidate_count = min(candidate_k, eligible_indices.size)
        result_count = min(top_k, candidate_count)
        eligible_scores = relevance_scores[eligible_indices]

        local_indices = np.argpartition(
            eligible_scores,
            -candidate_count,
        )[-candidate_count:]

        candidate_indices = eligible_indices[local_indices]

        logger.info(
            "MMR eligible chunks: %d/%d",
            eligible_indices.size,
            len(self.metadata),
        )

        # Sort candidate indices by relevance before MMR selection.
        candidate_indices = candidate_indices[
            np.argsort(relevance_scores[candidate_indices])[::-1]
        ]

        selected_indices: list[int] = []
        remaining_indices = [int(i) for i in candidate_indices]

        mmr_scores: dict[int, float] = {}

        while remaining_indices and len(selected_indices) < result_count:
            best_index: int | None = None
            best_score = -float("inf")

            for candidate_index in remaining_indices:
                relevance = float(relevance_scores[candidate_index])

                if selected_indices:
                    # Similarity between the candidate and all selected
                    # child vectors. The largest value is the redundancy.
                    selected_vectors = self.embeddings[selected_indices]
                    candidate_vector = self.embeddings[candidate_index]

                    redundancy = float(
                        np.max(selected_vectors @ candidate_vector)
                    )
                else:
                    redundancy = 0.0

                score = (
                    lambda_mult * relevance
                    - (1.0 - lambda_mult) * redundancy
                )

                if score > best_score:
                    best_score = score
                    best_index = candidate_index

            if best_index is None:
                break

            selected_indices.append(best_index)
            remaining_indices.remove(best_index)
            mmr_scores[best_index] = best_score

        results = [
            self._make_result(
                index=index,
                similarity=float(relevance_scores[index]),
                mmr_score=mmr_scores[index],
            )
            for index in selected_indices
        ]

        return RetrievalResult(
            query=query,
            results=results,
            candidate_count=candidate_count,
            retrieval_method="semantic_search_with_mmr",
        )

    # --------------------------------------------------------
    # Parent-child grouping
    # --------------------------------------------------------

    def group_by_parent(
        self,
        chunks: list[RetrievedChunk],
    ) -> dict[str, list[RetrievedChunk]]:
        """
        Group retrieved child chunks by their parent IDs.

        This method groups retrieved results only. It does not claim
        to reconstruct the complete original parent text.
        """

        grouped: dict[str, list[RetrievedChunk]] = {}

        for chunk in chunks:
            parent_id = chunk.parent_id or chunk.child_id
            grouped.setdefault(parent_id, []).append(chunk)

        return grouped

    # --------------------------------------------------------
    # Public convenience methods
    # --------------------------------------------------------

    def retrieve(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        candidate_k: int = DEFAULT_CANDIDATE_K,
        lambda_mult: float = DEFAULT_MMR_LAMBDA,
    ) -> RetrievalResult:
        """Run MMR retrieval and return source-aware results."""

        return self.mmr_search(
            query=query,
            top_k=top_k,
            candidate_k=candidate_k,
            lambda_mult=lambda_mult,
        )

    def retrieve_as_dict(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        candidate_k: int = DEFAULT_CANDIDATE_K,
        lambda_mult: float = DEFAULT_MMR_LAMBDA,
    ) -> dict[str, Any]:
        """Return retrieval results in a JSON-friendly structure."""

        result = self.retrieve(
            query=query,
            top_k=top_k,
            candidate_k=candidate_k,
            lambda_mult=lambda_mult,
        )

        return {
            "query": result.query,
            "retrieval_method": result.retrieval_method,
            "candidate_count": result.candidate_count,
            "result_count": len(result.results),
            "results": [
                asdict(chunk)
                for chunk in result.results
            ],
        }


# ============================================================
# Command-line test
# ============================================================

def main() -> None:
    """Run a small retrieval smoke test from the terminal."""

    logger.info("Starting RAG retrieval smoke test.")

    retriever = RAGRetriever()

    test_queries = [
        "What are the different stages of diabetic retinopathy?",
        "What screening recommendations exist for diabetic retinopathy?",
        "How have AI systems been evaluated for diabetic retinopathy screening?",
    ]

    for query in test_queries:
        print("\n" + "=" * 80)
        print(f"QUESTION: {query}")
        print("=" * 80)

        result = retriever.retrieve(
            query=query,
            top_k=5,
            candidate_k=20,
            lambda_mult=0.70,
        )

        for rank, chunk in enumerate(result.results, start=1):
            print(f"\n[{rank}] {chunk.source_file}")
            print(f"Category: {chunk.category}")
            print(f"Pages: {chunk.page_numbers or 'Not specified'}")
            print(f"Child ID: {chunk.child_id}")
            print(f"Parent ID: {chunk.parent_id or 'Not specified'}")
            print(f"Similarity: {chunk.similarity_score:.4f}")
            print(f"MMR score: {chunk.mmr_score:.4f}")
            print(f"Text: {chunk.text[:700]}")

    logger.info("Retrieval smoke test completed.")


if __name__ == "__main__":
    main()