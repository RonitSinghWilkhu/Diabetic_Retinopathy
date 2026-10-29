"""
Cross-encoder reranking for the diabetic retinopathy RAG pipeline.

The reranker receives candidate chunks from semantic/MMR retrieval,
scores each query-chunk pair with a cross-encoder, and returns the
highest-ranked chunks.

Reranker scores indicate estimated text relevance. They are not
medical confidence scores and should not be interpreted as probabilities.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Iterable, Optional, Sequence

from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)


DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


@dataclass
class RerankedChunk:
    """A retrieved chunk with its cross-encoder relevance score."""

    chunk: Any
    rerank_score: float
    rerank_rank: int

    @property
    def text(self) -> str:
        """Return the chunk's text."""
        return str(_get_value(self.chunk, ("text", "content", "chunk_text"), ""))

    @property
    def metadata(self) -> dict:
        """Return chunk metadata without modifying the original chunk."""
        metadata = _get_value(self.chunk, ("metadata",), {})
        return metadata if isinstance(metadata, dict) else {}

    def to_dict(self) -> dict:
        """Serialize the reranked result while preserving useful source data."""
        result = {
            "text": self.text,
            "rerank_score": self.rerank_score,
            "rerank_rank": self.rerank_rank,
            "metadata": self.metadata,
        }

        # Include common source fields when they are available directly
        # on the chunk object.
        for field in (
            "chunk_id",
            "child_id",
            "parent_id",
            "source",
            "source_file",
            "document_name",
            "page_numbers",
            "pages",
            "score",
            "similarity",
        ):
            value = _get_value(self.chunk, (field,), None)
            if value is not None:
                result[field] = value

        return result


@dataclass
class RerankingResult:
    """Results and basic statistics from a reranking operation."""

    query: str
    results: list[RerankedChunk]
    candidates_count: int
    returned_count: int

    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "candidates_count": self.candidates_count,
            "returned_count": self.returned_count,
            "results": [item.to_dict() for item in self.results],
        }


def _get_value(obj: Any, names: Sequence[str], default: Any = None) -> Any:
    """Read a field from a dictionary or object using possible field names."""
    for name in names:
        if isinstance(obj, dict) and name in obj:
            return obj[name]
        if hasattr(obj, name):
            return getattr(obj, name)
    return default


def _extract_text(chunk: Any) -> str:
    """Extract passage text from supported chunk representations."""
    value = _get_value(chunk, ("text", "content", "chunk_text"), "")

    if value is None:
        return ""

    return str(value).strip()


class RAGReranker:
    """
    Rerank semantic retrieval results using a cross-encoder.

    Parameters
    ----------
    model_name:
        Hugging Face sentence-transformers cross-encoder model identifier.
    device:
        Optional device, such as "cpu" or "cuda".
        If omitted, sentence-transformers selects a suitable device.
    batch_size:
        Number of query-passage pairs processed per inference batch.
    max_length:
        Maximum token length used by the cross-encoder.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_RERANKER_MODEL,
        device: Optional[str] = None,
        batch_size: int = 16,
        max_length: int = 512,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1.")

        if max_length < 1:
            raise ValueError("max_length must be at least 1.")

        self.model_name = model_name
        self.batch_size = batch_size
        self.max_length = max_length

        logger.info("Loading cross-encoder reranker: %s", model_name)

        model_kwargs = {"max_length": max_length}
        if device is not None:
            model_kwargs["device"] = device

        self.model = CrossEncoder(model_name, **model_kwargs)

        logger.info("Cross-encoder reranker loaded.")

    def rerank(
        self,
        query: str,
        candidates: Iterable[Any],
        top_k: Optional[int] = 5,
        text_key: Optional[str] = None,
    ) -> RerankingResult:
        """
        Rerank candidate passages for a query.

        Parameters
        ----------
        query:
            The user's question.
        candidates:
            Retrieved chunks, dictionaries, or objects with a text field.
        top_k:
            Maximum number of results to return. Use None to return all.
        text_key:
            Optional text field name for dictionary candidates.

        Returns
        -------
        RerankingResult
            Candidates sorted from highest to lowest cross-encoder score.
        """
        query = query.strip()

        if not query:
            raise ValueError("query must not be empty.")

        if top_k is not None and top_k < 1:
            raise ValueError("top_k must be at least 1 or None.")

        candidates = list(candidates)
        candidates_count = len(candidates)

        if not candidates:
            return RerankingResult(
                query=query,
                results=[],
                candidates_count=0,
                returned_count=0,
            )

        candidate_texts = []

        for index, candidate in enumerate(candidates):
            if text_key and isinstance(candidate, dict):
                text = str(candidate.get(text_key, "") or "").strip()
            else:
                text = _extract_text(candidate)

            if not text:
                logger.warning(
                    "Candidate at index %d has no usable text; "
                    "it will receive the lowest score.",
                    index,
                )

            candidate_texts.append(text)

        # Empty-text candidates are retained to preserve traceability,
        # but are not sent to the model.
        valid_indices = [
            index for index, passage in enumerate(candidate_texts) if passage
        ]

        scores = [float("-inf")] * candidates_count

        if valid_indices:
            pairs = [
                [query, candidate_texts[index]]
                for index in valid_indices
            ]

            model_scores = self.model.predict(
                pairs,
                batch_size=self.batch_size,
                show_progress_bar=False,
            )

            # CrossEncoder normally returns a 1-D score array for a
            # single-output relevance model. Convert scores to Python floats.
            for index, score in zip(valid_indices, model_scores):
                scores[index] = float(score)

        ranked_indices = sorted(
            range(candidates_count),
            key=lambda index: scores[index],
            reverse=True,
        )

        if top_k is not None:
            ranked_indices = ranked_indices[:top_k]

        ranked_results = [
            RerankedChunk(
                chunk=candidates[index],
                rerank_score=scores[index],
                rerank_rank=rank,
            )
            for rank, index in enumerate(ranked_indices, start=1)
        ]

        return RerankingResult(
            query=query,
            results=ranked_results,
            candidates_count=candidates_count,
            returned_count=len(ranked_results),
        )

    def rerank_retrieval_result(
        self,
        query: str,
        retrieval_result: Any,
        top_k: Optional[int] = 5,
    ) -> RerankingResult:
        """
        Rerank the output of a retrieval call.

        Accepts:
        - a list of chunks;
        - an object exposing a `results` attribute;
        - a dictionary containing a `results` list.
        """
        if isinstance(retrieval_result, dict):
            candidates = retrieval_result.get("results", [])
        elif hasattr(retrieval_result, "results"):
            candidates = retrieval_result.results
        else:
            candidates = retrieval_result

        return self.rerank(query=query, candidates=candidates, top_k=top_k)


def rerank_chunks(
    query: str,
    candidates: Iterable[Any],
    top_k: int = 5,
    model_name: str = DEFAULT_RERANKER_MODEL,
) -> RerankingResult:
    """
    Convenience function for a one-off reranking operation.

    For production use, instantiate RAGReranker once and reuse it so the
    cross-encoder model is not reloaded for every query.
    """
    reranker = RAGReranker(model_name=model_name)
    return reranker.rerank(query, candidates, top_k=top_k)


def _smoke_test() -> None:
    """Run a small local test using the existing retrieval index."""
    from app.rag.retrieval import RAGRetriever

    test_queries = [
        "What are the stages of diabetic retinopathy?",
        "What screening recommendations exist for diabetic retinopathy?",
        "How have AI systems for diabetic retinopathy been evaluated?",
    ]

    retriever = RAGRetriever()
    reranker = RAGReranker()

    for query in test_queries:
        print("\n" + "=" * 90)
        print(f"QUERY: {query}")
        print("=" * 90)

        # Retrieve a wider candidate pool, then rerank down to five.
        retrieval_result = retriever.retrieve(query, top_k=15)

        reranking_result = reranker.rerank_retrieval_result(
            query=query,
            retrieval_result=retrieval_result,
            top_k=5,
        )

        print(
            f"Candidates: {reranking_result.candidates_count} | "
            f"Returned: {reranking_result.returned_count}"
        )

        for item in reranking_result.results:
            chunk = item.chunk

            source = _get_value(
                chunk,
                ("source", "source_file", "document_name", "document"),
                "Unknown source",
            )

            pages = _get_value(
                chunk,
                ("page_numbers", "pages"),
                [],
            )

            print(f"\nRank: {item.rerank_rank}")
            print(f"Reranker score: {item.rerank_score:.4f}")
            print(f"Source: {source}")
            print(f"Pages: {pages}")
            print(f"Text: {item.text[:500].replace(chr(10), ' ')}")

    print("\nReranker smoke test completed.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    _smoke_test()