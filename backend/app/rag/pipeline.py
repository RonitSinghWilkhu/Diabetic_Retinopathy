"""
RAG query pipeline.
Flow:
User question
-> Semantic retrieval + MMR
-> Cross-encoder reranking
-> Evidence filtering
-> Context assembly
-> Structured result for the answer generator
This module does not call the LLM yet.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from app.rag.retrieval import RAGRetriever
from app.rag.reranker import RAGReranker


@dataclass
class EvidenceChunk:
    """A retrieved passage prepared for use as evidence."""
    text: str
    source: str
    pages: list[int] = field(default_factory=list)
    parent_id: str = ""
    child_id: str = ""
    retrieval_score: float | None = None
    rerank_score: float | None = None
    evidence_id: int = 0 


@dataclass
class RAGPipelineResult:
    """Complete output of a RAG query."""
    query: str
    evidence: list[EvidenceChunk]
    context: str
    retrieval_count: int
    reranked_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "context": self.context,
            "retrieval_count": self.retrieval_count,
            "reranked_count": self.reranked_count,
            "evidence": [
                {
                    "evidence_id": item.evidence_id,
                    "text": item.text,
                    "source": item.source,
                    "pages": item.pages,
                    "parent_id": item.parent_id,
                    "child_id": item.child_id,
                    "retrieval_score": item.retrieval_score,
                    "rerank_score": item.rerank_score,
                }
                for item in self.evidence
            ],
        }


def _get_value(obj: Any, *names: str, default: Any = None) -> Any:
    """Read a field from a dictionary or an object."""
    for name in names:
        if isinstance(obj, dict) and name in obj:
            return obj[name]
        if hasattr(obj, name):
            return getattr(obj, name)
    return default


def _get_results(result: Any) -> list[Any]:
    """Normalize the retrieval/reranking result into a list."""
    if isinstance(result, list):
        return result
    results = _get_value(result, "results", default=None)
    if results is None:
        return []
    return list(results)


def _clean_text(text: str) -> str:
    """Remove excessive whitespace without altering wording."""
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _normalise_pages(value: Any) -> list[int]:
    """Convert page metadata to a list of integers when possible."""
    if value is None:
        return []
    if isinstance(value, int):
        return [value]
    if isinstance(value, (list, tuple, set)):
        pages = []
        for page in value:
            try:
                pages.append(int(page))
            except (TypeError, ValueError):
                continue
        return sorted(set(pages))
    return []


def _to_evidence(item: Any) -> EvidenceChunk | None:
    """Convert a retrieved or reranked item into a standard evidence record."""
    text = _get_value(
        item,
        "text",
        "child_text",
        "content",
        "page_content",
        default="",
    )
    if not isinstance(text, str):
        return None
    
    text = _clean_text(text)
    if len(text) < 40:
        return None
        
    source = _get_value(
        item,
        "source",
        "source_file",
        "filename",
        "document_name",
        default="Unknown source",
    )
    
    metadata = _get_value(item, "metadata", default={})
    if not isinstance(metadata, dict):
        metadata = {}
        
    if source == "Unknown source":
        source = _get_value(
            metadata,
            "source",
            "source_file",
            "filename",
            "document_name",
            default=source,
        )
        
    pages = _get_value(
        item,
        "pages",
        "page_numbers",
        "page",
        default=None,
    )
    if pages is None:
        pages = _get_value(
            metadata,
            "pages",
            "page_numbers",
            "page",
            default=[],
        )
        
    parent_id = _get_value(
        item,
        "parent_id",
        default=metadata.get("parent_id", ""),
    )
    
    child_id = _get_value(
        item,
        "child_id",
        "chunk_id",
        "id",
        default=metadata.get("child_id", ""),
    )
    
    retrieval_score = _get_value(
        item,
        "retrieval_score",
        "similarity",
        "score",
        default=None,
    )
    
    rerank_score = _get_value(
        item,
        "rerank_score",
        "relevance_score",
        "cross_encoder_score",
        default=None,
    )

    def as_float(value: Any) -> float | None:
        try:
            return float(value) if value is not None else None
        except (TypeError, ValueError):
            return None

    return EvidenceChunk(
        text=text,
        source=str(source),
        pages=_normalise_pages(pages),
        parent_id=str(parent_id or ""),
        child_id=str(child_id or ""),
        retrieval_score=as_float(retrieval_score),
        rerank_score=as_float(rerank_score),
        evidence_id=int(_get_value(item, "evidence_id", default=0) or 0),
    )


class RAGPipeline:
    """
    Combines semantic retrieval, MMR, reranking, and context assembly.
    The retrieval and reranker classes are injected optionally so they can
    be tested independently or replaced later.
    """
    def __init__(
        self,
        retriever: RAGRetriever | None = None,
        reranker: RAGReranker | None = None,
        retrieval_top_k: int = 15,
        final_top_k: int = 5,
        max_context_chars: int = 12000,
    ):
        if retrieval_top_k < 1:
            raise ValueError("retrieval_top_k must be at least 1.")
        if final_top_k < 1:
            raise ValueError("final_top_k must be at least 1.")
        if max_context_chars < 100:
            raise ValueError("max_context_chars must be at least 100.")
            
        self.retriever = retriever or RAGRetriever()
        self.reranker = reranker or RAGReranker()
        self.retrieval_top_k = retrieval_top_k
        self.final_top_k = final_top_k
        self.max_context_chars = max_context_chars

    def query(self, question: str) -> RAGPipelineResult:
        """Retrieve, rerank, and assemble evidence for a question."""
        question = question.strip()
        if not question:
            raise ValueError("Question cannot be empty.")

        # 1. Retrieve candidates using the existing retrieval module.
        retrieved_result = self.retriever.retrieve(
            question,
            top_k=self.retrieval_top_k,
        )
        retrieved_items = _get_results(retrieved_result)
        retrieval_count = len(retrieved_items)
        
        if not retrieved_items:
            return RAGPipelineResult(
                query=question,
                evidence=[],
                context="",
                retrieval_count=0,
                reranked_count=0,
            )

        # 2. Rerank the retrieved candidates.
        reranked_result = self.reranker.rerank(
            question,
            retrieved_items,
            top_k=self.final_top_k,
        )
        reranked_items = _get_results(reranked_result)

        # 3. Normalize the output and discard empty/noisy passages.
        evidence: list[EvidenceChunk] = []
        for item in reranked_items:
            record = _to_evidence(item)
            if record is not None:
                evidence.append(record)

        # 4. Assemble a bounded context for the future LLM call.
        context_parts: list[str] = []
        used_chars = 0
        accepted_evidence: list[EvidenceChunk] = []
        
        for index, item in enumerate(evidence, start=1):
            # Assign the same ID used in the context heading.
            item.evidence_id = index
            
            page_text = ""
            if item.pages:
                # Page metadata is included as recorded by the index.
                # Verify chunking provenance before treating it as exact.
                page_text = (
                    " | Pages: "
                    + ", ".join(str(page) for page in item.pages)
                )
                
            heading = (
                f"[Evidence {item.evidence_id}] "
                f"Source: {item.source}{page_text}"
            )
            block = f"{heading}\n{item.text}\n"
            
            if used_chars + len(block) > self.max_context_chars:
                remaining = self.max_context_chars - used_chars
                if remaining > len(heading) + 80:
                    truncated_text = item.text[
                        : remaining - len(heading) - 10
                    ].rstrip()
                    block = f"{heading}\n{truncated_text}\n"
                    context_parts.append(block)
                    accepted_evidence.append(item)
                break
                
            context_parts.append(block)
            accepted_evidence.append(item)
            used_chars += len(block)
            
        context = "\n".join(context_parts).strip()

        return RAGPipelineResult(
            query=question,
            evidence=accepted_evidence,
            context=context,
            retrieval_count=retrieval_count,
            reranked_count=len(reranked_items),
        )


if __name__ == "__main__":
    pipeline = RAGPipeline()
    test_questions = [
        "What are the clinical stages of diabetic retinopathy?",
        "What do guidelines say about diabetic retinopathy screening?",
        "How should an AI system for diabetic retinopathy grading be evaluated?",
    ]

    for question in test_questions:
        print("\n" + "=" * 80)
        print(f"QUESTION: {question}")
        try:
            result = pipeline.query(question)
            print(f"Retrieved candidates: {result.retrieval_count}")
            print(f"Reranked candidates: {result.reranked_count}")
            print(f"Final evidence chunks: {len(result.evidence)}")
            print("\nASSEMBLED CONTEXT:\n")
            print(result.context[:5000])
        except Exception as exc:
            print(f"Pipeline test failed: {type(exc).__name__}: {exc}")
            raise