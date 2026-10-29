import logging

from fastapi import APIRouter, HTTPException

from app.schemas import ChatRequest, ChatResponse
from app.rag.generation import RAGAnswerGenerator

logger = logging.getLogger(__name__)

router = APIRouter()

_generator = None


def get_generator() -> RAGAnswerGenerator:
    """Initialize the RAG answer generator once."""
    global _generator

    if _generator is None:
        try:
            _generator = RAGAnswerGenerator()
        except Exception:
            logger.exception("Failed to initialize RAG answer generator")
            raise HTTPException(
                status_code=503,
                detail="The educational assistant is temporarily unavailable.",
            )

    return _generator


@router.post("/", response_model=ChatResponse)
def educational_chat(request: ChatRequest):
    """Answer an educational question using RAG."""
    generator = get_generator()

    try:
        result = generator.answer(
            request.question,
            image_context=request.image_context.model_dump()
            if request.image_context
            else None,
        )

        # Convert source metadata to strings for ChatResponse.
        raw_sources = result.sources
        sources = []
        seen_sources = set()

        for source in raw_sources:
            if isinstance(source, str):
                label = source
            elif isinstance(source, dict):
                filename = source.get("source", "Unknown source")
                pages = source.get("pages", [])
                page_text = ", ".join(map(str, pages))
                label = f"{filename} (pages: {page_text})" if pages else filename
            if label not in seen_sources:
                sources.append(label)
                seen_sources.add(label)
            else:
                label = str(source)

        return ChatResponse(
            answer=result.answer,
            sources=sources,
        )

    except HTTPException:
        raise
    except Exception:
        logger.exception("Educational chat request failed")
        raise HTTPException(
            status_code=500,
            detail="Unable to answer the question at this time.",
        )