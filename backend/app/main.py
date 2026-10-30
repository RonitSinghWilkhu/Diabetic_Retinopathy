from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import auth, chat, explanations, history, predictions
from app.schemas import HealthResponse

from pathlib import Path
from fastapi import HTTPException
from fastapi.responses import FileResponse

app = FastAPI(
    title="DR AI Platform API",
    description=(
        "An educational diabetic retinopathy classification platform "
        "with model explanations and evidence-grounded educational chat."
    ),
    version="0.1.0",
)

# Development configuration. Restrict allowed origins before deployment.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(predictions.router, prefix="/api/predictions", tags=["Predictions"])
app.include_router(explanations.router, prefix="/api/explanations", tags=["Explanations"])
app.include_router(chat.router, prefix="/api/chat", tags=["Educational Chat"])
app.include_router(auth.router, prefix="/api/auth", tags=["Authentication"])
app.include_router(history.router, prefix="/api/history", tags=["Analysis History"])


@app.get("/", response_model=HealthResponse)
def root():
    return {
        "status": "ok",
        "message": "DR AI Platform API is running.",
    }


@app.get("/health", response_model=HealthResponse)
def health_check():
    return {
        "status": "ok",
        "message": "Backend is healthy.",
    }

RAG_DOCUMENTS_DIR = (
    Path(__file__).resolve().parents[2] / "data" / "rag_documents"
)

@app.get("/api/documents/{filename}")
def serve_rag_document(filename: str):
    """Serve an indexed PDF for viewing in the browser."""

    # Only accept PDF filenames, not arbitrary filesystem paths.
    if (
        not filename.lower().endswith(".pdf")
        or Path(filename).name != filename
        or "/" in filename
        or "\\" in filename
    ):
        raise HTTPException(
            status_code=400,
            detail="Invalid filename. Only PDF files are allowed.",
        )

    # Search the category subfolders for the requested PDF.
    matches = [
        path
        for path in RAG_DOCUMENTS_DIR.rglob(filename)
        if path.is_file() and path.suffix.lower() == ".pdf"
    ]

    if not matches:
        raise HTTPException(
            status_code=404,
            detail=f"File '{filename}' not found in the RAG documents directory.",
        )

    if len(matches) > 1:
        raise HTTPException(
            status_code=409,
            detail=f"Multiple PDFs found with the filename '{filename}'.",
        )

    return FileResponse(
        path=matches[0],
        media_type="application/pdf",
        filename=filename,
        content_disposition_type="inline",
    )