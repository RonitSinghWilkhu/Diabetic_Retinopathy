from pathlib import Path
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from sqlalchemy import select
from sqlalchemy.orm import Session
from app.api.auth import get_current_user
from app.database import get_db
from app.models import Analysis, AnalysisArtifact, User

router = APIRouter()
BACKEND_DIR = Path(__file__).resolve().parents[2]
ANALYSES_DIR = (BACKEND_DIR / "storage" / "analyses").resolve()
ALLOWED_ARTIFACT_TYPES = {"heatmap", "overlay"}

def safe_path(relative_path: str) -> Path:
    candidate = (BACKEND_DIR / relative_path).resolve()
    try:
        candidate.relative_to(ANALYSES_DIR)
    except ValueError:
        raise HTTPException(status_code=404, detail="Stored file not found.")
    return candidate

def owned_analysis(db: Session, user_id: int, analysis_id: int) -> Analysis:
    item = db.scalar(select(Analysis).where(Analysis.id == analysis_id, Analysis.user_id == user_id))
    if item is None:
        raise HTTPException(status_code=404, detail="Analysis not found.")
    return item

@router.get("/")
def list_history(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    analyses = db.scalars(select(Analysis).where(Analysis.user_id == current_user.id).order_by(Analysis.created_at.desc(), Analysis.id.desc())).all()
    output = []
    for item in analyses:
        artifacts = db.scalars(select(AnalysisArtifact).where(
            AnalysisArtifact.analysis_id == item.id,
            AnalysisArtifact.artifact_type.in_(ALLOWED_ARTIFACT_TYPES)
        )).all()
        status = {"heatmap": False, "overlay": False}
        for artifact in artifacts:
            try:
                if safe_path(artifact.file_path).is_file():
                    status[artifact.artifact_type] = True
            except HTTPException:
                pass
        try:
            original_available = safe_path(item.image_path).is_file()
        except HTTPException:
            original_available = False
        output.append({
            "id": item.id, "original_filename": item.original_filename,
            "predicted_class": item.predicted_class,
            "predicted_class_index": item.predicted_class_index,
            "confidence": item.confidence, "probabilities": item.probabilities,
            "model_name": item.model_name, "created_at": item.created_at,
            "original_image_available": original_available, "artifacts": status
        })
    return output

@router.get("/{analysis_id}/image")
def get_original(analysis_id: int, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    item = owned_analysis(db, current_user.id, analysis_id)
    path = safe_path(item.image_path)
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Original image is unavailable.")
    return FileResponse(path=path, media_type="application/octet-stream")

@router.get("/{analysis_id}/artifacts/{artifact_type}")
def get_artifact(analysis_id: int, artifact_type: str, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    if artifact_type not in ALLOWED_ARTIFACT_TYPES:
        raise HTTPException(status_code=400, detail="Unsupported artifact type.")
    owned_analysis(db, current_user.id, analysis_id)
    artifact = db.scalar(select(AnalysisArtifact).where(
        AnalysisArtifact.analysis_id == analysis_id,
        AnalysisArtifact.artifact_type == artifact_type
    ).order_by(AnalysisArtifact.created_at.desc(), AnalysisArtifact.id.desc()))
    if artifact is None:
        raise HTTPException(status_code=404, detail=f"{artifact_type.title()} is unavailable.")
    path = safe_path(artifact.file_path)
    if not path.is_file():
        raise HTTPException(status_code=404, detail=f"{artifact_type.title()} is unavailable.")
    return FileResponse(path=path, media_type="image/png")
