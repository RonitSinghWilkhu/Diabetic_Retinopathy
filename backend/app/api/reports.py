
import logging
from pathlib import Path
from io import BytesIO

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.auth import get_current_user
from app.database import get_db
from app.models import Analysis, AnalysisArtifact, User
from app.services.report_service import generate_analysis_report


router = APIRouter(tags=["Patient Reports"])

logger = logging.getLogger(__name__)

# backend/app/api/reports.py -> backend/
BACKEND_DIR = Path(__file__).resolve().parents[2]

ANALYSES_DIR = (
    BACKEND_DIR / "storage" / "analyses"
).resolve()

ALLOWED_ARTIFACT_TYPES = {"heatmap", "overlay"}


def safe_path(relative_path: str) -> Path:
    """
    Resolve a stored file path safely.

    Only files inside storage/analyses are allowed.
    """

    candidate = (BACKEND_DIR / relative_path).resolve()

    try:
        candidate.relative_to(ANALYSES_DIR)
    except ValueError:
        raise HTTPException(
            status_code=404,
            detail="Stored report file not found.",
        )

    return candidate


def get_owned_analysis(
    db: Session,
    user_id: int,
    analysis_id: int,
) -> Analysis:
    """
    Retrieve an analysis only if it belongs to the current user.
    """

    analysis = db.scalar(
        select(Analysis).where(
            Analysis.id == analysis_id,
            Analysis.user_id == user_id,
        )
    )

    if analysis is None:
        raise HTTPException(
            status_code=404,
            detail="Analysis not found.",
        )

    return analysis


def get_gradcam_artifacts(
    db: Session,
    analysis_id: int,
) -> dict:
    """
    Retrieve the latest available Grad-CAM heatmap and overlay.
    """

    artifacts = db.scalars(
        select(AnalysisArtifact)
        .where(
            AnalysisArtifact.analysis_id == analysis_id,
            AnalysisArtifact.artifact_type.in_(
                ALLOWED_ARTIFACT_TYPES
            ),
        )
        .order_by(
            AnalysisArtifact.created_at.desc(),
            AnalysisArtifact.id.desc(),
        )
    ).all()

    result = {}

    for artifact in artifacts:
        artifact_type = artifact.artifact_type

        if artifact_type in result:
            continue

        path = safe_path(artifact.file_path)

        if path.is_file():
            result[artifact_type] = path

    return result


@router.get("/{analysis_id}/download")
def download_patient_report(
    analysis_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Generate and download a patient-friendly DR screening report.
    """

    # 1. Verify analysis ownership.
    analysis = get_owned_analysis(
        db=db,
        user_id=current_user.id,
        analysis_id=analysis_id,
    )

    # 2. Retrieve the original retinal image.
    original_image_path = safe_path(
        analysis.image_path
    )

    if not original_image_path.is_file():
        raise HTTPException(
            status_code=404,
            detail="Original retinal image is unavailable.",
        )

    # 3. Retrieve available Grad-CAM artifacts.
    gradcam_paths = get_gradcam_artifacts(
        db=db,
        analysis_id=analysis.id,
    )

    # 4. Generate the complete report using the report service.
    try:
        pdf_bytes = generate_analysis_report(
            analysis=analysis,
            original_image_path=original_image_path,
            gradcam_paths=gradcam_paths,
            patient_name=current_user.full_name,
        )

    except HTTPException:
        raise

    except Exception as exc:
        logger.exception(
            "Failed to generate patient report for analysis %s",
            analysis_id,
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "Unable to generate the patient report. "
                "Check the backend logs."
            ),
        ) from exc

    # 5. Return the generated PDF.
    filename = f"DR_Patient_Report_{analysis.id}.pdf"

    return StreamingResponse(
        BytesIO(pdf_bytes),
        media_type="application/pdf",
        headers={
            "Content-Disposition": (
                f'attachment; filename="{filename}"'
            )
        },
    )