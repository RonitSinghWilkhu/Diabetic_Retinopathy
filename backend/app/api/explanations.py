import base64
import io
import logging
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.auth import get_current_user
from app.database import get_db
from app.models import Analysis, AnalysisArtifact, User
from app.ml.explainability import generate_gradcam
from app.ml.model_loader import load_model
from app.ml.preprocessing import preprocess_image


router = APIRouter(tags=["Explanations"])

logger = logging.getLogger(__name__)

# Load the model once when this module is imported.
model, class_mapping, model_config, device = load_model()

MAX_FILE_SIZE = 10 * 1024 * 1024

ALLOWED_CONTENT_TYPES = {
    "image/jpeg",
    "image/png",
    "image/webp",
}

# backend/app/api/explanations.py -> backend/
BACKEND_DIR = Path(__file__).resolve().parents[2]
ARTIFACTS_DIR = BACKEND_DIR / "storage" / "analyses"


def decode_png_base64(value: str) -> bytes:
    """Decode a Base64-encoded PNG returned by the Grad-CAM generator."""
    try:
        return base64.b64decode(value, validate=True)
    except (ValueError, TypeError) as exc:
        raise ValueError("Grad-CAM returned invalid Base64 image data.") from exc


@router.post("/")
async def explain_prediction(
    image: UploadFile = File(...),
    analysis_id: int = Form(...),
    class_index: int | None = Form(None),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Generate and persist Grad-CAM artifacts for an existing analysis.

    The analysis must belong to the authenticated user.
    """

    if image.content_type and image.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=415,
            detail="Upload a JPEG, PNG, or WebP image.",
        )

    # Verify ownership before processing the image.
    analysis = db.scalar(
        select(Analysis).where(
            Analysis.id == analysis_id,
            Analysis.user_id == current_user.id,
        )
    )

    if analysis is None:
        raise HTTPException(
            status_code=404,
            detail="Analysis not found.",
        )

    image_bytes = await image.read()

    if not image_bytes:
        raise HTTPException(
            status_code=400,
            detail="The uploaded image is empty.",
        )

    if len(image_bytes) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail="Image exceeds the 10 MB upload limit.",
        )

    try:
        original_image = Image.open(
            io.BytesIO(image_bytes)
        ).convert("RGB")

        input_tensor = preprocess_image(original_image)

        result = generate_gradcam(
            model=model,
            input_tensor=input_tensor,
            original_image=original_image,
            class_index=class_index,
            device=device,
        )

    except (UnidentifiedImageError, OSError):
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is not a valid image.",
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        )

    predicted_index = result["predicted_class_index"]
    explained_index = result["explained_class_index"]

    result["predicted_class"] = class_mapping[predicted_index]
    result["explained_class"] = class_mapping[explained_index]
    result["model_name"] = model_config["model_name"]
    result["analysis_id"] = analysis.id

    result["disclaimer"] = (
        "Grad-CAM is an experimental visualization of model "
        "sensitivity, not a confirmed disease localization "
        "or clinical diagnosis."
    )

    # Preserve the existing response fields while persisting the PNGs.
    encoded_images = {
        "heatmap": result.get("heatmap_png_base64"),
        "overlay": result.get("overlay_png_base64"),
    }

    if not all(encoded_images.values()):
        raise HTTPException(
            status_code=500,
            detail="Grad-CAM did not return all required images.",
        )

    artifact_directory = ARTIFACTS_DIR / str(analysis.id)
    saved_paths = []

    try:
        artifact_directory.mkdir(parents=True, exist_ok=True)

        for artifact_type, encoded_image in encoded_images.items():
            png_bytes = decode_png_base64(encoded_image)

            # Validate that the decoded data is actually an image.
            with Image.open(io.BytesIO(png_bytes)) as generated_image:
                if generated_image.format != "PNG":
                    raise ValueError(
                        f"The generated {artifact_type} is not a PNG."
                    )
                generated_image.verify()

            filename = f"{artifact_type}_{uuid.uuid4().hex}.png"
            absolute_path = artifact_directory / filename

            with absolute_path.open("xb") as output_file:
                output_file.write(png_bytes)

            saved_paths.append(absolute_path)

            relative_path = absolute_path.relative_to(BACKEND_DIR).as_posix()

            db.add(
                AnalysisArtifact(
                    analysis_id=analysis.id,
                    artifact_type=artifact_type,
                    file_path=relative_path,
                )
            )

        db.commit()

    except (OSError, ValueError, TypeError) as exc:
        db.rollback()

        for saved_path in saved_paths:
            try:
                saved_path.unlink(missing_ok=True)
            except OSError:
                logger.exception(
                    "Could not clean up failed artifact: %s",
                    saved_path,
                )

        logger.exception("Failed to persist Grad-CAM artifacts.")
        raise HTTPException(
            status_code=500,
            detail="Failed to save the Grad-CAM artifacts.",
        ) from exc

    except Exception as exc:
        db.rollback()

        for saved_path in saved_paths:
            try:
                saved_path.unlink(missing_ok=True)
            except OSError:
                logger.exception(
                    "Could not clean up failed artifact: %s",
                    saved_path,
                )

        logger.exception("Database error while saving Grad-CAM artifacts.")
        raise HTTPException(
            status_code=500,
            detail="Failed to save the Grad-CAM artifacts.",
        ) from exc

    return result