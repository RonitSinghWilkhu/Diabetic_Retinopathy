import logging
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.api.auth import get_current_user
from app.database import get_db
from app.models import Analysis, User
from app.ml.inference import predict_image
from app.schemas import PredictionResponse

router = APIRouter()
logger = logging.getLogger(__name__)

ALLOWED_CONTENT_TYPES = {
    "image/jpeg",
    "image/png",
    "image/webp",
}

MAX_IMAGE_SIZE = 10 * 1024 * 1024  # 10 MB

# Store uploaded images under backend/storage/analyses.
STORAGE_DIR = Path(__file__).resolve().parents[2] / "storage" / "analyses"


@router.post("/", response_model=PredictionResponse)
async def predict_retinopathy(
    image: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    if image.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=415,
            detail="Unsupported image format. Upload a JPEG, PNG, or WebP image.",
        )

    image_bytes = await image.read(MAX_IMAGE_SIZE + 1)

    if not image_bytes:
        raise HTTPException(
            status_code=400,
            detail="The uploaded image is empty.",
        )

    if len(image_bytes) > MAX_IMAGE_SIZE:
        raise HTTPException(
            status_code=413,
            detail="Image exceeds the 10 MB upload limit.",
        )

    # Validate the actual file contents, not just the declared content type.
    try:
        from io import BytesIO

        with Image.open(BytesIO(image_bytes)) as uploaded_image:
            uploaded_image.verify()

    except (UnidentifiedImageError, OSError, SyntaxError) as exc:
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is not a valid image.",
        ) from exc

    except Exception as exc:
        logger.exception("Unexpected image validation error")
        raise HTTPException(
            status_code=400,
            detail="The uploaded file could not be validated.",
        ) from exc

    try:
        result = predict_image(image_bytes)

    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    except Exception as exc:
        logger.exception("Prediction inference failed")
        raise HTTPException(
            status_code=500,
            detail="Prediction failed. Check the backend logs.",
        ) from exc

    # Derive the storage extension from the validated image format.
    try:
        from io import BytesIO

        with Image.open(BytesIO(image_bytes)) as uploaded_image:
            image_format = uploaded_image.format

        extensions = {
            "JPEG": ".jpg",
            "PNG": ".png",
            "WEBP": ".webp",
        }

        extension = extensions.get(image_format)

        if extension is None:
            raise HTTPException(
                status_code=415,
                detail="Unsupported image format.",
            )

    except HTTPException:
        raise

    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is not a valid image.",
        ) from exc

    # Generate a server-controlled filename; never use the supplied
    # filename as a filesystem path.
    file_id = uuid.uuid4().hex
    relative_image_path = Path("storage") / "analyses" / f"{file_id}{extension}"
    absolute_image_path = STORAGE_DIR / f"{file_id}{extension}"

    try:
        STORAGE_DIR.mkdir(parents=True, exist_ok=True)
        absolute_image_path.write_bytes(image_bytes)

        analysis = Analysis(
            user_id=current_user.id,
            original_filename=Path(image.filename or "upload").name[:255],
            image_path=relative_image_path.as_posix(),
            predicted_class=result["predicted_class"],
            predicted_class_index=result["predicted_class_index"],
            confidence=result["confidence"],
            probabilities=result["probabilities"],
            model_name=result["model_name"],
        )

        db.add(analysis)
        db.commit()
        db.refresh(analysis)

    except (OSError, SQLAlchemyError) as exc:
        db.rollback()

        # Avoid leaving an unreferenced file if persistence fails.
        try:
            absolute_image_path.unlink(missing_ok=True)
        except OSError:
            logger.exception("Could not remove orphaned uploaded image")

        logger.exception("Could not persist prediction analysis")
        raise HTTPException(
            status_code=500,
            detail="Prediction succeeded, but the analysis could not be saved.",
        ) from exc

    result["analysis_id"] = analysis.id
    return PredictionResponse(**result)