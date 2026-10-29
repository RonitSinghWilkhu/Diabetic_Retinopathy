from fastapi import APIRouter, File, HTTPException, UploadFile

from app.ml.inference import predict_image
from app.schemas import PredictionResponse

router = APIRouter()

ALLOWED_CONTENT_TYPES = {
    "image/jpeg",
    "image/png",
    "image/webp",
}

MAX_IMAGE_SIZE = 10 * 1024 * 1024  # 10 MB


@router.post("/", response_model=PredictionResponse)
async def predict_retinopathy(image: UploadFile = File(...)):
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

    try:
        result = predict_image(image_bytes)
        return PredictionResponse(**result)

    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    except Exception as exc:
        # Keep internal model errors out of the response.
        raise HTTPException(
            status_code=500,
            detail="Prediction failed. Check the backend logs.",
        ) from exc