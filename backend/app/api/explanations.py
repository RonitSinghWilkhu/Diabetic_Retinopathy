import io

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError

from app.api.auth import get_current_user
from app.models import User

from app.ml.explainability import generate_gradcam
from app.ml.model_loader import load_model
from app.ml.preprocessing import preprocess_image


router = APIRouter(tags=["Explanations"])

# Load the model once when this module is imported.
model, class_mapping, model_config, device = load_model()


MAX_FILE_SIZE = 10 * 1024 * 1024

ALLOWED_CONTENT_TYPES = {
    "image/jpeg",
    "image/png",
    "image/webp",
}


@router.post("/")
async def explain_prediction(
    image: UploadFile = File(...),
    class_index: int | None = None,
    current_user: User = Depends(get_current_user),
):
    """
    Generate a Grad-CAM explanation for an uploaded retinal image.

    Optionally supply class_index to explain a particular class.
    Otherwise, explain the model's predicted class.
    """

    if (
        image.content_type
        and image.content_type not in ALLOWED_CONTENT_TYPES
    ):
        raise HTTPException(
            status_code=415,
            detail="Upload a JPEG, PNG, or WebP image.",
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

        # Use the same preprocessing function as prediction.
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

    result["disclaimer"] = (
        "Grad-CAM is an experimental visualization of model "
        "sensitivity, not a confirmed disease localization "
        "or clinical diagnosis."
    )

    return result