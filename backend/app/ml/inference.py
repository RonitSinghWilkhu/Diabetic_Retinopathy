import torch

from app.ml.model_loader import load_model
from app.ml.preprocessing import load_and_preprocess_image


# Load the model once when this module is imported.
model, class_mapping, model_config, device = load_model()


def predict_image(image_bytes: bytes) -> dict:
    """
    Run diabetic retinopathy classification on uploaded image bytes.

    Returns the predicted class, confidence, and class probabilities.
    """
    input_tensor = load_and_preprocess_image(image_bytes)
    input_tensor = input_tensor.to(device)

    with torch.inference_mode():
        logits = model(input_tensor)
        probabilities = torch.softmax(logits, dim=1)[0]

    predicted_index = int(torch.argmax(probabilities).item())

    probability_dict = {
        class_mapping[index]: float(probabilities[index].item())
        for index in sorted(class_mapping)
    }

    return {
        "predicted_class": class_mapping[predicted_index],
        "predicted_class_index": predicted_index,
        "confidence": float(probabilities[predicted_index].item()),
        "probabilities": probability_dict,
        "model_name": model_config.get("model_name", "convnext_tiny"),
    }