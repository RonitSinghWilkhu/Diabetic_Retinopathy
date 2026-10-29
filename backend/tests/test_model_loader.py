from pathlib import Path
import json

import torch
from torchvision.models import convnext_tiny


PROJECT_ROOT = Path(__file__).resolve().parents[3]
MODEL_DIR = PROJECT_ROOT / "artifacts" / "model"


def load_model():
    config_path = MODEL_DIR / "model_config.json"
    mapping_path = MODEL_DIR / "class_mapping.json"

    if not config_path.exists():
        raise FileNotFoundError(f"Model config not found: {config_path}")

    if not mapping_path.exists():
        raise FileNotFoundError(f"Class mapping not found: {mapping_path}")

    with config_path.open("r", encoding="utf-8") as f:
        config = json.load(f)

    with mapping_path.open("r", encoding="utf-8") as f:
        raw_mapping = json.load(f)

    class_mapping = {
        int(key): value for key, value in raw_mapping.items()
    }

    num_classes = int(config["num_classes"])
    image_size = int(config["image_size"])

    expected_classes = [
        "No_DR",
        "Mild",
        "Moderate",
        "Severe",
        "Proliferative_DR",
    ]

    if num_classes != 5:
        raise ValueError(f"Expected 5 classes, got {num_classes}")

    if image_size != 224:
        raise ValueError(f"Expected image size 224, got {image_size}")

    if [class_mapping.get(i) for i in range(5)] != expected_classes:
        raise ValueError(
            f"Unexpected class mapping: {class_mapping}"
        )

    if config.get("weights_format") != "state_dict_only":
        raise ValueError(
            "model_config.json must specify weights_format='state_dict_only'"
        )

    checkpoint_path = MODEL_DIR / config["checkpoint_filename"]

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found: {checkpoint_path}"
        )

    model = convnext_tiny(weights=None)

    in_features = model.classifier[2].in_features
    model.classifier[2] = torch.nn.Linear(in_features, num_classes)

    state_dict = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )

    if not isinstance(state_dict, dict):
        raise TypeError("Checkpoint must contain a state dictionary.")

    if not all(
        isinstance(value, torch.Tensor)
        for value in state_dict.values()
    ):
        raise ValueError("Checkpoint contains non-tensor values.")

    model.load_state_dict(state_dict, strict=True)

    if model.classifier[2].out_features != num_classes:
        raise ValueError("Model output layer does not match class count.")

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = model.to(device)
    model.eval()

    return model, class_mapping, config, device