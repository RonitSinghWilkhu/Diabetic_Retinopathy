import json
from pathlib import Path

import torch
from torchvision.models import convnext_tiny

import pickle


PROJECT_ROOT = Path(__file__).resolve().parents[3]
MODEL_DIR = PROJECT_ROOT / "artifacts" / "model"


def load_model():
    """
    Load the trained ConvNeXt-Tiny model and its configuration.

    The checkpoint must be a tensors-only state_dict (see
    kaggle/export_deployment_checkpoint.py). It is always loaded with
    weights_only=True; the restricted loader is never disabled.

    Returns:
        model: Model in evaluation mode.
        class_mapping: Dictionary mapping class indices to class names.
        model_config: Model configuration dictionary.
        device: CPU or CUDA device.
    """
    checkpoint_path = MODEL_DIR / "dr_classifier.pth"
    model_config_path = MODEL_DIR / "model_config.json"
    class_mapping_path = MODEL_DIR / "class_mapping.json"

    required_files = [
        checkpoint_path,
        model_config_path,
        class_mapping_path,
    ]

    missing_files = [str(path) for path in required_files if not path.exists()]

    if missing_files:
        raise FileNotFoundError(
            "Required model files are missing:\n" + "\n".join(missing_files)
        )

    with model_config_path.open("r", encoding="utf-8") as file:
        model_config = json.load(file)

    with class_mapping_path.open("r", encoding="utf-8") as file:
        raw_mapping = json.load(file)

    class_mapping = {
        int(index): class_name for index, class_name in raw_mapping.items()
    }

    num_classes = int(model_config.get("num_classes", len(class_mapping)))

    if len(class_mapping) != num_classes:
        raise ValueError(
            f"Configuration specifies {num_classes} classes, "
            f"but mapping contains {len(class_mapping)}."
        )

    if sorted(class_mapping) != list(range(num_classes)):
        raise ValueError(
            "Class mapping indices must be contiguous from 0 to "
            f"{num_classes - 1}; got {sorted(class_mapping)}."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # The checkpoint was trained using torchvision ConvNeXt-Tiny.
    model = convnext_tiny(weights=None)

    # Replace the classification head for the five DR classes.
    in_features = model.classifier[2].in_features
    model.classifier[2] = torch.nn.Linear(in_features, num_classes)

    # weights_only=True is kept on purpose: it blocks arbitrary code
    # execution from a pickle file.
    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )
    except pickle.UnpicklingError as exc:
        raise RuntimeError(
            "The checkpoint contains objects that PyTorch's restricted "
            "weights-only loader does not allow (for example NumPy "
            "metadata). Re-export a tensors-only state_dict from the "
            "trusted training environment with "
            "kaggle/export_deployment_checkpoint.py instead of disabling "
            "weights_only."
        ) from exc

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    head_key = "classifier.2.weight"
    if head_key in state_dict and state_dict[head_key].shape[0] != num_classes:
        raise ValueError(
            f"Checkpoint head has {state_dict[head_key].shape[0]} outputs, "
            f"but the configuration expects {num_classes}."
        )

    # strict=True (default): any missing/unexpected key raises an error.
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return model, class_mapping, model_config, device