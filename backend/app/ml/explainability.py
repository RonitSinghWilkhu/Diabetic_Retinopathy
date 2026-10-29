from __future__ import annotations

import base64
import io

import numpy as np
import torch
import torch.nn.functional as F

from PIL import Image
from matplotlib import colormaps


def generate_gradcam(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    original_image: Image.Image,
    class_index: int | None = None,
    device: torch.device | None = None,
) -> dict:
    """
    Generate a Grad-CAM heatmap for a ConvNeXt-Tiny classifier.

    Args:
        model: Loaded ConvNeXt-Tiny model.
        input_tensor: Preprocessed image tensor [1, 3, 224, 224].
        original_image: Original retinal image as a PIL image.
        class_index: Class to explain. Defaults to predicted class.
        device: Model execution device.

    Returns:
        Prediction metadata and base64-encoded PNG images.
    """

    if input_tensor.ndim != 4 or input_tensor.shape[0] != 1:
        raise ValueError(
            "Expected a single image tensor with shape [1, 3, H, W]."
        )

    if not hasattr(model, "features"):
        raise ValueError("Expected a model with a 'features' module.")

    if device is None:
        device = next(model.parameters()).device

    model.eval()

    # ConvNeXt-Tiny's final feature stage produces spatial feature maps.
    target_layer = model.features[-1]

    captured = {}

    def capture_activations(module, inputs, output):
        captured["activations"] = output

        # Capture gradients flowing back into the feature maps.
        output.register_hook(
            lambda gradient: captured.__setitem__(
                "gradients", gradient
            )
        )

    handle = target_layer.register_forward_hook(capture_activations)

    try:
        # Grad-CAM needs gradients, so do not use inference_mode().
        model.zero_grad(set_to_none=True)

        x = input_tensor.to(device)

        with torch.enable_grad():
            logits = model(x)

            if logits.ndim != 2 or logits.shape[0] != 1:
                raise ValueError(
                    f"Unexpected model output shape: {tuple(logits.shape)}"
                )

            probabilities = torch.softmax(logits, dim=1)
            predicted_index = int(
                probabilities.argmax(dim=1).item()
            )

            selected_index = (
                predicted_index
                if class_index is None
                else int(class_index)
            )

            if not 0 <= selected_index < logits.shape[1]:
                raise ValueError(
                    f"class_index must be between 0 and "
                    f"{logits.shape[1] - 1}."
                )

            # Differentiate the selected class score.
            score = logits[0, selected_index]
            score.backward()

        activations = captured.get("activations")
        gradients = captured.get("gradients")

        if activations is None or gradients is None:
            raise RuntimeError(
                "Could not capture activations and gradients."
            )

        # Average gradients spatially to obtain channel weights.
        weights = gradients.mean(dim=(2, 3), keepdim=True)

        # Weighted sum of activation maps.
        cam = (weights * activations).sum(dim=1, keepdim=True)

        # Retain positive contributions.
        cam = F.relu(cam)

        # Resize the heatmap to the input tensor dimensions.
        cam = F.interpolate(
            cam,
            size=input_tensor.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        cam = cam[0, 0].detach().cpu().numpy()

        # Normalize safely, including the all-zero case.
        cam_min = float(cam.min())
        cam_max = float(cam.max())

        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam, dtype=np.float32)

        heatmap = Image.fromarray(
            np.uint8(np.clip(cam, 0, 1) * 255)
        ).resize(
            original_image.size,
            Image.Resampling.BILINEAR,
        )

        # Create a colored heatmap.
        heatmap_array = np.asarray(heatmap, dtype=np.float32) / 255.0
        colored = colormaps["jet"](heatmap_array)[..., :3]
        colored = np.uint8(np.clip(colored, 0, 1) * 255)

        heatmap_image = Image.fromarray(colored, mode="RGB")
        original_rgb = original_image.convert("RGB")

        # Blend the heatmap with the original image.
        overlay = Image.blend(
            original_rgb,
            heatmap_image,
            alpha=0.40,
        )

        def image_to_base64(image: Image.Image) -> str:
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            return base64.b64encode(
                buffer.getvalue()
            ).decode("utf-8")

        return {
            "predicted_class_index": predicted_index,
            "explained_class_index": selected_index,
            "class_score": float(
                probabilities[0, selected_index].detach().item()
            ),
            "probabilities": [
                float(value)
                for value in probabilities[0].detach().cpu().tolist()
            ],
            "heatmap_png_base64": image_to_base64(heatmap_image),
            "overlay_png_base64": image_to_base64(overlay),
            "heatmap_method": "Grad-CAM",
            "target_layer": "features[-1]",
        }

    finally:
        handle.remove()
        model.zero_grad(set_to_none=True)