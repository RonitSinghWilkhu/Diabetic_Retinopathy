from io import BytesIO

from PIL import Image, UnidentifiedImageError
from torchvision import transforms


IMAGE_SIZE = 224

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


eval_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD,
    ),
])


def preprocess_image(image: Image.Image):
    """
    Convert a PIL image into a normalized model input tensor.

    Returns a tensor with shape [1, 3, 224, 224].
    """
    if not isinstance(image, Image.Image):
        raise TypeError("Expected a PIL Image.")

    image = image.convert("RGB")
    tensor = eval_transform(image)

    return tensor.unsqueeze(0)


def load_and_preprocess_image(image_bytes: bytes):
    """
    Decode uploaded image bytes and preprocess them.

    Raises:
        ValueError: If the uploaded content is not a valid image.
    """
    if not image_bytes:
        raise ValueError("The uploaded image is empty.")

    try:
        with Image.open(BytesIO(image_bytes)) as image:
            image.load()
            return preprocess_image(image)

    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ValueError(
            "The uploaded file could not be decoded as a valid image."
        ) from exc