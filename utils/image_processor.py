import io
from typing import Optional, Tuple, Union

import numpy as np
import requests
from PIL import Image, ImageOps

ImageSource = Union[str, bytes, bytearray, Image.Image]


def load_image(source: ImageSource) -> Image.Image:
    """Load an image from a file path, URL, raw bytes, or PIL Image."""
    if isinstance(source, Image.Image):
        return source

    if isinstance(source, (bytes, bytearray)):
        return Image.open(io.BytesIO(source))

    if isinstance(source, str):
        if source.startswith(('http://', 'https://')):
            response = requests.get(source, timeout=10)
            response.raise_for_status()
            return Image.open(io.BytesIO(response.content))

        return Image.open(source)

    raise TypeError(f'Unsupported image source type: {type(source)}')


def normalize_image(image: Image.Image) -> np.ndarray:
    """Convert a PIL image to a normalized float32 numpy array."""
    image_array = np.asarray(image, dtype=np.float32)

    if image_array.ndim == 2:
        image_array = np.stack([image_array] * 3, axis=-1)

    if image_array.shape[-1] == 4:
        image_array = image_array[..., :3]

    return image_array / 255.0


def resize_image(
    image: Image.Image,
    target_size: Tuple[int, int] = (224, 224),
    keep_aspect_ratio: bool = True,
    fill_color: Tuple[int, int, int] = (0, 0, 0)
) -> Image.Image:
    """Resize the image for the model input while optionally preserving aspect ratio."""
    if keep_aspect_ratio:
        image = ImageOps.contain(image, target_size, Image.Resampling.LANCZOS)
        background = Image.new('RGB', target_size, fill_color)
        x = (target_size[0] - image.width) // 2
        y = (target_size[1] - image.height) // 2
        background.paste(image, (x, y))
        return background

    return image.resize(target_size, Image.Resampling.LANCZOS)


def preprocess_image(
    source: ImageSource,
    target_size: Tuple[int, int] = (224, 224),
    keep_aspect_ratio: bool = True
) -> Optional[np.ndarray]:
    """Load and preprocess an image for the plant disease model.

    Returns a numpy array of shape (1, height, width, 3) with values in [0, 1].
    """
    try:
        image = load_image(source)
        image = ImageOps.exif_transpose(image)
        image = image.convert('RGB')
        image = resize_image(
            image,
            target_size=target_size,
            keep_aspect_ratio=keep_aspect_ratio
        )
        image_array = normalize_image(image)
        return np.expand_dims(image_array, axis=0)
    except Exception as exc:
        print(f'Error in image preprocessing: {exc}')
        return None
