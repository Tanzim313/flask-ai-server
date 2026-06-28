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
    """Convert a PIL image to a float32 numpy array in [0, 255] range."""
    image_array = np.asarray(image, dtype=np.float32)

    if image_array.ndim == 2:
        image_array = np.stack([image_array] * 3, axis=-1)

    if image_array.shape[-1] == 4:
        image_array = image_array[..., :3]

    return image_array


def is_likely_plant_image(image: Image.Image, min_plant_pixel_pct: float = 10.0) -> bool:
    """Check if the image contains a significant amount of plant-like colors (green, yellow, brown, orange, red)."""
    try:
        hsv_img = image.convert('HSV')
        hsv_array = np.array(hsv_img)
        h = hsv_array[:, :, 0]
        s = hsv_array[:, :, 1]
        v = hsv_array[:, :, 2]

        # Scale H to degrees [0, 360]
        h_deg = (h.astype(np.float32) * 360.0) / 255.0

        # Plant colors: green, yellow, brown, orange, red
        is_green = (h_deg >= 35) & (h_deg <= 160) & (s > 20) & (v > 20)
        is_yellow_brown = (h_deg >= 10) & (h_deg < 35) & (s > 25) & (v > 20)
        is_red_dark_brown = ((h_deg >= 340) | (h_deg < 10)) & (s > 25) & (v > 20)

        plant_pixels = is_green | is_yellow_brown | is_red_dark_brown
        plant_pct = (np.sum(plant_pixels) / plant_pixels.size) * 100.0

        return plant_pct >= min_plant_pixel_pct
    except Exception:
        return True  # Fallback to True on error


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

    Returns a numpy array of shape (1, height, width, 3) with values in [0, 255].
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
