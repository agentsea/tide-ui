import base64
from io import BytesIO
from typing import List

from PIL import Image


def calculate_normalized_distance(
    pred: List[int], target: List[int], resolution: List[int]
) -> float:
    """
    Calculate the distance between a prediction and target point, normalized by the diagonal of the image.
    """
    pred_x, pred_y = pred
    target_x, target_y = target
    res_x, res_y = resolution

    return (
        ((pred_x - target_x) ** 2 + (pred_y - target_y) ** 2)
        / ((res_x**2) + (res_y**2))
    ) ** 0.5


def image_to_b64(img: Image.Image, image_format="PNG") -> str:
    """Converts a PIL Image to a base64-encoded string with MIME type included.

    Args:
        img (Image.Image): The PIL Image object to convert.
        image_format (str): The format to use when saving the image (e.g., 'PNG', 'JPEG').

    Returns:
        str: A base64-encoded string of the image with MIME type.
    """
    buffer = BytesIO()
    img.save(buffer, format=image_format)
    image_data = buffer.getvalue()
    buffer.close()

    mime_type = f"image/{image_format.lower()}"
    base64_encoded_data = base64.b64encode(image_data).decode("utf-8")
    return f"data:{mime_type};base64,{base64_encoded_data}"
