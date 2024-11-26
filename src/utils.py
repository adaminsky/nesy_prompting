from dataclasses import dataclass
from typing import Any, Optional
from io import BytesIO
import base64
from PIL import Image


@dataclass
class RawInput:
    """Dataclass to store raw input for a function."""

    image_input: Optional[Image.Image]
    text_input: Optional[str]


@dataclass
class IOExamples:
    """Dataclass to store input-output examples for a function."""

    inputs: list[RawInput]
    outputs: list[list[Any]]


def img2base64(img):
    buffer = BytesIO()
    if img.mode != "RGB":
        img = img.convert("RGB")
    img.save(buffer, format="JPEG")
    return base64.b64encode(buffer.getvalue()).decode()
