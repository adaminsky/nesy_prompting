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

    # # if image larger than 600x600, resize it
    # if img.width > 500 or img.height > 500:
    #     # make largest dimension 600
    #     new_width = 500
    #     new_height = 500
    #     if img.width > img.height:
    #         new_height = int((500 / img.width) * img.height)
    #     else:
    #         new_width = int((500 / img.height) * img.width)
    #     img = img.resize((new_width, new_height))

    img.save(buffer, format="JPEG")
    return base64.b64encode(buffer.getvalue()).decode()
