from dataclasses import dataclass
from typing import Any, Optional
from io import BytesIO
import base64
import contextlib
from PIL import Image


@dataclass
class RawInput:
    """Dataclass to store raw input for a function."""

    image_input: Optional[Image.Image]
    text_input: Optional[str]


@dataclass
class IOExamples:
    """Dataclass to store input-output examples for a function."""

    description: str
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

    # if width or height < 28, resize it keeping aspect ratio
    if img.width < 28 or img.height < 28:
        # make smallest dimension 28
        new_width = 28
        new_height = 28
        if img.width < img.height:
            new_height = int((28 / img.width) * img.height)
        else:
            new_width = int((28 / img.height) * img.width)
        img = img.resize((new_width, new_height))


    img.save(buffer, format="JPEG")
    return base64.b64encode(buffer.getvalue()).decode()


def base642img(base64_str):
    imgdata = base64.b64decode(base64_str)
    return Image.open(BytesIO(imgdata))


def eval_extracted_code(code):
    try:
        locs = {'__name__':'__main__'}
        with contextlib.redirect_stdout(None):
            exec(code, locs, locs)
        return locs["answer"]
    except Exception as e:
        return "None"
