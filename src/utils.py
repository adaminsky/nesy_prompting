from dataclasses import dataclass
from typing import Any, Optional
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
