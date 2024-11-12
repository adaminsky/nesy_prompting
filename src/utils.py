from dataclasses import dataclass
from typing import Any


@dataclass
class IOExamples:
    """Dataclass to store input-output examples for a function."""

    inputs: list[str]
    outputs: list[list[Any]]


@dataclass
class RawInput:
    """Dataclass to store raw input for a function."""

    raw_input: Any
