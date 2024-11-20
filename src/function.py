from typing import Union, Any, Optional
from collections.abc import Callable
from .symbol_mapping import (
    prompting_mapper,
    prompting_mapper_structure,
    function_mapper,
)
from .utils import IOExamples, RawInput


class Function:
    def __init__(
        self,
        model,
        processor,
        fn: Union[Callable[..., Any], str, IOExamples],
        args: Optional[Union[IOExamples, str]],
    ):
        self.model = model
        self.processor = processor
        self.fn = fn
        self.args = args
        self.symbol_mapper = prompting_mapper
        self.fn_mapper = function_mapper
        self.symbol_mapper_structure = prompting_mapper_structure
        self.learned_mapper = None

    def apply_two_stage(self, args: Union[list[Any], RawInput], return_symbols=False):
        """NeSy apply function which first converts raw input to symbolic form
        and then applies a symbolic function."""

        if isinstance(args, RawInput):
            # extract the symbols from the raw input
            symbols = self.symbol_mapper(
                args, self.args, self.fn, self.model, self.processor
            )
            # print("Symbols:", symbols)

        else:
            symbols = ", ".join(args)

        # apply the function to the symbols
        output = self.fn_mapper(args, symbols, self.fn, self.model, self.processor)
        # print("Output:", output)

        if return_symbols:
            return output, symbols

        return output

    # def get_arg_symbols(self, i):
    #     if self.args is None:
    #         return None

    #     assert i < len(
    #         self.args
    #     ), f"Index {i} is out of bounds for {len(self.args)} arguments"
    #     return self.args[i]

    # def nesy_v1(self, *inputs):
    #     """Rather than symbolic input, we now have raw (non-symbolic) inputs
    #     corresponding to each of the original inputs to the function."""

    #     assert self.args is None or len(inputs) == len(
    #         self.args
    #     ), f"Expected {len(self.args)} inputs, got {len(inputs)}"

    #     symbols = []
    #     for i in range(len(self.args)):
    #         symbols.append(
    #             self.symbol_mapper(
    #                 inputs[i], self.get_arg_symbols(i), self.model, self.processor
    #             )
    #         )

    #     return self.fn(*symbols), *symbols

    def nesy_v2(self, input):
        """Rather than symbolic input, we now have a single non-symbolic input
        containing all the required information."""

        if isinstance(self.args, IOExamples):
            # Explicit symbols
            pass
        elif isinstance(self.args, list):
            # Implicit symbols
            pass
        else:
            # Inferred symbols
            pass

        if isinstance(self.fn, IOExamples):
            # Inferred function
            pass
        elif isinstance(self.fn, str):
            # Implicit function
            pass
        else:
            # Explicit function
            pass

        symbols = []
        for i in range(len(self.args)):
            symbols.append(
                self.symbol_mapper_structure(
                    input,
                    len(self.args),
                    i,
                    self.get_arg_symbols(i),
                    self.model,
                    self.processor,
                )
            )

        return self.fn(*symbols), *symbols
