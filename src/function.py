from typing import Union, Any, Optional
import logging
import re
from vllm import LLM
from collections.abc import Callable
from .symbol_mapping import (
    prompting_mapper,
    function_mapper,
    single_prompt_mapper,
)
from .utils import IOExamples, RawInput

logger = logging.getLogger(__name__)

def replace_base64_urls(text):
    pattern = r"\{'url': 'data:image/jpeg;base64.*?(//|==)"
    return re.sub(pattern, 'OMITTED', text, flags=re.DOTALL)

class Function:
    def __init__(
        self,
        model: LLM,
        fn: Union[Callable[..., Any], str, IOExamples],
        args: Optional[Union[IOExamples, str]],
    ):
        self.model = model
        self.fn = fn
        self.args = args
        self.symbol_mapper = prompting_mapper
        self.fn_mapper = function_mapper
        self.single_prompt_mapper = single_prompt_mapper
        self.learned_mapper = None

    def apply(
        self, args: Union[list[Any], RawInput], return_symbols=False, print_log=False
    ):
        """NeSy apply function"""

        # apply the function to the raw input
        output, log, prompt_content = self.single_prompt_mapper(args, self.args, self.fn, self.model)
        prompt_content = replace_base64_urls(str(prompt_content))
        if print_log:
            logger.info("Prompt: %s", prompt_content)
            logger.info("Output: %s", log)

        if return_symbols:
            return output, log, prompt_content

        return output, None, None

    def apply_two_stage(
        self, args: Union[list[Any], RawInput], return_symbols=False, print_log=False
    ):
        """NeSy apply function which first converts raw input to symbolic form
        and then applies a symbolic function."""

        if isinstance(args, RawInput):
            # extract the symbols from the raw input
            symbols = self.symbol_mapper(args, self.args, self.fn, self.model)
            if print_log:
                logger.info("Symbols: %s", symbols)

        else:
            symbols = ", ".join(args)

        # apply the function to the symbols
        output = self.fn_mapper(args, symbols, self.fn, self.model)
        if print_log:
            logger.info("Output: %s", output)

        if return_symbols:
            return output, symbols

        return output
