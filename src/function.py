from typing import Union, Any, Optional
import logging
import re
from vllm import LLM
from collections.abc import Callable
from .symbol_mapping import (
    code_mapper,
    function_mapper,
    single_prompt_mapper,
)
from .function_evaluation import llm_simulate, python_eval
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
        self.code_mapper = code_mapper
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
        self, args: Union[list[Any], RawInput], simulate_code=False, return_code=False, print_log=False
    ):
        """NeSy apply function which first converts raw input to symbolic form
        and then applies a symbolic function."""

        # extract the symbols from the raw input
        code, _, _ = self.code_mapper(args, self.args, self.fn, self.model)
        if print_log:
            logger.info("Code output: %s", code)

        # evaluate the code
        if simulate_code:
            output = llm_simulate(code, self.model)
        else:
            output = python_eval(code)

        if print_log:
            logger.info("Output: %s", output)

        if return_code:
            return output, code

        return output
