from .utils import IOExamples, RawInput, img2base64
from typing import Union, Optional, Callable, Any
from vllm import LLM, SamplingParams
import json
import inspect
import re
import logging
logger = logging.getLogger(__name__)



def single_prompt_mapper(
    raw_input: RawInput,
    symbols: Optional[Union[IOExamples, str]],
    fn_desc: Union[IOExamples, str, Callable[..., Any]],
    model: LLM,
):
    # Adding the input to the prompt
    prompt_content = []
    prompt_content.append(
        {
            "type": "text",
            "text": "Given an input, compute the result following the steps below. ",
        }
    )

    # Adding symbol description to the prompt
    if isinstance(symbols, IOExamples):
        prompt_content.append(
            {
                "type": "text",
                "text": "First, convert the input to symbolic form. The following are examples of symbol extraction on other inputs:",
            }
        )
        for i, (input, output) in enumerate(zip(symbols.inputs, symbols.outputs)):
            symbol_str = ", ".join([json.dumps(o) for o in output])
            if input.text_input is not None and input.image_input is None:
                prompt_content.append(
                    {
                        "type": "text",
                        "text": f"\nExample {i + 1}: {input.text_input}\nSymbols: {symbol_str}",
                    }
                )
            elif input.text_input is None and input.image_input is not None:
                prompt_content.extend(
                    [
                        {"type": "text", "text": f"\nExample {i + 1}: "},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{img2base64(input.image_input)}"
                            },
                        },
                        {"type": "text", "text": f"\nSymbols: {symbol_str}"},
                    ]
                )
            else:
                prompt_content.extend(
                    [
                        {"type": "text", "text": f"\nExample {i + 1}: "},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{img2base64(input.image_input)}"
                            },
                        },
                        {
                            "type": "text",
                            "text": f", {input.text_input}\nSymbols: {symbol_str}",
                        },
                    ]
                )
    elif isinstance(symbols, str):
        prompt_content.append(
            {"type": "text", "text": f"First, convert the input to symbolic form. The symbols extracted should be {symbols}."}
        )
    else:
        prompt_content.append(
            {
                "type": "text",
                "text": "First, convert the input to symbolic form.",
            }
        )

    # Adding function description to the prompt
    if isinstance(fn_desc, IOExamples):
        prompt_content.append(
            {
                "type": "text",
                "text": "\nAfter extracting symbols, apply the following function. Output the result in the format: {\"result\": <result>}. The following are input-output examples for the function:\n",
            }
        )
        for i, (input, output) in enumerate(zip(fn_desc.inputs, fn_desc.outputs)):
            if input.text_input is not None and input.image_input is None:
                prompt_content.append(
                    {
                        "type": "text",
                        "text": f"Example {i + 1}: {input.text_input}\nOutput: {{\"result\": {output[0]}}}\n",
                    }
                )
            elif input.text_input is None and input.image_input is not None:
                prompt_content.extend(
                    [
                        {"type": "text", "text": f"Example {i + 1}: "},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{img2base64(input.image_input)}"
                            },
                        },
                        {"type": "text", "text": f"\nOutput: {{\"result\": {output[0]}}}\n"},
                    ]
                )
            else:
                prompt_content.extend(
                    [
                        {"type": "text", "text": f"Example {i + 1}: "},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{img2base64(input.image_input)}"
                            },
                        },
                        {
                            "type": "text",
                            "text": f", {input.text_input}\nOutput: {{\"result\": {output[0]}}}\n",
                        },
                    ]
                )
    elif isinstance(fn_desc, str):
        prompt_content.append({"type": "text", "text": f"\nAfter extracting symbols, apply the following function. The function is: {fn_desc}. Output the result in the format: {{\"result\": <result>}}."})
    else:
        prompt_content.append(
            {"type": "text", "text": f"\nAfter extracting symbols, apply the following function. The function is:\n{inspect.getsource(fn_desc)}Output the result in the format: {{\"result\": <result>}}."}
        )

    prompt_content.append(
        {"type": "text", "text": "Input: "},
    )
    if raw_input.text_input is not None and raw_input.image_input is None:
        prompt_content.append(
            {"type": "text", "text": f"{raw_input.text_input}"}
        )
    elif raw_input.text_input is None and raw_input.image_input is not None:
        prompt_content.extend(
            [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{img2base64(raw_input.image_input)}"
                    },
                },
            ]
        )
    else:
        prompt_content.extend(
            [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{img2base64(raw_input.image_input)}"
                    },
                },
                {"type": "text", "text": f", {raw_input.text_input}"},
            ]
        )

    prompt = [{"role": "user", "content": prompt_content}]
    sampling_params = SamplingParams(temperature=0.0, max_tokens=2500, top_p=1.0)
    output = (
        model.chat(prompt, sampling_params=sampling_params, use_tqdm=False)[0]
        .outputs[0]
        .text
    )

    try:
        json_str = re.findall(r"\{\s*\"result\"(?:.|\s)*?\}", output)[-1]
        return json.loads(json_str)["result"], output, prompt_content
    except Exception:
        return None, output, prompt_content


# TODO: Currently this doesn't take into account any information about the symbols
def function_mapper(
    raw_input: RawInput,
    symbols: str,
    fn_desc: Union[IOExamples, str, Callable[..., Any]],
    model: LLM,
):
    prompt_content = [
        {
            "type": "text",
            "text": f'Evaluate the following function on the symbols {symbols} and output the result in the following JSON format: {{"result": <result>}}.',
        }
    ]
    if isinstance(fn_desc, IOExamples):
        # Inferred function
        prompt_content.append(
            {
                "type": "text",
                "text": "Here are some input-output examples to define the function:",
            }
        )
        for input, output in zip(fn_desc.inputs, fn_desc.outputs):
            if input.text_input is not None and input.image_input is None:
                prompt_content.append(
                    {
                        "type": "text",
                        "text": f'Input: {input.text_input}\nOutput: {{"result": {output[0]}}}',
                    }
                )
            elif input.text_input is None and input.image_input is not None:
                prompt_content.extend(
                    [
                        {"type": "text", "text": "Input image:"},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{img2base64(input.image_input)}"
                            },
                        },
                        {"type": "text", "text": f'Output: {{"result": {output[0]}}}'},
                    ]
                )
            else:
                prompt_content.extend(
                    [
                        {"type": "text", "text": "Input image:"},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{img2base64(input.image_input)}"
                            },
                        },
                        {
                            "type": "text",
                            "text": f'Input text: {input.text_input}\nOutput: {{"result": {output[0]}}}',
                        },
                    ]
                )
    elif isinstance(fn_desc, str):
        # Implicit function
        prompt_content.append({"type": "text", "text": f"The function is: {fn_desc}."})
    else:
        # Explicit function
        prompt_content.append(
            {"type": "text", "text": f"The function is:\n{inspect.getsource(fn_desc)}."}
        )

    prompt = [{"role": "user", "content": prompt_content}]

    sampling_params = SamplingParams(temperature=0.0, max_tokens=2500, top_p=1.0)
    output = (
        model.chat(prompt, sampling_params=sampling_params, use_tqdm=False)[0]
        .outputs[0]
        .text
    )

    try:
        json_str = re.findall(r"\{\s*\"result\"(?:.|\s)*?\}", output)[-1]
        return json.loads(json_str)["result"]
    except Exception:
        return None


def prompting_mapper(
    fn_input: RawInput,
    symbol_desc: Optional[Union[IOExamples, str]],
    fn_desc: Union[IOExamples, str, Callable[..., Any]],
    model: LLM,
):
    prompt_content = [
        {
            "type": "text",
            "text": 'Extract symbols from the following input and output their symbolic value in the following JSON format: {"symbols": [symbol1, symbol2, ...]}',
        }
    ]
    if fn_input.text_input is not None and fn_input.image_input is None:
        prompt_content.append({"type": "text", "text": f"Input: {fn_input.text_input}"})
    elif fn_input.text_input is None and fn_input.image_input is not None:
        prompt_content.extend(
            [
                {"type": "text", "text": "Input image:"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{img2base64(fn_input.image_input)}"
                    },
                },
            ]
        )
    else:
        prompt_content.extend(
            [
                {"type": "text", "text": "Input image:"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{img2base64(fn_input.image_input)}"
                    },
                },
                {"type": "text", "text": f"Input text: {fn_input.text_input}"},
            ]
        )

    if isinstance(symbol_desc, IOExamples):
        # Explicit symbols
        prompt_content.append(
            {
                "type": "text",
                "text": "The following are some examples showing how to extract symbols:",
            }
        )
        for input, output in zip(symbol_desc.inputs, symbol_desc.outputs):
            symbol_str = ", ".join([json.dumps(o) for o in output])
            if input.text_input is not None and input.image_input is None:
                prompt_content.append(
                    {
                        "type": "text",
                        "text": f"Input: {input.text_input}\nSymbols: {symbol_str}",
                    }
                )
            elif input.text_input is None and input.image_input is not None:
                prompt_content.extend(
                    [
                        {"type": "text", "text": "Example image:"},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{img2base64(input.image_input)}"
                            },
                        },
                        {"type": "text", "text": f"Symbols: {symbol_str}"},
                    ]
                )
            else:
                prompt_content.extend(
                    [
                        {"type": "text", "text": "Example input:"},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{img2base64(input.image_input)}"
                            },
                        },
                        {
                            "type": "text",
                            "text": f"{input.text_input}\nSymbols: {symbol_str}",
                        },
                    ]
                )

    elif isinstance(symbol_desc, str):
        # Implicit symbols
        prompt_content.append(
            {"type": "text", "text": f"The symbols are {symbol_desc}."}
        )
    else:
        # Inferred symbols
        prompt_content.append(
            {
                "type": "text",
                "text": "The symbols will be used in the function described by the following examples:",
            }
        )
        if isinstance(fn_desc, IOExamples):
            for input, output in zip(fn_desc.inputs, fn_desc.outputs):
                if input.text_input is not None and input.image_input is None:
                    prompt_content.append(
                        {
                            "type": "text",
                            "text": f"Input: {input.text_input}\nOutput: {', '.join([str(o) for o in output])}",
                        }
                    )
                elif input.text_input is None and input.image_input is not None:
                    prompt_content.extend(
                        [
                            {"type": "text", "text": "Example image:"},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{img2base64(input.image_input)}"
                                },
                            },
                            {
                                "type": "text",
                                "text": f"Output: {', '.join([str(o) for o in output])}",
                            },
                        ]
                    )
                else:
                    prompt_content.extend(
                        [
                            {"type": "text", "text": "Example input:"},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{img2base64(input.image_input)}"
                                },
                            },
                            {
                                "type": "text",
                                "text": f"{input.text_input}\nOutput: {', '.join([str(o) for o in output])}",
                            },
                        ]
                    )

        elif isinstance(fn_desc, str):
            prompt_content.append(
                {"type": "text", "text": f"The function is: {fn_desc}."}
            )
        else:
            prompt_content.append(
                {
                    "type": "text",
                    "text": f"The function is:\n{inspect.getsource(fn_desc)}.",
                }
            )

    prompt = [{"role": "user", "content": prompt_content}]

    sampling_params = SamplingParams(temperature=0.0, max_tokens=2500, top_p=1.0)
    output = (
        model.chat(prompt, sampling_params=sampling_params, use_tqdm=False)[0]
        .outputs[0]
        .text
    )

    try:
        json_str = re.findall(r'{.*?"symbols".*}', output, re.DOTALL)[-1]
        return json.loads(json_str)["symbols"]
    except Exception:
        # print(e)
        return None


def prompting_mapper_structure(
    input, total, obj_num, allowed_symbols, model, processor
):
    # Simplest prompting strategy: just ask for which symbol is present in the image
    obj_str = [
        "first",
        "second",
        "third",
        "fourth",
        "fifth",
        "sixth",
        "seventh",
        "eighth",
        "ninth",
        "tenth",
    ]
    prompt = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {
                    "type": "text",
                    "text": f"The input image contains {total} objects. The {obj_str[obj_num]} object is one of the following symbols: {', '.join([str(s) for s in allowed_symbols])}. Which symbol is {obj_str[obj_num]} object? Output only the symbol.",
                },
            ],
        }
    ]
    input_text = (
        processor.apply_chat_template(prompt, add_generation_prompt=True) + "\nSymbol:"
    )
    inputs = processor(
        input, input_text, add_special_tokens=False, return_tensors="pt"
    )  # .to(model.device)
    model_output = model.generate(
        **inputs, max_new_tokens=10, do_sample=False, temperature=None, top_p=None
    )
    model_output = (
        processor.decode(model_output[0])[len(input_text) :]
        .strip()
        .replace("<|eot_id|>", "")
        .strip(".")
    )
    if model_output not in [str(s) for s in allowed_symbols]:
        print(model_output)
        return allowed_symbols[0]
    return allowed_symbols[[str(s) for s in allowed_symbols].index(model_output)]
