from .utils import IOExamples, RawInput, img2base64
from typing import Union, Optional, Callable, Any
from vllm import LLM, SamplingParams
import json
import inspect
import re
import logging
logger = logging.getLogger(__name__)


def code_mapper(
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
            "text": "Analyze the provided input and output self-contained Python code at the end enclosed in a markdown code block such that executing the code stores the answer in the variable 'answer'. Do not use a main() function.",
        }
    )

    # Adding symbol description to the prompt
    if isinstance(symbols, IOExamples):
        prompt_content.append(
            {
                "type": "text",
                "text": f" Based on the input, define {symbols.description}. For example:",
            }
        )
        for i, (input, output) in enumerate(zip(symbols.inputs, symbols.outputs)):
            symbol_str = ", ".join([json.dumps(o).encode('utf-8').decode('unicode_escape') for o in output])
            if input.text_input is not None and input.image_input is None:
                prompt_content.append(
                    {
                        "type": "text",
                        "text": f"\nExample input {i + 1}: {input.text_input}\nExtracted symbols {i + 1}: {symbol_str}",
                    }
                )
            elif input.text_input is None and input.image_input is not None:
                prompt_content.extend(
                    [
                        {"type": "text", "text": f"\nExample input {i + 1}: "},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{img2base64(input.image_input)}"
                            },
                        },
                        {"type": "text", "text": f"\nExtracted symbols {i + 1}: {symbol_str}"},
                    ]
                )
            else:
                prompt_content.extend(
                    [
                        {"type": "text", "text": f"\nExample input {i + 1}: "},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{img2base64(input.image_input)}"
                            },
                        },
                        {
                            "type": "text",
                            "text": f", {input.text_input}\nExtracted symbols {i + 1}: {symbol_str}",
                        },
                    ]
                )
    elif isinstance(symbols, str):
        prompt_content.append(
            {"type": "text", "text": f" Based on the input, define {symbols}."}
        )
    else:
        pass
        # prompt_content.append(
        #     {
        #         "type": "text",
        #         "text": " First, process the input to understand its contents.",
        #     }
        # )

    # Adding function description to the prompt
    if isinstance(fn_desc, IOExamples):
        prompt_content.append(
            {
                "type": "text",
                "text": f"\nTo derive the final answer, write Python code. The following are some examples of the expected answer:\n",
            }
        )
        for i, (input, output) in enumerate(zip(fn_desc.inputs, fn_desc.outputs)):
            if input.text_input is not None and input.image_input is None:
                prompt_content.append(
                    {
                        "type": "text",
                        "text": f"Example {i + 1}: {input.text_input}\nAnswer: {output[0]}\n",
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
                        {"type": "text", "text": f"\nAnswer: {output[0]}\n"},
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
                            "text": f", {input.text_input}\nAnswer: {output[0]}\n",
                        },
                    ]
                )
    elif isinstance(fn_desc, str):
        prompt_content.append({"type": "text", "text": f"\nTo derive the final answer, write a Python function to {fn_desc}."})
    else:
        prompt_content.append(
            {"type": "text", "text": f"\nTo derive the final answer, call the following function:\n{inspect.getsource(fn_desc)}Include the above function in the code block and assume that any provided methods are already implemented."},
        )

    prompt_content.append(
        {"type": "text", "text": "\nThe input is: "},
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
    sampling_params = SamplingParams(temperature=0.0, max_tokens=5000, top_p=1.0)
    print(prompt)
    output = (
        model.chat(prompt, sampling_params=sampling_params, use_tqdm=False)[0]
        .outputs[0]
        .text
    )

    # Extracting the final answer from the output after "FINAL ANSWER:", "**FINAL ANSWER:**", or "*FINAL ANSWER:*"
    try:
        # json_str = re.findall(r"\{\s*\"answer\"(?:.|\s)*?\}", output)[-1]
        # return json.loads(json_str)["answer"], output, prompt_content
        if "\\[ \\boxed{" in output:
            ans_str = re.findall(r"\[ \\boxed{(.*)}", output, re.DOTALL)[-1]

        if "```python" in output:
            ans_str = re.findall(r"```python(.*?)```", output, re.DOTALL)[-1]
        elif "```" in output:
            ans_str = re.findall(r"```(.*?)```", output, re.DOTALL)[-1]

        return ans_str.strip(), output, prompt_content
    except Exception:
        return "None", output, prompt_content


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
            "text": "Analyze the provided input and think through the answer step-by-step. Once the final answer is found, write it at the end after \"FINAL ANSWER:\".",
        }
    )

    # Adding symbol description to the prompt
    if isinstance(symbols, IOExamples):
        prompt_content.append(
            {
                "type": "text",
                "text": f" Based on the input, define {symbols.description}. For example:",
            }
        )
        for i, (input, output) in enumerate(zip(symbols.inputs, symbols.outputs)):
            symbol_str = ", ".join([json.dumps(o).encode('utf-8').decode('unicode_escape') for o in output])
            if input.text_input is not None and input.image_input is None:
                prompt_content.append(
                    {
                        "type": "text",
                        "text": f"\nExample input {i + 1}: {input.text_input}\nExtracted symbols {i + 1}: {symbol_str}",
                    }
                )
            elif input.text_input is None and input.image_input is not None:
                prompt_content.extend(
                    [
                        {"type": "text", "text": f"\nExample input {i + 1}: "},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{img2base64(input.image_input)}"
                            },
                        },
                        {"type": "text", "text": f"\nExtracted symbols {i + 1}: {symbol_str}"},
                    ]
                )
            else:
                prompt_content.extend(
                    [
                        {"type": "text", "text": f"\nExample input {i + 1}: "},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{img2base64(input.image_input)}"
                            },
                        },
                        {
                            "type": "text",
                            "text": f", {input.text_input}\nExtracted symbols {i + 1}: {symbol_str}",
                        },
                    ]
                )
    elif isinstance(symbols, str):
        prompt_content.append(
            {"type": "text", "text": f" Based on the input, define {symbols}."}
        )
    else:
        pass
        # prompt_content.append(
        #     {
        #         "type": "text",
        #         "text": " First, process the input to understand its contents.",
        #     }
        # )

    # Adding function description to the prompt
    if isinstance(fn_desc, IOExamples):
        prompt_content.append(
            {
                "type": "text",
                "text": f"\nThe following are some examples of the expected answer:\n",
            }
        )
        for i, (input, output) in enumerate(zip(fn_desc.inputs, fn_desc.outputs)):
            if input.text_input is not None and input.image_input is None:
                prompt_content.append(
                    {
                        "type": "text",
                        "text": f"Example {i + 1}: {input.text_input}\nAnswer: {output[0]}\n",
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
                        {"type": "text", "text": f"\nAnswer: {output[0]}\n"},
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
                            "text": f", {input.text_input}\nAnswer: {output[0]}\n",
                        },
                    ]
                )
            # prompt_content.append({"type": "text", "text": f"\nWrite a Python function to compute the final answer."})
    elif isinstance(fn_desc, str):
        prompt_content.append({"type": "text", "text": f"\nTo derive the final answer write and then simulate Python code to {fn_desc}. Write out all intermediate steps to get the answer."})
        # prompt_content.append({"type": "text", "text": f"\nTo derive the final answer, write a Python function to {fn_desc}."})
    else:
        prompt_content.append(
            {"type": "text", "text": f"\nTo derive the final answer, simulate the following Python function:\n{inspect.getsource(fn_desc)}Write out all intermediate steps in simulating the program to get the answer."}
        )
        # prompt_content.append(
        #     {"type": "text", "text": f"\nTo derive the final answer, we will call the following function:\n{inspect.getsource(fn_desc)}"}
        # )

    prompt_content.append(
        {"type": "text", "text": "\nThe input is: "},
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
    # prompt_content.append(
    #     {"type": "text", "text": ". IMPORTANT: Output the final answer directly like {\"answer\": <answer>} using JSON syntax."},
    # )

    prompt = [{"role": "user", "content": prompt_content}]
    sampling_params = SamplingParams(temperature=0.0, max_tokens=5000, top_p=1.0)
    output = (
        model.chat(prompt, sampling_params=sampling_params, use_tqdm=False)[0]
        .outputs[0]
        .text
    )

    # Extracting the final answer from the output after "FINAL ANSWER:", "**FINAL ANSWER:**", or "*FINAL ANSWER:*"
    try:
        # json_str = re.findall(r"\{\s*\"answer\"(?:.|\s)*?\}", output)[-1]
        # return json.loads(json_str)["answer"], output, prompt_content
        if "\\[ \\boxed{" in output:
            ans_str = re.findall(r"\[ \\boxed{(.*)}", output, re.DOTALL)[-1]
        elif "**FINAL ANSWER:**" in output:
            ans_str = re.findall(r"\*\*FINAL ANSWER:\*\*(.*)(?:<|$)", output, re.DOTALL)[-1]
        elif "*FINAL ANSWER:*" in output:
            ans_str = re.findall(r"\*FINAL ANSWER:\*(.*)(?:<|$)", output, re.DOTALL)[-1]
        elif "**Answer:**" in output:
            ans_str = re.findall(r"\*\*Answer:\*\*(.*)(?:<|$)", output, re.DOTALL)[-1]
        elif "*Answer:*" in output:
            ans_str = re.findall(r"\*Answer:(.*)(?:<|$)", output, re.DOTALL)[-1]
        elif "**Answer**:" in output:
            ans_str = re.findall(r"\*\*Answer\*\*:(.*)(?:<|$)", output, re.DOTALL)[-1]
        elif "*Answer*:" in output:
            ans_str = re.findall(r"\*Answer\*:(.*)(?:<|$)", output, re.DOTALL)[-1]
        else:
            ans_str = re.findall(r"FINAL ANSWER:(.*)(?:<|$)", output, re.DOTALL)[-1]

        if "```" in ans_str:
            ans_str = re.findall(r"```(.*)```", ans_str, re.DOTALL)[-1]
        return ans_str.strip(), output, prompt_content
    except Exception:
        return "None", output, prompt_content


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


class LLMNet:
    def __init__(self, model: LLM, input_desc: str, output_desc: str) -> str:
        self.model = model
        self.input_desc = input_desc
        self.output_desc = output_desc

    def forward(self, input: RawInput) -> str:
        prompt_content = [
            {"type": "text", "text": "Analyze the following input: "},
        ]
        if input.text_input is not None and input.image_input is None:
            prompt_content.append({"type": "text", "text": input.text_input})
        elif input.text_input is None and input.image_input is not None:
            prompt_content.extend(
                [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img2base64(input.image_input)}"}},
                ]
            )
        else:
            prompt_content.extend(
                [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img2base64(input.image_input)}"}},
                    {"type": "text", "text": input.text_input},
                ]
            )
        prompt_content.append({"type": "text", "text": f" Output just {self.output_desc} after 'FINAL ANSWER:'."})

        prompt = [{"role": "user", "content": prompt_content}]

        sampling_params = SamplingParams(temperature=0.0, max_tokens=5000, top_p=1.0)
        output = (
            self.model.chat(prompt, sampling_params=sampling_params, use_tqdm=False)[0]
            .outputs[0]
            .text
        )
        print("out:", output)

        extra_args = [re.DOTALL]
        try:
            if "\\[ \\boxed{" in output:
                res = re.findall(r"\[ \\boxed{(.*)}", output, *extra_args)[-1]
                pred = res.strip()
            elif "**FINAL ANSWER:**" in output:
                res = re.findall(r"\*\*FINAL ANS.*:\*\*(.*)(?:<|$)", output, *extra_args)[-1]
                pred = res.strip()
            elif "*FINAL ANSWER:*" in output:
                res = re.findall(r"\*FINAL ANS.*:(.*)(?:<|$)", output, *extra_args)[-1]
                pred = res.strip()
            elif "**Final Answer:**" in output:
                res = re.findall(r"\*\*Final Ans.*:\*\*(.*)(?:<|$)", output, *extra_args)[-1]
                pred = res.strip()
            elif "*Final Answer:*" in output:
                res = re.findall(r"\*Final Ans.*:(.*)(?:<|$)", output, *extra_args)[-1]
                pred = res.strip()
            elif "Final answer:" in output:
                res = re.findall(r"Final ans.*:(.*)(?:<|$)", output, *extra_args)[-1]
                pred = res.strip()
            elif "*Final answer:*" in output:
                res = re.findall(r"\*Final ans.*:(.*)(?:<|$)", output, *extra_args)[-1]
                pred = res.strip()
            elif "**Final answer:**" in output:
                res = re.findall(r"\*\*Final ans.*:\*\*(.*)(?:<|$)", output, *extra_args)[-1]
                pred = res.strip()
            elif "**Answer:**" in output:
                res = re.findall(r"\*\*Answer:\*\*(.*)(?:<|$)", output, *extra_args)[-1]
                pred = res.strip()
            elif "*Answer:*" in output:
                res = re.findall(r"\*Answer:\*(.*)(?:<|$)", output, *extra_args)[-1]
                pred = res.strip()
            elif "**Answer**:" in output:
                res = re.findall(r"\*\*Answer\*\*:(.*)(?:<|$)", output, *extra_args)[-1]
                pred = res.strip()
            elif "*Answer*:" in output:
                res = re.findall(r"\*Answer\*:(.*)(?:<|$)", output, *extra_args)[-1]
                pred = res.strip()
            else:
                # print("here", re.findall(r"FINAL ANS.*:(.*)(?:<|$)", output, *extra_args))
                res = re.findall(r"FINAL ANS.*:(.*)(?:<|$)", output, *extra_args)[-1]
                pred = res.strip()

            if "```" in pred:
                pred = re.sub(r"```", "", pred).strip()
            if "<|eot_id|>" in pred:
                pred = re.sub(r"<\|eot_id\|>", "", pred).strip()
            return pred
        except Exception:
            return "None"
