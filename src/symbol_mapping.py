from .utils import IOExamples
from typing import Union, Optional, Callable, Any
import json
import inspect


def function_mapper(
    symbols: str, fn_desc: Union[IOExamples, str, Callable[..., Any]], model, processor
):
    image_inputs = []
    fn_desc_str = ""
    if isinstance(fn_desc, IOExamples):
        # Implicit function
        # fn_desc_str += "Here are some input-output examples to define the function:"
        for input, output in zip(fn_desc.inputs, fn_desc.outputs):
            fn_desc_str += f"Passing the input symbols from the image <|image|> to the function results in an output of {output[0]}."
            image_inputs.append(input)
            fn_desc_str += "\n"
    elif isinstance(fn_desc, str):
        # Implicit function
        fn_desc_str += f"'{fn_desc}'."
    else:
        # Explicit function
        fn_desc_str += inspect.getsource(fn_desc)

    prompt = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": f'Apply the following function to the given input and give the result. The function is:\n{fn_desc_str}\nGive the result of this function on the input {symbols} in the format {{"result": <result>}}.',
                },
            ],
        }
    ]
    input_text = (
        processor.apply_chat_template(prompt, add_generation_prompt=True) + '{"result":'
    )
    # print(input_text)
    inputs = processor(
        image_inputs if len(image_inputs) > 0 else None,
        input_text,
        add_special_tokens=False,
        return_tensors="pt",
    ).to(model.device)
    model_output = model.generate(
        **inputs, max_new_tokens=500, do_sample=False, temperature=None, top_p=None
    )
    model_output = (
        processor.decode(model_output[0])[len(input_text) :]
        .replace("<|eot_id|>", "")
        .strip()
    )
    # inputs = processor(
    #     image_inputs if len(image_inputs) > 0 else None, input_text, add_special_tokens=False
    # )
    # params = SamplingParams(max_tokens=500, temperature=0.0, stop_token_ids=[128009, 128001])
    # if len(image_inputs) > 0:
    #     model_output = model.generate({"prompt_token_ids": inputs["input_ids"][0], "multi_modal_data": {"image": image_inputs}}, params, use_tqdm=False)[0].outputs[0].text
    # else:
    #     model_output = model.generate({"prompt_token_ids": input["input_ids"][0]}, params, use_tqdm=False)[0].outputs[0].text

    # print(model_output)
    # get the json from the output
    try:
        # json_str = re.findall(r"\{.*\}", model_output)[-1]
        # return json.loads(json_str)["result"]
        return json.loads('{"result":' + model_output)["result"]
    except Exception:
        # print(model_output)
        return None


def prompting_mapper(
    fn_input,
    symbol_desc: Optional[Union[IOExamples, str]],
    fn_desc: Union[IOExamples, str, Callable[..., Any]],
    model,
    processor,
):
    image_inputs = []
    symbol_extraction_prompt = "Extract symbols from the provided image."
    if isinstance(symbol_desc, IOExamples):
        # Explicit symbols
        symbol_extraction_prompt += " Examples:"
        for input, output in zip(symbol_desc.inputs, symbol_desc.outputs):
            symbol_extraction_prompt += f"\nFor the image <|image|>, extract the following {len(output)} symbols: {', '.join([str(o) for o in output])}."
            image_inputs.append(input)
        symbol_extraction_prompt += "\n"
    elif isinstance(symbol_desc, str):
        # Implicit symbols
        symbol_extraction_prompt += f" The symbols are {symbol_desc}."
    else:
        # Inferred symbols
        symbol_extraction_prompt += (
            " The symbols will be used in the function described by:\n"
        )
        if isinstance(fn_desc, IOExamples):
            for input, output in zip(fn_desc.inputs, fn_desc.outputs):
                symbol_extraction_prompt += f"\nFor the image <|image|>, the function output is: {', '.join([str(o) for o in output])}."
                image_inputs.append(input)
            symbol_extraction_prompt += "\n"
        elif isinstance(fn_desc, str):
            symbol_extraction_prompt += f"'{fn_desc}'."
        else:
            symbol_extraction_prompt += inspect.getsource(fn_desc)

    symbol_extraction_prompt += '\nOutput the symbols for the image <|image|> in the following format: {"symbols": [symbol1, symbol2, ...]}.'

    prompt = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": symbol_extraction_prompt,
                },
            ],
        }
    ]
    input_text = processor.apply_chat_template(prompt, add_generation_prompt=True) + "{"
    inputs = processor(
        image_inputs + [fn_input],
        input_text,
        add_special_tokens=False,
        return_tensors="pt",
    ).to(model.device)
    model_output = model.generate(
        **inputs, max_new_tokens=50, do_sample=False, temperature=None, top_p=None
    )
    model_output = (
        processor.decode(model_output[0])[len(input_text) :]
        .strip()
        .replace("<|eot_id|>", "")
    )
    # inputs = processor(
    #     image_inputs + [fn_input], input_text, add_special_tokens=False
    # )
    # params = SamplingParams(max_tokens=500, temperature=0.0, stop_token_ids=[128009, 128001])
    # model_output = model.generate({"prompt_token_ids": inputs["input_ids"][0], "multi_modal_data": {"image": image_inputs + [fn_input]}}, params, use_tqdm=False)[0].outputs[0].text

    # print(input_text)
    # print(model_output)
    try:
        # json_str = re.findall(r"\{.*\}", model_output)[-1]
        # return json.loads(json_str)["symbols"]
        return json.loads("{" + model_output)["symbols"]
    except Exception:
        return None
    # if model_output not in [str(s) for s in allowed_symbols]:
    #     print(model_output)
    #     return allowed_symbols[0]
    # return allowed_symbols[[str(s) for s in allowed_symbols].index(model_output)]


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
    ).to(model.device)
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
