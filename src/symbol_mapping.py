def prompting_mapper(input, allowed_symbols, model, processor):
    # Simplest strategy: just prompt for which symbol is present in the image
    # out of a set of options
    prompt = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {
                    "type": "text",
                    "text": f"In the input image, which of the following symbols is present: {', '.join([str(s) for s in allowed_symbols])}? Output only the symbol.",
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
    )
    if model_output not in [str(s) for s in allowed_symbols]:
        print(model_output)
        return allowed_symbols[0]
    return allowed_symbols[[str(s) for s in allowed_symbols].index(model_output)]


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
