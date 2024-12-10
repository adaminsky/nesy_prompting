from vllm import LLM, SamplingParams


def llm_simulate(code: str, model: LLM):
    prompt_content = []
    prompt_content.append(
        {
            "type": "text",
            "text": f"Simulate the following code and output the value of the variable `answer` at the end after 'FINAL ANSWER'. Code:\n{code}",
        }
    )
    prompt = [{"role": "user", "content": prompt_content}]
    sampling_params = SamplingParams(temperature=0.0, max_tokens=5000, top_p=1.0)
    output = (
        model.chat(prompt, sampling_params=sampling_params, use_tqdm=False)[0]
        .outputs[0]
        .text
    )
    return output


def python_eval(code: str):
    try:
        locs = {}
        exec(code, locs, locs)
        return locs["answer"]
    except Exception as e:
        return "None"

