from vllm import LLM, SamplingParams
import re
import contextlib
import timeout_decorator
import multiprocessing


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


@timeout_decorator.timeout(0.5)
def my_exec(code, locs):
    exec(code, locs, locs)


def run_with_timeout(code, timeout):
    def target(queue):
        locs = {}  # Standard dictionary for local variables
        locs['__name__'] = '__main__'
        try:
            exec(code, locs, locs)  # Execute the code with locs as locals
            queue.put(locs.get("answer", None))  # Retrieve the value of "answer"
        except Exception as e:
            queue.put(f"Error: {e}")

    queue = multiprocessing.Queue()  # Queue for communication
    process = multiprocessing.Process(target=target, args=(queue,))
    process.start()
    process.join(timeout)

    if process.is_alive():
        process.terminate()
        process.join()
        print("Code execution timed out.")
        return None

    # Retrieve result from the queue
    if not queue.empty():
        result = queue.get()
        if isinstance(result, str) and result.startswith("Error:"):
            print(result)  # Print error if there is one
        else:
            return result  # Return the value of "answer"
    return None


def python_eval(code: str):
    try:
        if "main():" in code:
            code = code.replace("if __name__ == '__main__':\n    main()", "    return answer\nif __name__ == '__main__':\n    answer = main()")
            code = code.replace("if __name__ == \"__main__\":\n    main()", "    return answer\nif __name__ == '__main__':\n    answer = main()")
            code = "answer = None\n" + code
        # print(code)
        with contextlib.redirect_stdout(None):
            # my_exec(code, locs)
            return run_with_timeout(code, 0.5)
        # return locs["answer"]
    except Exception as e:
        return "None"

