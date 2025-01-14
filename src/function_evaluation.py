from vllm import LLM, SamplingParams
import timeout_decorator
import multiprocessing
from io import StringIO
from contextlib import redirect_stdout


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


def run_with_timeout(code, timeout, code_context=None):
    def target(queue):
        locs = {}  # Standard dictionary for local variables
        locs["__name__"] = "__main__"
        try:
            if code_context:
                exec(code_context, locs, locs)
        except Exception as e:
            pass

        try:
            # store stdout in a variable
            f = StringIO()
            with redirect_stdout(f):
                exec(code, locs, locs)  # Execute the code with locs as locals
            if "answer" in locs:
                queue.put(locs.get("answer", None))  # Retrieve the value of "answer"
            else:
                queue.put(f.getvalue())  # Retrieve the output
        except Exception as e:
            queue.put(f"Error: {e}")

    queue = multiprocessing.Queue()  # Queue for communication
    process = multiprocessing.Process(target=target, args=(queue,))
    process.start()
    process.join(timeout)

    if process.is_alive():
        process.terminate()
        process.join()
        return None, "Error: Code execution timed out"

    # Retrieve result from the queue
    errors = None
    if not queue.empty():
        result = queue.get()
        if isinstance(result, str) and result.startswith("Error:"):
            # print(result)  # Print error if there is one
            errors = result
        else:
            return result, None  # Return the value of "answer"
    return None, errors


def python_eval(code: str, code_context: str=None):
    try:
        if "if __name__ == '__main__'" in code:
            code = code.replace(
                "if __name__ == '__main__':\n    main()",
                "    return answer\nif __name__ == '__main__':\n    answer = main()",
            )
            code = code.replace(
                'if __name__ == "__main__":\n    main()',
                "    return answer\nif __name__ == '__main__':\n    answer = main()",
            )
            code = "answer = None\n" + code
        if "main():" in code:
            code += "\nmain()"
        # with contextlib.redirect_stdout(None):
        # my_exec(code, locs)
        # run the code with a timeout of 5 seconds and store any output or errors
        return run_with_timeout(code, 5, code_context)
        # return locs["answer"]
    except Exception as e:
        print("Exception:", e)
        return "None"
