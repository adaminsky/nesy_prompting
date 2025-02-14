import os
import logging
from src.dataset import MNISTSumKOrigDataset, GSM8KDataset, ChartQADataset, ClevrDataset, HWFDataset, BlocksWorldDataset, BBHDataset, FOLIODataset, LongSortDataset, ListSynthesisDataset, ClutrrDataset, LeafDataset, PathFinder128Dataset, SVHNSumKDataset
from tqdm import tqdm
import argparse
from vllm import LLM, SamplingParams
import re
import json
import numpy as np
import torch
import ast
from transformers import MllamaForConditionalGeneration, AutoProcessor, AutoModelForCausalLM, AutoTokenizer, Qwen2_5_VLForConditionalGeneration
from src.symbol_mapping import LLMNet
from src.function import LLMNesy
from src.utils import IOExamples, RawInput, img2base64, base642img, eval_extracted_code
from src.pddl import eval_solution_files, find_solution
from src.function_evaluation import python_eval
from src.program_gen import ProgramSynthesisSolver, DSLOp
import time
import itertools
import json
import PIL
from PIL import Image, ImageDraw, ImageFont
from openai import OpenAI
from time import sleep

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class OurLLM:
    def __init__(self, model_name):
        self.model_name = model_name
        if "Llama-3.2" in model_name:
            self.model = MllamaForConditionalGeneration.from_pretrained(model_name, torch_dtype="auto", device_map="auto", token="***REMOVED***",)
            self.processor = AutoProcessor.from_pretrained(model_name, token="***REMOVED***")
        elif "Qwen" in model_name:
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_name, torch_dtype="auto", device_map="auto", attn_implementation="flash_attention_2", token="***REMOVED***")
            self.processor = AutoProcessor.from_pretrained(model_name, token="***REMOVED***")

    def chat(self, prompt, sampling_params, use_tqdm):
        # parse prompt content
        imgs = []
        processed_prompt = []
        for j in range(len(prompt)):
            prompt_content = []
            for i in range(len(prompt[j]["content"])):
                if prompt[j]["content"][i]["type"] == "text":
                    prompt_content.append(prompt[j]["content"][i])
                elif prompt[j]["content"][i]["type"] == "image_url":
                    prompt_content.append({"type": "image"})
                    img_base64 = prompt[j]["content"][i]["image_url"]["url"].split(",")[1]
                    imgs.append(base642img(img_base64))
            processed_prompt.append({"role": prompt[j]["role"], "content": prompt_content})

        # prompt = [{"role": "user", "content": prompt_content}]
        print("prompt:", processed_prompt)

        input_text = self.processor.apply_chat_template(processed_prompt, add_generation_prompt=True)
        inputs = self.processor(imgs if len(imgs) > 0 else None, input_text, add_special_tokens=False, return_tensors="pt").to("cuda:0")

        print(self.processor.decode(inputs["input_ids"][0]))
        start = time.time()
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=2500, temperature=0.0, do_sample=False, top_p=1.0)
        elapsed = time.time() - start

        print(f"Tokens per second: {(len(outputs[0][len(inputs['input_ids'][0]):])) / elapsed}")

        output_text = self.processor.decode(outputs[0][len(inputs["input_ids"][0]):][:-1])
        print("output:", output_text)

        class Outputs:
            def __init__(self, outputs):
                self.outputs = outputs

        class Text:
            def __init__(self, text):
                self.text = text

        return [Outputs([Text(output_text)])]


class APIModel:
    def __init__(self, model_name):
        self.model_name = model_name
        if "gemini" in model_name:
            self.client = OpenAI(
                api_key="***REMOVED***",
                base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
            )
        else:
            self.client = OpenAI(api_key="***REMOVED***")

    def chat(self, prompt, sampling_params, use_tqdm):
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=prompt,
            temperature=0,
            max_tokens=2500,
            top_p=1.0
        )
        class Outputs:
            def __init__(self, outputs):
                self.outputs = outputs

        class Text:
            def __init__(self, text):
                self.text = text

        return [Outputs([Text(response.choices[0].message.content)])]



def get_predictions(model: LLM, data, prompt):
    sampling_params = SamplingParams(temperature=0.0, max_tokens=5000, top_p=1.0)
    logs = []
    preds = []
    gt = []
    for i in (pbar := tqdm(range(len(data)))):
        input = data[i][0]
        label = data[i][1]
        gt.append(label)

        prompt = [{"role": "user", "content": prompt(*input)}]

        output = (
            model.chat(prompt, sampling_params=sampling_params, use_tqdm=False)[0]
            .outputs[0]
            .text
        )

        try:
            res = re.findall(r"\{\s*\"result\"(?:.|\s)*?\}", output)[-1]
            pred = json.loads(res)["result"]
        except Exception:
            pred = -100000

        logs.append((output, pred))
        preds.append(pred)
        pbar.set_description(
            f"Acc: {sum([str(gt[i]) == str(preds[i]) for i in range(len(preds))]) / len(preds)}"
        )

    return preds, gt, logs


def get_raw_predictions(model: LLM, data, log=False, equiv = None, instruction=None):
    if equiv is None:
        equiv = lambda x, y: str(x) == str(y)
    sampling_params = SamplingParams(temperature=0.0, max_tokens=10000, top_p=1.0)
    logs = []
    preds = []
    gt = []
    for i in (pbar := tqdm(range(len(data)))):
        input = data[i][0]
        label = data[i][1]
        gt.append(label)

        if instruction is not None:
            prompt = [{"type": "text", "text": f"{instruction} Once the final answer is found, write it at the end after \"FINAL ANSWER:\". The input is: "}]
        else:
            prompt = [{"type": "text", "text": "Analyze the provided input and think through the answer step-by-step. Once the final answer is found, write it at the end after \"FINAL ANSWER:\". The input is: "}]

        # Add the input to the prompt
        if input[1] is not None and input[0] is None:
            prompt.append(
                {"type": "text", "text": input[1]}
            )
        elif input[1] is None and input[0] is not None:
            prompt.extend(
                [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{img2base64(input[0])}"
                        },
                    },
                ]
            )
        else:
            prompt.extend(
                [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{img2base64(input[0])}"
                        },
                    },
                    {"type": "text", "text": f"{input[1]}"},
                ]
            )

        prompt = [{"role": "user", "content": prompt}]
        output = (
            model.chat(prompt, sampling_params=sampling_params, use_tqdm=False)[0]
            .outputs[0]
            .text
        )

        try:
            # res = re.findall(r"\{\s*\"result\"(?:.|\s)*?\}", output)[-1]
            # pred = json.loads(res)["result"]
            if "\\[ \\boxed{" in output:
                res = re.findall(r"\[ \\boxed{(.*)}", output)[-1]
                pred = res.strip()
            elif "**FINAL ANSWER:**" in output:
                res = re.findall(r"\*\*FINAL ANSWER:\*\*(.*)[<$]", output)[-1]
                pred = res.strip()
            elif "*FINAL ANSWER:*" in output:
                res = re.findall(r"\*FINAL ANSWER:(.*)[<$]", output)[-1]
                pred = res.strip()
            elif "**Answer:**" in output:
                res = re.findall(r"\*\*Answer:\*\*(.*)[<$]", output)[-1]
                pred = res.strip()
            elif "*Answer:*" in output:
                res = re.findall(r"\*Answer:(.*)[<$]", output)[-1]
                pred = res.strip()
            elif "**Answer**:" in output:
                res = re.findall(r"\*\*Answer\*\*:(.*)[<$]", output)[-1]
                pred = res.strip()
            elif "*Answer*:" in output:
                res = re.findall(r"\*Answer\*:(.*)[<$]", output)[-1]
                pred = res.strip()
            else:
                res = re.findall(r"FINAL ANSWER:(.*)[<$]", output)[-1]
            pred = res.strip()
        except Exception:
            pred = str(-100000)

        if log:
            logger.info("Output: %s", output)
            logger.info("GT: %s, Pred: %s", repr(gt[-1]), repr(pred))
        logs.append((output, pred))
        preds.append(pred)
        pbar.set_description(
            f"Acc: {sum([equiv(data[i][1], preds[i], i) for i in range(len(preds))]) / len(preds)}"
        )

    return preds, gt, logs


def get_task_predictions(task: LLMNesy, data, log=False, equiv=None):
    preds = []
    logs = []

    if equiv is None:
        equiv = lambda x, y: str(x) == str(y)
    
    for i in (pbar := tqdm(range(len(data)))):
        try:
            print(data[i])
            # data[i] = ((['[Bonita] asked her daughter, [Maryann], if she would like to go to a movie with her on Saturday night', '[Maryann] went to the bakery with her uncle [John] to pick up some bread for lunch'], "('Bonita', 'John')"), 'grandmother') 
            if type(data[i][0]) == str or (type(data[i][0]) == list and type(data[i][0][0]) == str) or (type(data[i][0]) == list and type(data[i][0][0]) == list and type(data[i][0][0][0]) == str):
                out, symbols = task.apply(
                    [RawInput(image_input=None, text_input=d) for d in data[i][0]], return_symbols=True, print_log=log
                )
            else:
                if len(data[i][0]) > 1 and type(data[i][0][1]) == list and len(data[i][0][1]) > 0 and type(data[i][0][1][0]) == dict:
                    out, symbols = task.apply(
                        [data[i][0][0], data[i][0][1]], return_symbols=True, print_log=log
                    )
                else:
                    out, symbols = task.apply(
                        [RawInput(image_input=d, text_input=None) for d in data[i][0]], return_symbols=True, print_log=log
                    )
            logs.append((out, symbols))
            preds.append(out)
            if log:
                logger.info("GT: %s, Pred: %s", repr(data[i][1]), repr(out))

                if len(data[i]) > 2:
                    logger.info("Metadata: %s", repr(data[i][2]))
        except Exception as e:
            print("Error in get predictions:", e)
            # print stacktrace
            import traceback
            traceback.print_exc()
            # get line number
            preds.append(str(-100000))
        pbar.set_description(
            f"Acc: {sum([equiv(data[i][1], preds[i], i) for i in range(len(preds))]) / len(preds)}"
        )
    return preds, logs


def sum2_extract(data, model):
    examples = None
    if args.few_shot:
        examples = IOExamples(
            description=None,
            inputs=[RawInput(image_input=data[101][0][0], text_input=None), RawInput(image_input=data[101][0][1], text_input=None), RawInput(image_input=data[101][0][3], text_input=None), RawInput(image_input=data[101][0][4], text_input=None), RawInput(image_input=data[102][0][0], text_input=None), RawInput(image_input=data[102][0][1], text_input=None), RawInput(image_input=data[102][0][2], text_input=None), ],
            outputs=[[3], [1], [6], [5], [7], [4], [2]],
        )

    mnist_extract = LLMNet(
        model,
        "an image of a handwritten digit",
        "the digit as an integer from 0 to 9",
        examples
    )

    def parse(*imgs):
        digits = []
        for img in imgs:
            digits.append(int(re.sub(r"[^0-9]", "", mnist_extract.forward(img))))
        return digits

    def function(*digits):
        return sum(digits)

    return parse, function


def svhn_extract(data, model):
    examples = None
    if args.few_shot:
        examples = IOExamples(
            description=None,
            inputs=[RawInput(image_input=data[101][0][0], text_input=None), RawInput(image_input=data[101][0][1], text_input=None), RawInput(image_input=data[101][0][2], text_input=None), RawInput(image_input=data[102][0][0], text_input=None), RawInput(image_input=data[102][0][1], text_input=None), RawInput(image_input=data[102][0][2], text_input=None)],
            outputs=[[7], [5], [6], [6], [2], [3]],
        )

    mnist_extract = LLMNet(
        model,
        "an image of a number",
        "the number in the center of the image from 0 to 9. Ignore any other digits in the image and make a best guess if the number is unclear.",
        examples
    )

    def parse(*imgs):
        digits = []
        for img in imgs:
            digits.append(int(re.sub(r"[^0-9]", "", mnist_extract.forward(img))))
        return digits

    def function(*digits):
        return sum(digits)

    return parse, function


def FOLIO_settings(data):
    def evaluate_conclusion(premises_list, conclusion_str):
        from nltk.sem import Expression
        from nltk.inference import ResolutionProver
        read_expr = Expression.fromstring
        
        # Parse premises into expressions
        premises = [read_expr(premise) for premise in premises_list]
        
        # Parse conclusion
        conclusion = read_expr(conclusion_str)
        
        # Attempt to prove the conclusion from the premises
        result = ResolutionProver().prove(conclusion, premises)
        return result
    
    explicit_fn = evaluate_conclusion
    implicit_fn = "evaluate the conclusion given the premises.  Return True, False, or Uncertain. Choose one."
    inferred_fn = IOExamples(
        description=None,
        inputs=[
            RawInput(None, text_input=data[101][0][1]),
            RawInput(None, text_input=data[102][0][1]),
        ],
        outputs=[[data[101][1]], [data[102][1]]],
    )
    explicit_s = IOExamples(
        description="The premise and conclusion each converted to first order logic formulas which share non-logical symbols.",
        inputs=[RawInput(None, data[101][0][1]), 
                RawInput(None, data[102][0][1])],
        outputs=[[data[101][2]],[data[102][2]]],
    )
    implicit_s = "The premise and conclusion each converted to first order logic formulas which share non-logical symbols."
    inferred_s = None
    return (explicit_s, implicit_s, inferred_s, explicit_fn, implicit_fn, inferred_fn)


def chartqa_settings(data):
    def apply_query_to_chart(chart, query):
        """Apply the query to the chart to get the answer as a string."""
        res = query(chart)
        if isinstance(res, bool):
            return "Yes" if res else "No"
        return f"{res:.2f}"

    explicit_fn = apply_query_to_chart
    implicit_fn = "apply a query to the chart to get the answer. The answer should be 'Yes', 'No', or a number as a string."
    inferred_fn = IOExamples(
        description=None,
        inputs=[
            RawInput(image_input=data[101][0][0], text_input=data[101][0][1]),
            RawInput(image_input=data[1001][0][0], text_input=data[1001][0][1]),
        ],
        outputs=[["Yes"], ["0.46"]],
    )

    example_dict = {
        "categories": [
            {"label": "China's economic strength", "color": "green"},
            {"label": "China's military strength", "color": "blue"},
        ],
        "data_points": [
            {"year": 2012, "economic_strength": 59, "military_strength": 28},
            {"year": 2014, "economic_strength": 50, "military_strength": 36},
            {"year": 2016, "economic_strength": 52, "military_strength": 37},
            {"year": 2018, "economic_strength": 58, "military_strength": 29},
        ],
    }
    query_str = """def query(chart):
    values_above_55 = [
        point["economic_strength"]
        for point in chart["data_points"]
        if point["economic_strength"] > 55
    ]

    # Sum the values
    total = sum(values_above_55)

    # Check if the sum is greater than 100
    is_greater_than_100 = total > 100

    # Output the result
    return is_greater_than_100"""

    example_dict2 = {
        "Democratic Party": {
            "Mostly divided": ("light yellow", 39),
            "Mostly united": ("dark yellow", 58),
        },
        "Republican Party": {
            "Mostly divided": ("light yellow", 80),
            "Mostly united": ("dark yellow", 17),
        },
    }
    query_str2 = """def query(chart):
    smallest_val = min([chart[party][category][1] for party in chart for category in chart[party]])
    for color, val in [chart[party][category] for party in chart for category in chart[party]]:
        if val == smallest_val:
            return color"""

    explicit_s = IOExamples(
        description="a chart represented as a Python dict and a query which is a Python function taking a single dict as input representing the chart",
        inputs=[RawInput(*data[101][0]), RawInput(*data[1001][0])],
        outputs=[[example_dict, query_str], [example_dict2, query_str2]],
    )
    implicit_s = "a chart represented as a Python dict and a query which is a Python function taking a single dict as input representing the chart"
    inferred_s = None

    return (explicit_s, implicit_s, inferred_s, explicit_fn, implicit_fn, inferred_fn)


def hwf_settings(data):
    def eval_expression(expr):
        return eval(expr)

    explicit_fn = eval_expression
    implicit_fn = "evaluate an expression to get the answer"
    inferred_fn = IOExamples(
        description=None,
        inputs=[RawInput(*data[101][0]), RawInput(*data[102][0])], outputs=[[0.88], [5]]
    )

    explicit_s = IOExamples(
        description="a mathematical expression as a string",
        inputs=[RawInput(*data[101][0]), RawInput(*data[102][0])],
        outputs=[
            ["2 / 9 * 4"],
            ["6 / 1 - 1"],
        ],
    )
    implicit_s = "a mathematical expression as a string consisting of three numbers between 0 and 9 inclusive and two operators in the set {'+', '-', '*', '/'}"
    inferred_s = None

    return (explicit_s, implicit_s, inferred_s, explicit_fn, implicit_fn, inferred_fn)


def hwf_extract(data, model):
    digit_examples = None
    if args.few_shot:
        digit_examples = IOExamples(
            description=None,
            inputs=[RawInput(image_input=data[102][0][2], text_input=None), RawInput(image_input=data[101][0][0], text_input=None), RawInput(image_input=data[104][0][4], text_input=None), RawInput(image_input=data[101][0][4], text_input=None), RawInput(image_input=data[106][0][0], text_input=None)],
            outputs=[[1], [2], [3], [4], [5]],
        )
    extract_digit = LLMNet(
        model,
        "a handwritten number from 0 to 9",
        "the value of the number as an integer from 0 to 9",
        digit_examples
    )
    operator_examples = None
    if args.few_shot:
        operator_examples = IOExamples(
            description=None,
            inputs=[RawInput(image_input=data[101][0][1], text_input=None), RawInput(image_input=data[101][0][3], text_input=None), RawInput(image_input=data[102][0][3], text_input=None), RawInput(image_input=data[103][0][1], text_input=None)],
            outputs=[['/'], ['*'], ['-'], ['+']],
        )
    extract_operator = LLMNet(
        model,
        "a handwritten arithmetic operator",
        "the operator as a string in the set {'+', '-', '*', '/'} (note that the division operator can look like a line with a dot above and below it and multiplication can look like an 'x')",
        operator_examples
    )

    def parse(*imgs):
        expr = []
        for i, img in enumerate(imgs):
            if i % 2 == 0:
                expr.append(re.sub(r"[^0-9]", "", extract_digit.forward(img)))
            else:
                expr.append(re.sub(r"[^\*/\+\-]", "", extract_operator.forward(img)))
        return [expr]

    def function(expr):
        expr_str = " ".join(expr)
        return eval(expr_str)

    return parse, function


def pathfinder_extract(data, model):
    def create_adjacency(img: PIL.Image):
        num_block_x = 6
        num_block_y = 6
        wx = 128 // num_block_x
        wy = 128 // num_block_y
        def block_coord_to_block_id(x, y):
            return y * num_block_x + x

        adjacency_imgs = []
        adjacency_ids = []
        img = torch.tensor(np.array(img))
        for i, j in itertools.product(range(num_block_x), range(num_block_y)):
            for dx, dy in [(-1, 0), (0, -1), (0, 1), (1, 0)]:
                x, y = i + dx, j + dy
                if x >= 0 and x < num_block_x and y >= 0 and y < num_block_y:
                    source_id = (i * wx, j * wy, i * wx + wx, j * wy + wy)
                    target_id = (x * wx, y * wy, x * wx + wx, y * wy + wy)

                    img_black = img.clone()
                    # draw a rectangular outline of the block
                    img_black[source_id[0], source_id[1]:source_id[3]] = 255
                    img_black[source_id[2], source_id[1]:source_id[3]] = 255
                    img_black[source_id[0]:source_id[2], source_id[1]] = 255
                    img_black[source_id[0]:source_id[2], source_id[3]] = 255

                    img_black[target_id[0], target_id[1]:target_id[3]] = 255
                    img_black[target_id[2], target_id[1]:target_id[3]] = 255
                    img_black[target_id[0]:target_id[2], target_id[1]] = 255
                    img_black[target_id[0]:target_id[2], target_id[3]] = 255

                    # img_black = torch.zeros_like(img)
                    # img_black[source_id[0]:source_id[2], source_id[1]:source_id[3]] = img[source_id[0]:source_id[2], source_id[1]:source_id[3]]
                    # img_black[target_id[0]:target_id[2], target_id[1]:target_id[3]] = img[target_id[0]:target_id[2], target_id[1]:target_id[3]]
                    # img_black = img_black[min(source_id[0], target_id[0]):max(source_id[2], target_id[2]), min(source_id[1], target_id[1]):max(source_id[3], target_id[3])]

                    # if dy == 0: # horizontal
                    #     img_black[wy - 1, :] = 255
                    # else: # vertical
                    #     img_black[:, wx - 1] = 255

                    img_black = PIL.Image.fromarray(img_black.numpy())
                    adjacency_imgs.append(img_black)

                    source_id = block_coord_to_block_id(i, j)
                    target_id = block_coord_to_block_id(x, y)
                    adjacency_ids.append((source_id, target_id))
        return adjacency_imgs, adjacency_ids

    def create_blocks(img: PIL.Image):
        num_block_x = 6
        num_block_y = 6
        wx = 128 // num_block_x
        wy = 128 // num_block_y
        blocks = []
        for i, j in itertools.product(range(num_block_x), range(num_block_y)):
            # blocks.append(img[i * wx:(i + 1) * wx, j * wy:(j + 1) * wy])
            # blocks.append(img.crop((i * wx, j * wy, (i + 1) * wx, (j + 1) * wy)))
            img_block = img.copy()
            draw = ImageDraw.Draw(img_block)
            draw.rectangle([i * wx, j * wy, (i + 1) * wx, (j + 1) * wy], outline=255)
            blocks.append(img_block)
        return blocks
    
    def overlay_grid(img: PIL.Image):
        img = np.array(img)
        img = img.copy()
        for i in range(21, img.shape[0]-10, 21):
            img[i, :] = 255
            img[:, i] = 255
        
        img = Image.fromarray(img)
        # 9pt font
        font = ImageFont.load_default(size=7)
        # write a number in each cell
        for i in range(0, 128, 21):
            for j in range(0, 128, 21):
                draw = ImageDraw.Draw(img)

                draw.text((i+2, j+2), f"{i//21*6 + j//21}", fill="white", font=font)

        return img

    edge_examples = None
    node_examples = None
    if args.few_shot:
        edge_examples = IOExamples(
            description=None,
            inputs=[RawInput(image_input=create_adjacency(data[101][0][0])[0][0], text_input=None), RawInput(image_input=create_adjacency(data[101][0][0])[0][1], text_input=None), RawInput(image_input=create_adjacency(data[101][0][0])[0][3], text_input=None), RawInput(image_input=create_adjacency(data[101][0][0])[0][4], text_input=None), RawInput(image_input=create_adjacency(data[101][0][0])[0][6], text_input=None), RawInput(image_input=create_adjacency(data[101][0][0])[0][7], text_input=None)],
            outputs=[['0'], ['1'], ['0'], ['0'], ['1'], ['1']],
        )
        node_examples = IOExamples(
            description=None,
            inputs=[RawInput(image_input=create_blocks(data[101][0][0])[12], text_input=None), RawInput(image_input=create_blocks(data[101][0][0])[13], text_input=None), RawInput(image_input=create_blocks(data[101][0][0])[14], text_input=None), RawInput(image_input=create_blocks(data[101][0][0])[15], text_input=None)],
            outputs=[['0'], ['0'], ['1'], ['0']],
        )
        # edge_examples = IOExamples(
        #     description=None,
        #     inputs=[RawInput(image_input=data[101][0][0], text_input=None)],
        #     outputs=[['[(0, 1), (1, 2), (3, 9), (4, 5), (5, 11), (8, 14), (8, 9), (10, 16), (11, 17), (12, 18), (12, 13), (14, 20), (14, 15), (15, 16), (16, 17), (20, 26), (24, 30), (24, 25), (25, 31), (26, 32), (27, 28), (29, 35), (33, 34), (34, 35)]']],
        # )
        # node_examples = IOExamples(
        #     description=None,
        #     inputs=[RawInput(image_input=data[101][0][0], text_input=None)],
        #     outputs=[['[4, 14]']],
        # )

    edge_model = LLMNet(
        model,
        "an image with dashed lines with two adjacent cells outlined",
        "1 if the two outlined cells contain a contiguous dashed line going from one square to the other, and 0 otherwise. The output should be 0 if each cell contains separate dashed lines that do not connect to the other cell.",
        # "an image containing dashed lines and circular nodes",
        # "an adjacency matrix as a list of tuples representing splitting the input into a 6x6 grid and connecting the adjacent cells which are connected by a dashed line. The cells are numbered from 0 to 35 starting from the top left corner and going down column by column so cell 1 is the cell below the top left corner.",
        edge_examples,
    )

    node_model = LLMNet(
        model,
        "an image with a square cell outlined",
        "1 if there is the majority of a large circular node lies within the outlined square cell, and 0 if there is nothing, only dashed lines, or a small part of the circle in the square cell",
        # "an image containing dashed lines and circular nodes",
        # "a list of two values representing which of the 36 blocks of the input image (after splitting it into a 6x6 grid) have a circular node. The numbering of the cells is from the top left corner and goes down column by column so cell 1 is the cell below the top left corner.",
        node_examples,
    )

    def parse(img: RawInput):
        blocks = create_blocks(img.image_input)
        adjacency_imgs, adjacency_ids = create_adjacency(img.image_input)
        adjacency_graph = []
        for i, adj_img in enumerate(adjacency_imgs):
            edge = edge_model.forward(RawInput(image_input=adj_img, text_input=None))
            edge = re.sub(r"[^01]", "", edge)
            if edge == '1':
                adjacency_graph.append(adjacency_ids[i]) 
        nodes = [re.sub(r"[^01]", "", node_model.forward(RawInput(image_input=block, text_input=None))) for block in blocks]
        # adjacency_graph = ast.literal_eval(edge_model.forward(img))
        # nodes = ast.literal_eval(node_model.forward(img))

        return adjacency_graph, nodes

    def function(adjacency_graph, nodes):
        print("adjacency_graph:", adjacency_graph)
        print("nodes:", nodes)

        # check if there is a path from one node to the other node
        nodes = np.array([int(node) for node in nodes])
        if np.sum(nodes) != 2:
            return 0
        # if len(nodes) != 2:
        #     return 0

        node_ids = np.where(nodes == 1)[0]
        print("node_ids:", node_ids)
        start_node = node_ids[0]
        end_node = node_ids[1]

        # check if there is a path from start_node to end_node
        visited = set()
        stack = [start_node]
        while stack:
            node = stack.pop()
            if node == end_node:
                return 1
            if node in visited:
                continue
            visited.add(node)
            stack.extend([adj[1] for adj in adjacency_graph if adj[0] == node])
        return 0

    return parse, function


def leaf_extract(data, model):
    margin_examples = None
    shape_examples = None
    texture_examples = None
    if args.few_shot:
        margin_examples = IOExamples(
            description=None,
            inputs=[RawInput(image_input=data[103][0][0], text_input=None), RawInput(image_input=data[126][0][0], text_input=None), RawInput(image_input=data[104][0][0], text_input=None)],
            outputs=[['entire'], ["lobed"], ["indented"]],
        )
        shape_examples = IOExamples(
            description=None,
            inputs=[RawInput(image_input=data[101][0][0], text_input=None), RawInput(image_input=data[15][0][0], text_input=None), RawInput(image_input=data[3][0][0], text_input=None), RawInput(image_input=data[0][0][0], text_input=None), RawInput(image_input=data[11][0][0], text_input=None)],
            outputs=[['elliptical'], ['lanceolate'], ['ovate'], ['obovate'], ['oblong']],
        )
        texture_examples = IOExamples(
            description=None,
            inputs=[RawInput(image_input=data[14][0][0], text_input=None), RawInput(image_input=data[110][0][0], text_input=None), RawInput(image_input=data[137][0][0], text_input=None)],
            outputs=[['leathery'], ['smooth'], ['glossy']],
        )

    margin_net = LLMNet(
        model,
        "an image of a leaf",
        "the classification of the leaf's margin as one of the following: {'entire', 'indented', 'lobed', 'serrate', 'serrulate', 'undulate'}.",
        margin_examples
    )
    shape_net = LLMNet(
        model,
        "an image of a leaf",
        "the classification of the leaf's shape as one of the following: {'elliptical', 'lanceolate', 'oblong', 'obovate', 'ovate'}.",
        shape_examples
    )
    texture_net = LLMNet(
        model,
        "an image of a leaf",
        "the classification of the leaf's texture as one of the following: {'glossy', 'leathery', 'smooth', 'rough'}.",
        texture_examples
    )

    def parse(img):
        margin = margin_net.forward(img)
        shape = shape_net.forward(img)
        texture = texture_net.forward(img)

        # Clean up the output
        margin = re.sub(r"[^a-zA-Z]", "", margin)
        shape = re.sub(r"[^a-zA-Z]", "", shape)
        texture = re.sub(r"[^a-zA-Z]", "", texture)
        return [margin, shape, texture]
    
    def function(margin, shape, texture):
        if margin == 'serrate': return 'Ocimum basilicum'
        elif margin == 'indented': return 'Jatropha curcas'
        elif margin == 'lobed': return 'Platanus orientalis'
        elif margin == 'serrulate': return "Citrus limon"
        elif margin == 'entire':
          if shape == 'ovate': return 'Pongamia Pinnata'
          elif shape == 'lanceolate': return 'Mangifera indica'
          elif shape == 'oblong': return 'Syzygium cumini'
          elif shape == 'obovate': return "Psidium guajava"
          else:
            if texture == 'leathery': return "Alstonia Scholaris"
            elif texture == 'rough': return "Terminalia Arjuna"
            elif texture == 'glossy': return "Citrus limon"
            else: return "Punica granatum"
        else:
          if shape == 'elliptical': return 'Terminalia Arjuna'
          elif shape == 'lanceolate': return "Mangifera indica"
          else: return 'Syzygium cumini'

    return parse, function


def clevr_extract(data, model):
    init_objects_examples = None
    object_bbox_examples = None
    if args.few_shot:
        objs_100 = [
                    ('cyan','cube','rubber','large'),
                    ('yellow','cylinder','metal','small'),
                    ('red','sphere','rubber','large'),
                    ('gray','cylinder','rubber','small'),
                    ('brown','cylinder','rubber','small'),
                    ('brown','cylinder','metal','small'),
                    ('brown','sphere','rubber','large'),
                    ('cyan','cylinder','metal','small'),
                    ('gray','sphere','rubber','small'),
                    ('brown','cylinder','metal','large')
        ]
        bbox_100 = {
                    ('cyan','cube','rubber','large'):(60,320-195,121,320-96),
                    ('yellow','cylinder','metal','small'):(154,320-273,193,320-215),
                    ('red','sphere','rubber','large'):(139,320-155,204,320-89),
                    ('gray','cylinder','rubber','small'):(242,320-236,285,320-178),
                    ('brown','cylinder','rubber','small'):(210,320-131,238,320-93),
                    ('brown','cylinder','metal','small'):(231,320-97,257,320-66),
                    ('brown','sphere','rubber','large'):(270,320-116,326,320-58),
                    ('cyan','cylinder','metal','small'):(332,320-152,336,320-110),
                    ('gray','sphere','rubber','small'):(319,320-194,357,320-156),
                    ('brown','cylinder','metal','large'):(397,320-185,480,320-93)
                }
        objs_103 = [
                    ('purple','cube','metal','large'),
                    ('cyan','cylinder','rubber','small'),
                    ('brown','sphere','metal','large'),
                    ('brown','sphere','metal','large'),
                    ('red','cylinder','metal','large'),
                    ('cyan','cylinder','metal','large')
        ]
        bbox_103 = {
                    ('purple','cube','metal','large'):(114,320-219,218,320-111),
                    ('cyan','cylinder','rubber','small'):(186,320-124,216,320-105),
                    ('brown','sphere','metal','large'):(195,320-105,244,320-53),
                    ('brown','sphere','metal','large'):(288,320-131,280,320-73),
                    ('red','cylinder','metal','large'):(262,320-173,330,320-87),
                    ('cyan','cylinder','metal','large'):(365,320-191,448,320-98)
                }
        
        init_objects_examples = IOExamples(
            description=None,
            inputs=[RawInput(image_input=data[100][0][0], text_input=None), RawInput(image_input=data[103][0][0], text_input=None)],
            outputs=[objs_100, objs_103],
        )
        
        object_bbox_examples = IOExamples(
            description=None,
            inputs=[RawInput(image_input=data[100][0][0], text_input=str(objs_100)), RawInput(image_input=data[103][0][0], text_input=str(objs_103))],
            outputs=[bbox_100, bbox_103],
        )
        
        
    
    init_objects_net = LLMNet(
        model,
        "an image of geometric objects",
        "the extraction of all of the objects and each of their attributes.  Colors can be one of ['gray','green','blue','red','brown','purple','yellow','cyan']. Each shape can be one of ['cube','cylinder','sphere']. Material can be one of ['rubber','metal'].  Size can be one of ['small','large'].  Your final answer should be a list of tuples of shape (number of objects, number of attributes). The attributes should be organized as (color, shape, material, size).  For example, for an image with two objects, the final answer could look like: [('green','cube','metal','small'),('red','cylinder','rubber','large')].",
        init_objects_examples
    )
    
    object_bbox_net = LLMNet(
        model,
        "an image of geometric objects",
        "the extraction of each object's bounding box given the attributes of each object. The format of the bounding box should be (x1,y1,x2,y2).  The final answer should look look like a dictionary; if you are given the image and the objects [('purple','cube','metal','large'),(cyan','cylinder','rubber','small')], your final answer should look something like {('purple','cube','metal','large'):(114,101,218,209),('cyan','cylinder','rubber','small'):(186,196,216,215)}. The image dimensions are (480,320).",
        object_bbox_examples
    )
    
    
    def parse_data_from_string(s):
        # Collapse multiple whitespace characters into a single space
        cleaned_str = re.sub(r'\s+', ' ', s.strip())
        # Safely evaluate the cleaned string into a Python object
        return ast.literal_eval(cleaned_str)
    
    

    def parse(img, program):
        
        try:
            init_objects_s = init_objects_net.forward(RawInput(image_input=img,text_input=None))
            init_objects = parse_data_from_string(init_objects_s)
            scene_dict_s = object_bbox_net.forward(RawInput(image_input=img,text_input=str(init_objects)))
            scene_dict = parse_data_from_string(scene_dict_s)
            return [scene_dict, program]
        except:
            return [None, None]
        
    
    def function(data, program):
        """
        data: a dict where each key is a list/tuple of attributes in order
            [color, shape, material, size] and the value is the object's bbox.
            For example:
                {
                    ('purple','cube','metal','large'): [114, 320-219, 218, 320-111],
                    ('cyan','cylinder','rubber','small'): [186, 320-124, 216, 320-105],
                    ...
                }
        program: a list of instructions (dictionaries) with keys:
                - "inputs": list of indices referring to previous outputs
                - "function": the function name to call
                - "value_inputs": a list of constant values (e.g. a color or size)
        The final output (the output of the last instruction) will be returned.
        """
        if not data:
            return "-999"
        # Preprocess the input scene data into a list of objects.
        # Each object will have keys: color, shape, material, size, bbox, and a unique id.
        scene_objs = []
        obj_id = 0
        for key, bbox in data.items():
            # Ensure key is a list/tuple with [color, shape, material, size]
            obj = {
                "color": key[0],
                "shape": key[1],
                "material": key[2],
                "size": key[3],
                "bbox": bbox,  # assume bbox is a list like [x_min, y_min, x_max, y_max]
                "id": obj_id
            }
            scene_objs.append(obj)
            obj_id += 1

        # We'll store each intermediate result in a list (memory),
        # so that later instructions can refer to earlier ones by index.
        memory = []

        # ---------------- Helper Functions ----------------

        def scene_fn():
            # Return the full scene.
            return scene_objs

        def filter_color(objects, color):
            return [obj for obj in objects if obj["color"] == color]

        def filter_size(objects, size):
            return [obj for obj in objects if obj["size"] == size]

        def filter_material(objects, material):
            return [obj for obj in objects if obj["material"] == material]

        def filter_shape(objects, shape):
            return [obj for obj in objects if obj["shape"] == shape]

        def unique(objects):
            if len(objects) == 1:
                return objects[0]
            raise ValueError("unique() expected exactly one object, but got {} objects.".format(len(objects)))

        def query_color(obj):
            return obj["color"]

        def query_shape(obj):
            return obj["shape"]

        def query_material(obj):
            return obj["material"]

        def query_size(obj):
            return obj["size"]

        def same_size(obj):
            # Return all objects in the scene that have the same size as obj (excluding obj itself)
            return [o for o in scene_objs if o["size"] == obj["size"] and o["id"] != obj["id"]]

        def same_material(obj):
            return [o for o in scene_objs if o["material"] == obj["material"] and o["id"] != obj["id"]]

        def same_shape(obj):
            return [o for o in scene_objs if o["shape"] == obj["shape"] and o["id"] != obj["id"]]

        def same_color(obj):
            return [o for o in scene_objs if o["color"] == obj["color"] and o["id"] != obj["id"]]

        def relate(obj, relation):
            """
            Given a reference object, return all objects in the scene that are in the given spatial relation.
            For simplicity, we define:
            - "left": object's center x < reference center x
            - "right": object's center x > reference center x
            - "front": object's center y > reference center y (i.e. lower on image)
            - "behind": object's center y < reference center y
            """
            def center(o):
                bbox = o["bbox"]
                # Assuming bbox = [x_min, y_min, x_max, y_max]
                return ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
            ref_center = center(obj)
            if relation == "left":
                return [o for o in scene_objs if center(o)[0] < ref_center[0]]
            elif relation == "right":
                return [o for o in scene_objs if center(o)[0] > ref_center[0]]
            elif relation == "front":
                return [o for o in scene_objs if center(o)[1] > ref_center[1]]
            elif relation == "behind":
                return [o for o in scene_objs if center(o)[1] < ref_center[1]]
            else:
                raise ValueError("Unknown relation: " + relation)

        def union(list1, list2):
            # Return the union of two lists (removing duplicate objects based on id)
            seen = set()
            result = []
            for obj in list1 + list2:
                if obj["id"] not in seen:
                    seen.add(obj["id"])
                    result.append(obj)
            return result

        def intersect(list1, list2):
            # Return the intersection of two lists (objects that appear in both, based on id)
            set1 = {obj["id"] for obj in list1}
            return [obj for obj in list2 if obj["id"] in set1]

        def count_fn(objects):
            return str(len(objects))

        # Map program function names to our helper functions.
        function_map = {
            "scene": scene_fn,
            "filter_color": filter_color,
            "filter_size": filter_size,
            "filter_material": filter_material,
            "filter_shape": filter_shape,
            "unique": unique,
            "query_color": query_color,
            "query_shape": query_shape,
            "query_material": query_material,
            "query_size": query_size,
            "same_size": same_size,
            "same_material": same_material,
            "same_shape": same_shape,
            "same_color": same_color,
            "relate": relate,
            "union": union,
            "intersect": intersect,
            "count": count_fn
        }

        # ---------------- Program Execution ----------------

        # Execute each instruction in order.
        for instruction in program:
            # Get the outputs of prior instructions as specified by "inputs"
            inputs = [memory[i] for i in instruction.get("inputs", [])]
            func_name = instruction["function"]
            val_inputs = instruction.get("value_inputs", [])
            func = function_map.get(func_name)
            if func is None:
                raise ValueError("Unknown function: " + func_name)
            # Call the function with the unpacked inputs and any constant value parameters.
            # For example, filter_color(objects, 'purple') is called as:
            # filter_color(*inputs, *val_inputs)
            result = func(*inputs, *val_inputs)
            memory.append(result)

        # The output of the final instruction is our answer.
        return memory[-1]


    return parse, function


def clutrr_extract(data, model):
    examples = None
    if args.few_shot:
        examples = IOExamples(
            description=None,
            inputs=[RawInput(image_input=None, text_input="Bob is the son of John. Bob is John's what?"), RawInput(image_input=None, text_input="Bob is the son of John. John is Bob's what?")],
            outputs=[['son'], ['father']],
        )
    extract_relation = LLMNet(
        model,
        "a description of a relationship between two people and a query about the two people's relationship",
        "the described relationship which answers the question. The output relationship is one of the following: {'brother', 'sister', 'father', 'mother', 'son', 'daughter', 'grandfather', 'grandmother', 'uncle', 'aunt', 'nephew', 'niece', 'husband', 'wife', 'brother-in-law', 'sister-in-law', 'son-in-law', 'daughter-in-law', 'father-in-law', 'mother-in-law', 'grandson', 'granddaughter'}. For example, for the input 'John took his sister Mary to the store. John is Mary\'s what?' the output should be 'brother.' Output just the relationship as a word.",
        examples
    )
    # extract_relations = LLMNet(
    #     model,
    #     "a story about how people are related to each other",
    #     "a string representing a Python dictionary mapping pairs of people to their relationship where the relationship is one of the following: {'brother', 'sister', 'father', 'mother', 'son', 'daughter', 'grandfather', 'grandmother', 'uncle', 'aunt', 'nephew', 'niece', 'husband', 'wife', 'brother-in-law', 'sister-in-law', 'son-in-law', 'daughter-in-law', 'father-in-law', 'mother-in-law', 'grandson', 'granddaughter'}",
    #     IOExamples(
    #         description=None,
    #         inputs=[RawInput(image_input=None, text_input="Bob is the son of John.")],
    #         outputs=[['{("Bob", "John"): "son", ("John", "Bob"): "father"}']],
    #     )
    # )

    # def parse(context: RawInput, query: RawInput):
    #     out_dict = extract_relations.forward(context)
    #     print("out_dict:", out_dict)
    #     out_dict = ast.literal_eval(out_dict)
    #     return out_dict, tuple(query.text_input.replace("[", "").replace("]", "").replace("'", "").split(", "))

    def parse(context: RawInput, query: RawInput):
        # Preprocess sentences
        relation_sentences = []
        relation_name_pairs = []
        curr_relation_sentences = []
        curr_name_pairs = []
        skip_next = False
        skip_until = 0
        context = [s.strip() for s in context.text_input.split(".") if s.strip() != ""]
        for (j, sentence) in enumerate(context):
            # It is possible to skip a sentence because the previous one includes the current one.
            if skip_next:
                if j >= skip_until:
                    skip_next = False
                continue

            # Get all the names of the current sentence
            names = re.findall(r"\[(\w+)\]", sentence)

            # Check if we need to include the next sentence(s) as well
            num_sentences_limit = 4
            num_sentences = 1
            union_sentence = f"{sentence}"
            for k in range(j + 1, len(context)):
                next_sentence = context[k]
                next_sentence_names = re.findall(r"\[(\w+)\]", next_sentence)
                if (len(names) == 1 or len(next_sentence_names) == 1) and num_sentences < num_sentences_limit:
                    if len(next_sentence_names) > 0:
                        num_sentences += 1
                        union_sentence += f". {next_sentence}"
                        names += next_sentence_names
                    skip_next = True
                    if len(next_sentence_names) == 1:
                        skip_until = k - 1
                    else:
                        skip_until = k
                else:
                    break

            # Deduplicate the names
            names = list(dict.fromkeys(names))

            # Clean up the sentence and add it to the batch
            clean_sentence = union_sentence.replace("[", "").replace("]", "")
            curr_relation_sentences += [f"{clean_sentence}. {names[k]} is {names[l]}'s what?" for k in range(len(names)) for l in range(len(names)) if k != l]
            curr_name_pairs += [(k, l) for k in names for l in names if k != l]

        # Construct the current datatpoint
        relation_sentences += curr_relation_sentences
        relation_name_pairs += curr_name_pairs

        facts = []
        for i in range(len(relation_sentences)):
            rel = extract_relation.forward(RawInput(image_input=None, text_input=relation_sentences[i]))
            rel = re.sub(r"[^a-zA-Z\-]", "", rel)
            facts.append((relation_name_pairs[i], rel))

        return facts, query.text_input #tuple(query.text_input.replace("[", "").replace("]", "").replace("'", "").split(", "))

    def function(facts, query):
        rules = {('sister-in-law', 'brother'): 'brother-in-law', ('sister-in-law', 'sister'): 'sister-in-law', ('brother-in-law', 'brother'): 'brother-in-law', ('brother-in-law', 'sister'): 'sister-in-law', ('daughter-in-law', 'daughter'): 'granddaughter', ('daughter-in-law', 'son'): 'grandson', ('son-in-law', 'daughter'): 'granddaughter', ('son-in-law', 'son'): 'grandson', ('nephew', 'sister'): 'niece', ('nephew', 'brother'): 'nephew', ('niece', 'sister'): 'niece', ('niece', 'brother'): 'nephew', ('grandson', 'aunt'): 'daughter', ('grandson', 'uncle'): 'son', ('granddaughter', 'aunt'): 'daughter', ('granddaughter', 'uncle'): 'son', ('brother-in-law', 'son'): 'nephew', ('sister-in-law', 'son'): 'nephew', ('brother-in-law', 'daughter'): 'niece', ('sister-in-law', 'daughter'): 'niece', ('sister-in-law', 'father'): 'father-in-law', ('grandson', 'brother'): 'grandson', ('grandson', 'father'): 'son', ('grandson', 'sister'): 'granddaughter', ('niece', 'grandfather'): 'father', ('grandmother', 'husband'): 'grandfather', ('wife', 'mother'): 'mother-in-law', ('wife', 'brother'): 'brother-in-law', ('wife', 'sister'): 'sister-in-law', ('wife', 'mother-in-law'): 'mother', ('wife', 'daughter-in-law'): 'daughter-in-law', ('wife', 'father-in-law'): 'father', ('wife', 'son-in-law'): 'son-in-law', ('wife', 'grandson'): 'grandson', ('wife', 'granddaughter'): 'granddaughter', ('wife', 'father'): 'father-in-law', ('wife', 'son'): 'son', ('wife', 'daughter'): 'daughter', ('grandfather', 'wife'): 'grandmother', ('grandfather', 'son'): 'father', ('grandfather', 'daughter'): 'mother', ('uncle', 'mother'): 'grandmother', ('uncle', 'brother'): 'uncle', ('uncle', 'father'): 'grandfather', ('uncle', 'sister'): 'aunt', ('mother', 'mother-in-law'): 'grandmother', ('mother', 'daughter-in-law'): 'wife', ('mother', 'father-in-law'): 'grandfather', ('mother', 'son-in-law'): 'husband', ('mother', 'grandson'): 'son', ('mother', 'mother'): 'grandmother', ('mother', 'brother'): 'uncle', ('mother', 'granddaughter'): 'daughter', ('mother', 'husband'): 'father', ('mother', 'father'): 'grandfather', ('mother', 'son'): 'brother', ('mother', 'sister'): 'aunt', ('mother', 'daughter'): 'sister', ('nephew', 'grandmother'): 'mother', ('nephew', 'grandfather'): 'father', ('nephew', 'uncle'): 'brother', ('nephew', 'mother'): 'sister', ('nephew', 'father'): 'brother', ('nephew', 'aunt'): 'sister', ('brother', 'niece'): 'niece', ('brother', 'grandmother'): 'grandmother', ('brother', 'grandfather'): 'grandfather', ('brother', 'uncle'): 'uncle', ('brother', 'mother'): 'mother', ('brother', 'nephew'): 'nephew', ('brother', 'brother'): 'brother', ('brother', 'father'): 'father', ('brother', 'aunt'): 'aunt', ('brother', 'son'): 'nephew', ('brother', 'sister'): 'sister', ('brother', 'daughter'): 'niece', ('granddaughter', 'grandmother'): 'wife', ('granddaughter', 'grandfather'): 'husband', ('granddaughter', 'mother'): 'daughter', ('granddaughter', 'brother'): 'grandson', ('granddaughter', 'father'): 'son', ('granddaughter', 'sister'): 'granddaughter', ('husband', 'grandson'): 'grandson', ('husband', 'mother'): 'mother-in-law', ('husband', 'brother'): 'brother-in-law', ('husband', 'granddaughter'): 'granddaughter', ('husband', 'father'): 'father-in-law', ('husband', 'son'): 'son', ('husband', 'sister'): 'sister-in-law', ('husband', 'daughter'): 'daughter', ('father', 'wife'): 'mother', ('father', 'mother'): 'grandmother', ('father', 'brother'): 'uncle', ('father', 'granddaughter'): 'daughter', ('father', 'father'): 'grandfather', ('father', 'son'): 'brother', ('father', 'sister'): 'aunt', ('father', 'daughter'): 'sister', ('aunt', 'mother'): 'grandmother', ('aunt', 'brother'): 'uncle', ('aunt', 'father'): 'grandfather', ('aunt', 'sister'): 'aunt', ('son', 'grandmother'): 'mother', ('son', 'wife'): 'daughter-in-law', ('son', 'grandfather'): 'father', ('son', 'uncle'): 'brother', ('son', 'mother'): 'wife', ('son', 'brother'): 'son', ('son', 'father'): 'husband', ('son', 'aunt'): 'sister', ('son', 'son'): 'grandson', ('son', 'sister'): 'daughter', ('son', 'daughter'): 'granddaughter', ('sister', 'niece'): 'niece', ('sister', 'grandmother'): 'grandmother', ('sister', 'grandfather'): 'grandfather', ('sister', 'uncle'): 'uncle', ('sister', 'mother'): 'mother', ('sister', 'nephew'): 'nephew', ('sister', 'brother'): 'brother', ('sister', 'husband'): 'brother-in-law', ('sister', 'father'): 'father', ('sister', 'aunt'): 'aunt', ('sister', 'son'): 'nephew', ('sister', 'sister'): 'sister', ('sister', 'daughter'): 'niece', ('daughter', 'grandmother'): 'mother', ('daughter', 'grandfather'): 'father', ('daughter', 'uncle'): 'brother', ('daughter', 'mother'): 'wife', ('daughter', 'brother'): 'son', ('daughter', 'husband'): 'son-in-law', ('daughter', 'father'): 'husband', ('daughter', 'aunt'): 'sister', ('daughter', 'son'): 'grandson', ('daughter', 'sister'): 'daughter', ('daughter', 'daughter'): 'granddaughter', ('niece', 'father'): 'brother', ('niece', 'mother'): 'sister', ('sister', 'wife'): 'sister-in-law', ('brother', 'husband'): 'brother-in-law', ('brother', 'wife'): 'sister-in-law', ('sister-in-law', 'mother'): 'mother-in-law', ('brother-in-law', 'father'): 'father-in-law', ('brother-in-law', 'mother'): 'mother-in-law', ('grandmother', 'daughter'): 'mother', ('grandmother', 'son'): 'father', ('mother-in-law', 'daughter'): 'wife', ('father-in-law', 'daughter'): 'wife', ('sister-in-law', 'husband'): 'brother-in-law', ('brother-in-law', 'wife'): 'sister-in-law', ('mother-in-law', 'son'): 'brother-in-law', ('husband', 'wife'): 'self', ('wife', 'husband'): 'self', ('grandson', 'mother'): 'daughter', ('cousin', 'grandmother'): 'grandmother', ('cousin', 'grandfather'): 'grandfather', ('aunt', 'son'): 'cousin', ('aunt', 'daughter'): 'cousin', ('uncle', 'son'): 'cousin', ('uncle', 'daughter'): 'cousin', ('niece', 'grandmother'): 'mother', ('niece', 'uncle'): 'brother', ('niece', 'aunt'): 'sister', ('cousin', 'uncle'): 'uncle', ('cousin', 'aunt'): 'aunt', ('grandson', 'grandfather'): 'husband', ('grandson', 'grandmother'): 'wife', ('cousin', 'mother'): 'aunt', ('cousin', 'father'): 'uncle', ('cousin', 'sister'): 'sister', ('cousin', 'brother'): 'brother', ('father-in-law', 'son'): 'husband', ('mother-in-law', 'husband'): 'father-in-law', ('father-in-law', 'wife'): 'mother-in-law', ('aunt', 'husband'): 'uncle', ('uncle', 'wife'): 'aunt'}

        facts = {(pair[0], pair[1]): rel for pair, rel in facts}
        last_facts = {}
        while query not in facts:
            added_facts = {}
            for fact1 in facts.items():
                for fact2 in facts.items():
                    if fact1[0][0] != fact2[0][1] and fact1[0][1] == fact2[0][0] and (fact2[1], fact1[1]) in rules:
                        new_fact = rules[(fact2[1], fact1[1])]
                        added_facts[(fact1[0][0], fact2[0][1])] = new_fact
            facts.update(added_facts)
            if last_facts == facts:
                break
            last_facts = facts
        print("final facts:", facts)

        if query in facts:
            return facts[query]
        else:
            return "Uncertain"

    return parse, function


def clevr_settings(data):
    # (question, image, answer, program_seq, scene)
    # ((image, question), answer, (program_seq, scene))
    def apply_query_to_objects(objects, query):
        ans = query(objects)
        if ans == True or ans == "True" or ans == "true":
            return "yes"
        elif ans == False or ans == "False" or ans == "false":
            return "no"
        elif type(ans) == int:
            return str(ans)
        elif type(ans) == str:
            return ans.lower()
        else:
            return ans

    explicit_fn = apply_query_to_objects
    implicit_fn = "apply a query to over objects and relationships to get an answer in the set {'0', '1', '10', '2', '3', '4', '5', '6', '7', '8', '9', 'blue', 'brown', 'cube', 'cyan', 'cylinder', 'gray', 'green', 'large', 'metal', 'no', 'purple', 'red', 'rubber', 'small', 'sphere', 'yellow', 'yes'}"
    inferred_fn = IOExamples(
        description=None,
        inputs=[
            RawInput(image_input=data[101][0][0], text_input=data[101][0][1]),
            RawInput(image_input=data[102][0][0], text_input=data[102][0][1]),
        ],
        outputs=[[data[101][1]], [data[102][1]]],
    )

    example_dict = {
        "objects": [
            {"color": "brown", "size": "small", "shape": "cube", "material": "rubber"},
            {
                "color": "yellow",
                "size": "small",
                "shape": "cylinder",
                "material": "metal",
            },
            {
                "color": "purple",
                "size": "small",
                "shape": "cylinder",
                "material": "rubber",
            },
            {"color": "gray", "size": "large", "shape": "sphere", "material": "metal"},
            {"color": "red", "size": "large", "shape": "sphere", "material": "rubber"},
            {
                "color": "purple",
                "size": "small",
                "shape": "sphere",
                "material": "rubber",
            },
            {"color": "red", "size": "large", "shape": "cube", "material": "metal"},
            {
                "color": "blue",
                "size": "large",
                "shape": "cylinder",
                "material": "rubber",
            },
        ],
        "relationships": {
            "right": [
                [2, 3],
                [0, 2, 3, 6],
                [3],
                [],
                [0, 1, 2, 3, 5, 6, 7],
                [0, 1, 2, 3, 6, 7],
                [0, 2, 3],
                [0, 1, 2, 3, 6],
            ],
            "behind": [
                [2, 3, 4, 5, 6, 7],
                [0, 2, 3, 4, 5, 6, 7],
                [5, 6, 7],
                [2, 4, 5, 6, 7],
                [2, 5, 6, 7],
                [6, 7],
                [7],
                [],
            ],
            "front": [
                [1],
                [],
                [0, 1, 3, 4],
                [0, 1],
                [0, 1, 3],
                [0, 1, 2, 3, 4],
                [0, 1, 2, 3, 4, 5],
                [0, 1, 2, 3, 4, 5, 6],
            ],
            "left": [
                [1, 4, 5, 6, 7],
                [4, 5, 7],
                [0, 1, 4, 5, 6, 7],
                [0, 1, 2, 4, 5, 6, 7],
                [],
                [4],
                [1, 4, 5, 7],
                [4, 5],
            ],
        },
    }
    example_dict2 = {"objects": [{"color": "purple", "size": "small", "shape": "cube", "material": "rubber"}, {"color": "gray", "size": "large", "shape": "sphere", "material": "metal"}, {"color": "blue", "size": "small", "shape": "sphere", "material": "rubber"}, {"color": "red", "size": "small", "shape": "cube", "material": "rubber"}, {"color": "yellow", "size": "small", "shape": "cylinder", "material": "metal"}, {"color": "cyan", "size": "large", "shape": "cube", "material": "rubber"}, {"color": "purple", "size": "large", "shape": "cube", "material": "rubber"}, {"color": "purple", "size": "small", "shape": "cylinder", "material": "rubber"}], "relationships": {"right": [[2, 3, 6], [0, 2, 3, 4, 6, 7], [3, 6], [], [0, 2, 3, 6, 7], [0, 1, 2, 3, 4, 6, 7], [3], [0, 2, 3, 6]], "behind": [[4], [0, 4], [0, 1, 3, 4, 5, 6], [0, 1, 4, 6], [], [0, 1, 3, 4, 6], [0, 1, 4], [0, 1, 2, 3, 4, 5, 6]], "front": [[1, 2, 3, 5, 6, 7], [2, 3, 5, 6, 7], [7], [2, 5, 7], [0, 1, 2, 3, 5, 6, 7], [2, 7], [2, 3, 5, 7], []], "left": [[1, 4, 5, 7], [5], [0, 1, 4, 5, 7], [0, 1, 2, 4, 5, 6, 7], [1, 5], [], [0, 1, 2, 4, 5, 7], [1, 4, 5]]}}
    query_str = """def query(data):
    objects = data["objects"]
    
    # Step 0: scene()
    # Get indices of all objects
    step0 = list(range(len(objects)))  # Indices 0 to 8

    # Step 1: filter_size with size "small"
    # Filter objects by size == "small"
    step1 = [idx for idx in step0 if objects[idx]["size"] == "small"]
    # step1 contains indices: [1, 2, 3, 4, 6, 7, 8]

    # Step 2: filter_color with color "red"
    # Filter the previous result by color == "red"
    step2 = [idx for idx in step1 if objects[idx]["color"] == "red"]
    # step2 contains indices: [7]

    # Step 3: unique()
    # Expect exactly one object; if not, return False
    if len(step2) != 1:
        # No unique object found; return False as per the context
        return False
    unique_idx = step2[0]

    # Step 4: same_shape()
    # Find all other objects with the same shape as the unique object
    target_shape = objects[unique_idx]["shape"]
    step4 = [idx for idx in range(len(objects)) 
             if idx != unique_idx and objects[idx]["shape"] == target_shape]
    # step4 contains indices: [1, 4, 5, 6]

    # Step 5: filter_color with color "green"
    # Filter the previous result by color == "green"
    step5 = [idx for idx in step4 if objects[idx]["color"] == "green"]
    # step5 contains indices: [1]

    # Step 6: filter_material with material "metal"
    # Filter the previous result by material == "metal"
    step6 = [idx for idx in step5 if objects[idx]["material"] == "metal"]
    # step6 is empty: []

    # Step 7: exist()
    # Check if there exists any object in step 6
    result = len(step6) > 0

    return result
    """
    query_str2 = """def query(data):
    objects = data["objects"]
    relationships = data["relationships"]
    
    # Get indices of all objects
    all_indices = list(range(len(objects)))  # Indices 0 to 7
    
    # Part 1: Small green matte cylinders
    # Filter objects that are small, green, matte (rubber), and cylinders
    small_green_matte_cylinders = [
        idx for idx in all_indices
        if objects[idx]['size'] == 'small'
        and objects[idx]['color'] == 'green'
        and objects[idx]['material'] == 'rubber'
        and objects[idx]['shape'] == 'cylinder'
    ]
    count1 = len(small_green_matte_cylinders)
    
    # Part 2: Small shiny objects to the right of the green matte thing
    # First, find the green matte thing
    green_matte_objects = [
        idx for idx in all_indices
        if objects[idx]['color'] == 'green'
        and objects[idx]['material'] == 'rubber'
    ]
    
    count2 = 0
    if green_matte_objects:
        # Assuming there is only one green matte object
        green_matte_idx = green_matte_objects[0]
        # Get objects to the right of the green matte thing
        right_indices = relationships['right'][green_matte_idx]
        # Filter for small shiny objects (material is metal)
        small_shiny_objects = [
            idx for idx in right_indices
            if objects[idx]['size'] == 'small' and objects[idx]['material'] == 'metal'
        ]
        count2 = len(small_shiny_objects)
    
    # Total count
    total_count = count1 + count2
    return total_count
    """
    implicit_s = """a image of 3D objects represented as a Python dict and a query which is a Python function which takes the objects dict as input representing the objects and computes the answer to the question.
The image dict contains two entries.
'objects', which maps to a list of dicts containing information about each object, including 'color','size','shape', and 'material'.
'relationships', which tells us about the spatial relationships 'right','behind','front', and 'left' between objects, and is a dict that maps each spatial relationship to a 2D-list.
In the relationships dictionary, each spatial relation (like 'right', 'left', 'front', 'behind') maps to a list where each element at index i contains a list of object indices that are in that relation with object i.
The index of an object is it's position in 'objects'.
The attributes and their possible values are as follows.
'color': 'blue', 'brown', 'cyan', 'gray', 'green', 'purple', 'red', 'yellow'
'material': 'metal', 'rubber'
'shape': 'cube', 'cylinder', 'sphere'
'size': 'small', 'large'"""
    explicit_s = IOExamples(
        description=implicit_s,
        inputs=[
                RawInput(image_input=data[101][0][0], text_input=data[101][0][1]),
                RawInput(image_input=data[102][0][0], text_input=data[102][0][1])
                ],
        outputs=[[example_dict, query_str], [example_dict2, query_str2]],
    )
    inferred_s = None

    return (explicit_s, implicit_s, inferred_s, explicit_fn, implicit_fn, inferred_fn)


def gsm8k_zs(data):
    expl_s = [
        {
            "type": "text",
            "text": "First, extract python code from the input. The following are some examples of the desired python code extraction:",
        },
        {"type": "text", "text": f"Example 1: {data[101][0][1]}"},
        {
            "type": "text",
            "text": """# Each tub of lip gloss holds 2 tubes
tubes_per_tub = 2

# Marcy is bringing 6 tubs
num_tubs = 6

# Each tube of lip gloss can be used for 3 people
people_per_tube = 3

# Calculate the total number of tubes
total_tubes = tubes_per_tub * num_tubs

# Calculate the total number of people
total_people = total_tubes * people_per_tube""",
        },
        {"type": "text", "text": f"Example 2: {data[102][0][1]}"},
        {
            "type": "text",
            "text": """# Oil needed for each wheel
oil_per_wheel = 10  # in ml

# Number of wheels
num_wheels = 2

# Oil needed for the rest of the bike
oil_for_rest = 5  # in ml

# Calculate the total amount of oil
total_oil = (oil_per_wheel * num_wheels) + oil_for_rest""",
        },
    ]
    impl_s = [
        {
            "type": "text",
            "text": """First, extract Python code from the input.""",
        }
    ]
    inf_s = [
        {"type": "text", "text": """First, convert the input to a symbolic form."""}
    ]

    expl_fn = [
        {
            "type": "text",
            "text": """Next, compute the following function:
    def eval_python(code_str):
    return eval(code_str)""",
        }
    ]
    impl_fn = [
        {
            "type": "text",
            "text": """Next, compute the following function on the input:
    the output of the Python code""",
        }
    ]
    inf_fn = [
        {
            "type": "text",
            "text": """Next, compute a function on the input by following the examples below:""",
        },
        {"type": "text", "text": f"Example 1: {data[101][0][1]}"},
        {"type": "text", "text": "The output for this input is 36."},
        {"type": "text", "text": f"Example 2: {data[102][0][1]}"},
        {"type": "text", "text": "The output for this input is 25."},
    ]

    return (expl_s, impl_s, inf_s, expl_fn, impl_fn, inf_fn)


def blocksworld_settings(data):
    def planner(problem: str, domain: str):
        """Finds a sequence of actions to achieve the goal. Outputs the final plan as a newline separated list of actions. For example:
            (attack a)
            (overcome a b)"""
        return find_solution(problem, domain)

    explicit_fn = planner

    implicit_fn = """find a sequence of actions which achieve the goal and output the final plan as a newline separated list of actions. For example:
(attack a)
(overcome a b)"""
    inferred_fn = IOExamples(
        description=None,
        inputs=[RawInput(*data[101][0])], outputs=[[["""(feast c d)
(succumb c)
(feast d b)
(succumb d)
(attack c)
(overcome c d)
(feast b a)
(overcome b c)"""]]]
    )

    explicit_s = IOExamples(
        description="a string representing a complete PDDL problem definition and another string representing a complete PDDL domain definition",
        inputs=[RawInput(*data[101][0])],
        outputs=[
            [
                """(define (problem pb1)
(:domain blocksworld)
(:objects a b c d )
(:init
(craves b a)
(craves c d)
(craves d b)
(harmony)
(planet a)
(province c)
)
(:goal
(and
(craves b c)
(craves c d))
)
)""",
                """(define (domain blocksworld)
  (:requirements :strips)
(:predicates (province ?x)
             (planet ?x)
             (harmony)
             (pain ?x)
             (craves ?x ?y))

(:action attack
  :parameters (?ob)
  :precondition (and (province ?ob) (planet ?ob) (harmony))
  :effect (and (pain ?ob) (not (province ?ob)) (not (planet ?ob))
               (not (harmony))))

(:action succumb
  :parameters  (?ob)
  :precondition (pain ?ob)
  :effect (and (province ?ob) (harmony) (planet ?ob)
               (not (pain ?ob))))

(:action overcome
  :parameters  (?ob ?underob)
  :precondition (and (province ?underob) (pain ?ob))
  :effect (and (harmony) (province ?ob) (craves ?ob ?underob)
               (not (province ?underob)) (not (pain ?ob))))

(:action feast
  :parameters  (?ob ?underob)
  :precondition (and (craves ?ob ?underob) (province ?ob) (harmony))
  :effect (and (pain ?ob) (province ?underob)
               (not (craves ?ob ?underob)) (not (province ?ob)) (not (harmony)))))"""
            ],
        ],
    )
    implicit_s = "a PDDL problem definition as a string and a PDDL domain definition as a string"
    inferred_s = None

    return (explicit_s, implicit_s, inferred_s, explicit_fn, implicit_fn, inferred_fn)


def long_sort_settings(data):
    def sort_words(words):
        """Sort the words in alphabetical order and return the sorted word list."""
        return sorted(words)

    explicit_fn = sort_words
    implicit_fn = "sort the words in alphabetical order and return the sorted word list"
    inferred_fn = IOExamples(
        description=None,
        inputs=[RawInput(*data[101][0]), RawInput(*data[102][0])], outputs=[[['ad hoc', 'bankruptcy', 'bawdy', 'beard', 'cage', 'carving', 'damaging', 'damp', 'deliberation', 'dispatch', 'dough', 'dramaturge', 'drug', 'efficacious', 'footrest', 'framework', 'ginseng', 'harbor', 'ikebana', 'jicama', 'liner', 'motion', 'otter', 'penny', 'peony', 'pink', 'pit', 'publicity', 'raspy', 'recondite', 'refuge', 'revascularisation', 'sadness', 'shop', 'silica', 'spool', 'uncertainty', 'value', 'weird']], [['advocate', 'biology', 'bonsai', 'chopsticks', 'clan', 'codpiece', 'compassionate', 'dispense', 'duty', 'hold', 'homework', 'honorable', 'idiom', 'inexpensive', 'inside', 'landmine', 'licensing', 'link', 'liquidity', 'minor-league', 'nondisclosure', 'operating', 'penny', 'provision', 'reciprocity', 'rocker', 'seizure', 'starter', 'subgroup', 'tickle', 'towel', 'tram', 'trout', 'wave']]]
    )

    explicit_s = IOExamples(
        description="a string of digits",
        inputs=[RawInput(*data[101][0]), RawInput(*data[102][0])],
        outputs=[[['weird', 'drug', 'damp', 'ad hoc', 'framework', 'harbor', 'refuge', 'damaging', 'dispatch', 'beard', 'penny', 'value', 'bawdy', 'bankruptcy', 'silica', 'liner', 'dramaturge', 'peony', 'recondite', 'deliberation', 'uncertainty', 'publicity', 'jicama', 'carving', 'pit', 'pink', 'otter', 'cage', 'shop', 'dough', 'raspy', 'footrest', 'spool', 'revascularisation', 'sadness', 'ikebana', 'efficacious', 'ginseng', 'motion']], [['inside', 'subgroup', 'reciprocity', 'minor-league', 'operating', 'provision', 'towel', 'link', 'penny', 'trout', 'compassionate', 'biology', 'bonsai', 'homework', 'tram', 'rocker', 'landmine', 'duty', 'advocate', 'hold', 'seizure', 'inexpensive', 'idiom', 'nondisclosure', 'licensing', 'clan', 'starter', 'dispense', 'chopsticks', 'liquidity', 'tickle', 'honorable', 'codpiece', 'wave']]],
    )
    implicit_s = "a list of words"
    inferred_s = None

    return (explicit_s, implicit_s, inferred_s, explicit_fn, implicit_fn, inferred_fn)


def listsynthesis_settings(data):
    def get_output(ops, examples, input_list):
        """Return the output list."""
        solver = ProgramSynthesisSolver(ops)
        solution = solver.solve(examples)
        return solution.evaluate(input_list)

    explicit_fn = get_output
    implicit_fn = "produce the output when given the input list"
    inferred_fn = IOExamples(
        description=None,
        inputs=[RawInput(None, """Available operations:
  - min: (int, int) -> int
Each input to the above operations is a value from the list 'input'. The values are accessed by their index in the list. For example, 'input[0]' is the first value in the list. A function f: int -> int can be called like f(input[4]) which would apply f to the 5th value in input.

Input-Output Examples:
  Example 1: [6, 5, 2, -1, 5] -> -1
  Example 2: [1, 8, -4, 6, -6] -> 1
  Example 3: [-1, -6, -7, 9, -2] -> -1
  Example 4: [7, 9, -6, -1, -7] -> -1
  Example 5: [-8, 0, 5, 7, -7] -> -8
Given the above examples, what is the output for the following input?
Input: [1, 3, 0, 9, 10] -> ?""")], outputs=[[1]]
    )

    explicit_s = IOExamples(
        description="a list of allowed operations, a list of input output examples, and the test input list",
        inputs=[RawInput(None, """Available operations:
  - min: (int, int) -> int
Each input to the above operations is a value from the list 'input'. The values are accessed by their index in the list. For example, 'input[0]' is the first value in the list. A function f: int -> int can be called like f(input[4]) which would apply f to the 5th value in input.

Input-Output Examples:
  Example 1: [6, 5, 2, -1, 5] -> -1
  Example 2: [1, 8, -4, 6, -6] -> 1
  Example 3: [-1, -6, -7, 9, -2] -> -1
  Example 4: [7, 9, -6, -1, -7] -> -1
  Example 5: [-8, 0, 5, 7, -7] -> -8
Given the above examples, what is the output for the following input?
Input: [1, 3, 0, 9, 10] -> ?""")],
        outputs=[["""[{
        "name": "min",
        "arg_indices": [0,1],
        "output_type": "int",
        "implementation": lambda x,y: min(x,y)
    }]""", [
        ([6, 5, 2, -1, 5], -1),
        ([1, 8, -4, 6, -6], 1),
        ([-1, -6, -7, 9, -2], -1),
        ([7, 9, -6, -1, -7], -1),
        ([-8, 0, 5, 7, -7], -8)
    ], [1, 3, 0, 9, 10]]],
    )
    implicit_s = "a list of input output examples, the allowed operations with their input and output types, and the test input list"
    inferred_s = None

    return (explicit_s, implicit_s, inferred_s, explicit_fn, implicit_fn, inferred_fn)


def bbh_sort_settings(data):
    def sort_list(lst):
        """Sort the list of words in lexicographical order and return the sorted list."""
        return sorted(lst)
    
    def insertion_sort(arr):
        """
        Sorts the given array using the insertion sort algorithm.
        """

        for i in range(1, len(arr)):
            key = arr[i]
            j = i - 1

            # Move elements of arr[0..i-1] that are greater than key
            # to one position ahead of their current position
            while j >= 0 and key < arr[j]:
                arr[j + 1] = arr[j]
                j -= 1

            arr[j + 1] = key

        return arr

    explicit_fn = insertion_sort
    implicit_fn = "sort a list of words in lexicographical order"
    inferred_fn = IOExamples(
        description=None,
        inputs=[RawInput(*data[101][0]), RawInput(*data[102][0])], outputs=[[["administer", "aeneid", "coachman", "decadent", "delhi", "dey", "gradate", "grim", "jacky", "littleneck", "phosphorescent", "pristine", "shrunk", "sinh", "systemwide", "tasting", "thrown", "torpedo", "verdict"]], [["absorption", "align", "anastasia", "anastomotic", "apache", "award", "bobbin", "burrow", "calumny", "epaulet", "execrable", "hostelry", "hun", "macedon", "omnipotent", "putty", "roughshod", "smooth", "spontaneity"]]]
    )

    explicit_s = IOExamples(
        description="a list of words",
        inputs=[RawInput(*data[101][0]), RawInput(*data[102][0])],
        outputs=[
            [
                ["torpedo", "phosphorescent", "pristine", "decadent", "shrunk", "dey", "administer", "gradate", "littleneck", "thrown", "jacky", "coachman", "aeneid", "verdict", "tasting", "sinh", "delhi", "systemwide", "grim"]
            ],
            [
                ["spontaneity", "smooth", "execrable", "epaulet", "bobbin", "calumny", "hun", "anastasia", "omnipotent", "award", "hostelry", "macedon", "roughshod", "burrow", "align", "apache", "putty", "absorption", "anastomotic"]
            ],
        ],
    )
    implicit_s = "a list of words"
    inferred_s = None

    return (explicit_s, implicit_s, inferred_s, explicit_fn, implicit_fn, inferred_fn)


# def bbh_logic7_settings(data):
#     def get_correct_answer(facts, answers):
#         current_state = {}
#         for fact in facts:
#             # fact is a tuple of (subject, relation, object)
#             if fact[1] == "POS":
#                 current_state[fact[0]] = fact[2]
#             elif fact[1] == "BEFORE":
#                 if fact[0] in current_state:


def bbh_shuffle7_settings(data):
    def get_answer(initial_state, swaps, target, condition_to_answer):
        # Iteratively apply the swaps to the initial positions and return the answer choice letter for the target item
        positions = initial_state
        for swap in swaps:
            positions[swap[0]], positions[swap[1]] = positions[swap[1]], positions[swap[0]]
        return condition_to_answer[positions[target]]

    explicit_fn = get_answer
    implicit_fn = "iteratively apply the swaps to the initial positions and return the answer choice letter for the target item"
    inferred_fn = IOExamples(
        description=None,
        inputs=[RawInput(*data[101][0]), RawInput(*data[102][0])], outputs=[["(E)"], ["(B)"]]
    )

    explicit_s = IOExamples(
        description="a dictionary of items and their initial conditions, a list of item swaps, the target item, and a mapping from condition to answer choice letter",
        inputs=[RawInput(*data[101][0]), RawInput(*data[102][0])],
        outputs=[
            [{"Alice": "Lolita", "Bob": "Ulysses", "Claire": "Hound of the Baskervilles", "Dave": "The Great Gatsby", "Eve": "Catch-22", "Fred": "The Fellowship of the Ring", "Gertrude": "The Odyssey"}, [["Alice", "Eve"], ["Bob", "Gertrude"], ["Claire", "Dave"], ["Gertrude", "Dave"], ["Alice", "Fred"], ["Eve", "Fred"], ["Gertrude", "Bob"]], "Eve", {"Lolita": "(A)", "Ulysses": "(B)", "Hound of the Baskervilles": "(C)", "The Great Gatsby": "(D)", "Catch-22": "(E)", "The Fellowship of the Ring": "(F)", "The Odyssey": "(G)"}],
            [{"Alice": "black", "Bob": "blue", "Claire": "red", "Dave": "purple", "Eve": "pink", "Fred": "brown", "Gertrude": "white"}, [["Fred", "Gertrude"], ["Fred", "Alice"], ["Alice", "Claire"], ["Bob", "Dave"], ["Fred", "Eve"], ["Gertrude", "Fred"], ["Gertrude", "Bob"]], "Dave", {"black": "(A)", "blue": "(B)", "red": "(C)", "purple": "(D)", "pink": "(E)", "brown": "(F)", "white": "(G)"}],
        ],
    )
    implicit_s = "a dictionary of items and their initial conditions, a list of item swaps, the target item, and a mapping from condition to answer choice letter"
    inferred_s = None

    return (explicit_s, implicit_s, inferred_s, explicit_fn, implicit_fn, inferred_fn)


def gsm8k_settings(data):
    def eval_extracted_code(code_str):
        """Evaluate the Python code and output the value of the variable 'answer'."""
        locs = {}
        exec(code_str, locs, locs)
        return locs["answer"]

    explicit_fn = eval_extracted_code
    implicit_fn = "evaluate Python code to get the answer"
    inferred_fn = IOExamples(
        description=None,
        inputs=[RawInput(*data[101][0]), RawInput(*data[102][0])], outputs=[[36], [25]]
    )

    explicit_s = IOExamples(
        description="a string representing a complete Python code snippet with the answer stored in a variable named 'answer'",
        inputs=[RawInput(*data[101][0]), RawInput(*data[102][0])],
        outputs=[
            [
                """# Each tub of lip gloss holds 2 tubes
tubes_per_tub = 2

# Marcy is bringing 6 tubs
num_tubs = 6

# Each tube of lip gloss can be used for 3 people
people_per_tube = 3

# Calculate the total number of tubes
total_tubes = tubes_per_tub * num_tubs

# Calculate the total number of people
answer = total_tubes * people_per_tube"""
            ],
            [
                """# Oil needed for each wheel
oil_per_wheel = 10  # in ml

# Number of wheels
num_wheels = 2

# Oil needed for the rest of the bike
oil_for_rest = 5  # in ml

# Calculate the total amount of oil
answer = (oil_per_wheel * num_wheels) + oil_for_rest"""
            ],
        ],
    )
    implicit_s = "a string representing a complete Python code snippet with the answer stored in a variable named 'answer'"
    inferred_s = None

    return (explicit_s, implicit_s, inferred_s, explicit_fn, implicit_fn, inferred_fn)


def get_equivalence(args):
    if args.dataset == "gsm8k":
        def equiv(x, y, i):
            try:
                y = re.findall(r"(?:-)?(\d+(?:\.\d+)?)", y)[-1]
                if not abs(float(x) - float(y)) < 0.01:
                    print(f"GT: {x}, Pred: {y}")
                return abs(float(x) - float(y)) < 0.01
            except:
                print(f"GT: {x}, Pred: {y}")
                return False
        return equiv
    elif args.dataset == "chartqa":
        def relaxed_equiv(gt, pred, i):
            # check if x and y are strings that can be converted to floats
            if pred is None:
                return False

            pred = str(pred)

            if gt.replace(".", "", 1).replace("%", "", 1).replace("$", "", 1).isdigit() and pred.replace(".", "", 1).replace("%", "", 1).replace("$", "", 1).isdigit():
                return abs(float(gt) - float(pred.replace("%", "", 1).replace("$", "", 1))) / max(abs(float(gt)), abs(float(pred.replace("%", "", 1).replace("$", "", 1)))) < 0.05
            elif pred == "True":
                return gt == "yes"
            elif pred == "False":
                return gt == "no"
            else:
                return gt.lower() == pred.lower()
        return relaxed_equiv
    elif args.dataset == "blocksworld":
        def equiv(gt, pred, i):
            # find pddl for problem
            problem = f"data/mystery_blocksworld/mystery_pddl/instance-{i+2}.pddl"
            domain = "data/mystery_blocksworld/mystery_pddl/domain.pddl"

            try:
                if "pddl" in pred:
                    pred = pred.split("pddl")[-1].strip()
                if "\\text" in pred:
                    pred = pred.split("\\text")[-1].strip()
                pred = pred.replace(" from", "").strip().lower()
                pred = pred.replace(" object", "").strip().lower()
                pred = pred.replace(" then", "").strip().lower()
                pred = pred.replace(",", "\n")
                plan = [re.sub(r"[^a-zA-Z]", " ", s).strip() for s in pred.split("\n")]
                print("Inside equiv:", i, plan)
            except:
                return False
            return eval_solution_files(problem, domain, plan)
        return equiv
    elif args.dataset == "bbh_sort":
        def equiv(gt, pred, i):
            try:
                if "\\text" in pred:
                    pred = re.findall(r"\\text{(.*)}", pred)[0]

                if "1. " in pred:
                    # extract the list of words from numbered list
                    words = re.findall(r"\d\. (.*)", pred)
                    pred = " ".join(words)
                else:
                    pred = pred.replace("[", "").replace(",", " ").replace("]", "").replace("'", "").replace("\"", "").strip()
                    pred = " ".join([p.strip() for p in pred.split()])
                gt = gt.replace("'", "")
                if pred != gt:
                    print(f"GT: {gt}, Pred: {pred}")
                return pred == gt
            except:
                print(f"GT: {gt}, Pred: {pred}")
                return False
        return equiv
    elif args.dataset == "bbh_shuffle7":
        def equiv(gt, pred, i):
            try:
                gt = re.findall(r"[A-Z]", gt)[0]
                if "(" in pred:
                    pred = re.findall(r"([a-zA-Z])", pred)[0]
                else:
                    pred = re.findall(r"[a-zA-Z]", pred)[-1]
                return gt == pred
            except:
                return False
        return equiv
    elif args.dataset == "clevr":
        def equiv(gt, pred, i):
            gt = gt.strip().lower()
            pred = pred.strip().lower()
            if gt.isdigit() and not pred.isdigit():
                return False
            if gt.isdigit() and pred.isdigit():
                return int(gt) == int(pred)
            return gt in pred  
        return equiv
    elif args.dataset == "folio":
        def equiv(gt, pred, i):
            def conv(string):
                if type(string) is not str:
                    return None
                string = string.strip().lower()
                if any([i in string for i in ["true"]]):
                    return "True"
                elif any([i in string for i in ["false"]]):
                    return "False"
                elif any([i in string for i in ["uncertain"]]):
                    return "Uncertain"
                else:
                    return None

            gt2 = conv(gt)
            pred2 = conv(pred)

            if gt2 is None or pred2 is None:
                return False
            
            return gt2 == pred2
        return equiv
    elif args.dataset == "longsort":
        def equiv(gt, pred, i):
            gt = "[" + ", ".join([f"'{i}'" for i in gt]) + "]"
            # print(f"GT: {gt}, Pred: {pred}", type(pred))
            return gt == str(pred)
        return equiv
    elif args.dataset == "listsynthesis":
        def equiv(gt, pred, i):
            if type(pred) == "str" and "\n" in pred:
                pred = pred.split("\n")[0]
            print(f"GT: {gt}, Pred: {pred}")
            return str(gt) == str(pred)
        return equiv
    else:
        def equiv(gt, pred, i):
            return gt == pred
        return equiv


def create_symbol_extractor(args, model):
    if args.dataset == "sum2":
        data = MNISTSumKOrigDataset(root="data", train=False, download=True, k=5)
    elif args.dataset == "svhn":
        data = SVHNSumKDataset(root="data", k=5, train=False)
    elif args.dataset == "hwf":
        data = HWFDataset(root="data", split="train", length=5)
    elif args.dataset == "clutrr":
        data = ClutrrDataset()
    elif args.dataset == "leaf":
        data = LeafDataset()
    elif args.dataset == "pathfinder":
        data = PathFinder128Dataset("./data/pathfinder/", "128", difficulty="easy")
    elif args.dataset == "clevr":
        data = ClevrDataset(max_samples=500)
    elif args.dataset == "chartqa":
        data = ChartQADataset()
    elif args.dataset == "gsm8k":
        data = GSM8KDataset()
    elif args.dataset == "blocksworld":
        data = BlocksWorldDataset()
    elif args.dataset == "bbh_sort":
        data = BBHDataset("word_sorting")
    elif args.dataset == "longsort":
        data = LongSortDataset()
    elif args.dataset == "listsynthesis":
        data = ListSynthesisDataset()
    elif args.dataset == "bbh_logic7":
        data = BBHDataset("logical_deduction_seven_objects")
    elif args.dataset == "bbh_shuffle7":
        data = BBHDataset("tracking_shuffled_objects_seven_objects")
    elif args.dataset == "folio":
        data = FOLIODataset()
    else:
        raise NotImplementedError

    all_get_symbols = {
        "sum2": sum2_extract,
        "svhn": svhn_extract,
        "hwf": hwf_extract,
        "clutrr": clutrr_extract,
        "clevr": clevr_extract,
        "leaf": leaf_extract,
        "pathfinder": pathfinder_extract,
        "chartqa": chartqa_settings,
        "gsm8k": gsm8k_settings,
        "blocksworld": blocksworld_settings,
        "bbh_sort": bbh_sort_settings,
        "longsort": long_sort_settings,
        "listsynthesis": listsynthesis_settings,
        # "bbh_logic7": bbh_logic7_settings,
        "bbh_shuffle7": bbh_shuffle7_settings,
        "folio": FOLIO_settings
    }
    get_symbols, function = all_get_symbols[args.dataset](data, model)

    return data, get_symbols, function



def eval_cached(args):
    data, settings = get_dataset_settings(args)
    equiv = get_equivalence(args)

    settings = [
        "Expl-fn,Expl-S",
        "Expl-fn,Impl-S",
        "Expl-fn,Inf-S",
        "Impl-fn,Expl-S",
        "Impl-fn,Impl-S",
        "Impl-fn,Inf-S",
        "Inf-fn,Expl-S",
        "Inf-fn,Impl-S",
        "Inf-fn,Inf-S",
    ]

    if args.setting:
        settings = [args.setting]
    elif args.raw:
        settings = ["none,none"]

    # Select random subset of 200 samples
    if args.dataset == "chartqa":
        test_data_ids = (
            list(range(100))
            + list(range(103, 1001))
            + list(range(1002, len(data)))
        )
    else:
        test_data_ids = list(range(100)) + list(range(103, len(data)))
    shuf = np.random.permutation(test_data_ids)
    print(shuf[:10])
    test_data = [data[int(i)] for i in shuf[:200]]
    gt = [test_data[i][1] for i in range(len(test_data))]

    model_name = args.model.split("/")[-1]
    prompt_type = "multi_prompt" if args.multi_prompt else "single_prompt"
    for setting_name in settings:
        setting_name = setting_name.lower()
        with open(f"logs/{model_name}/{args.dataset}/{prompt_type}/{setting_name}{'_handwritten' if args.handwritten else ''}{'_raw' if args.raw else ''}.txt", "r") as f:
            preds = f.readlines()

        def get_pred(output):
            extra_args = []
            # if not args.dataset == "gsm8k":
            extra_args.append(re.DOTALL)
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
            except Exception as e:
                # print("Failed:", output, e)
                pred = "None"
            return pred

        if args.raw:
            predictions = [get_pred(ast.literal_eval(pred)[0]) for pred in preds]
        elif args.multi_prompt:
            code = [ast.literal_eval(pred)[1] for pred in preds]
            llm_predictions = [get_pred(ast.literal_eval(pred)[0]) for pred in preds]
            print("Finished LLM predictions")
            if args.dataset == "blocksworld":
                for i in range(len(code)):
                    code[i] = "from src.pddl import find_solution\n" + code[i].replace("?", " ?")
                    code[i] = re.sub(r"def find_solution\(problem, domain\):.*?\n\n", "", code[i], flags=re.DOTALL)
            elif args.dataset == "listsynthesis":
                code = ["from src.program_gen import ProgramSynthesisSolver\n" + c for c in code]
            eval_predictions = [python_eval(c) for c in code]
            print("Finished Eval predictions")
        else:
            predictions = [get_pred(ast.literal_eval(pred)[-1]) for pred in preds]

        if args.multi_prompt and not args.raw:
            acc = sum([equiv(gt[i], llm_predictions[i], shuf[i]) for i in range(len(llm_predictions))]) / len(llm_predictions)
            print(f"Simulate Setting: {setting_name}, Acc: {acc}")
            with open(f"logs/{model_name}/{args.dataset}/{prompt_type}/results.txt", "a") as f:
                f.write(
                    f"simulate_{model_name},{args.dataset},{prompt_type},{setting_name.lower().replace(', ', '_')},{acc}\n"
                )

            acc = sum([equiv(gt[i], eval_predictions[i], shuf[i]) for i in range(len(eval_predictions))]) / len(eval_predictions)
            print(f"Eval Setting: {setting_name}, Acc: {acc}")
            with open(f"logs/{model_name}/{args.dataset}/{prompt_type}/results.txt", "a") as f:
                f.write(
                    f"eval_{model_name},{args.dataset},{prompt_type},{setting_name.lower().replace(', ', '_')},{acc}\n"
                )
        else:
            acc = sum([equiv(gt[i], predictions[i], shuf[i]) for i in range(len(predictions))]) / len(predictions)
            print(f"Setting: {setting_name}, Acc: {acc}")

            with open(f"logs/{model_name}/{args.dataset}/{prompt_type}/results.txt", "a") as f:
                f.write(
                    f"{model_name},{args.dataset},{prompt_type},{setting_name.lower().replace(', ', '_')},{acc}\n"
                )

        if args.raw:
            return


def main(args):
    # model = MllamaForConditionalGeneration.from_pretrained(
    #     args.model, torch_dtype=torch.bfloat16
    # ).to("cuda:0")
    # processor = AutoProcessor.from_pretrained(args.model)
    if not args.use_hf and not "gemini" in args.model.lower() and not "gpt" in args.model.lower() and not "o1" in args.model.lower():
        model = LLM(
            model=args.model,
            max_model_len=12288,
            limit_mm_per_prompt={"image": 10},
            max_num_seqs=1,
            enforce_eager=True if "llama" in args.model.lower() else False,
            trust_remote_code=True,
            tensor_parallel_size=args.num_gpus,
        )
    elif "gemini" in args.model.lower() or "gpt" in args.model.lower() or "o1" in args.model.lower():
        model = APIModel(args.model)
    else:
        model = OurLLM(model_name=args.model)

    data, get_symbols, function = create_symbol_extractor(args, model)
    equiv = get_equivalence(args)

    results = []

    # Select random subset of 200 samples
    test_data_ids = list(range(100)) + list(range(103, len(data)))
    shuf = np.random.permutation(test_data_ids)
    test_data = [data[int(i)] for i in shuf[:200]]
    gt = [test_data[i][1] for i in range(len(test_data))]

    # Run the NeSy task
    task = LLMNesy(get_symbols, function)
    preds, logs = get_task_predictions(
        task,
        test_data,
        log=args.log,
        equiv=equiv,
    )

    acc = sum([equiv(gt[i], preds[i], shuf[i]) for i in range(len(preds))]) / len(preds)
    print(f"Accuracy:", acc)
    results.append(acc)
    
    # check if logs/model dir exists
    model_name = args.model.split("/")[-1]
    if not os.path.exists(f"logs/{('debug/' if args.debug else '') + model_name}/{args.dataset}/"):
        os.makedirs(f"logs/{('debug/' if args.debug else '') + model_name}/{args.dataset}/")
    with open(
        f"logs/{('debug/' if args.debug else '') + model_name}/{args.dataset}/llm_symbolic_{'fs' if args.few_shot else 'zs'}.txt",
        "w",
    ) as f:
        for log in logs:
            f.write(str(log) + "\n")

    # append to results file
    with open(f"logs/{('debug/' if args.debug else '') + model_name}/{args.dataset}/results.txt", "a") as f:
        f.write(
            f"{('debug_' if args.debug else '') + model_name},{args.few_shot},{args.dataset},{acc}\n"
        )


if __name__ == "__main__":
    # set seeds
    np.random.seed(0)

    args = argparse.ArgumentParser()
    args.add_argument("--dataset", type=str, default="clevr")
    args.add_argument(
        "--model", type=str, default="Qwen/Qwen2.5-VL-72B-Instruct"
    )
    args.add_argument("--log", action="store_true")
    args.add_argument("--raw", action="store_true")
    args.add_argument("--num_gpus", type=int, default=1)
    args.add_argument("--use_hf", action="store_true")
    args.add_argument("--debug", action="store_true")
    args.add_argument("--eval", action="store_true")
    args.add_argument("--few_shot", action="store_true")
    args = args.parse_args()

    logger.info("Starting")

    if args.eval:
        eval_cached(args)
    else:
        main(args)
