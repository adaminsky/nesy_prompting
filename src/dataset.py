from typing import Optional, List, Tuple
import torchvision
import torch
import logging
import json
import os
from PIL import Image
import numpy as np
from wonderwords import RandomWord
import random
from datasets import load_dataset
from typing import Optional, Callable
import subprocess
import pddlpy
import tempfile
from src.program_gen import demonstrate_generator
import random
import itertools
import copy
from sudoku import Sudoku
import re
import csv
import ast
logger = logging.getLogger(__name__)


class MNISTSumKOrigDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        root,
        train=True,
        transform=None,
        target_transform=None,
        download=False,
        k=2,
        concatenate=False,
    ):
        self.mnist = torchvision.datasets.MNIST(
            root, train, transform, target_transform, download
        )
        self.train = train
        self.k = k
        self.concatenate = concatenate

    def __getitem__(self, index):
        imgs = []
        labels = []
        for i in range(self.k):
            img, label = self.mnist[index * self.k + i]
            imgs.append(img)
            labels.append(label)
        sum_label = sum(labels)

        return imgs, sum_label

    def __len__(self):
        return len(self.mnist) // self.k


class MNISTSumKDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        root,
        train=True,
        transform=None,
        target_transform=None,
        download=False,
        k=2,
    ):
        self.mnist = torchvision.datasets.MNIST(
            root, train, transform, target_transform, download
        )
        self.train = train
        self.k = k

    def __getitem__(self, index):
        imgs = []
        labels = []
        for i in range(self.k):
            img, label = self.mnist[index * self.k + i]
            imgs.append(img)
            labels.append(label)
        sum_label = sum(labels)
        img = Image.new("RGB", (28 * self.k, 28))
        for i in range(self.k):
            img.paste(imgs[i], (28 * i, 0))
        return [img, None], sum_label, *labels

    def __len__(self):
        return len(self.mnist) // self.k


class HWFDataset(torch.utils.data.Dataset):
    def __init__(self, root: str, split: str, length: int, concatenate=False):
        super(HWFDataset, self).__init__()
        self.root = root
        self.split = split
        self.concatenate = concatenate
        md = json.load(open(os.path.join(root, f"HWF/expr_{split}.json")))

        # finding only the metadata with length == 1
        if length > 0:
            self.metadata = [m for m in md if len(m["img_paths"]) == length]
        else:
            self.metadata = md

        # self.img_transform = torchvision.transforms.Compose(
        #     [
        #         torchvision.transforms.ToTensor(),
        #         # torchvision.transforms.Normalize((0.5,), (1,)),
        #     ]
        # )

    def __getitem__(self, index):
        sample = self.metadata[index]

        # Input is a sequence of images
        img_seq = []
        for img_path in sample["img_paths"]:
            img_full_path = os.path.join(
                self.root, "HWF/Handwritten_Math_Symbols", img_path
            )
            img = Image.open(img_full_path).convert("RGB")
            # img = self.img_transform(img)
            # print(img.shape)
            img_seq.append(img)
        # img_seq_len = len(img_seq)

        if self.concatenate:
            img = np.concatenate([np.array(img) for img in img_seq], axis=1)
            img_seq = Image.fromarray(img * 255)

        # Output is the "res" in the sample of metadata
        res = sample["res"]

        # Return (input, output) pair
        return img_seq, res, sample['expr']

    def __len__(self):
        return len(self.metadata)

    @staticmethod
    def collate_fn(batch):
        max_len = max([img_seq_len for (_, img_seq_len, _) in batch])
        zero_img = torch.zeros_like(batch[0][0][0])

        def pad_zero(img_seq):
            return img_seq + [zero_img] * (max_len - len(img_seq))

        img_seqs = torch.stack(
            [torch.stack(pad_zero(img_seq)) for (img_seq, _, _) in batch]
        )
        img_seq_len = torch.stack(
            [torch.tensor(img_seq_len).long() for (_, img_seq_len, _) in batch]
        )
        results = torch.stack([torch.tensor(res) for (_, _, res) in batch])
        return (img_seqs, img_seq_len, results)


class PathFinder128Dataset(torch.utils.data.Dataset):
    pathfinder_img_transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            # torchvision.transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )

    """
    :param data_root, the root directory of the data folder
    :param data_dir, the directory to the pathfinder dataset under the root folder
    :param difficulty, can be picked from "easy", "normal", "hard", and "all"
    """

    def __init__(
        self,
        data_root: str,
        data_dir: str = "128",
        difficulty: str = "all",
        transform: Optional[Callable] = pathfinder_img_transform,
    ):
        # Store
        self.transform = transform

        # Get subdirectories
        easy, normal, hard = (
            ("curv_baseline", 0),
            ("curv_contour_length_9", 1),
            ("curv_contour_length_14", 2),
        )
        if difficulty == "all":
            sub_dirs = [easy, normal, hard]
        elif difficulty == "easy":
            sub_dirs = [easy]
        elif difficulty == "normal":
            sub_dirs = [normal]
        elif difficulty == "hard":
            sub_dirs = [hard]
        else:
            raise Exception(f"Unrecognized difficulty {difficulty}")

        # Get all image paths and their labels
        self.samples = []
        for sub_dir, difficulty_id in sub_dirs:
            metadata_dir = os.path.join(data_root, data_dir, sub_dir, "metadata")
            for sample_group_file in os.listdir(metadata_dir):
                sample_group_dir = os.path.join(metadata_dir, sample_group_file)
                sample_group_file = open(sample_group_dir, "r")
                sample_group_lines = np.load(sample_group_dir)
                for sample_line in sample_group_lines:
                    sample_img_path = os.path.join(
                        data_root, data_dir, sub_dir, sample_line[0], sample_line[1]
                    )
                    sample_label = int(sample_line[3])
                    self.samples.append((sample_img_path, difficulty_id, sample_label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        (img_path, difficulty_id, label) = self.samples[idx]
        img = Image.open(open(img_path, "rb"))
        if self.transform is not None:
            img = self.transform(img)
        return (img, difficulty_id, label)


def _dataset_to_tensor(dset, mask=None):
    arr = np.asarray(dset, dtype=np.int64)
    if mask is not None:
        arr = arr[mask]
    tensor = torch.LongTensor(arr)
    return tensor


class ClevrDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        questions_path="./data/CLEVR_v1.0/questions/CLEVR_train_questions.json",
        images_path="./data/CLEVR_v1.0/images/train/",
        scene_path="./data/CLEVR_v1.0/scenes/CLEVR_train_scenes.json",
        max_samples=None,
    ):
        self.images_path = images_path
        self.scene_path = scene_path
        self.max_samples = max_samples

        question_json = json.load(open(questions_path, "r"))
        if scene_path:
            scene_json = json.load(open(scene_path, "r"))

        # Data from the question file is small, so read it all into memory
        logger.info("Reading question data into memory")
        self.all_questions = [
            question_json["questions"][i]["question"]
            for i in range(len(question_json["questions"]))
        ]
        self.all_image_idxs = [
            question_json["questions"][i]["image_index"]
            for i in range(len(question_json["questions"]))
        ]
        self.all_programs = None
        if "program" in question_json["questions"][0]:
            self.all_programs = [
                question_json["questions"][i]["program"]
                for i in range(len(question_json["questions"]))
            ]
        self.all_answers = [
            question_json["questions"][i]["answer"]
            for i in range(len(question_json["questions"]))
        ]
        self.all_scenes = (
            [
                {key: d[key] for key in ["objects", "relationships"]}
                for d in scene_json["scenes"]
            ]
            if scene_path
            else None
        )
        if self.all_scenes is not None:
            for scene in self.all_scenes:
                for i in scene["objects"]:
                    i.pop("rotation")
                    i.pop("3d_coords")
                    i.pop("pixel_coords")

        # remove yes/no questions
        keep_idx = [
            True if ans != "yes" and ans != "no" else False for ans in self.all_answers
        ]
        self.all_questions = [
            self.all_questions[i] for i in range(len(self.all_questions)) if keep_idx[i]
        ]
        self.all_image_idxs = [
            self.all_image_idxs[i]
            for i in range(len(self.all_image_idxs))
            if keep_idx[i]
        ]
        self.all_programs = [
            self.all_programs[i] for i in range(len(self.all_programs)) if keep_idx[i]
        ]
        self.all_answers = [
            self.all_answers[i] for i in range(len(self.all_answers)) if keep_idx[i]
        ]
        if self.all_scenes is not None:
            self.all_scenes = [
                self.all_scenes[i] for i in range(len(self.all_scenes)) if keep_idx[i]
            ]

    def __getitem__(self, index):
        question = self.all_questions[index]
        image_idx = self.all_image_idxs[index]
        answer = self.all_answers[index]
        program_seq = None
        if self.all_programs is not None:
            program_seq = self.all_programs[index]

        image = None
        if self.images_path is not None:
            image = Image.open(
                os.path.join(
                    self.images_path, f"CLEVR_train_{str(image_idx).zfill(6)}.png"
                )
            )
            # image = torch.FloatTensor(np.asarray(image, dtype=np.float32))

        # program_json = None
        # if program_seq is not None:
        #     program_json_seq = []
        #     for fn_idx in program_seq:
        #         fn_str = self.vocab["program_idx_to_token"][fn_idx]
        #         if fn_str == "<START>" or fn_str == "<END>":
        #             continue
        #         fn = iep.programs.str_to_function(fn_str)
        #         program_json_seq.append(fn)
        #     if self.mode == "prefix":
        #         program_json = iep.programs.prefix_to_list(program_json_seq)
        #     elif self.mode == "postfix":
        #         program_json = iep.programs.postfix_to_list(program_json_seq)

        if self.all_scenes is not None:
            scene = self.all_scenes[index]

        # return (question, image, answer, program_seq, scene)
        return ((image, question), answer, (program_seq, scene))

    def __len__(self):
        if self.max_samples is None:
            return len(self.all_questions)
        else:
            return min(self.max_samples, len(self.all_questions))


class GSM8KDataset(torch.utils.data.Dataset):
    def __init__(self):
        self.data = load_dataset("openai/gsm8k", "main", split="train")

    def __getitem__(self, index):
        return [
            [None, self.data[index]["question"]],
            int(self.data[index]["answer"].split("#### ")[-1].replace(",", "")),
        ]

    def __len__(self):
        return len(self.data)


class ChartQADataset(torch.utils.data.Dataset):
    def __init__(self):
        self.data = load_dataset("HuggingFaceM4/ChartQA", split="train")

    def __getitem__(self, index):
        return (self.data[index]["image"], self.data[index]["query"]), self.data[index][
            "label"
        ][0]

    def __len__(self):
        return len(self.data)


class BlocksWorldDataset(torch.utils.data.Dataset):
    def __init__(
        self, root="./data/mystery_blocksworld/", min_objects=3, max_objects=10
    ):
        # self.data_json = json.load(open(os.path.join(root, "task_1_plan_generation.json"), "r"))["instances"]
        self.fast_downward = "../downward/fast-downward.py"
        self.solver_cmd = f'timeout 60s {self.fast_downward} {{domain}} {{problem}} --search "astar(lmcut())" > /dev/null'

        if not os.path.exists(
            os.path.join(root, f"mysteryblocks_{min_objects}_{max_objects}.json")
        ):
            np.random.seed(0)
            num_objects = np.random.randint(min_objects, 1 + max_objects, size=400)
            self.data = []
            cnt = 0
            while cnt < 400:
                res = subprocess.run(
                    [
                        "bash",
                        "data/mystery_blocksworld/generate/blocksworld",
                        "4",
                        str(num_objects[cnt]),
                    ],
                    capture_output=True,
                    text=True,
                )
                res = res.stdout.strip()
                res = (
                    res.replace("handempty", "harm-ny")
                    .replace("ontable", "planet")
                    .replace("clear", "province")
                    .replace("on", "craves")
                    .replace("harm-ny", "harmony")
                )
                initial = []
                goal = []
                with tempfile.NamedTemporaryFile(
                    mode="w", delete=False, encoding="ascii"
                ) as f:
                    f.write(res)
                    problem_file = f.name
                    f.seek(0)

                    # check if the problem is solvable
                    subprocess.run(
                        self.solver_cmd.format(
                            domain="data/mystery_blocksworld/generate/domain.pddl",
                            problem=problem_file,
                        ),
                        shell=True,
                    )
                    try:
                        with open("sas_plan", "r") as f:
                            plan = "\n".join([l.strip() for l in f.readlines()][:-1])
                        os.remove("sas_plan")
                    except:
                        # unsolvable problem
                        continue

                    dp = pddlpy.DomainProblem(
                        "data/mystery_blocksworld/generate/domain.pddl", problem_file
                    )
                    initial = dp.initialstate()
                    goal = dp.goals()
                query = "As initial conditions I have that"
                goal = [
                    f"object {g.predicate[1]} {g.predicate[0]} object {g.predicate[2]}"
                    for g in goal
                ]
                for cond in initial:
                    if len(cond.predicate) == 3:
                        query += f", object {cond.predicate[1]} {cond.predicate[0]} object {cond.predicate[2]}"
                    elif len(cond.predicate) == 2:
                        query += f", {cond.predicate[0]} object {cond.predicate[1]}"
                    else:
                        query += f", {cond.predicate[0]}"
                query += f". My goal is to have that {', '.join(goal)}."
                self.data.append(
                    {
                        "query": query,
                        "num_objects": str(num_objects[cnt]),
                        "ground_truth_plan": plan,
                        "pddl_problem": res,
                    }
                )

                cnt += 1
                if cnt < 400:
                    print("Finding solvable problem of size", num_objects[cnt])

            with open(
                os.path.join(root, f"mysteryblocks_{min_objects}_{max_objects}.json"),
                "w",
            ) as f:
                # write formatted json
                json.dump({"data": self.data}, f, indent=4)
        else:
            with open(
                os.path.join(root, f"mysteryblocks_{min_objects}_{max_objects}.json"),
                "r",
            ) as f:
                self.data = json.load(f)["data"]

    def __getitem__(self, index):
        instruction = """Here are the actions I can do:
- attack object
- feast object from another object
- succumb object
- overcome object from another object
   
I have the following restrictions on my actions:
- To perform attack action, the following facts need to be true: province object, planet object, harmony.
- Once attack action is performed the following facts will be true: pain object.
- Once attack action is performed the following facts will be false: province object, planet object, harmony.
- To perform succumb action, the following facts need to be true: Pain object.
- Once succumb action is performed the following facts will be true: province object, planet object, harmony.
- Once succumb action is performed the following facts will be false: pain object.
- To perform overcome action, the following needs to be true: province other object, pain object.
- Once overcome action is performed the following will be true: harmony, province object, object craves other object.
- Once overcome action is performed the following will be false: province other object, pain object.
- To perform feast action, the following needs to be true: object craves other object, province object, harmony.
- Once feast action is performed the following will be true: pain object, province other object.
- Once feast action is performed the following will be false:, object craves other object, province object, harmony."""

        #         example = """As initial conditions I have that, object b craves object a, object c craves object d, object d craves object b, harmony, planet object a and province object c.
        # My goal is to have that object b craves object c and object c craves object d.
        # My plan is as follows:
        # (feast c d)
        # (succumb c)
        # (feast d b)
        # (succumb d)
        # (attack c)
        # (overcome c d)
        # (feast b a)
        # (overcome b c)"""

        # raw_prompt = self.data_json[index]["query"]
        # query = re.findall(r"\[STATEMENT\](.*?)My plan is as follows:\n\n\[PLAN\]", raw_prompt, re.DOTALL)[-1].strip()
        query = self.data[index]["query"]

        prompt = f"{instruction}\n\nQuery: {query} Find a sequence of actions to achieve this goal."

        return (
            (None, prompt),
            (self.data[index]["ground_truth_plan"],
            self.data[index]["num_objects"],
            self.data[index]["num_objects"],)
        )  # self.data_json[index]["ground_truth_plan"]

    def __len__(self):
        return len(self.data)


class BBHDataset(torch.utils.data.Dataset):
    def __init__(self, subset):
        self.data = load_dataset("maveriq/bigbenchhard", subset, split="train")

    def __getitem__(self, index):
        return (None, self.data[index]["input"]), self.data[index]["target"]

    def __len__(self):
        return len(self.data)


class LongSortDataset(torch.utils.data.Dataset):
    def __init__(self, dir="./", min_length=5, max_length=50):
        if not os.path.exists(f"{dir}/data/long_sort.json"):
            # generate lists of random words of given lengths
            np.random.seed(0)
            random.seed(0)
            torch.manual_seed(0)
            lengths = np.random.randint(min_length, 1 + max_length, size=400)
            r = RandomWord()

            self.data = []
            for i in range(400):
                self.data.append({"input": r.random_words(amount=lengths[i])})
            for d in self.data:
                d["target"] = sorted(d["input"])

            # save to json
            with open(f"{dir}/data/long_sort.json", "w") as f:
                json.dump(self.data, f)
        else:
            with open(f"{dir}/data/long_sort.json", "r") as f:
                self.data = json.load(f)

    def __getitem__(self, index):
        template = "Sort the following words in alphabetical order: {}"
        return (
            None,
            template.format(", ".join(map(str, self.data[index]["input"]))),
        ), self.data[index]["target"]

    def __len__(self):
        return len(self.data)


class FOLIODataset(torch.utils.data.Dataset):
    def __init__(self, split="train", transform=None):
        # Load the specified split of the dataset
        self.dataset = load_dataset("yale-nlp/FOLIO", split=split)
        self.transform = transform

    def __len__(self):
        # Return the total number of samples
        return len(self.dataset)

    def __getitem__(self, idx):
        # Get the sample at the specified index
        item = self.dataset[idx]

        # Extract the relevant fields
        # sample = {
        #     'story_id': item['story_id'],
        #     'premises': item['premises'],
        #     'premises_FOL': item['premises-FOL'],
        #     'conclusion': item['conclusion'],
        #     'conclusion_FOL': item['conclusion-FOL'],
        #     'label': item['label'],
        #     'example_id': item['example_id']
        # }
        image = None
        question = f"Premise: {item['premises']} Conclusion: {item['conclusion']}  Is the conclusion True, False, or Uncertain? Choose one."
        answer = item["label"]
        metadata = (
            f"Premise: {item['premises-FOL']} Conclusion: {item['conclusion-FOL']}"
        )

        sample = [[image, question], answer, metadata]

        return sample
    
    
class MultiplicationDataset(torch.utils.data.Dataset):
    def __init__(self, num_digit=4, max_sequence=3000, generate_on_the_fly=True, transform=None):
        """
        Initializes the dataset by generating all data during initialization.

        Args:
            num_digit (int): Maximum number of digits for the numbers.
            max_sequence (int): Maximum number of inputs per combination.
            generate_on_the_fly (bool): Ignored (data is always generated in __init__).
            transform (callable, optional): Optional transform to be applied to the data.
        """
        self.num_digit = num_digit
        self.max_sequence = max_sequence
        self.transform = transform

        self.data = self._generate_dataset()

    def _generate_dataset(self):
        """
        Generate the entire dataset during initialization.

        Returns:
            list: A list of generated samples.
        """
        data = []
        for _ in range(self.max_sequence):
            a = random.randint(10**(self.num_digit - 1), 10**self.num_digit - 1)
            b = random.randint(10**(self.num_digit - 1), 10**self.num_digit - 1)
            question = f"What is {a} times {b}?"
            answer = str(a * b)
            metadata = None
            data.append([[None, question], answer, metadata])
        return data

    def __len__(self):
        """
        Returns the total number of samples in the dataset.
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        Retrieve a sample by its index.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            sample (list): A list containing the following elements:
                - [None, question]: Image placeholder (None) and the multiplication question.
                - answer: The multiplication result.
                - metadata: Metadata (optional, can be extended).
        """
        sample = self.data[idx]

        if self.transform:
            sample = self.transform(sample)

        return sample


class SubsequenceSumDataset(torch.utils.data.Dataset):
    def __init__(self, num_numbers=5, min_value=-5, max_value=5, num_samples=1000, generate_scratchpad=False, transform=None):
        """
        Initializes the dataset by generating all data during initialization.

        Args:
            num_numbers (int): Number of integers in each input sequence.
            min_value (int): Minimum value of integers in the sequence.
            max_value (int): Maximum value of integers in the sequence.
            num_samples (int): Number of samples to generate.
            generate_scratchpad (bool): Whether to include scratchpad explanations in the data.
            transform (callable, optional): Optional transform to be applied to the data.
        """
        self.num_numbers = num_numbers
        self.min_value = min_value
        self.max_value = max_value
        self.num_samples = num_samples
        self.generate_scratchpad = generate_scratchpad
        self.transform = transform

        self.data = self._generate_dataset()
        
    def find_max_sum_nonadjacent(self, arr):
        """
        When there are many results, choose the one that appears first lexicographically,
        where 1=picking the number and 2=not picking it.

        dp[i][0] = maximum subsequence of arr[i:] where we do not use arr[i]
        dp[i][1] = maximum subsequence of arr[i:] where we do use arr[i]

        dp[i][0] = max(dp[i+1][0], dp[i+1][1])
        dp[i][1] = arr[i] + dp[i+1][0]
        """
        N = len(arr)

        dp = [[0 for _ in range(2)] for _ in range(N)]
        dp[N-1][0] = 0
        dp[N-1][1] = arr[N-1]
        for i in range(N-2, -1, -1):
            dp[i][1] = dp[i + 1][0] + arr[i]
            dp[i][0] = max(dp[i + 1][0], dp[i + 1][1])

        max_sum = max(dp[0][0], dp[0][1])

        result = []
        remaining_sum = max_sum
        can_access_next_item = True
        for i in range(N):
            if dp[i][1] == remaining_sum and can_access_next_item:
                result.append(1)
                remaining_sum -= arr[i]
                can_access_next_item = False
            elif dp[i][0] == remaining_sum:
                result.append(2)
                can_access_next_item = True
            else:
                assert False

        return result, max_sum

    def _sample_entries(self, num_numbers, min_value, max_value, num_samples):
        result = set()
        while len(result) < num_samples:
            input_list = [random.randint(min_value, max_value) for _ in range(num_numbers)]

            output_sequence, my_max_sum = self.find_max_sum_nonadjacent(input_list)

            result.add((tuple(input_list), tuple(output_sequence)))

        result = [(list(input_list), list(output_seq)) for input_list, output_seq in result]
        return result

    def _all_entries(self, num_numbers, min_value, max_value):
        all_inputs = itertools.product(list(range(min_value, max_value + 1)), repeat=num_numbers)

        result = []
        for input_list in all_inputs:
            input_list = list(input_list)
            output_sequence, my_max_sum = self.find_max_sum_nonadjacent(input_list)

            result.append((input_list, output_sequence))

        return result

    def _generate_dataset(self):
        """
        Generate the entire dataset during initialization.

        Returns:
            list: A list of generated samples.
        """
        total_combinations = (self.max_value - self.min_value + 1) ** self.num_numbers

        if total_combinations > self.num_samples:
            entries = self._sample_entries(self.num_numbers, self.min_value, self.max_value, self.num_samples)
        else:
            entries = self._all_entries(self.num_numbers, self.min_value, self.max_value)

        data = []
        for input_list, output_list in entries:
            if self.generate_scratchpad:
                raise NotImplementedError()
            else:
                prompt = f"input = {input_list}\n\n###\n\n"
                completion = ' ' + f"output = {output_list}" + ' ###'

            data.append([[None, prompt], completion, None])
        return data

    def __len__(self):
        """
        Returns the total number of samples in the dataset.
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        Retrieve a sample by its index.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            sample (list): A list containing the following elements:
                - [None, prompt]: Image placeholder (None) and the task prompt.
                - completion: The task completion (with or without scratchpad).
                - metadata: Metadata (currently None but can be extended).
        """
        sample = self.data[idx]

        if self.transform:
            sample = self.transform(sample)

        return sample


class ZebraPuzzleDataset(torch.utils.data.Dataset):
    def __init__(self, mode='test_id_xl', data_dir='./data/einstein_puzzles', transform=None):
        """
        Initializes the dataset by loading puzzles from the specified mode.

        Args:
            mode (str): The mode of the dataset (e.g., 'train', 'dev', 'test').
            data_dir (str): Path to the directory containing the puzzle data.
            transform (callable, optional): Optional transform to be applied to the data.
        """
        self.mode = mode
        self.data_dir = data_dir
        self.transform = transform

        # Load puzzles
        with open(f"{self.data_dir}/logic_grid_puzzles.{self.mode}.json", "r") as f:
            self.dataset = json.load(f)

    def __len__(self):
        """
        Returns the total number of samples in the dataset.
        """
        return len(self.dataset)

    def __getitem__(self, idx):
        """
        Retrieve a sample by its index.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            sample (list): A list containing the following elements:
                - [None, question]: Image placeholder (None) and the puzzle question context without revealing the answer.
                - answer: The correct answer to the question.
                - metadata: Additional metadata including question data.
        """
        item = self.dataset[idx]

        # Extract fields
        table_data = item['solution']['table_rows']
        col_names = item['solution']['table_header']
        questions = item['questions']

        # Select a random question
        question_data = random.choice(questions)
        raw_question = question_data['question']
        choices = question_data['choices']
        truth_idx = question_data['truth_idx']
        answer = choices[truth_idx]

        # Modify the question to exclude the answer
        question_parts = raw_question.split("?")
        question = question_parts[0] + "?"  # Remove any specifics beyond the question mark

        # Metadata
        metadata = {
            "choices": choices,
            "truth_idx": truth_idx,
            "table_data": table_data,
            "col_names": col_names
        }

        sample = [
            [None, question],  # No image; puzzle question without answer
            answer,             # Correct answer
            metadata            # Additional metadata
        ]

        if self.transform:
            sample = self.transform(sample)

        return sample
    
    
class MultiplicationDataset(torch.utils.data.Dataset):
    def __init__(self, num_digit=4, max_sequence=3000, generate_on_the_fly=True, transform=None):
        """
        Initializes the dataset by generating all data during initialization.

        Args:
            num_digit (int): Maximum number of digits for the numbers.
            max_sequence (int): Maximum number of inputs per combination.
            generate_on_the_fly (bool): Ignored (data is always generated in __init__).
            transform (callable, optional): Optional transform to be applied to the data.
        """
        self.num_digit = num_digit
        self.max_sequence = max_sequence
        self.transform = transform

        self.data = self._generate_dataset()

    def _generate_dataset(self):
        """
        Generate the entire dataset during initialization.

        Returns:
            list: A list of generated samples.
        """
        data = []
        for _ in range(self.max_sequence):
            a = random.randint(10**(self.num_digit - 1), 10**self.num_digit - 1)
            b = random.randint(10**(self.num_digit - 1), 10**self.num_digit - 1)
            question = f"What is {a} times {b}?"
            answer = str(a * b)
            metadata = None
            data.append([[None, question], answer, metadata])
        return data

    def __len__(self):
        """
        Returns the total number of samples in the dataset.
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        Retrieve a sample by its index.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            sample (list): A list containing the following elements:
                - [None, question]: Image placeholder (None) and the multiplication question.
                - answer: The multiplication result.
                - metadata: Metadata (optional, can be extended).
        """
        sample = self.data[idx]

        if self.transform:
            sample = self.transform(sample)

        return sample

class SubsequenceSumDataset(torch.utils.data.Dataset):
    def __init__(self, num_numbers=5, min_value=-5, max_value=5, num_samples=1000, generate_scratchpad=False, transform=None):
        """
        Initializes the dataset by generating all data during initialization.

        Args:
            num_numbers (int): Number of integers in each input sequence.
            min_value (int): Minimum value of integers in the sequence.
            max_value (int): Maximum value of integers in the sequence.
            num_samples (int): Number of samples to generate.
            generate_scratchpad (bool): Whether to include scratchpad explanations in the data.
            transform (callable, optional): Optional transform to be applied to the data.
        """
        self.num_numbers = num_numbers
        self.min_value = min_value
        self.max_value = max_value
        self.num_samples = num_samples
        self.generate_scratchpad = generate_scratchpad
        self.transform = transform

        self.data = self._generate_dataset()
        
    def find_max_sum_nonadjacent(self, arr):
        """
        When there are many results, choose the one that appears first lexicographically,
        where 1=picking the number and 2=not picking it.

        dp[i][0] = maximum subsequence of arr[i:] where we do not use arr[i]
        dp[i][1] = maximum subsequence of arr[i:] where we do use arr[i]

        dp[i][0] = max(dp[i+1][0], dp[i+1][1])
        dp[i][1] = arr[i] + dp[i+1][0]
        """
        N = len(arr)

        dp = [[0 for _ in range(2)] for _ in range(N)]
        dp[N-1][0] = 0
        dp[N-1][1] = arr[N-1]
        for i in range(N-2, -1, -1):
            dp[i][1] = dp[i + 1][0] + arr[i]
            dp[i][0] = max(dp[i + 1][0], dp[i + 1][1])

        max_sum = max(dp[0][0], dp[0][1])

        result = []
        remaining_sum = max_sum
        can_access_next_item = True
        for i in range(N):
            if dp[i][1] == remaining_sum and can_access_next_item:
                result.append(1)
                remaining_sum -= arr[i]
                can_access_next_item = False
            elif dp[i][0] == remaining_sum:
                result.append(2)
                can_access_next_item = True
            else:
                assert False

        return result, max_sum

    def _sample_entries(self, num_numbers, min_value, max_value, num_samples):
        result = set()
        while len(result) < num_samples:
            input_list = [random.randint(min_value, max_value) for _ in range(num_numbers)]

            output_sequence, my_max_sum = self.find_max_sum_nonadjacent(input_list)

            result.add((tuple(input_list), tuple(output_sequence)))

        result = [(list(input_list), list(output_seq)) for input_list, output_seq in result]
        return result

    def _all_entries(self, num_numbers, min_value, max_value):
        all_inputs = itertools.product(list(range(min_value, max_value + 1)), repeat=num_numbers)

        result = []
        for input_list in all_inputs:
            input_list = list(input_list)
            output_sequence, my_max_sum = self.find_max_sum_nonadjacent(input_list)

            result.append((input_list, output_sequence))

        return result

    def _generate_dataset(self):
        """
        Generate the entire dataset during initialization.

        Returns:
            list: A list of generated samples.
        """
        total_combinations = (self.max_value - self.min_value + 1) ** self.num_numbers

        if total_combinations > self.num_samples:
            entries = self._sample_entries(self.num_numbers, self.min_value, self.max_value, self.num_samples)
        else:
            entries = self._all_entries(self.num_numbers, self.min_value, self.max_value)

        data = []
        for input_list, output_list in entries:
            if self.generate_scratchpad:
                raise NotImplementedError()
            else:
                prompt = f"input = {input_list}\n\n###\n\n"
                completion = ' ' + f"output = {output_list}" + ' ###'

            data.append([[None, prompt], completion, None])
        return data

    def __len__(self):
        """
        Returns the total number of samples in the dataset.
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        Retrieve a sample by its index.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            sample (list): A list containing the following elements:
                - [None, prompt]: Image placeholder (None) and the task prompt.
                - completion: The task completion (with or without scratchpad).
                - metadata: Metadata (currently None but can be extended).
        """
        sample = self.data[idx]

        if self.transform:
            sample = self.transform(sample)

        return sample


class ZebraPuzzleDataset(torch.utils.data.Dataset):
    def __init__(self, mode='test_id_xl', data_dir='./data/einstein_puzzles', transform=None):
        """
        Initializes the dataset by loading puzzles from the specified mode.

        Args:
            mode (str): The mode of the dataset (e.g., 'train', 'dev', 'test').
            data_dir (str): Path to the directory containing the puzzle data.
            transform (callable, optional): Optional transform to be applied to the data.
        """
        self.mode = mode
        self.data_dir = data_dir
        self.transform = transform

        # Load puzzles
        with open(f"{self.data_dir}/logic_grid_puzzles.{self.mode}.json", "r") as f:
            self.dataset = json.load(f)

    def __len__(self):
        """
        Returns the total number of samples in the dataset.
        """
        return len(self.dataset)

    def __getitem__(self, idx):
        """
        Retrieve a sample by its index.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            sample (list): A list containing the following elements:
                - [None, question]: Image placeholder (None) and the puzzle question context without revealing the answer.
                - answer: The correct answer to the question.
                - metadata: Additional metadata including question data.
        """
        item = self.dataset[idx]

        # Extract fields
        table_data = item['solution']['table_rows']
        col_names = item['solution']['table_header']
        questions = item['questions']

        # Select a random question
        question_data = random.choice(questions)
        raw_question = question_data['question']
        choices = question_data['choices']
        truth_idx = question_data['truth_idx']
        answer = choices[truth_idx]

        # Modify the question to exclude the answer
        question_parts = raw_question.split("?")
        question = question_parts[0] + "?"  # Remove any specifics beyond the question mark

        # Metadata
        metadata = {
            "choices": choices,
            "truth_idx": truth_idx,
            "table_data": table_data,
            "col_names": col_names
        }

        sample = [
            [None, question],  # No image; puzzle question without answer
            answer,             # Correct answer
            metadata            # Additional metadata
        ]

        if self.transform:
            sample = self.transform(sample)

        return sample


class ListSynthesisDataset(torch.utils.data.Dataset):
    def __init__(self, min_width=1, max_width=5, min_depth=1, max_depth=5):
        self.data = demonstrate_generator(min_width, max_width, min_depth, max_depth)

    def __getitem__(self, index):
        return (None, self.data[index][0]), self.data[index][1]

    def __len__(self):
        return len(self.data)

    def check_correct(self, index, fn):
        return all([fn(*ex) == out for ex, out in self.data[index][2]])


def difficulty(sudoku, difficulty):
    """
    Sets the difficulty of the Sudoku board by removing cells.

    This method modifies the current Sudoku instance by removing cells from the solved puzzle to achieve the desired difficulty level. The difficulty is specified as a float value between 0 and 1, where 0 represents the easiest puzzle (fully solved) and 1 represents the most difficult puzzle (almost empty).

    :param difficulty: A float value between 0 and 1 representing the desired difficulty level of the Sudoku puzzle.
    :return: A new Sudoku instance representing the puzzle with adjusted difficulty.
    :raises AssertionError: If the provided difficulty value is not within the range of 0 to 1.
    """
    assert 0 < difficulty < 1, 'Difficulty must be between 0 and 1'
    indices = list(range(sudoku.size * sudoku.size))
    random.shuffle(indices)
    problem_board = sudoku.solve().board
    removed = 0
    for index in indices[:int(difficulty * sudoku.size * sudoku.size)]:
        row_index = index // sudoku.size
        col_index = index % sudoku.size
        saved_board = copy.deepcopy(problem_board)
        problem_board[row_index][col_index] = Sudoku._empty_cell_value
        if not Sudoku(sudoku.width, sudoku.height, problem_board).has_multiple_solutions():
            removed += 1
        else:
            problem_board = saved_board
        if removed >= difficulty * sudoku.size * sudoku.size:
            break
    return Sudoku(sudoku.width, sudoku.height, problem_board, difficulty)


class SudokuDataset(torch.utils.data.Dataset):
    def __init__(self, min_clues=40, max_clues=70, num_samples=200):
        if os.path.exists(f"data/sudoku/data_{min_clues}_{max_clues}.json"):
            with open(f"data/sudoku/data_{min_clues}_{max_clues}.json", "r") as f:
                self.data = json.load(f)["data"]
        else:
            self.data = []
            clues = np.random.randint(min_clues, max_clues, num_samples)
            i = 0
            for clue in clues:
                puzzle = difficulty(Sudoku(3, seed=i), (81 - clue) / 81)
                query = str(puzzle)
                query = re.sub(r"\n---------------------------\n9x9 \(3x3\) SUDOKU PUZZLE\nDifficulty: 0\.\d\d\n---------------------------\n", "", query).strip()
                solution = str(puzzle.solve()).replace("\n---------------------------\n9x9 (3x3) SUDOKU PUZZLE\nDifficulty: SOLVED\n---------------------------\n", "").strip()
                self.data.append({"board": query, "solution": solution, "clues": str(clue)})
                i += 1
            with open(f"data/sudoku/data_{min_clues}_{max_clues}.json", "w") as f:
                json.dump({"data": self.data}, f, indent=4)
    
    def __getitem__(self, index):
        prompt = "Solve the following Sudoku puzzle:\n```\n"
        return (None, prompt + self.data[index]["board"] + "\n```"), self.data[index]["solution"]

    def __len__(self):
        return len(self.data)


class GenClutrrDataset(torch.utils.data.Dataset):
    def __init__(self):
        # load jsonlines
        self.data = []
        with open("data/CLUTRR/train.jsonl", "r") as f:
            for line in f:
                self.data.append(json.loads(line))

        # subsample to 300
        self.data = [d for d in self.data if len(d["descriptor"].split(",")) >= 8]
        print("Number of samples:", len(self.data))
        random.seed(0)
        self.data = random.sample(self.data, 300)
        
        num_people = [len(d["name_map"]) for d in self.data]
        path_len = [len(d["descriptor"].split(",")) for d in self.data]
        # print histogram as text
        print("Number of people histogram:")
        print(np.histogram(num_people))
        print("Path length histogram:")
        print(np.histogram(path_len))

    
    def __getitem__(self, index):
        story = self.data[index]["text_story"]
        context = [s.strip() for s in story.split(".") if s.strip() != ""]
        query = self.data[index]["query"]
        name_map = self.data[index]["name_map"]
        # query = f"How is {name_map[str(query[1])]} related to {name_map[str(query[0])]}?"
        query = (name_map[str(query[1])], name_map[str(query[0])])
        return (context, query), self.data[index]["target_gender"], len(name_map), len(self.data[index]["descriptor"].split(","))

    def __len__(self):
        return len(self.data)
    

class ClutrrDataset(torch.utils.data.Dataset):
    def __init__(self):
        # load jsonlines
        # self.data = load_dataset("CLUTRR/v1", "gen_train234_test2to10", split="test").to_list()
        self.data = []
        # with open("data/CLUTRR/test.jsonl", "r") as f:
        #     for line in f:
        #         self.data.append(json.loads(line))
        # load from csv
        with open("data/CLUTRR/clutrr_4.csv", "r") as f:
            # read the first line to get the keys
            reader = csv.DictReader(f)
            for row in reader:
                self.data.append({"question": row['story'], "answer": row['target'], "query": row['query']})

        # subsample to 300
        print("Number of samples:", len(self.data))
        # random.seed(0)
        # self.data = random.sample(self.data, 300)
        
        # num_people = [len(d["genders"].split(",")) for d in self.data]
        # num_people = [len(d["name_map"]) for d in self.data]
        # # print histogram as text
        # print("Number of people histogram:")
        # print(np.histogram(num_people))
    
    def __getitem__(self, index):
        # story = self.data[index]["question"].split("\n")[0]
        story = self.data[index]["question"]
        context = [s.strip() for s in story.split(".") if s.strip() != ""]
        query = ast.literal_eval(self.data[index]["query"])[::-1]
        # query = self.data[index]["question"].split("\n")[1]
        # get the two names in [] from the query as a tuple
        # query = str(re.findall(r"\[(.*?)\]", query))

        # return (story, query), self.data[index]["answer"].split("#### ")[1]
        return (story, query), self.data[index]["answer"]

    def __len__(self):
        return len(self.data)


class LeafDataset(torch.utils.data.Dataset):
    def __init__(self, root="./"):
        # data is stored in a directory data/leaf-11 where each subdirectory is a class
        self.data = []
        for i, class_dir in enumerate(os.listdir(root + "data/leaf_11")):
            for img_file in os.listdir(root + f"data/leaf_11/{class_dir}"):
                img_path = root + f"data/leaf_11/{class_dir}/{img_file}"
                label = class_dir
                self.data.append(([Image.open(img_path).convert("RGB")], label))
        
        # subsample to 200 samples
        random.seed(0)
        self.data = random.sample(self.data, 200)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)
    

class TwentyFourGameDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        n: int,
        data_dir: str = "./data/24game/",
        max_samples: Optional[int] = None,
        filename_pattern: str = "num({})_samples({}).jsonl",
        num_samples_per_file: int = 200,
    ):
        """
        Initializes the TwentyFourGameDataset.

        Args:
            data_dir (str): Directory where the JSONL files are stored.
            n (int): Number of numbers in each sample (corresponds to 'num' in the filename).
            max_samples (Optional[int], optional): Maximum number of samples to load. Defaults to None.
            filename_pattern (str, optional): Pattern to format the filename. Defaults to "num({})_samples({}).jsonl".
            num_samples_per_file (int, optional): Number of samples per file. Defaults to 200.
        """
        self.data_dir = data_dir
        self.n = n
        self.max_samples = max_samples
        self.target = 24  # Fixed target for the 24 Game

        # Construct the filename based on 'n' and 'num_samples_per_file'
        filename = filename_pattern.format(n, num_samples_per_file)
        file_path = os.path.join(data_dir, filename)

        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"Data file not found: {file_path}")

        # Load all samples into memory
        self.samples = []
        with open(file_path, 'r') as f:
            for line in f:
                data = json.loads(line.strip())
                numbers = data.get("numbers", [])
                if len(numbers) != self.n:
                    raise ValueError(
                        f"Sample index {data.get('sample_index')} has {len(numbers)} numbers, expected {self.n}."
                    )
                self.samples.append(numbers)

        # Apply max_samples if specified
        if self.max_samples is not None:
            self.samples = self.samples[:self.max_samples]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[Tuple[Optional[torch.Tensor], List[int]], int, List]:
        """
        Retrieves the sample at the specified index.

        Args:
            index (int): Index of the sample to retrieve.

        Returns:
            Tuple containing:
                - Tuple (image, numbers): image is None, numbers is the list of integers.
                - target: Fixed integer 24.
                - Additional info: Empty list.
        """
        if index < 0 or index >= len(self):
            raise IndexError(f"Index {index} out of range for dataset of size {len(self)}.")

        numbers = self.samples[index]
        
        query = f"""Play the 24 game with the following list of numbers: {numbers}.  Only use +, -, *, and /.  Every list index must be used. You can assume a solution always exists. Find a valid expression using all of the numbers and return what that expression is."""
        
        return ((None, query), self.target, numbers)

    

# TODO: This dataset is currently unsupported because it requires GPT-4 to
# evaluate the generated plans. We can either implement this or find a
# workaround.
class TravelPlannerDataset(torch.utils.data.Dataset):
    def __init__(self):
        self.data = load_dataset("osunlp/TravelPlanner", "validation")
        self.prompt = """You are a proficient planner. Based on the provided information and query, please give me a detailed plan, including specifics such as flight numbers (e.g., F0123456), restaurant names, and accommodation names. Note that all the information in your plan should be derived from the provided data. You must adhere to the format given in the example. Additionally, all details should align with commonsense. The symbol '-' indicates that information is unnecessary. For example, in the provided sample, you do not need to plan after returning to the departure city. When you travel to two cities in one day, you should note it in the 'Current City' section as in the example (i.e., from A to B).

***** Example *****
Query: Could you create a travel plan for 7 people from Ithaca to Charlotte spanning 3 days, from March 8th to March 14th, 2022, with a budget of $30,200?
Travel Plan:
Day 1:
Current City: from Ithaca to Charlotte
Transportation: Flight Number: F3633413, from Ithaca to Charlotte, Departure Time: 05:38, Arrival Time: 07:46
Breakfast: Nagaland's Kitchen, Charlotte
Attraction: The Charlotte Museum of History, Charlotte
Lunch: Cafe Maple Street, Charlotte
Dinner: Bombay Vada Pav, Charlotte
Accommodation: Affordable Spacious Refurbished Room in Bushwick!, Charlotte

Day 2:
Current City: Charlotte
Transportation: -
Breakfast: Olive Tree Cafe, Charlotte
Attraction: The Mint Museum, Charlotte;Romare Bearden Park, Charlotte.
Lunch: Birbal Ji Dhaba, Charlotte
Dinner: Pind Balluchi, Charlotte
Accommodation: Affordable Spacious Refurbished Room in Bushwick!, Charlotte

Day 3:
Current City: from Charlotte to Ithaca
Transportation: Flight Number: F3786167, from Charlotte to Ithaca, Departure Time: 21:42, Arrival Time: 23:26
Breakfast: Subway, Charlotte
Attraction: Books Monument, Charlotte.
Lunch: Olive Tree Cafe, Charlotte
Dinner: Kylin Skybar, Charlotte
Accommodation: -

***** Example Ends *****

Given information: {text}
Query: {query}
Travel Plan:"""

    def __getitem__(self, index):
        query = self.data[index]["query"]
        info = self.data[index]["reference_information"]
        return (None, self.prompt.format(text=info, query=query)), (self.data[index]["level"], self.data[index]["local_constraint"])

    def __len__(self):
        return len(self.data)


class OmniMathDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        split: str = "test",
        max_samples: Optional[int] = None,
        allowed_difficulties: Optional[List[float]] = None
    ):
        """
        A Dataset wrapper around the Hugging Face dataset "KbsdJames/Omni-MATH".

        Args:
            split (str): Which split of the dataset to load ("train", "validation", "test", etc.).
            max_samples (Optional[int]): If provided, cap the number of samples to this value.
            allowed_difficulties (Optional[List[float]]): List of allowed difficulties. If provided,
                only samples whose 'difficulty' is in this list will be retained.
        """
        # Load the specified split of the Omni-MATH dataset
        self.dataset = load_dataset("KbsdJames/Omni-MATH", split=split)

        # If allowed_difficulties is specified, filter the dataset
        if allowed_difficulties is not None:
            self.dataset = self.dataset.filter(
                lambda x: x["difficulty"] in allowed_difficulties
            )

        # Apply max_samples if specified
        if max_samples is not None:
            self.dataset = self.dataset.select(range(min(max_samples, len(self.dataset))))

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> Tuple[Tuple[None, str], str, List[str]]:
        """
        Retrieve a single sample from the Omni-MATH dataset.

        Returns:
            A tuple of the form:
                (
                    (None, problem),
                    answer,
                    [solution],
                )
        """
        if index < 0 or index >= len(self):
            raise IndexError(f"Index {index} out of range for dataset of size {len(self)}.")

        item = self.dataset[index]

        problem = item["problem"]
        answer = item["answer"]
        solution = item["solution"]

        # Return ((None, problem), answer, [solution])
        return ( (None, problem), answer, [solution] )


def main():
    # d = FOLIODataset()
    # d[0]
    # d = LongSortDataset()
    # d = BlocksWorldDataset()
    # d = SudokuDataset()
    d = ClutrrDataset()
    print(d[10][0][0])
    print(d[10][0][1])
    print(d[10][0][2])
    print()
    print(d[10][1])


if __name__ == "__main__":
    main()
