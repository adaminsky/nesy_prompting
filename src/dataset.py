import torchvision
import torch
import logging
import json
import os
from PIL import Image
import numpy as np
from datasets import load_dataset
from typing import Optional, Callable

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

        return *imgs, sum_label

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
    def __init__(self, root: str, split: str, length: int):
        super(HWFDataset, self).__init__()
        self.root = root
        self.split = split
        md = json.load(open(os.path.join(root, f"HWF/expr_{split}.json")))

        # finding only the metadata with length == 1
        if length > 0:
            self.metadata = [m for m in md if len(m["img_paths"]) == length]
        else:
            self.metadata = md

        self.img_transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                # torchvision.transforms.Normalize((0.5,), (1,)),
            ]
        )

    def __getitem__(self, index):
        sample = self.metadata[index]

        # Input is a sequence of images
        img_seq = []
        for img_path in sample["img_paths"]:
            img_full_path = os.path.join(
                self.root, "HWF/Handwritten_Math_Symbols", img_path
            )
            img = Image.open(img_full_path)
            img = self.img_transform(img)
            print(img.shape)
            img_seq.append(img[0])
        img_seq_len = len(img_seq)

        img = np.concatenate(img_seq, axis=1)
        img = Image.fromarray(img * 255)

        # Output is the "res" in the sample of metadata
        res = sample["res"]

        # Return (input, output) pair
        return ((img, None), res)

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


def main():
    d = ClevrDataset(
        "./data/CLEVR_v1.0/questions/CLEVR_train_questions.json",
        "./data/CLEVR_v1.0/images/train/",
        "./data/CLEVR_v1.0/scenes/CLEVR_train_scenes.json",
        max_samples=100,
    )
    d[0]


if __name__ == "__main__":
    main()
