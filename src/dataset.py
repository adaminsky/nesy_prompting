import torchvision
import torch
import logging
import json
import os
from PIL import Image

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
        return img, sum_label, *labels

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
            img = Image.open(img_full_path).convert("L")
            img = self.img_transform(img)
            img_seq.append(img)
        img_seq_len = len(img_seq)

        # Output is the "res" in the sample of metadata
        res = sample["res"]

        # Return (input, output) pair
        return (img_seq, img_seq_len, res)

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


def main():
    logger.info("Downloading required datasets")
    MNISTSumKDataset(root="./data", train=True, download=True)


if __name__ == "__main__":
    main()
