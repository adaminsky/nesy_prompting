import torchvision
import torch
import logging
from PIL import Image
logger = logging.getLogger(__name__)


class MNISTSumKOrigDataset(torch.utils.data.Dataset):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False, k=2):
        self.mnist = torchvision.datasets.MNIST(root, train, transform, target_transform, download)
        self.train = train
        self.k = 2

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
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False, k=2):
        self.mnist = torchvision.datasets.MNIST(root, train, transform, target_transform, download)
        self.train = train
        self.k = 2

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


def main():
    logger.info("Downloading required datasets")
    MNISTSumKDataset(root="./data", train=True, download=True)


if __name__ == "__main__":
    main()
