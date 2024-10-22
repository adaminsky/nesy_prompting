import torchvision
import torch
import logging
from PIL import Image
logger = logging.getLogger(__name__)


class MNISTSum2Dataset(torch.utils.data.Dataset):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        self.mnist = torchvision.datasets.MNIST(root, train, transform, target_transform, download)
        self.train = train

    def __getitem__(self, index):
        img1, label1 = self.mnist[index * 2]
        img2, label2 = self.mnist[index * 2 + 1]
        sum_label = (label1 + label2)

        img = Image.new("RGB", (img1.width + img2.width, img1.height))
        img.paste(img1, (0, 0))
        img.paste(img2, (img1.width, 0))
        return img, sum_label

    def __len__(self):
        return len(self.mnist) // 2



def main():
    logger.info("Downloading required datasets")
    MNISTSum2Dataset(root="./data", train=True, download=True)


if __name__ == "__main__":
    main()
