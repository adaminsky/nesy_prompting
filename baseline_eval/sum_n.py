from datetime import datetime
import os
import random
from typing import *
import time
import string
from src.dataset import MNISTSumKOrigDataset
import numpy as np

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from argparse import ArgumentParser
from tqdm import tqdm
import pickle

# import scallopy

mnist_img_transform = torchvision.transforms.Compose([
  torchvision.transforms.ToTensor(),
  torchvision.transforms.Normalize(
    (0.1307,), (0.3081,)
  )
])

class MNISTSumNDataset(torch.utils.data.Dataset):
  def __init__(
    self,
    root: str,
    sum_n: int,
    train: bool = True,
    transform: Optional[Callable] = None,
    target_transform: Optional[Callable] = None,
    download: bool = False,
  ):
    # Contains a MNIST dataset
    self.mnist_dataset = torchvision.datasets.MNIST(
      root,
      train=train,
      transform=transform,
      target_transform=target_transform,
      download=download,
    )
    self.sum_n = sum_n
    self.index_map = list(range(len(self.mnist_dataset)))
    random.shuffle(self.index_map)

  def __len__(self):
    return int(len(self.mnist_dataset) / self.sum_n)

  def __getitem__(self, idx):
    # Get n data points
    imgs = ()
    sum = 0
    for i in range(self.sum_n):
      img, digit = self.mnist_dataset[self.index_map[idx*self.sum_n + i]]
      imgs = imgs + (img,)
      sum += digit 
    # Each data has two images and the GT is the sum of n digits
    return (*imgs, sum)

  @staticmethod
  def collate_fn(batch):
    imgs = ()
    for i in range(len(batch[0])-1):
      a = torch.stack([item[i] for item in batch])
      imgs = imgs + (a,)
    digits = torch.stack([torch.tensor(item[len(batch[0])-1]).long() for item in batch])
    return ((imgs), digits)


def mnist_sum_n_loader(data_dir, sum_n, batch_size_train, batch_size_test):
  train_loader = torch.utils.data.DataLoader(
    MNISTSumNDataset( 
      data_dir,
      sum_n,
      train=True,
      download=True,
      transform=mnist_img_transform,
    ),
    collate_fn=MNISTSumNDataset.collate_fn,
    batch_size=batch_size_train,
    shuffle=True
  )

  test_loader = MNISTSumKOrigDataset(root="data", train=False, transform=mnist_img_transform, download=True, k=5, noise=0.01)

  test_data_ids = list(range(min(200, len(test_loader)))) #+ list(range(103, len(data)))
  shuf = np.random.permutation(test_data_ids)
  test_loader = [test_loader[int(i)] for i in shuf[:200]]

  return train_loader, test_loader


class MNISTNet(nn.Module):
  def __init__(self):
    super(MNISTNet, self).__init__()
    self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
    self.conv2 = nn.Conv2d(32, 64, kernel_size=5)
    self.fc1 = nn.Linear(1024, 1024)
    self.fc2 = nn.Linear(1024, 10)

  def forward(self, x):
    x = F.max_pool2d(self.conv1(x), 2)
    x = F.max_pool2d(self.conv2(x), 2)
    x = x.view(-1, 1024)
    x = F.relu(self.fc1(x))
    x = F.dropout(x, p = 0.5, training=self.training)
    x = self.fc2(x)
    return F.softmax(x, dim=1)


class MNISTSumNNet(nn.Module):
  def __init__(self, provenance, k, sum_n):
    super(MNISTSumNNet, self).__init__()

    # MNIST Digit Recognition Network
    self.mnist_net = MNISTNet()

    # # Scallop Context
    # self.scl_ctx = scallopy.ScallopContext(provenance=provenance, k=k)
    # self.relations = [] 
    # self.variables = []
    # for x in range(1, sum_n+1):
    #      self.scl_ctx.add_relation(f'digit_{x}', int, input_mapping=list(range(10)))
    #      a =  f'{random.choice(string.ascii_letters)}{x}'
    #      self.variables += [a]
    #      self.relations += [f'digit_{x}({a})']
    # self.scl_ctx.add_rule(f'sum_{sum_n}({"+".join(self.variables)}) = {", ".join(self.relations)}')
    # # The `sum_n` logical reasoning module
    # self.sum_n = self.scl_ctx.forward_function(f'sum_{sum_n}', output_mapping=[(i,) for i in range((sum_n*9)+1)], jit=args.jit, dispatch=args.dispatch)

  def forward(self, x: Tuple[torch.Tensor, ...]):
    # First recognize the n digits
    """distr = ()
    for i in x:
      distr = distr + (self.mnist_net(i),)
    # Then execute the reasoning module
    parameters = {}
    for i in range(len(distr)):
      parameter_name = f"digit_{i+1}"
      parameters[parameter_name] = distr[i]"""
    outputs = []
    for i in range(len(x)):
      outputs.append(self.mnist_net(x[i]))
    return outputs

def bce_loss(output, ground_truth):
  (_, dim) = output.shape
  gt = torch.stack([torch.tensor([1.0 if i == t else 0.0 for i in range(dim)]) for t in ground_truth])
  return F.binary_cross_entropy(output, gt)


def nll_loss(output, ground_truth):
  return F.nll_loss(output, ground_truth)


class Trainer():
  def __init__(self, train_loader, test_loader, model_dir, learning_rate, loss, k, provenance, sum_n):
    self.model_dir = model_dir
    self.network = MNISTSumNNet(provenance, k, sum_n)
    self.network.mnist_net = torch.load(os.path.join(model_dir, "sum_5_cnn_best.pt"), weights_only=False)
    self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)
    self.train_loader = train_loader
    self.test_loader = test_loader
    self.best_loss = 10000000000
    self.sum_n = sum_n
    self.provenance = provenance
    self.best_acc = 0
    if loss == "nll":
      self.loss = nll_loss
    elif loss == "bce":
      self.loss = bce_loss
    else:
      raise Exception(f"Unknown loss function `{loss}`")

  def train_epoch(self, epoch):
    self.network.train()
    iter = tqdm(self.train_loader, total=len(self.train_loader))
    t_begin_epoch = time.time()
    for (data, target) in iter:
      self.optimizer.zero_grad()
      output = self.network(data)
      loss = self.loss(output, target)
      loss.backward()
      self.optimizer.step()
      iter.set_description(f"[Train Epoch {epoch}] Loss: {loss.item():.4f}")
    total_epoch_time = time.time() - t_begin_epoch
    wandb.log(
      {
        "epoch": epoch,
        "total_epoch_time": total_epoch_time,
      }
    )
    print(f"Total Epoch Time: {total_epoch_time}")
  def test_epoch(self, epoch):
    self.network.eval()
    num_items = len(self.test_loader)
    test_loss = 0
    correct = 0
    outputs = []
    with torch.no_grad():
      iter = tqdm(self.test_loader, total=len(self.test_loader))
      for (data, target, _) in iter:
        output = self.network(data)
        outputs.append(output)
        pred = sum(torch.argmax(o, dim=1) for o in output)
        correct += 1 if pred == target else 0
        perc = 100. * correct / num_items
        if perc > 97.00:
           # record sum_n + epoch number combination when accuracy is high
          file_path = f'scallop_mnist_sum_n_{self.provenance}_epoch_count.log'
          current_timestamp = datetime.now()
          formatted_timestamp = current_timestamp.strftime("%Y-%m-%d %H:%M:%S")
          if os.path.exists(file_path):
            with open(file_path, 'a') as file:
             file.write(f'sum n={self.sum_n}, epoch num={epoch}, {formatted_timestamp}\n')
          else:
            with open(file_path, 'w') as file:
             file.write(f'sum n={self.sum_n}, epoch num={epoch}, {formatted_timestamp}\n') 
        iter.set_description(f"[Test Epoch {epoch}] Total loss: {test_loss:.4f}, Accuracy: {correct}/{num_items} ({perc:.2f}%)")

      # pickle.dump(outputs, open(f"baseline_outputs/scallop/mnist.pkl", "wb"))

      print(f"Best loss: {self.best_loss:.4f}")
      print(f"Best acc: {self.best_acc:.2f}%")

  def train(self, n_epochs):
    self.test_epoch(0)
    for epoch in range(1, n_epochs + 1):
      self.train_epoch(epoch)
      self.test_epoch(epoch)

if __name__ == "__main__":
  # Argument parser
  parser = ArgumentParser("mnist_sum_n")
  parser.add_argument("--sum-n", type=int, default=2)
  parser.add_argument("--n-epochs", type=int, default=10)
  parser.add_argument("--batch-size-train", type=int, default=64)
  parser.add_argument("--batch-size-test", type=int, default=64)
  parser.add_argument("--learning-rate", type=float, default=0.001)
  parser.add_argument("--loss-fn", type=str, default="bce")
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument("--provenance", type=str, default="difftopkproofs")
  parser.add_argument("--top-k", type=int, default=1)
  parser.add_argument("--jit", action="store_true")
  parser.add_argument("--dispatch", type=str, default="parallel")
  args = parser.parse_args()

  # Parameters
  sum_n = args.sum_n
  n_epochs = args.n_epochs
  batch_size_train = args.batch_size_train
  batch_size_test = args.batch_size_test
  learning_rate = args.learning_rate
  loss_fn = args.loss_fn
  k = args.top_k
  provenance = args.provenance
  torch.manual_seed(args.seed)
  np.random.seed(args.seed)
  random.seed(args.seed)


  config = {
    "sum_n": sum_n,
    "n_epochs": n_epochs,
    "batch_size_train": batch_size_train, 
    "batch_size_test": batch_size_test,
    "provenance": provenance,
    "seed": args.seed, 
    "experiment_type": "scallop",
  }

  # Data
  data_dir = os.path.abspath(os.path.join(os.path.abspath(__file__), "../../data"))
  model_dir = "/home/steinad/common-data/aadityanaik/scallop_models/mnist/"
  os.makedirs(model_dir, exist_ok=True)

  # Dataloaders
  train_loader, test_loader = mnist_sum_n_loader(data_dir, sum_n, batch_size_train, batch_size_test)
  timestamp = datetime.now()
  id = f'scallop_sum{sum_n}_{args.seed}_{provenance}_{timestamp.strftime("%Y-%m-%d %H-%M-%S")}'
  # Create trainer and train
  trainer = Trainer(train_loader, test_loader, model_dir, learning_rate, loss_fn, k, provenance, sum_n)
  trainer.test_epoch(0)
