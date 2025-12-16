import numpy
import torch

from datasets import load_dataset

ds = load_dataset("starhopp3r/TinyChat")


print(ds["train"][:100])
#loc = torch.device("mps")