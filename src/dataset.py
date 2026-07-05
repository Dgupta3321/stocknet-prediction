import torch
import numpy as np
from torch.utils.data import Dataset

class StockSequenceDataset(Dataset):
    def __init__(self, features, targets, sequence_length=5):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)
        self.sequence_length = sequence_length

    def __len__(self):
        return len(self.features) - self.sequence_length

    def __getitem__(self, idx):
        x = self.features[idx : idx + self.sequence_length]
        y = self.targets[idx + self.sequence_length]
        return x, y
