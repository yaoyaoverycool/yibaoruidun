import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
class IdentifyDataset(Dataset):
    def __init__(self, data: pd.DataFrame):
        self.features = data
        self.data = torch.tensor(self.features.values[:, 1:-1], dtype=torch.float32)
        self.label = torch.tensor(self.features.values[:, -1], dtype=torch.long)

    def __getitem__(self, index):
        x = self.data[index]
        y = self.label[index]
        return x, y

    def __len__(self):
        return len(self.features)