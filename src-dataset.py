import torch
from torch.utils.data import Dataset

class PriceDataset(Dataset):
    def __init__(self, sequences, targets):
        self.sequences = torch.from_numpy(sequences).float()
        self.targets = torch.from_numpy(targets).float()

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]
