
import torch
from torch.utils.data import Dataset
import numpy as np
import os

class ACDCDataset(Dataset):
    def __init__(self, data_dir, split="trn"):
        self.data = np.transpose(np.load(os.path.join(data_dir, f"{split}_dat.npy")), (0, 1, 4, 3, 2))[:, :, None]
        self.frames = self.data.shape[1]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        target_idx = np.random.randint(1, self.frames)
        missing_mask = [1]
        if target_idx>1:
            if target_idx>2:
                missing_mask = np.append(missing_mask, np.random.randint(0, 2, size=(target_idx-2,)))
            missing_mask = np.append(missing_mask, [1])
        missing_mask = np.append(missing_mask, np.zeros(self.frames - len(missing_mask))).astype(np.float32)

        x_prev = np.clip(self.data[idx, :-1] * missing_mask[:-1, None, None ,None, None], 0., 1.)
        x = self.data[idx, target_idx]
        return torch.from_numpy(x), torch.from_numpy(x_prev)