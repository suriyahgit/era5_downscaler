from typing import Tuple
import numpy as np
import torch
from torch.utils.data import Dataset


def extract_patches(
    arr: np.ndarray,
    patch_size_y: int,
    patch_size_x: int,
    stride_y: int,
    stride_x: int,
) -> np.ndarray:
    """
    arr: (T, C, Y, X)
    returns: (N_patches, C, patch_y, patch_x)
    """
    T, C, Y, X = arr.shape
    patches = []
    for t in range(T):
        for y in range(0, Y - patch_size_y + 1, stride_y):
            for x in range(0, X - patch_size_x + 1, stride_x):
                patch = arr[t, :, y : y + patch_size_y, x : x + patch_size_x]
                patches.append(patch)
    if not patches:
        raise ValueError("No patches extracted — check patch size/stride vs domain size.")
    return np.stack(patches, axis=0)


class PatchDataset(Dataset):
    def __init__(
        self,
        predictors: np.ndarray,  # (T, Cx, Y, X)
        targets: np.ndarray,     # (T, Cy, Y, X)
        patch_size: Tuple[int, int],
        stride: Tuple[int, int],
    ):
        patch_y, patch_x = patch_size
        stride_y, stride_x = stride

        self.X = extract_patches(predictors, patch_y, patch_x, stride_y, stride_x)
        self.Y = extract_patches(targets, patch_y, patch_x, stride_y, stride_x)

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int):
        x = torch.from_numpy(self.X[idx]).float()
        y = torch.from_numpy(self.Y[idx]).float()
        return x, y
