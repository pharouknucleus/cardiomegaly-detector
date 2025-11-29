import os
from typing import Callable, Optional

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class CXRCsvDataset(Dataset):
    """
    Chest X-ray dataset using a CSV file with columns:
      - path: relative path to image
      - cardiomegaly_label: 0 or 1
      - split: 'train' or 'val' (handled outside this class)
    """

    def __init__(
        self,
        csv_path: str,
        img_root: str,
        split: str,
        transform: Optional[Callable] = None,
    ):
        self.df = pd.read_csv(csv_path)
        self.df = self.df[self.df["split"] == split].reset_index(drop=True)
        self.img_root = img_root
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_root, row["path"])
        img = Image.open(img_path).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        label = float(row["cardiomegaly_label"])
        return img, label
