import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=str, required=True, help="Path to NIH metadata CSV")
    parser.add_argument("--out_csv", type=str, required=True)
    parser.add_argument("--val_fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)

    df = pd.read_csv(args.metadata)

    # Expect columns: "Image Index", "Finding Labels"
    df["path"] = df["Image Index"]
    df["cardiomegaly_label"] = df["Finding Labels"].str.contains("Cardiomegaly").astype(int)

    # Stratified split by cardiomegaly label
    train_idx = []
    val_idx = []

    for label, group in df.groupby("cardiomegaly_label"):
        idx = np.arange(len(group))
        np.random.shuffle(idx)
        n_val = int(len(group) * args.val_fraction)
        val_local = idx[:n_val]
        train_local = idx[n_val:]

        val_idx.extend(group.index[val_local].tolist())
        train_idx.extend(group.index[train_local].tolist())

    df["split"] = "train"
    df.loc[val_idx, "split"] = "val"

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Saved splits to {out_path}")


if __name__ == "__main__":
    main()
