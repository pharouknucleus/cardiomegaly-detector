import argparse
import os
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from .config import TrainConfig
from .data.cxr_dataset import CXRCsvDataset
from .models.densenet_cardiomegaly import CardiomegalyDenseNet


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Train cardiomegaly detector")
    parser.add_argument("--csv", type=str, required=True, help="Path to splits CSV")
    parser.add_argument("--img_root", type=str, required=True, help="Root dir with images")
    parser.add_argument("--output_dir", type=str, default="runs/cardiomegaly_densenet")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument("--multi_gpu", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--img_size", type=int, default=320)
    parser.add_argument("--pos_weight", type=float, default=None)

    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    return TrainConfig(
        csv=args.csv,
        img_root=args.img_root,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        use_cuda=use_cuda,
        multi_gpu=args.multi_gpu,
        seed=args.seed,
        img_size=args.img_size,
        pos_weight=args.pos_weight,
    )


def set_seed(seed: int):
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_dataloaders(cfg: TrainConfig) -> Tuple[DataLoader, DataLoader]:
    train_tf = transforms.Compose([
        transforms.Resize((cfg.img_size, cfg.img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    val_tf = transforms.Compose([
        transforms.Resize((cfg.img_size, cfg.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    train_ds = CXRCsvDataset(cfg.csv, cfg.img_root, split="train", transform=train_tf)
    val_ds = CXRCsvDataset(cfg.csv, cfg.img_root, split="val", transform=val_tf)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=cfg.use_cuda,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=cfg.use_cuda,
    )
    return train_loader, val_loader


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion,
    optimizer,
    device: torch.device,
) -> float:
    model.train()
    running_loss = 0.0

    for imgs, labels in tqdm(loader, desc="Train", leave=False):
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(imgs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * imgs.size(0)

    return running_loss / len(loader.dataset)


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion,
    device: torch.device,
):
    model.eval()
    running_loss = 0.0
    all_logits = []
    all_labels = []

    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc="Val", leave=False):
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device)

            logits = model(imgs)
            loss = criterion(logits, labels)

            running_loss += loss.item() * imgs.size(0)
            all_logits.append(logits.detach().cpu())
            all_labels.append(labels.detach().cpu())

    val_loss = running_loss / len(loader.dataset)
    logits = torch.cat(all_logits).numpy()
    labels = torch.cat(all_labels).numpy()

    probs = 1 / (1 + np.exp(-logits))

    try:
        auc = roc_auc_score(labels, probs)
    except ValueError:
        auc = float("nan")

    preds = (probs >= 0.5).astype(int)

    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, zero_division=0)
    prec = precision_score(labels, preds, zero_division=0)
    rec = recall_score(labels, preds, zero_division=0)

    metrics = {
        "val_loss": val_loss,
        "auc": auc,
        "accuracy": acc,
        "f1": f1,
        "precision": prec,
        "recall": rec,
    }
    return metrics


def main():
    cfg = parse_args()
    set_seed(cfg.seed)

    os.makedirs(cfg.output_dir, exist_ok=True)
    device = torch.device("cuda" if cfg.use_cuda else "cpu")

    train_loader, val_loader = get_dataloaders(cfg)

    model = CardiomegalyDenseNet(pretrained=True)

    if cfg.multi_gpu and cfg.use_cuda and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    model = model.to(device)

    pos_weight = None
    if cfg.pos_weight is not None:
        pos_weight = torch.tensor([cfg.pos_weight], device=device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )

    best_auc = -1.0

    for epoch in range(1, cfg.epochs + 1):
        print(f"\nEpoch {epoch}/{cfg.epochs}")

        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        metrics = evaluate(model, val_loader, criterion, device)

        print(
            f"Train loss: {train_loss:.4f} | "
            f"Val loss: {metrics['val_loss']:.4f} | "
            f"AUC: {metrics['auc']:.4f} | "
            f"F1: {metrics['f1']:.4f}"
        )

        # Save last
        ckpt = {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "epoch": epoch,
            "cfg": cfg.__dict__,
            "metrics": metrics,
        }
        torch.save(ckpt, Path(cfg.output_dir) / "last_model.pt")

        # Save best by AUC
        if metrics["auc"] > best_auc:
            best_auc = metrics["auc"]
            torch.save(ckpt, Path(cfg.output_dir) / "best_model.pt")
            print(f"New best AUC: {best_auc:.4f} (checkpoint saved)")

    print("Training finished.")


if __name__ == "__main__":
    main()
