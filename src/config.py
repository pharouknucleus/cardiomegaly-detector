from dataclasses import dataclass

@dataclass
class TrainConfig:
    csv: str
    img_root: str
    output_dir: str = "runs/cardiomegaly_densenet"
    batch_size: int = 32
    num_workers: int = 4
    epochs: int = 20
    lr: float = 1e-4
    weight_decay: float = 1e-4
    use_cuda: bool = True
    multi_gpu: bool = False
    seed: int = 42
    img_size: int = 320
    pos_weight: float | None = None  # for BCEWithLogitsLoss (handles imbalance)
