import csv
import os
import random
from dataclasses import dataclass
from typing import List, Tuple

import sys

import numpy as np

# Ensure project root is importable when running as a script.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from perception.offline_vision_classifier import LABELS, LABEL_TO_IDX


def _lazy_import_torch():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    return torch, nn, F


def _lazy_import_pil():
    from PIL import Image
    return Image


def load_dataset_rows(labels_csv: str) -> List[dict]:
    with open(labels_csv, "r", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)


def preprocess_image(path: str, size: int = 128) -> np.ndarray:
    Image = _lazy_import_pil()
    img = Image.open(path).convert("RGB").resize((size, size))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1))  # CHW
    return arr


def build_model(num_classes: int):
    torch, nn, F = _lazy_import_torch()

    class TinyCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2, padding=2)
            self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
            self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
            self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
            self.fc1 = nn.Linear(128 * 8 * 8, 256)
            self.fc2 = nn.Linear(256, num_classes)

        def forward(self, x):
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
            x = F.relu(self.conv3(x))
            x = F.relu(self.conv4(x))
            x = x.flatten(1)
            x = F.relu(self.fc1(x))
            return self.fc2(x)

    return TinyCNN()


@dataclass
class Config:
    data_dir: str = "data/offline_vlm"
    image_size: int = 128
    batch_size: int = 64
    epochs: int = 8
    lr: float = 1e-3
    seed: int = 0
    val_split: float = 0.1
    out_path: str = "models/offline_vlm/tinycnn_direction.pt"


def main(cfg: Config = Config()):
    torch, nn, F = _lazy_import_torch()

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    labels_csv = os.path.join(cfg.data_dir, "labels.csv")
    rows = load_dataset_rows(labels_csv)

    # filter rows with known labels
    usable = [r for r in rows if r["label"] in LABEL_TO_IDX]
    random.shuffle(usable)

    n_val = max(1, int(len(usable) * cfg.val_split))
    val_rows = usable[:n_val]
    train_rows = usable[n_val:]

    if len(train_rows) < 50:
        raise RuntimeError(
            f"Not enough training samples ({len(train_rows)}). "
            "Collect more with: python experiments/collect_offline_vision_data.py"
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)
    print("Train samples:", len(train_rows), "Val samples:", len(val_rows))

    model = build_model(num_classes=len(LABELS)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    criterion = nn.CrossEntropyLoss()

    def iter_batches(rows_list):
        for i in range(0, len(rows_list), cfg.batch_size):
            batch = rows_list[i : i + cfg.batch_size]
            xs = []
            ys = []
            for r in batch:
                img_path = os.path.join(cfg.data_dir, r["filename"])
                x = preprocess_image(img_path, size=cfg.image_size)
                xs.append(x)
                ys.append(LABEL_TO_IDX[r["label"]])
            x_t = torch.from_numpy(np.stack(xs)).to(device)
            y_t = torch.tensor(ys, dtype=torch.long).to(device)
            yield x_t, y_t

    def eval_accuracy(rows_list):
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x_t, y_t in iter_batches(rows_list):
                logits = model(x_t)
                pred = torch.argmax(logits, dim=-1)
                correct += int((pred == y_t).sum().item())
                total += int(y_t.numel())
        model.train()
        return correct / max(1, total)

    for epoch in range(cfg.epochs):
        losses = []
        for x_t, y_t in iter_batches(train_rows):
            opt.zero_grad()
            logits = model(x_t)
            loss = criterion(logits, y_t)
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))

        train_acc = eval_accuracy(train_rows[: min(len(train_rows), 1000)])
        val_acc = eval_accuracy(val_rows)
        print(
            f"Epoch {epoch+1}/{cfg.epochs} "
            f"loss={np.mean(losses):.4f} train_acc={train_acc:.3f} val_acc={val_acc:.3f}"
        )

    os.makedirs(os.path.dirname(cfg.out_path), exist_ok=True)
    torch.save(model.state_dict(), cfg.out_path)
    print("Saved model:", cfg.out_path)


if __name__ == "__main__":
    main()

