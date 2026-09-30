import random
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pandas as pd
import pydicom
import timm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from ultralytics.nn.modules import Detect
from ultralytics.utils.loss import v8DetectionLoss

from mammo_prep.windowing import preprocess_window

# ---------------------------------------------------------------
# 1. CONFIGURACIÓN
# ---------------------------------------------------------------
IMAGES_DIR = Path("/home/enric/Datasets/Original/vindr/images")
CSV_PATH = Path("/home/enric/Datasets/Original/vindr/finding_annotations.csv")
LABELS_DIR = Path("/home/enric/Mammo/Lesion Detection/Yolo v8/vindr_yolo/labels")
OUTPUT_DIR = Path("/home/enric/Mammo/Lesion Detection/Yolo v8/vindr_yolo_swin/train_simple")

IMG_SIZE = 1024
BATCH_SIZE = 4
EPOCHS = 100
LR = 1e-4
NUM_WORKERS = 8
TRAIN_MAX_IMAGES = 4000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------
# 2. FUNCIONES DE DATOS
# ---------------------------------------------------------------
def read_dicom(path):
    """DICOM -> imagen uint8 2D."""
    ds = pydicom.dcmread(str(path))
    image = ds.pixel_array.astype(np.float32)
    if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
        image = image.max() - image
    return preprocess_window(
        image,
        dicom_dataset=ds,
        method="breast_tissue",
        voi_func="LINEAR",
        exclude_background=True,
        output_dtype=np.uint8,
    )


def load_labels(path):
    """Fichero .txt -> array (n, 5): clase, cx, cy, w, h (normalizados)."""
    rows = []
    for line in path.read_text().splitlines():
        if line.strip():
            rows.append([float(v) for v in line.split()])
    return np.array(rows, dtype=np.float32).reshape(-1, 5)


def resize_and_pad(image, labels, size):
    """Reescala la imagen a caber en size x size, rellena con negro y ajusta las cajas."""
    h, w = image.shape
    scale = size / max(h, w)
    new_w, new_h = round(w * scale), round(h * scale)
    resized = cv2.resize(image, (new_w, new_h))

    canvas = np.zeros((size, size), dtype=np.uint8)
    pad_x = (size - new_w) // 2
    pad_y = (size - new_h) // 2
    canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized

    # Las cajas van normalizadas (0-1) respecto a la imagen original.
    # Las pasamos a píxeles del canvas y volvemos a normalizar por size.
    labels = labels.copy()
    labels[:, 1] = (labels[:, 1] * new_w + pad_x) / size  # cx
    labels[:, 2] = (labels[:, 2] * new_h + pad_y) / size  # cy
    labels[:, 3] = labels[:, 3] * new_w / size            # w
    labels[:, 4] = labels[:, 4] * new_h / size            # h
    return canvas, labels


# ---------------------------------------------------------------
# 3. DATASET
# ---------------------------------------------------------------
class VindrDataset(Dataset):
    def __init__(self, split, max_images=None, seed=42):
        label_dir = LABELS_DIR / split
        df = pd.read_csv(CSV_PATH)
        df = df[df["split"] == "training"].drop_duplicates("image_id")

        # items: lista de (ruta_dicom, ruta_label, tiene_cajas)
        self.items = []
        for study_id, image_id in zip(df["study_id"], df["image_id"]):
            dicom_path = IMAGES_DIR / str(study_id) / f"{image_id}.dicom"
            label_path = label_dir / f"{image_id}.txt"
            if dicom_path.exists() and label_path.exists():
                positive = len(load_labels(label_path)) > 0
                self.items.append((dicom_path, label_path, positive))

        if max_images is not None:
            self.items = downsample(self.items, max_images, seed)

        n_pos = sum(item[2] for item in self.items)
        print(f"[{split}] {len(self.items)} imágenes | {n_pos} positivas | {len(self.items) - n_pos} negativas")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        dicom_path, label_path, _ = self.items[i]
        image = read_dicom(dicom_path)
        labels = load_labels(label_path)
        image, labels = resize_and_pad(image, labels, IMG_SIZE)

        image = np.stack([image] * 3) / 255.0  # (3, H, W), 3 canales iguales
        return torch.from_numpy(image).float(), torch.from_numpy(labels)


def downsample(items, max_images, seed):
    """Conserva toda la clase minoritaria y rellena con la mayoritaria al azar."""
    if len(items) <= max_images:
        return items
    rng = random.Random(seed)
    pos = [it for it in items if it[2]]
    neg = [it for it in items if not it[2]]
    minority, majority = sorted([pos, neg], key=len)
    keep = minority + rng.sample(majority, max_images - len(minority))
    rng.shuffle(keep)
    return keep


def collate_fn(batch):
    """Une las cajas de todo el batch en el formato que pide el loss de YOLO."""
    images = torch.stack([img for img, _ in batch])
    batch_idx, classes, boxes = [], [], []
    for i, (_, labels) in enumerate(batch):
        batch_idx.append(torch.full((len(labels),), i, dtype=torch.long))
        classes.append(labels[:, 0:1])
        boxes.append(labels[:, 1:5])
    targets = {
        "batch_idx": torch.cat(batch_idx),
        "cls": torch.cat(classes),
        "bboxes": torch.cat(boxes),
    }
    return images, targets


# ---------------------------------------------------------------
# 4. MODELO
# ---------------------------------------------------------------
class SwinYOLO(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model(
            "swin_tiny_patch4_window7_224",
            pretrained=True,
            features_only=True,
            out_indices=(1, 2, 3),
            img_size=IMG_SIZE,
        )
        self.detect = Detect(nc=1, ch=(192, 384, 768))
        self.detect.stride = torch.tensor([8.0, 16.0, 32.0])
        self.detect.bias_init()

    def forward(self, x):
        # Swin devuelve (B, H, W, C); YOLO espera (B, C, H, W)
        feats = [f.permute(0, 3, 1, 2).contiguous() for f in self.backbone(x)]
        return self.detect(feats)


class LossHelper(nn.Module):
    """Envoltorio mínimo para que v8DetectionLoss acepte nuestro modelo."""

    def __init__(self, detect):
        super().__init__()
        self.model = nn.ModuleList([detect])
        self.args = SimpleNamespace(box=7.5, cls=0.5, dfl=1.5)


# ---------------------------------------------------------------
# 5. ENTRENAMIENTO
# ---------------------------------------------------------------
def run_epoch(model, criterion, loader, optimizer=None):
    """Si le pasas optimizer entrena; si no, solo mide el loss (validación)."""
    training = optimizer is not None
    model.train()  # el loss de YOLO necesita la salida "de train"
    if not training:
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()  # no actualizar estadísticas al validar

    total = 0.0
    with torch.set_grad_enabled(training):
        for images, targets in tqdm(loader):
            images = images.to(DEVICE)
            targets = {k: v.to(DEVICE) for k, v in targets.items()}

            predictions = model(images)
            loss = criterion(predictions, targets)[0].sum()

            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total += loss.item()
    return total / len(loader)


def main():
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    train_ds = VindrDataset("train", max_images=TRAIN_MAX_IMAGES)
    val_ds = VindrDataset("val")
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, collate_fn=collate_fn, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, collate_fn=collate_fn, pin_memory=True)

    model = SwinYOLO().to(DEVICE)
    model.detect.stride = model.detect.stride.to(DEVICE)  # .to() no mueve tensores sueltos
    criterion = v8DetectionLoss(LossHelper(model.detect))  # después de .to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)

    best_val = float("inf")
    for epoch in range(1, EPOCHS + 1):
        train_loss = run_epoch(model, criterion, train_loader, optimizer)
        val_loss = run_epoch(model, criterion, val_loader)
        print(f"Epoch {epoch}/{EPOCHS} | train {train_loss:.4f} | val {val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), OUTPUT_DIR / "best.pt")


if __name__ == "__main__":
    main()



