import os, json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from src.models.classifier import FruitQualityClassifier
from src.utils.preprocess import get_train_transforms, get_val_transforms

TRAIN_DIR   = "data/classification/dataset/train"
VAL_DIR     = "data/classification/dataset/test"
WEIGHTS_DIR = "weights"
BATCH_SIZE  = 32
EPOCHS      = 10
LR          = 1e-4
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(WEIGHTS_DIR, exist_ok=True)

def train():
    train_dataset = ImageFolder(TRAIN_DIR, transform=get_train_transforms())
    val_dataset   = ImageFolder(VAL_DIR,   transform=get_val_transforms())
    train_loader  = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2)
    val_loader    = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    model     = FruitQualityClassifier().to(DEVICE)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_acc = 0.0
    history  = {"train_loss": [], "val_loss": [], "val_acc": []}

    for epoch in range(EPOCHS):
        # Train
        model.train()
        train_loss = 0.0
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} Train"):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validate
        model.eval()
        val_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} Val"):
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                out = model(images)
                val_loss += criterion(out, labels).item()
                _, pred = out.max(1)
                correct += pred.eq(labels).sum().item()
                total   += labels.size(0)

        acc = 100.0 * correct / total
        scheduler.step()

        tl = train_loss / len(train_loader)
        vl = val_loss   / len(val_loader)
        history["train_loss"].append(tl)
        history["val_loss"].append(vl)
        history["val_acc"].append(acc)
        print(f"Epoch {epoch+1}: Train Loss={tl:.4f} | Val Loss={vl:.4f} | Val Acc={acc:.2f}%")

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), f"{WEIGHTS_DIR}/efficientnet_best.pth")
            print(f"  💾 Saved! acc={acc:.2f}%")

    with open(f"{WEIGHTS_DIR}/history.json", "w") as f:
        json.dump(history, f)
    print(f"✅ Best Val Acc: {best_acc:.2f}%")

if __name__ == "__main__":
    train()