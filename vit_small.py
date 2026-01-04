import os
import time
import numpy as np
import pandas as pd
import torch
import timm
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

from dataset import AlbumentationsDataset
from transforms import get_train_transforms, get_test_transforms

#User Settings (EDIT THESE)
DATASET_DIR = r"D:\BdSL\dataset_224"
RESULTS_DIR = r"D:\BdSL\results\vit_small"
os.makedirs(RESULTS_DIR, exist_ok=True)

#Training Config (CPU-friendly)
NUM_CLASSES = 49
IMG_SIZE = 224
EPOCHS = 30
BATCH_SIZE = 16          
LR = 3e-4              
WEIGHT_DECAY = 0.05
DEVICE = "cpu"
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)


torch.set_num_threads(8)

#Data Load
train_ds = AlbumentationsDataset(
    root=os.path.join(DATASET_DIR, "train"),
    transform=get_train_transforms()
)
test_ds = AlbumentationsDataset(
    root=os.path.join(DATASET_DIR, "test"),
    transform=get_test_transforms()
)

train_loader = DataLoader(
    train_ds,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0  
)
test_loader = DataLoader(
    test_ds,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

#Model
model = timm.create_model(
    "vit_small_patch16_224",
    pretrained=True,
    num_classes=NUM_CLASSES
)
model.to(DEVICE)


criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)


scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

#Train
start_time = time.time()
for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0.0
    n_batches = 0

    for imgs, labels in train_loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        n_batches += 1

    scheduler.step()
    avg_loss = epoch_loss / max(1, n_batches)
    lr_now = scheduler.get_last_lr()[0]
    print(f"Epoch {epoch+1}/{EPOCHS} | Avg Loss: {avg_loss:.4f} | LR: {lr_now:.6f}")

train_time_sec = time.time() - start_time

#Evaluate
model.eval()
y_true, y_pred = [], []

with torch.no_grad():
    for imgs, labels in test_loader:
        imgs = imgs.to(DEVICE)
        outputs = model(imgs)
        preds = torch.argmax(outputs, dim=1)

        y_true.extend(labels.numpy())
        y_pred.extend(preds.cpu().numpy())

acc = accuracy_score(y_true, y_pred)
prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro")
cm = confusion_matrix(y_true, y_pred)

print(f"\n=== ViT-small Test Metrics ===")
print(f"Accuracy  : {acc:.4f}")
print(f"Precision : {prec:.4f}")
print(f"Recall    : {rec:.4f}")
print(f"F1 (macro): {f1:.4f}")
print(f"Train time (sec): {train_time_sec:.1f}")

#Save results
pd.DataFrame({
    "model": ["vit_small_patch16_224"],
    "epochs": [EPOCHS],
    "batch_size": [BATCH_SIZE],
    "lr": [LR],
    "weight_decay": [WEIGHT_DECAY],
    "accuracy": [acc],
    "precision_macro": [prec],
    "recall_macro": [rec],
    "f1_macro": [f1],
    "train_time_sec": [train_time_sec]
}).to_csv(os.path.join(RESULTS_DIR, "metrics.csv"), index=False)

pd.DataFrame(cm).to_csv(os.path.join(RESULTS_DIR, "confusion_matrix.csv"), index=False)

np.save(os.path.join(RESULTS_DIR, "y_true.npy"), np.array(y_true))
np.save(os.path.join(RESULTS_DIR, "y_pred.npy"), np.array(y_pred))
