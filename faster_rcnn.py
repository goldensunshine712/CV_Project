import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.data import DataLoader
from pycocotools.coco import COCO
import os
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as T

# ============================================================
# ✏️  CHANGE THESE — your actual folder paths
# ============================================================
TRAIN_IMG = "D:/vaishnavi/CV_Project/Dataset/train"
TRAIN_ANN = "D:/vaishnavi/CV_Project/Dataset/train/_annotations.coco.json"
VAL_IMG   = "D:/vaishnavi/CV_Project/Dataset/valid"
VAL_ANN   = "D:/vaishnavi/CV_Project/Dataset/valid/_annotations.coco.json"
SAVE_DIR  = "D:/vaishnavi/CV_Project/results"
# ============================================================


# -----------------------------
# SETTINGS
# -----------------------------
NUM_CLASSES   = 2
NUM_EPOCHS    = 30        # max epochs — early stopping will stop before this if needed
BATCH_SIZE    = 8
LR            = 0.005
TRAIN_LIMIT   = 3000      # ✅ 3000 training images
VAL_LIMIT     = 600       # 20% of 3000

# -----------------------------
# Early stopping settings
# -----------------------------
PATIENCE      = 3         # stop if val loss doesn't improve for 5 epochs
MIN_DELTA     = 0.001     # minimum improvement to count as "better"


# -----------------------------
# Dataset class
# -----------------------------
class CocoDataset(torch.utils.data.Dataset):
    def __init__(self, root, ann_file, limit=None):
        self.root = root
        self.coco = COCO(ann_file)
        self.transforms = T.ToTensor()

        all_ids = list(self.coco.imgs.keys())
        self.ids = [
            img_id for img_id in all_ids
            if len(self.coco.getAnnIds(imgIds=img_id)) > 0
        ]

        if limit is not None:
            self.ids = self.ids[:limit]

    def __getitem__(self, index):
        img_id  = self.ids[index]
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns    = self.coco.loadAnns(ann_ids)

        path = os.path.basename(self.coco.loadImgs(img_id)[0]['file_name'])
        full_path = f"{self.root}/{path}"
        img  = Image.open(full_path).convert("RGB")
        img  = self.transforms(img)

        boxes, labels = [], []
        for ann in anns:
            x, y, w, h = ann['bbox']
            if w > 0 and h > 0:
                boxes.append([x, y, x + w, y + h])
                labels.append(1)

        boxes  = torch.as_tensor(boxes,  dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {
            "boxes":    boxes,
            "labels":   labels,
            "image_id": torch.tensor([img_id]),
        }
        return img, target

    def __len__(self):
        return len(self.ids)


def collate_fn(batch):
    return tuple(zip(*batch))


def train_one_epoch(model, optimizer, loader, device, epoch):
    model.train()
    total_loss = 0.0

    for batch_idx, (images, targets) in enumerate(loader):
        images  = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses    = sum(loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        total_loss += losses.item()

        if batch_idx % 50 == 0:
            print(
                f"  Epoch [{epoch}/{NUM_EPOCHS}] "
                f"Step [{batch_idx}/{len(loader)}] "
                f"Loss: {losses.item():.4f}"
            )

    return total_loss / len(loader)


def validate(model, loader, device):
    model.train()
    total_loss = 0.0

    with torch.no_grad():
        for images, targets in loader:
            images  = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            loss_dict  = model(images, targets)
            total_loss += sum(loss_dict.values()).item()

    return total_loss / len(loader)


# ============================================================
if __name__ == '__main__':

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")

    os.makedirs(SAVE_DIR, exist_ok=True)

    # Datasets with limits
    train_dataset = CocoDataset(TRAIN_IMG, TRAIN_ANN, limit=TRAIN_LIMIT)
    val_dataset   = CocoDataset(VAL_IMG,   VAL_ANN,   limit=VAL_LIMIT)

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        collate_fn=collate_fn, num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=1, shuffle=False,
        collate_fn=collate_fn, num_workers=0
    )

    print(f"Train samples : {len(train_dataset)}")
    print(f"Val   samples : {len(val_dataset)}")

    # Load Faster R-CNN model
    weights = torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.COCO_V1
    model   = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=weights)

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, NUM_CLASSES)

    model.to(DEVICE)
    print("Model loaded ✅")

    # Optimizer & Scheduler
    params       = [p for p in model.parameters() if p.requires_grad]
    optimizer    = torch.optim.SGD(params, lr=LR, momentum=0.9, weight_decay=1e-4)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[10, 20], gamma=0.1  # adjusted for 30 max epochs
    )

    # Training loop
    history = {
        "epoch":          [],
        "train/box_loss": [],
        "val/box_loss":   [],
    }

    best_val_loss    = float("inf")
    patience_counter = 0             # counts epochs without improvement

    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss = train_one_epoch(model, optimizer, train_loader, DEVICE, epoch)
        val_loss   = validate(model, val_loader, DEVICE)
        lr_scheduler.step()

        history["epoch"].append(epoch)
        history["train/box_loss"].append(train_loss)
        history["val/box_loss"].append(val_loss)

        print(f"\nEpoch {epoch:02d}/{NUM_EPOCHS} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss:   {val_loss:.4f} | "
              f"Patience: {patience_counter}/{PATIENCE}\n")

        # ✅ Save best model if val loss improved
        if val_loss < best_val_loss - MIN_DELTA:
            best_val_loss    = val_loss
            patience_counter = 0      # reset counter
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, "fasterrcnn_best.pth"))
            print(f"  ✅ Best model saved (val_loss={val_loss:.4f})")

        else:
            patience_counter += 1
            print(f"  ⚠️  No improvement for {patience_counter}/{PATIENCE} epochs")

            # ✅ Early stopping — stop training if no improvement for PATIENCE epochs
            if patience_counter >= PATIENCE:
                print(f"\n🛑 Early stopping triggered at epoch {epoch}!")
                print(f"   Val loss hasn't improved for {PATIENCE} epochs.")
                print(f"   Best val loss was: {best_val_loss:.4f}")
                break

    print("\n✅ Training complete!")

    # Save results.csv
    df = pd.DataFrame(history)
    df.to_csv(os.path.join(SAVE_DIR, "results.csv"), index=False)
    print(f"Results saved to {SAVE_DIR}/results.csv")

    # Plot Loss Graph
    plt.figure()
    plt.plot(df['epoch'], df['train/box_loss'], label='Train Loss')
    plt.plot(df['epoch'], df['val/box_loss'],   label='Validation Loss')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss vs Epoch")
    plt.legend()
    plt.savefig(os.path.join(SAVE_DIR, "loss_curve.png"))
    plt.show()