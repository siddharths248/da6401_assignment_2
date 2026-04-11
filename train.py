"""Training entrypoint (Classification + Localization + Segmentation with W&B)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import wandb

from data.pets_dataset import (
    PetDataset,
    PetLocalizationDataset,
    get_default_transforms,
    get_localization_transforms,
    PetSegmentationDataset
)

from models.classification import VGG11Classifier
from models.localization import VGG11Localizer
from models.segmentation import VGG11UNet

from losses.iou_loss import IoULoss



def dice_score(pred, target, num_classes=3, eps=1e-6):
    pred = torch.argmax(pred, dim=1)

    dice = 0.0
    for cls in range(num_classes):
        pred_cls = (pred == cls).float()
        target_cls = (target == cls).float()

        intersection = (pred_cls * target_cls).sum()
        union = pred_cls.sum() + target_cls.sum()

        dice += (2 * intersection + eps) / (union + eps)

    return dice / num_classes


def pixel_accuracy(pred, target):
    pred = torch.argmax(pred, dim=1)
    correct = (pred == target).sum()
    total = torch.numel(target)
    return correct.float() / total


def compute_iou(preds, targets, eps=1e-6):
    px, py, pw, ph = preds.unbind(dim=1)
    tx, ty, tw, th = targets.unbind(dim=1)

    px1, py1 = px - pw/2, py - ph/2
    px2, py2 = px + pw/2, py + ph/2

    tx1, ty1 = tx - tw/2, ty - th/2
    tx2, ty2 = tx + tw/2, ty + th/2

    inter_x1 = torch.max(px1, tx1)
    inter_y1 = torch.max(py1, ty1)
    inter_x2 = torch.min(px2, tx2)
    inter_y2 = torch.min(py2, ty2)

    inter = (inter_x2 - inter_x1).clamp(min=0) * (inter_y2 - inter_y1).clamp(min=0)

    area_p = pw * ph
    area_t = tw * th

    union = area_p + area_t - inter + eps

    return (inter / union).mean()


# -------------------- CLASSIFICATION --------------------
def train_classification(dropout_p=0.5, use_bn=True):

    wandb.init(
        project="da6401-assignment2",
        name=f"cls_bn_{use_bn}_dropout_{dropout_p}",
        config={
            "task": "classification",
            "lr": 3e-4,
            "batch_size": 32,
            "epochs": 40,
            "dropout_p": dropout_p,
            "batchnorm": use_bn
        }
    )

    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = get_default_transforms()

    dataset = PetDataset(
        root_dir="/kaggle/input/datasets/siddharthsgeorge/oxford-iiit-pet-dataset",
        transform=transform
    )

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
    val_loader   = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)

    model = VGG11Classifier(
        num_classes=len(dataset.classes),
        dropout_p=dropout_p,
        use_bn=use_bn
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=3e-4)

    epochs = 40

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * images.size(0)

        avg_train_loss = total_loss / len(train_loader.dataset)

        model.eval()
        val_loss, correct, total = 0, 0, 0

        with torch.no_grad():
            iou_total = 0

            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)

                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
                iou_total += compute_iou(outputs, labels).item()

                if epoch == epochs - 1:
                    wandb.log({
                        "activations": wandb.Histogram(outputs.detach().cpu())
                    })

        avg_val_loss = val_loss / len(val_loader.dataset)
        accuracy = correct / total
        avg_iou = iou_total / len(val_loader)

        wandb.log({
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "val_accuracy": accuracy,
            "val_iou" : avg_iou
        })

        print(f"[CLS] Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Acc: {accuracy:.4f}")

    torch.save(model.state_dict(), "classifier.pth")
    wandb.finish()


# -------------------- LOCALIZATION --------------------
def train_localization():

    wandb.init(
        project="da6401-assignment2",
        name="loc_mse_2iou_partialfreeze",
        config={
            "task": "localization",
            "lr": 3e-4,
            "batch_size": 32,
            "epochs": 25,
            "loss": "MSE + 2*IoU",
            "freeze": "block5_only"
        }
    )

    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = PetLocalizationDataset(
        root_dir="/kaggle/input/datasets/siddharthsgeorge/oxford-iiit-pet-dataset",
        transform=get_localization_transforms()
    )

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
    val_loader   = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)

    model = VGG11Localizer().to(device)

    mse_loss = nn.MSELoss()
    iou_loss = IoULoss()

    def loss_fn(preds, targets):
        return mse_loss(preds, targets) + 2.0 * iou_loss(preds, targets)

    optimizer = optim.Adam(model.parameters(), lr=3e-4)

    epochs = 25

    for epoch in range(epochs):
        model.train()
        train_loss = 0

        for images, targets in train_loader:
            images, targets = images.to(device), targets.to(device)

            outputs = model(images)
            loss = loss_fn(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)

        avg_train_loss = train_loss / len(train_loader.dataset)

        model.eval()
        val_loss = 0

        with torch.no_grad():
            for images, targets in val_loader:
                images, targets = images.to(device), targets.to(device)

                outputs = model(images)
                loss = loss_fn(outputs, targets)

                val_loss += loss.item() * images.size(0)

        avg_val_loss = val_loss / len(val_loader.dataset)

        wandb.log({
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss
        })

        print(f"[LOC] Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

    torch.save(model.state_dict(), "localizer.pth")
    wandb.finish()


# -------------------- SEGMENTATION --------------------
def train_segmentation(freeze_mode="partial"):

    wandb.init(
        project="da6401-assignment2",
        name=f"seg_{freeze_mode}",
        config={
            "task": "segmentation",
            "lr": 3e-4,
            "batch_size": 16,
            "epochs": 25,
            "freeze_mode": freeze_mode,
            "loss": "CrossEntropy"
        }
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = PetSegmentationDataset(
        root_dir="/kaggle/input/datasets/siddharthsgeorge/oxford-iiit-pet-dataset",
        transform=get_localization_transforms()
    )

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=16, shuffle=False)

    model = VGG11UNet(num_classes=3).to(device)

    if freeze_mode == "freeze_all":
        for param in model.encoder.parameters():
            param.requires_grad = False

    elif freeze_mode == "partial":
        for param in model.encoder.parameters():
            param.requires_grad = False
        for name, param in model.encoder.named_parameters():
            if "block5" in name:
                param.requires_grad = True

    elif freeze_mode == "full":
        for param in model.parameters():
            param.requires_grad = True

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=3e-4)

    epochs = 25

    for epoch in range(epochs):
        model.train()
        train_loss = 0

        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)

            outputs = model(images)
            loss = criterion(outputs, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)

        avg_train_loss = train_loss / len(train_loader.dataset)

        model.eval()
        val_loss = 0
        dice_total = 0
        pixel_total = 0

        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)

                outputs = model(images)
                loss = criterion(outputs, masks)

                val_loss += loss.item() * images.size(0)

                dice_total += dice_score(outputs, masks).item()
                pixel_total += pixel_accuracy(outputs, masks).item()

        avg_val_loss = val_loss / len(val_loader.dataset)
        avg_dice = dice_total / len(val_loader)
        avg_pixel = pixel_total / len(val_loader)

        wandb.log({
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "val_dice": avg_dice,
            "val_pixel_acc": avg_pixel
        })

        print(f"[SEG] Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

    torch.save(model.state_dict(), "unet.pth")
    wandb.finish()


# -------------------- MAIN --------------------
def main():
    # Classification experiments
    # train_classification(dropout_p=0.0)
    # train_classification(dropout_p=0.2)
    train_classification(dropout_p=0.5)
    # train_classification(use_bn=False)

    # Localization
    # train_localization()

    # Segmentation experiments
    # train_segmentation("freeze_all")
    # train_segmentation("partial")
    # train_segmentation("full")


if __name__ == "__main__":
    main()