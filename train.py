"""Training entrypoint (Classification + Localization)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

from data.pets_dataset import (
    PetDataset,
    PetLocalizationDataset,
    get_default_transforms,
    get_localization_transforms
)

from models.classification import VGG11Classifier
from models.localization import VGG11Localizer

from losses.iou_loss import IoULoss


def train_classification():

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

    num_classes = len(dataset.classes)

    model = VGG11Classifier(num_classes=num_classes)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=3e-4)

    epochs = 25

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * images.size(0)

        avg_train_loss = total_loss / len(train_loader.dataset)

        model.eval()

        val_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)

                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        avg_val_loss = val_loss / len(val_loader.dataset)
        accuracy = correct / total

        print(
            f"[CLS] Epoch {epoch+1} | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} | "
            f"Acc: {accuracy:.4f}"
        )

    torch.save(model.state_dict(), "classifier.pth")
    print("Classifier training complete! Saved classifier.pth")



def train_localization():

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

    model = VGG11Localizer()
    model.to(device)

    mse_loss = nn.MSELoss()
    iou_loss = IoULoss(reduction="mean")

    def loss_fn(preds, targets):
        return 0.5*mse_loss(preds, targets) + iou_loss(preds,targets)

    optimizer = optim.Adam(model.parameters(), lr=3e-4)

    epochs = 15

    for epoch in range(epochs):
        model.train()
        train_loss = 0

        for images, targets in train_loader:
            images = images.to(device)
            targets = targets.to(device)

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
                images = images.to(device)
                targets = targets.to(device)

                outputs = model(images)
                loss = loss_fn(outputs, targets)

                val_loss += loss.item() * images.size(0)

        avg_val_loss = val_loss / len(val_loader.dataset)

        print(
            f"[LOC] Epoch {epoch+1} | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {avg_val_loss:.4f}"
        )

    torch.save(model.state_dict(), "localizer.pth")
    print("Localization training complete! Saved localizer.pth")



def main():
    train_classification()
    train_localization()


if __name__ == "__main__":
    main()