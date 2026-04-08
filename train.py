"""Training entrypoint
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

from data.pets_dataset import PetDataset, get_default_transforms
from models.classification import VGG11Classifier  


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ===== DATA =====
    transform = get_default_transforms()

    dataset = PetDataset(
        root_dir="/kaggle/input/datasets/tanlikesmath/the-oxfordiiit-pet-dataset",
        transform=transform
    )

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=32, shuffle=False)


    num_classes = len(dataset.classes)

    model = VGG11Classifier(num_classes=num_classes)  
    model.to(device)


    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)


    epochs = 3

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

            total_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

    print("Training complete!")


if __name__ == "__main__":
    main()