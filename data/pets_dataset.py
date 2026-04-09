"""Dataset skeleton for Oxford-IIIT Pet.
"""

import os
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms


class PetDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        self.image_dir = os.path.join(root_dir, "images")

        self.image_paths = [
            os.path.join(self.image_dir, fname)
            for fname in os.listdir(self.image_dir)
            if fname.endswith(".jpg")
        ]

        self.classes = sorted(list(set([
            fname.split("_")[0] for fname in os.listdir(self.image_dir)
        ])))

        self.class_to_idx = {
            cls_name: idx for idx, cls_name in enumerate(self.classes)
        }

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]

        image = Image.open(img_path).convert("RGB")

        fname = os.path.basename(img_path)
        class_name = fname.split("_")[0]
        label = self.class_to_idx[class_name]

        if self.transform:
            image = self.transform(image)

        return image, label


def get_default_transforms():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std =[0.229, 0.224, 0.225]
        )
    ])