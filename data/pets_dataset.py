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

        self.image_dir = os.path.join(root_dir, "images", "images")
        self.ann_dir = os.path.join(root_dir, "annotations", "annotations", "xmls")

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
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std =[0.229, 0.224, 0.225]
        )
    ])


import xml.etree.ElementTree as ET
import torch


class PetLocalizationDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        self.image_dir = os.path.join(root_dir, "images", "images")
        self.ann_dir = os.path.join(root_dir, "annotations", "annotations", "xmls")

        self.image_paths = [
            os.path.join(self.image_dir, fname)
            for fname in os.listdir(self.image_dir)
            if fname.endswith(".jpg")
        ]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]

        image = Image.open(img_path).convert("RGB")

        orig_w, orig_h = image.size

        fname = os.path.basename(img_path).replace(".jpg", ".xml")
        xml_path = os.path.join(self.ann_dir, fname)

        tree = ET.parse(xml_path)
        root = tree.getroot()

        bbox = root.find("object").find("bndbox")

        xmin = float(bbox.find("xmin").text)
        ymin = float(bbox.find("ymin").text)
        xmax = float(bbox.find("xmax").text)
        ymax = float(bbox.find("ymax").text)

        x_center = (xmin + xmax) / 2
        y_center = (ymin + ymax) / 2
        width = xmax - xmin
        height = ymax - ymin

        if self.transform:
            image = self.transform(image)

        new_w, new_h = 224, 224

        x_center = x_center * (new_w / orig_w)
        y_center = y_center * (new_h / orig_h)
        width    = width    * (new_w / orig_w)
        height   = height   * (new_h / orig_h)

        target = torch.tensor([x_center, y_center, width, height], dtype=torch.float32)

        return image, target