import os
import random
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageOps
import torchvision.transforms.functional as TF

# -------------------------------
# YOLO-compatible Dataset Class
# -------------------------------
class YOLODataset(Dataset):
    def __init__(self, img_dir, label_dir, img_size=(416, 416), augment=True):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.img_size = img_size
        self.augment = augment

        self.image_files = [
            os.path.join(img_dir, fname)
            for fname in os.listdir(img_dir)
            if fname.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
        self.image_files.sort()

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        img_path = self.image_files[index]
        basename = os.path.splitext(os.path.basename(img_path))[0]
        label_path = os.path.join(self.label_dir, basename + ".txt")

        image = Image.open(img_path).convert("RGB")
        orig_width, orig_height = image.size

        targets = []
        if os.path.isfile(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        class_id, x_center, y_center, width, height = map(float, parts)
                        targets.append([class_id, x_center, y_center, width, height])

        targets = torch.tensor(targets, dtype=torch.float32) if targets else torch.zeros((0, 5), dtype=torch.float32)

        # Random horizontal flip (augmentation)
        if self.augment and random.random() < 0.5:
            image = ImageOps.mirror(image)
            if targets.shape[0] > 0:
                targets[:, 1] = 1.0 - targets[:, 1]  # Flip x_center

        # Resize image
        if (orig_width, orig_height) != self.img_size:
            image = image.resize(self.img_size, resample=Image.BILINEAR)

        image_tensor = TF.to_tensor(image)
        return image_tensor, targets

# -------------------------------
# Collate Function for Dataloader
# -------------------------------
def yolo_collate_fn(batch):
    images = []
    targets_list = []
    for img, tgt in batch:
        images.append(img)
        targets_list.append(tgt)
    images = torch.stack(images, dim=0)
    return images, targets_list

# -------------------------------
# Dataloader Setup
# -------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_SIZE = 416
BATCH_SIZE = 8

TRAIN_IMG_DIR = "/content/drive/MyDrive/Number_Plate/dataset/train/images"
TRAIN_LABEL_DIR = "/content/drive/MyDrive/Number_Plate/dataset/train/labels"
VAL_IMG_DIR = "/content/drive/MyDrive/Number_Plate/dataset/val/images"
VAL_LABEL_DIR = "/content/drive/MyDrive/Number_Plate/dataset/val/labels"

def get_dataloaders():
    train_dataset = YOLODataset(TRAIN_IMG_DIR, TRAIN_LABEL_DIR, img_size=(IMAGE_SIZE, IMAGE_SIZE), augment=True)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=yolo_collate_fn)

    val_dataset = YOLODataset(VAL_IMG_DIR, VAL_LABEL_DIR, img_size=(IMAGE_SIZE, IMAGE_SIZE), augment=False)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=yolo_collate_fn)

    return train_loader, val_loader
