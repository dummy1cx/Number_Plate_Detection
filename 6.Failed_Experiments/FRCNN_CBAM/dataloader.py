import os
import torch
import random
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageOps
import torchvision.transforms.functional as TF

class YOLODataset(Dataset):
    def __init__(self, img_dir, label_dir, img_size=(416, 416), augment=True):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.img_size = img_size
        self.augment = augment
        self.image_files = []
        for fname in os.listdir(img_dir):
            if isinstance(fname, bytes):
                fname = fname.decode("utf-8")
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                self.image_files.append(os.path.join(img_dir, fname))
        self.image_files.sort()

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        img_path = self.image_files[index]
        if isinstance(img_path, bytes):
            img_path = img_path.decode("utf-8")
        basename = os.path.splitext(os.path.basename(img_path))[0]
        label_path = os.path.join(self.label_dir, basename + ".txt")
        image = Image.open(img_path).convert("RGB")
        orig_width, orig_height = image.size
        targets = []
        if os.path.isfile(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    if isinstance(line, bytes):
                        line = line.decode("utf-8")
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    class_id = int(parts[0])
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    width = float(parts[3])
                    height = float(parts[4])
                    targets.append([class_id, x_center, y_center, width, height])
        else:
            targets = []
        if len(targets) > 0:
            targets = torch.tensor(targets, dtype=torch.float32)
        else:
            targets = torch.zeros((0, 5), dtype=torch.float32)
        if self.augment:
            if random.random() < 0.5:
                image = ImageOps.mirror(image)
                if targets.shape[0] > 0:
                    targets[:, 1] = 1.0 - targets[:, 1]
        new_w, new_h = self.img_size
        if (orig_width, orig_height) != (new_w, new_h):
            image = image.resize((new_w, new_h), resample=Image.BILINEAR)
        image_tensor = TF.to_tensor(image)
        return image_tensor, targets

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_SIZE = 416
BATCH_SIZE = 8
NUM_CLASSES = 1
EPOCHS = 50
LEARNING_RATE = 1e-4
TRAIN_IMG_DIR = "/content/drive/MyDrive/Number_Plate/dataset/train/images"
TRAIN_LABEL_DIR = "/content/drive/MyDrive/Number_Plate/dataset/train/labels"
VAL_IMG_DIR = "/content/drive/MyDrive/Number_Plate/dataset/val/images"
VAL_LABEL_DIR = "/content/drive/MyDrive/Number_Plate/dataset/val/labels"
CHECKPOINT_PATH = "checkpoint.pth"

def yolo_collate_fn(batch):
    images = []
    targets_list = []
    for img, tgt in batch:
        images.append(img)
        targets_list.append(tgt)
    images = torch.stack(images, dim=0)
    return images, targets_list

train_dataset = YOLODataset(TRAIN_IMG_DIR, TRAIN_LABEL_DIR, img_size=(IMAGE_SIZE, IMAGE_SIZE), augment=True)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=yolo_collate_fn)

val_dataset = YOLODataset(VAL_IMG_DIR, VAL_LABEL_DIR, img_size=(IMAGE_SIZE, IMAGE_SIZE), augment=False)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=yolo_collate_fn)
