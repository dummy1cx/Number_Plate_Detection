import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import xml.etree.ElementTree as ET
import os
from PIL import Image
import torchvision.transforms.functional as TF


## ------------------------------------------------------------------------
## The previous dataset loader not supported yolo file format annoation
## This is new dataset loader for ximl annotation loader for FRCNN training
## -------------------------------------------------------------------------

class VOCDataset(Dataset):
    def __init__(self, image_dir, annot_dir, transforms=None):
        self.image_dir = image_dir
        self.annot_dir = annot_dir
        self.transforms = transforms
        self.images = list(sorted(os.listdir(image_dir)))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        annot_path = os.path.join(self.annot_dir, os.path.splitext(self.images[idx])[0] + '.xml')

        img = Image.open(img_path).convert("RGB")
        tree = ET.parse(annot_path)
        root = tree.getroot()

        boxes = []
        labels = []

        ##---------------------------------------------------------------------------
        ## The objectve is to retrive the co-ordiantes from pascal style data format
        ## --------------------------------------------------------------------------
        

        for obj in root.findall('object'): 
            bbox = obj.find('bndbox')
            xmin = float(bbox.find('xmin').text)
            ymin = float(bbox.find('ymin').text)
            xmax = float(bbox.find('xmax').text)
            ymax = float(bbox.find('ymax').text)
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(1)  

        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
        }

        if self.transforms:
            img = self.transforms(img)

        return img, target




def collate_fn(batch):
    images, targets = tuple(zip(*batch))
    images = [TF.to_tensor(img) for img in images]
    targets = [{k: v for k, v in t.items()} for t in targets]
    return images, targets

dataset = VOCDataset('/content/drive/MyDrive/INM_705_Practice_final/Datasets/images', '/content/drive/MyDrive/INM_705_Practice_final/Datasets/annotations')
data_loader = DataLoader(dataset, batch_size=16, shuffle=True,num_workers=2, collate_fn=collate_fn)
