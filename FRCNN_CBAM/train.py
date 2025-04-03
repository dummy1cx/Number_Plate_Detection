import os
import torch
import torch.nn as nn
import torchvision
import torch.optim as optim
from torchvision.ops import box_iou
from torch.utils.data import DataLoader
import wandb
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF
from torchmetrics.detection.mean_ap import MeanAveragePrecision
#from frcnn_cbam_setup import change_the_backbone
#from utils import print_model_size
#from dataset import YOLODataset

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

def convert_yolo_to_frcnn_targets(yolo_targets, image_size):
    frcnn_targets = []
    _, H, W = image_size
    for sample in yolo_targets:
        boxes = []
        labels = []
        for obj in sample:
            if isinstance(obj, bytes):
                obj = obj.decode("utf-8")
            class_id, x_c, y_c, w, h = obj.tolist()
            x1 = (x_c - w / 2) * W
            y1 = (y_c - h / 2) * H
            x2 = (x_c + w / 2) * W
            y2 = (y_c + h / 2) * H
            boxes.append([x1, y1, x2, y2])
            labels.append(int(class_id))
        target = {
            'boxes': torch.tensor(boxes).float().to(DEVICE),
            'labels': torch.tensor(labels).long().to(DEVICE)
        }
        frcnn_targets.append(target)
    return frcnn_targets

def train_loop(nr_epochs, model, train_dataloader, val_dataloader, optimizer):
    min_loss = float('inf')
    for epoch in range(nr_epochs):
        model.train()
        epoch_loss = 0
        for images, targets in train_dataloader:
            images = list(img.to(DEVICE) for img in images)
            frcnn_targets = convert_yolo_to_frcnn_targets(targets, images[0].shape)

            optimizer.zero_grad()
            loss_dict = model(images, frcnn_targets)
            total_loss = sum(loss for loss in loss_dict.values())
            total_loss.backward()
            optimizer.step()
            epoch_loss += total_loss.item()

        epoch_loss /= len(train_dataloader)
        if epoch_loss < min_loss:
            torch.save(model.state_dict(), CHECKPOINT_PATH)
            min_loss = epoch_loss

            artifact = wandb.Artifact("frcnn_cbam_best", type="model")
            artifact.add_file(CHECKPOINT_PATH)
            wandb.log_artifact(artifact)

        wandb.log({"train_loss": epoch_loss, "epoch": epoch})
        print(f"Epoch {epoch} | Train Loss: {epoch_loss:.4f}")
        evaluate(model, val_dataloader, epoch)

def evaluate(model, dataloader, epoch):
    model.eval()
    metric = MeanAveragePrecision(iou_type="bbox", iou_thresholds=[0.5])

    with torch.no_grad():
        for i, (images, targets) in enumerate(dataloader):
            images = list(img.to(DEVICE) for img in images)
            frcnn_targets = convert_yolo_to_frcnn_targets(targets, images[0].shape)
            outputs = model(images)

            for output, target in zip(outputs, frcnn_targets):
                pred = {
                    "boxes": output["boxes"].cpu(),
                    "scores": output["scores"].cpu(),
                    "labels": output["labels"].cpu()
                }
                target_metric = {
                    "boxes": target["boxes"].cpu(),
                    "labels": target["labels"].cpu()
                }
                metric.update([pred], [target_metric])

    results = metric.compute()
    wandb.log({
        "mAP@0.5": results["map_50"],
        "Precision": results["map"],
        "Recall": results["mar_100"],
        "epoch": epoch
    })
    print(f"Validation mAP@0.5: {results['map_50']:.4f} | Precision: {results['map']:.4f} | Recall: {results['mar_100']:.4f}")

def main():
    wandb.init(
        project="cbam_frcnn_project_new",
        name="cbam_frcnn_training",
        config={
            "image_size": IMAGE_SIZE,
            "batch_size": BATCH_SIZE,
            "num_classes": NUM_CLASSES,
            "epochs": EPOCHS,
            "lr": LEARNING_RATE,
        }
    )

    model = change_the_backbone(num_desired_classes=NUM_CLASSES + 1)
    model.to(DEVICE)

    train_dataset = YOLODataset(TRAIN_IMG_DIR, TRAIN_LABEL_DIR, img_size=(IMAGE_SIZE, IMAGE_SIZE), augment=True)
    val_dataset = YOLODataset(VAL_IMG_DIR, VAL_LABEL_DIR, img_size=(IMAGE_SIZE, IMAGE_SIZE), augment=False)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=yolo_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=yolo_collate_fn)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

    train_loop(
        nr_epochs=EPOCHS,
        model=model,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        optimizer=optimizer
    )

    print_model_size(model)
    print("Training complete.")

if __name__ == '__main__':
    main()