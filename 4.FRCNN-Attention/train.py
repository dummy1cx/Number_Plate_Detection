# run.py
import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import wandb
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from model import build_frcnn_with_attention, collate_fn, print_model_size
from dataset import VOCDataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 2
EPOCHS = 30
BATCH_SIZE = 4
LEARNING_RATE = 1e-4
TRAIN_IMG_DIR = "/content/drive/MyDrive/INM_705_Practice_final/Datasets/images"
TRAIN_ANNOT_DIR = "/content/drive/MyDrive/INM_705_Practice_final/Datasets/annotations"
CHECKPOINT_PATH = "attention_frcnn_best.pth"

def evaluate(model, dataloader, epoch):
    model.eval()
    metric = MeanAveragePrecision(iou_type="bbox")
    with torch.no_grad():
        for images, targets in dataloader:
            images = list(img.to(DEVICE) for img in images)
            targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]
            outputs = model(images)
            metric.update(outputs, targets)
    results = metric.compute()
    wandb.log({"val_mAP": results["map"], "val_mAP50": results["map_50"], "val_recall": results["mar_100"], "epoch": epoch})
    print(f"Epoch {epoch} - mAP: {results['map']:.4f}, Recall: {results['mar_100']:.4f}")

def main():
    wandb.init(project="FRCNN-x-Attention", name="FRCNN-x-Attention", config={"batch_size": BATCH_SIZE, "epochs": EPOCHS, "lr": LEARNING_RATE})

    train_dataset = VOCDataset(TRAIN_IMG_DIR, TRAIN_ANNOT_DIR)
    val_dataset = VOCDataset(TRAIN_IMG_DIR, TRAIN_ANNOT_DIR)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=2)

    model = build_frcnn_with_attention(num_classes=NUM_CLASSES)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_map = 0
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for images, targets in train_loader:
            images = list(img.to(DEVICE) for img in images)
            targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            total_loss += losses.item()

        avg_loss = total_loss / len(train_loader)
        # wandb performance tracking
        wandb.log({"train_loss": avg_loss, "epoch": epoch})
        print(f"Epoch {epoch} | Train Loss: {avg_loss:.4f}")

        evaluate(model, val_loader, epoch)

        current_map = wandb.run.summary.get("val_mAP", 0.0)
        if current_map > best_map:
            best_map = current_map
            #saving th e best model to artifacts in wandb
            torch.save(model.state_dict(), CHECKPOINT_PATH)
            artifact = wandb.Artifact("FRCNN-x-Attention", type="model")
            artifact.add_file(CHECKPOINT_PATH)
            wandb.log_artifact(artifact)

if __name__ == '__main__':
    main()
