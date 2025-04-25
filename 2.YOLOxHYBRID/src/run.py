import torch
import wandb
import optuna
import os
from model import YOLOv4DenseNetCBAM
from dataset import get_dataloaders
from loss import compute_yolo_loss
from evaluation import calculate_map_score
from utils import decode_yolo_output
from wandb_config import wandb_login, WANDB_PROJECT, WANDB_RUN_NAME

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Train function for model training
def train_epoch(model, dataloader, optimizer):
    model.train()
    total_loss = 0
    for images, targets in dataloader:
        images = images.to(DEVICE)
        targets = [t.to(DEVICE) for t in targets]

        preds = model(images)
        loss, obj_loss, cls_loss, box_loss = compute_yolo_loss(preds, targets, model)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    return total_loss / len(dataloader)

def validate(model, dataloader, conf_threshold=0.05):
    model.eval()
    val_loss = 0
    predictions = []
    ground_truths = []

    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(DEVICE)
            targets = [t.to(DEVICE) for t in targets]

            preds = model(images)  
            loss, obj_loss, cls_loss, box_loss = compute_yolo_loss(preds, targets, model)
            val_loss += loss.item()

            for i in range(images.shape[0]):
                pred_boxes_all = []

                for p, scale, anchors in zip(
                    preds, 
                    [8, 16, 32],
                    [model.anchors['small'], model.anchors['medium'], model.anchors['large']]
                ):
                    pred_boxes = decode_yolo_output(
                        output=p[i].unsqueeze(0),
                        anchors=anchors,
                        stride=scale,
                        conf_thresh=conf_threshold
                    )
                    if pred_boxes and len(pred_boxes[0]) > 0:
                        pred_boxes_all.extend(pred_boxes[0])

                if len(pred_boxes_all) == 0:
                    predictions.append(torch.empty((0, 4)))
                else:
                    predictions.append(torch.stack(pred_boxes_all))

                gt = targets[i].cpu()
                x1 = gt[:, 0] - gt[:, 2] / 2
                y1 = gt[:, 1] - gt[:, 3] / 2
                x2 = gt[:, 0] + gt[:, 2] / 2
                y2 = gt[:, 1] + gt[:, 3] / 2
                gt_boxes = torch.stack([x1, y1, x2, y2], dim=1)
                ground_truths.append(gt_boxes)

    avg_val_loss = val_loss / len(dataloader)
    map_score = calculate_map_score(predictions, ground_truths)
    return avg_val_loss, map_score

# Optuna Objective function
def objective(trial):
    config = {
        'learning_rate': trial.suggest_float('lr', 1e-5, 1e-3, log=True),
        'optimizer': trial.suggest_categorical('optimizer', ['adam', 'sgd']),
        'epochs': 50  # we have tried multiple epochs for each experiment for model training
    }

    wandb_login()  ## wandb initiation model performance tracking
    run = wandb.init(project=WANDB_PROJECT, name=WANDB_RUN_NAME or f"trial-{trial.number}", config=config)
    config = wandb.config

    train_loader, val_loader = get_dataloaders()
    model = YOLOv4DenseNetCBAM(num_classes=1).to(DEVICE)

    model.anchors = {
        'large':  torch.tensor([(10, 13), (16, 30), (33, 23)], dtype=torch.float32),
        'medium': torch.tensor([(30, 61), (62, 45), (59, 119)], dtype=torch.float32),
        'small':  torch.tensor([(116, 90), (156, 198), (373, 326)], dtype=torch.float32)
    }

    model.image_size = 416

    if config.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=config.learning_rate, momentum=0.9)

    best_val_loss = float('inf')
    best_model_path = f"best_model_trial_{trial.number}.pth"
    patience = 7
    no_improve_epochs = 0

    for epoch in range(config.epochs):
        train_loss = train_epoch(model, train_loader, optimizer)
        val_loss, map_score = validate(model, val_loader, conf_threshold=0.05)

        wandb.log({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "mAP": map_score
        })

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1

        if no_improve_epochs >= patience:
            print(f"Early stopping at epoch {epoch} for trial {trial.number}")
            break

    if os.path.exists(best_model_path):
        artifact = wandb.Artifact(f"best-model-trial-{trial.number}", type="model")
        artifact.add_file(best_model_path)
        run.log_artifact(artifact)

    wandb.finish()
    return best_val_loss

if __name__ == "__main__":
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=20)
    print("Best trial:", study.best_trial.params)
