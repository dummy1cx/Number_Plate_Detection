import wandb
import os
from ultralytics import YOLO
import argparse

# Initialize wandb run
wandb.init(project="YOLOv8_NumberPlate_Final", name="YOLOv8n_NP_Run1")

# configuration for model training
config = wandb.config

# Load YOLOv8 model 
# YOLOv8 nano model is used for the model training 
# Because of the less datasize nano model is opted
model = YOLO("yolov8n.pt")

# Train the model
results = model.train(
    data="/content/drive/MyDrive/INM_705_Practice_final/Datasets/data.yaml",
    epochs=config.epochs,
    imgsz=config.imgsz,
    batch=config.batch,
    optimizer=config.optimizer,
    lr0=config.lr0,
    project="YOLOv8_NumberPlate_Final",
    name="YOLOv8n_NP_Run1",
    verbose=True
)

# Log key metrics to wandb for performance eavluation
if hasattr(results, "results_dict"):
    for k, v in results.results_dict.items():
        if isinstance(v, (int, float)):
            wandb.log({k: v})

# model will get saved on artifacts
model_path = os.path.join(model.trainer.save_dir, "weights", "best.pt")

if os.path.exists(model_path):
    try:
        artifact = wandb.Artifact(
            name="yolov8n_numberplate_final",
            type="model",
            description="Best YOLOv8n model based on validation mAP",
            metadata={
                "epoch": getattr(model.trainer, "epoch", "N/A"),
                "mAP50": results.results_dict.get("metrics/mAP50(B)", "N/A"),
                "mAP50-95": results.results_dict.get("metrics/mAP50-95(B)", "N/A")
            }
        )
        artifact.add_file(model_path)
        wandb.log_artifact(artifact)
        print("WandB Artifact logged successfully.")
    except Exception as e:
        print(f"⚠️ Failed to log W&B artifact: {e}")
else:
    print(f"best.pt not found at: {model_path}")
