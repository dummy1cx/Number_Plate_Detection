import wandb
import yaml
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

print("W&B Project:", os.getenv("WANDB_PROJECT"))

# sweep configuration for model training with hyper-parameter
with open("sweep.yaml") as f:
    sweep_config = yaml.safe_load(f)

# initiating sweep id
sweep_id = wandb.sweep(sweep_config, project=os.getenv("WANDB_PROJECT"))
print("Sweep ID:", sweep_id)

# train function for training
def train():
    exec(open("src.py").read())

# initiating sweep agent training
wandb.agent(sweep_id, function=train)
