from dotenv import load_dotenv
import os
import wandb

load_dotenv()

# ---------------------------------------------------------
# information of the wandb credintials for model evaluation
# ---------------------------------------------------------

WANDB_API_KEY = os.getenv("WANDB_API_KEY")
WANDB_PROJECT = os.getenv("WANDB_PROJECT")
WANDB_RUN_NAME = os.getenv("WANDB_RUN_NAME")

def wandb_login():
    if WANDB_API_KEY:
        wandb.login(key=WANDB_API_KEY)
        print(f"W&B Login Successful for project: {WANDB_PROJECT}")
    else:
        raise EnvironmentError("WANDB_API_KEY not found in environment!")
