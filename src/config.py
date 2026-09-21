import os
import torch
from pathlib import Path

# Paths
ROOT_DIR = Path(__file__).parent.parent.resolve()
DATA_DIR = ROOT_DIR / "Data"
RAW_DIR = DATA_DIR / "Raw"
TRAIN_PATH = RAW_DIR / "train.csv"
TEST_PATH = RAW_DIR / "test.csv"

CHECKPOINTS_DIR = ROOT_DIR / "checkpoints"
CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
PREDICTIONS_PATH = ROOT_DIR / "predictions.csv"

# Training Parameters
SEED = 42
N_FOLDS = 3
EPOCHS = 30
PATIENCE = 4
BATCH_SIZE = 192
NUM_WORKERS = min(4, os.cpu_count() or 2)

# Model & Optimizer Parameters
LR = 2e-3
WEIGHT_DECAY = 2e-4
LABEL_SMOOTHING = 0.04
AMP = True
GRAD_CLIP = 1.0

# Augmentations & Inference
AUGMENT = True
TTA = True

# Columns
SIGNAL_COLS = [f"sig_{i}" for i in range(250)]
TIMING_COLS = ["pre_rr", "post_rr", "rr_ratio"]
REQUIRED_TRAIN = ["id", "label"] + SIGNAL_COLS + TIMING_COLS
REQUIRED_TEST = ["id"] + SIGNAL_COLS + TIMING_COLS

# Hardware
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
