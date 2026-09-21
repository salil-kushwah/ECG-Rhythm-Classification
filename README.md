# ECG Rhythm Classification

This repository contains the source code for an ECG Heartbeat Arrhythmia Classification PyTorch project.
It was originally a monolithic Jupyter Notebook, now refactored into a modular structure for maintainability and scalability.

## Directory Structure

- `Data/Raw/`: Place your `train.csv` and `test.csv` here.
- `src/`: The core source code.
  - `config.py`: Hyperparameters and configuration flags.
  - `data/`: Datasets and dataloader modules.
  - `models/`: Neural network architectures (CNN + Transformer).
  - `training/`: Loss functions, learning rate schedules, and the training loop logic.
  - `utils/`: Metrics, Exponential Moving Average (EMA), and calibration logic.
- `checkpoints/`: Stored models and metadata (created upon running `train.py`).

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Ensure you have PyTorch installed with the correct CUDA version for your hardware.

## How to Run

1. **Train the models:**
   Execute the K-fold cross-validation training script. This script will train the folds and serialize the best EMA model states and `meta.pkl` containing target encoders and probability biases into `checkpoints/`.
   ```bash
   python -m src.train
   ```

2. **Generate predictions:**
   Perform inference using the test set and the saved models to generate `predictions.csv` at the root directory.
   ```bash
   python -m src.inference
   ```
