# ECG Rhythm Classification

This repository provides an end-to-end deep learning pipeline for classifying Electrocardiogram (ECG) rhythms into four distinct arrhythmia categories. The project leverages a hybrid architecture combining Multi-Scale 1D Convolutional Neural Networks (ResNet) with a Transformer Encoder to capture both local morphological features and long-range temporal dependencies in ECG signals.

## Highlights
- **Hybrid Architecture:** Uses Multi-Scale ResNet blocks for local feature extraction and a Transformer Encoder with attention mechanisms for temporal context modeling.
- **Robust Training Pipeline:** Implements Stratified K-Fold Cross-Validation to ensure robust out-of-fold generalization.
- **Advanced Loss Functions:** Utilizes Focal Loss wrapped in a Class-Balanced Cross-Entropy module to effectively handle severe class imbalances.
- **Stable Convergence:** Incorporates Exponential Moving Average (EMA) of model weights and a Cosine Annealing learning rate schedule with warm restarts.
- **Prediction Calibration:** Includes automated post-training probability calibration (bias tuning) to maximize the Macro F1 score on imbalanced data.
- **Test-Time Augmentation (TTA):** Supports signal gain augmentations during inference for stabilized predictions.

## Performance
The model achieves state-of-the-art performance on the validation dataset (3-Fold CV):
- **Accuracy**: ~99.1%
- **Out-Of-Fold (OOF) Macro F1**: 0.9328 (Calibrated)

| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| **0** | 99.5%     | 99.5%  | 0.995    |
| **1** | 97.0%     | 95.5%  | 0.962    |
| **2** | 99.4%     | 99.2%  | 0.993    |
| **3** | 72.4%     | 81.9%  | 0.769    |

*Note: Class 3 is highly underrepresented in the raw data. The Focal Loss implementation actively works to balance precision and recall for this specific minority class.*

## Repository Structure
```text
├── Data/
│   └── Raw/               # Place train.csv and test.csv here
├── src/
│   ├── data/              # PyTorch Datasets and Dataloaders with dynamic augmentations
│   ├── models/            # Hybrid CNN (MultiScaleResBlock) + Transformer models
│   ├── training/          # Trainer loop, Focal Loss, and Schedulers
│   ├── utils/             # ModelEMA, Metrics, and Bias Calibration
│   ├── config.py          # Centralized configuration (hyperparameters, paths)
│   ├── train.py           # Entry point for K-Fold training
│   └── inference.py       # Entry point for Test-Time inference
├── checkpoints/           # Serialized models and meta.pkl (auto-generated)
├── requirements.txt
└── README.md
```

## Setup & Installation

**1. Clone the repository:**
```bash
git clone https://github.com/salil-kushwah/ECG-Rhythm-Classification.git
cd ECG-Rhythm-Classification
```

**2. Install dependencies:**
It is highly recommended to use a virtual environment (`venv` or `conda`).
```bash
pip install -r requirements.txt
```
*Make sure you have a CUDA-compatible version of PyTorch installed if you plan to train on an NVIDIA GPU.*

**3. Prepare the Data:**
Ensure your dataset is placed in the expected directory:
- `Data/Raw/train.csv`
- `Data/Raw/test.csv`

## Usage

### Training
To train the model from scratch, simply execute the training pipeline. This will run a 3-Fold Cross-Validation process, apply EMA, tune the class biases, and save the best weights to the `checkpoints/` folder.
```bash
python -m src.train
```

*Configuration:* If you want to modify hyperparameters (like learning rate, batch size, or epochs), edit the variables inside `src/config.py`.

### Inference
Once you have trained the models (or if you already have `.pth` files in the `checkpoints/` directory), you can run inference on the test set.
```bash
python -m src.inference
```
This will apply Test-Time Augmentation (TTA), aggregate the cross-validated model predictions, and output a `predictions.csv` file in the root directory.

## License
MIT License
