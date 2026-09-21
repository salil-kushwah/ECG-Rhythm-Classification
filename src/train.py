import os
import gc
import pickle
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, classification_report, confusion_matrix

from src.config import (
    TRAIN_PATH, SEED, N_FOLDS, EPOCHS, PATIENCE, 
    TIMING_COLS, DEVICE, LABEL_SMOOTHING, CHECKPOINTS_DIR
)
from src.data.dataset import ECGDataset
from src.data.dataloader import make_loader
from src.models.ecg_net import ECGRhythmNet
from src.training.loss import effective_num_weights, ClassBalancedCE
from src.training.trainer import train_one_fold
from src.utils.metrics import predict_logits
from src.utils.calibration import tune_class_bias

def seed_everything(seed=SEED):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def main():
    seed_everything()
    print("Loading data from:", TRAIN_PATH)
    train_df = pd.read_csv(TRAIN_PATH)
    
    original_labels = sorted(train_df["label"].dropna().unique().tolist())
    label_to_idx = {label: idx for idx, label in enumerate(original_labels)}
    idx_to_label = {idx: label for label, idx in label_to_idx.items()}
    n_classes = len(original_labels)
    
    train_df["_target_idx"] = train_df["label"].map(label_to_idx).astype(np.int64)
    work_train = train_df.reset_index(drop=True).copy()
    
    print(f"Working shape: {work_train.shape}")
    
    y_all = work_train["_target_idx"].to_numpy(np.int64)
    oof_logits = np.zeros((len(work_train), n_classes), dtype=np.float32)
    oof_filled = np.zeros(len(work_train), dtype=bool)
    
    fold_models = []
    fold_scores = []
    
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(work_train)), y_all), start=1):
        print(f"\n{'=' * 80}\nFOLD {fold}/{N_FOLDS}\n{'=' * 80}")
        
        fold_train = work_train.iloc[train_idx].reset_index(drop=True)
        fold_val = work_train.iloc[val_idx].reset_index(drop=True)
        
        timing_median = fold_train[TIMING_COLS].median().to_numpy(np.float32)
        timing_filled = fold_train[TIMING_COLS].fillna(pd.Series(timing_median, index=TIMING_COLS))
        timing_mean = timing_filled.mean().to_numpy(np.float32)
        timing_std = timing_filled.std().to_numpy(np.float32)
        timing_std = np.where(np.isfinite(timing_std) & (timing_std > 1e-6), timing_std, 1.0).astype(np.float32)
        
        train_dataset = ECGDataset(fold_train, timing_median, timing_mean, timing_std, train_mode=True)
        val_dataset = ECGDataset(fold_val, timing_median, timing_mean, timing_std, train_mode=False)
        
        train_loader = make_loader(train_dataset, shuffle=True)
        val_loader = make_loader(val_dataset, shuffle=False)
        
        class_weights, counts = effective_num_weights(fold_train["_target_idx"].to_numpy(np.int64), n_classes)
        
        model = ECGRhythmNet(n_classes).to(DEVICE)
        criterion = ClassBalancedCE(class_weights, LABEL_SMOOTHING).to(DEVICE)
        
        model, score, best_epoch = train_one_fold(
            model, train_loader, val_loader, criterion, fold, EPOCHS, PATIENCE
        )
        
        fold_val_logits = predict_logits(model, val_loader)
        oof_logits[val_idx] = fold_val_logits
        oof_filled[val_idx] = True
        
        # Save model and artifacts for this fold
        model_info = {
            "model_state_dict": model.state_dict(),
            "timing_median": timing_median,
            "timing_mean": timing_mean,
            "timing_std": timing_std,
            "fold": fold,
            "score": score
        }
        
        model_path = CHECKPOINTS_DIR / f"fold_{fold}_model.pth"
        torch.save(model_info, model_path)
        print(f"Saved fold {fold} model to {model_path}")
        
        fold_scores.append(score)
        
        del train_loader, val_loader, train_dataset, val_dataset, criterion, fold_train, fold_val
        gc.collect()
        torch.cuda.empty_cache()
        
    mask = oof_filled
    oof_y = y_all[mask]
    oof_logits_used = oof_logits[mask]
    oof_predictions = oof_logits_used.argmax(axis=1)
    
    oof_f1 = f1_score(oof_y, oof_predictions, average="macro")
    print(f"\n{'=' * 80}\nOOF RESULTS\n{'=' * 80}")
    print(f"Fold F1: {[round(float(x), 5) for x in fold_scores]}")
    print(f"OOF macro F1: {round(oof_f1, 6)}")
    print("\nClassification report:")
    print(classification_report(oof_y, oof_predictions, digits=4))
    
    # Tune calibration
    oof_bias, calibrated_f1 = tune_class_bias(oof_logits_used, oof_y)
    print(f"Raw OOF F1: {oof_f1}")
    print(f"Calibrated OOF F1: {calibrated_f1}")
    print(f"Class bias: {oof_bias}")
    
    # Save meta information
    meta_info = {
        "oof_bias": oof_bias,
        "label_to_idx": label_to_idx,
        "idx_to_label": idx_to_label,
        "original_labels": original_labels,
        "n_classes": n_classes
    }
    with open(CHECKPOINTS_DIR / "meta.pkl", "wb") as f:
        pickle.dump(meta_info, f)

if __name__ == "__main__":
    main()
