import pickle
import torch
import numpy as np
import pandas as pd

from src.config import (
    TEST_PATH, PREDICTIONS_PATH, CHECKPOINTS_DIR, 
    N_FOLDS, TTA, DEVICE, AMP
)
from src.data.dataset import ECGTestDataset
from src.data.dataloader import make_loader
from src.models.ecg_net import ECGRhythmNet
from src.utils.metrics import unpack_batch, move_features_to_device

def infer_test_logits(model, model_info, test_df):
    timing_median = model_info["timing_median"]
    timing_mean = model_info["timing_mean"]
    timing_std = model_info["timing_std"]
    
    test_dataset = ECGTestDataset(test_df, timing_median, timing_mean, timing_std)
    test_loader = make_loader(test_dataset, shuffle=False)
    
    model.eval()

    @torch.no_grad()
    def predict_with_gain(gain):
        outputs = []
        for raw_batch in test_loader:
            features, _ = unpack_batch(raw_batch)
            features = move_features_to_device(features)
            
            if gain != 1.0:
                features["signal"] = features["signal"] * gain
                
            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=(AMP and DEVICE.type == "cuda")):
                logits = model(features)
                
            outputs.append(logits.float().cpu())
        return torch.cat(outputs, dim=0).numpy()
        
    gains = [1.00, 0.99, 1.01] if TTA else [1.00]
    all_logits = []
    
    for gain in gains:
        print(f"TTA gain = {gain}")
        all_logits.append(predict_with_gain(gain))
        
    return np.mean(all_logits, axis=0)

def main():
    print("Loading test data from:", TEST_PATH)
    test_df = pd.read_csv(TEST_PATH)
    
    meta_path = CHECKPOINTS_DIR / "meta.pkl"
    if not meta_path.exists():
        raise FileNotFoundError("meta.pkl not found! Please run train.py first.")
        
    with open(meta_path, "rb") as f:
        meta_info = pickle.load(f)
        
    oof_bias = meta_info["oof_bias"]
    idx_to_label = meta_info["idx_to_label"]
    n_classes = meta_info["n_classes"]
    
    test_fold_logits = []
    
    for fold in range(1, N_FOLDS + 1):
        print(f"\n{'=' * 70}\nTEST INFERENCE — FOLD {fold}\n{'=' * 70}")
        model_path = CHECKPOINTS_DIR / f"fold_{fold}_model.pth"
        
        if not model_path.exists():
            print(f"Warning: Model for fold {fold} not found. Skipping...")
            continue
            
        model_info = torch.load(model_path, weights_only=True)
        model = ECGRhythmNet(n_classes).to(DEVICE)
        model.load_state_dict(model_info["model_state_dict"])
        
        logits = infer_test_logits(model, model_info, test_df)
        test_fold_logits.append(logits)
        
    if not test_fold_logits:
        raise RuntimeError("No models were found or successfully loaded for inference.")
        
    test_logits = np.mean(test_fold_logits, axis=0)
    test_calibrated_logits = test_logits + oof_bias[None, :]
    test_prediction_indices = test_calibrated_logits.argmax(axis=1)
    
    test_predictions = np.array([idx_to_label[int(idx)] for idx in test_prediction_indices])
    
    predictions = pd.DataFrame({
        "id": test_df["id"].astype(str),
        "label": test_predictions
    })
    
    predictions.to_csv(PREDICTIONS_PATH, index=False)
    print(f"\n{'=' * 80}\nPREDICTIONS READY\n{'=' * 80}")
    print(f"File: {PREDICTIONS_PATH}")
    print(f"Rows: {len(predictions)}")
    
if __name__ == "__main__":
    main()
