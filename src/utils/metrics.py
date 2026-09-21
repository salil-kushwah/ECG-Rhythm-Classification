import torch
from sklearn.metrics import f1_score
from src.config import DEVICE, AMP

def unpack_batch(batch):
    if isinstance(batch, (tuple, list)):
        features = batch[0]
        labels = batch[1] if len(batch) > 1 else None
    elif isinstance(batch, dict):
        features = batch
        labels = None
    else:
        raise TypeError(f"Unexpected batch type: {type(batch)}")
    return features, labels

def move_features_to_device(features):
    return {
        key: value.to(DEVICE, non_blocking=True)
        for key, value in features.items()
    }

def macro_f1(logits, labels):
    predictions = logits.argmax(axis=1)
    return f1_score(labels, predictions, average="macro")

@torch.no_grad()
def predict_logits(model, loader):
    model.eval()
    outputs = []
    
    for raw_batch in loader:
        features, _ = unpack_batch(raw_batch)
        features = move_features_to_device(features)
        
        with torch.amp.autocast("cuda", dtype=torch.float16, enabled=(AMP and DEVICE.type == "cuda")):
            logits = model(features)
            
        outputs.append(logits.float().cpu())
        
    return torch.cat(outputs, dim=0).numpy()
