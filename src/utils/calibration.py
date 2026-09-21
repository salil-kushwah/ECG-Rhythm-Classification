import numpy as np
from sklearn.metrics import f1_score

def tune_class_bias(logits, labels):
    bias = np.zeros(logits.shape[1], dtype=np.float32)
    
    best_score = f1_score(
        labels,
        (logits + bias).argmax(axis=1),
        average="macro"
    )
    
    search_grids = [
        np.linspace(-1.00, 1.00, 41),
        np.linspace(-0.25, 0.25, 41),
        np.linspace(-0.08, 0.08, 33)
    ]
    
    for grid in search_grids:
        changed = True
        
        while changed:
            changed = False
            for class_id in range(logits.shape[1]):
                local_best = best_score
                local_value = bias[class_id]
                
                for delta in grid:
                    candidate = bias.copy()
                    candidate[class_id] = bias[class_id] + np.float32(delta)
                    
                    prediction = (logits + candidate).argmax(axis=1)
                    score = f1_score(labels, prediction, average="macro")
                    
                    if score > local_best + 1e-7:
                        local_best = score
                        local_value = candidate[class_id]
                        
                if local_best > best_score + 1e-7:
                    bias[class_id] = local_value
                    best_score = local_best
                    changed = True
                    
    return bias, best_score
