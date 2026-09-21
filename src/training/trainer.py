import time
import torch
from src.config import LR, WEIGHT_DECAY, DEVICE, AMP, GRAD_CLIP
from src.training.scheduler import make_scheduler
from src.utils.ema import ModelEMA
from src.utils.metrics import unpack_batch, move_features_to_device, predict_logits, macro_f1

def train_one_fold(model, train_loader, val_loader, criterion, fold, epochs, patience):
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
        betas=(0.9, 0.99)
    )

    total_steps = epochs * len(train_loader)
    scheduler = make_scheduler(
        optimizer,
        total_steps,
        warmup_steps=max(100, len(train_loader))
    )

    scaler = torch.amp.GradScaler("cuda", enabled=(AMP and DEVICE.type == "cuda"))
    ema = ModelEMA(model, decay=0.995)

    best_f1 = -1.0
    best_epoch = -1
    best_ema_state = None
    patience_count = 0

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_items = 0
        t0 = time.time()

        for raw_batch in train_loader:
            features, labels = unpack_batch(raw_batch)
            features = move_features_to_device(features)
            labels = labels.to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=(AMP and DEVICE.type == "cuda")):
                logits = model(features)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            ema.update(model)

            batch_size = labels.shape[0]
            total_loss += loss.item() * batch_size
            total_items += batch_size

        ema.apply_to(model)
        
        val_logits = predict_logits(model, val_loader)
        val_labels_list = []
        
        for raw_batch in val_loader:
            _, labels = unpack_batch(raw_batch)
            val_labels_list.append(labels)
            
        val_labels = torch.cat(val_labels_list, dim=0).numpy()
        val_f1 = macro_f1(val_logits, val_labels)
        val_accuracy = (val_logits.argmax(axis=1) == val_labels).mean()

        if val_f1 > best_f1 + 1e-6:
            best_f1 = val_f1
            best_epoch = epoch
            best_ema_state = ema.state_dict()
            patience_count = 0
        else:
            patience_count += 1

        ema.restore(model)

        elapsed = time.time() - t0
        average_loss = total_loss / max(total_items, 1)

        print(
            f"Fold {fold} | "
            f"Epoch {epoch:02d}/{epochs} | "
            f"loss={average_loss:.5f} | "
            f"val_F1={val_f1:.5f} | "
            f"val_acc={val_accuracy:.5f} | "
            f"lr={optimizer.param_groups[0]['lr']:.2e} | "
            f"{elapsed:.1f}s"
        )

        if patience_count >= patience:
            print("Early stopping.")
            break

    ema.load_state_dict(best_ema_state, model)

    print(f"\nFold {fold} best F1 = {best_f1:.5f} at epoch {best_epoch}\n")
    return model, float(best_f1), int(best_epoch)
