import math
import torch

def make_scheduler(optimizer, total_steps, warmup_steps):
    def schedule(step):
        if step < warmup_steps:
            return max(
                1e-3,
                (step + 1) / max(1, warmup_steps)
            )

        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)

        return (
            0.05
            + 0.95
            * 0.5
            * (1.0 + math.cos(math.pi * progress))
        )

    return torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
