import torch

class ModelEMA:
    def __init__(self, model, decay=0.995):
        self.decay = decay
        self.shadow = {}
        
        for key, value in model.state_dict().items():
            if value.dtype.is_floating_point:
                self.shadow[key] = value.detach().clone()
                
        self.backup = {}

    @torch.no_grad()
    def update(self, model):
        state = model.state_dict()
        for key in self.shadow:
            self.shadow[key].mul_(self.decay)
            self.shadow[key].add_(
                state[key].detach(),
                alpha=(1.0 - self.decay)
            )

    @torch.no_grad()
    def apply_to(self, model):
        self.backup = {}
        state = model.state_dict()
        for key in self.shadow:
            self.backup[key] = state[key].detach().clone()
            state[key].copy_(self.shadow[key])

    @torch.no_grad()
    def restore(self, model):
        state = model.state_dict()
        for key, value in self.backup.items():
            state[key].copy_(value)
        self.backup = {}

    def state_dict(self):
        return {
            key: value.detach().cpu()
            for key, value in self.shadow.items()
        }

    def load_state_dict(self, state_dict, model):
        model_state = model.state_dict()
        with torch.no_grad():
            for key, value in state_dict.items():
                if key in model_state:
                    model_state[key].copy_(
                        value.to(
                            device=model_state[key].device,
                            dtype=model_state[key].dtype
                        )
                    )
