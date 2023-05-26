import torch
import torch.optim as optim


def make_optimizer(model: torch.nn.Module, learning_rate: float, kind: str, config: dict):
    optimizer = None
    if kind == 'AdamW':
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, **config)
    if kind == 'Adam':
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, **config)
    return optimizer
