from pathlib import Path

import torch

def load_torch(path: Path, default=None):
    if path.exists():
        with open(path, 'rb') as f:
            return torch.load(f)
    else:
        return default


def save_torch(path: Path, content):
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'wb') as f:
        torch.save(content, f)
