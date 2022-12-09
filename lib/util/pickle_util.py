import pickle
from pathlib import Path


def load_pickle(path: Path, default=None):
    if path.exists():
        with open(path, 'rb') as f:
            return pickle.load(f)
    else:
        return default


def save_pickle(path: Path, content):
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(content, f)
