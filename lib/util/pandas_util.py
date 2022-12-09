from pathlib import Path

import pandas as pd


def load_pandas(path: Path):
    return pd.read_pickle(path)


def save_pandas(path: Path, content: pd.DataFrame):
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
    content.to_pickle(path)
