import shutil
from functools import wraps
from pathlib import Path
from typing import Callable

from lib.util.load_util import load
from lib.util.torch_util import load_torch, save_torch


def experiment(model_dir: Path):
    def outer(func: Callable):
        def inner(*args, keep_results: bool = False, **kwargs):
            if keep_results:
                @load(model_dir, load_torch, save_torch)
                @wraps(func)
                def helper():
                    return func(*args, **kwargs)

                return helper()
            else:
                if model_dir.exists():
                    shutil.rmtree(model_dir)
                return func(*args, **kwargs)

        return inner

    return outer


def avg_fn(keep_ratio: float):
    def helper(average, new, _):
        return keep_ratio * average + (1.0 - keep_ratio) * new

    return helper
