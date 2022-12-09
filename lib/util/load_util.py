from functools import wraps
from pathlib import Path
from typing import Callable, Any, Optional

from lib.util.pickle_util import load_pickle, save_pickle

WITH_CACHING = True

_cache = {}


def load(
        path: Path,
        loader: Callable[[Path], Any] = load_pickle,
        saver: Optional[Callable[[Path, Any], None]] = save_pickle
):
    def outer(generate: Callable[[], Any]):
        @wraps(generate)
        def inner():
            if WITH_CACHING:
                result = _cache.get(path, None)
                if result is not None:
                    return result

            result = loader(path)
            if result is None:
                result = generate()
                saver(path, result)

            if WITH_CACHING:
                _cache[path] = result

            return result

        return inner

    return outer


def free_cache():
    _cache.clear()
