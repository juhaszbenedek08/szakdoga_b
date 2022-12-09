from functools import wraps, partial
from pathlib import Path
from typing import Callable, Any, Optional

from more_itertools import chunked

from lib.util.load_util import load
from lib.util.log_util import log_func, pretty_tqdm, logger
from lib.util.pickle_util import load_pickle, save_pickle


def safe_generate(
        source_reprs: dict[int, Any],
        generator: Callable[[Any], Any],
        name: str,
        ids: dict[int, Any],
        with_bar: bool
):
    if with_bar:
        result = {}
        with pretty_tqdm(total=len(source_reprs)) as pbar:
            for index, source_repr in source_reprs.items():
                try:
                    repr_ = generator(source_repr)
                    if repr_ is None:
                        raise Exception()
                    result[index] = repr_
                except:
                    logger.warning(f'No available {name} of {ids[index] : >20}')
                pbar.update()
            return result
    else:
        result = {}
        for index, source_repr in source_reprs.items():
            try:
                repr_ = generator(source_repr)
                if repr_ is None:
                    raise Exception()
                result[index] = repr_
            except:
                logger.warning(f'No available {name} of {ids[index] : >20}')
        return result


def batch_generate(
        load_source_reprs: Callable[[], dict[int, Any]],
        path: Path,
        batch_size: int,
        loader: Callable[[Path], Any] = partial(load_pickle, default={}),
        saver: Optional[Callable[[Path, Any], None]] = partial(save_pickle, default={})
):
    def outer(func: Callable[[dict[int, Any]], dict[int, Any]]):
        @log_func
        @load(path, loader, saver)
        @wraps(func)
        def inner() -> dict[int, Any]:
            source_reprs = load_source_reprs()
            done_reprs = load_pickle(path, {})
            remaining_reprs = {
                index: id_
                for index, id_
                in source_reprs.items()
                if index not in done_reprs or done_reprs[index] is None
            }

            with pretty_tqdm(total=len(source_reprs)) as pbar:
                pbar.update(len(source_reprs) - len(remaining_reprs))
                for batch in chunked(remaining_reprs.items(), n=batch_size):
                    batch_sources = dict(batch)
                    batch_reprs = func(batch_sources)
                    done_reprs |= batch_reprs
                    save_pickle(path, done_reprs)
                    pbar.update(len(batch_sources))

            return done_reprs

        return inner

    return outer
