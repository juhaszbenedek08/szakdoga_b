import logging
import pprint
import sys
from functools import wraps, partial
from pathlib import Path
from typing import Callable
import tqdm
from torch.utils.tensorboard import SummaryWriter

from tqdm.contrib import DummyTqdmFile

pretty_tqdm = partial(tqdm.tqdm, file=sys.stdout, dynamic_ncols=True)
sys.stdout, sys.stderr = map(DummyTqdmFile, (sys.stdout, sys.stderr))

##################################################

logger = logging.getLogger('thesis')
logger.setLevel(level=logging.INFO)
_sh = logging.StreamHandler(sys.stdout)
_sh.setFormatter(
    logging.Formatter(
        fmt='%(asctime)s %(levelname)-8s %(message)s',
        datefmt='%H:%M:%S'
    )
)
logger.addHandler(_sh)


def log_func(func: Callable):
    @wraps(func)
    def inner(*args, **kwargs):
        logger.info('Entering: ' + inner.__qualname__ + '...')
        result = func(*args, **kwargs)
        logger.info('Exiting: ' + inner.__qualname__)
        return result

    return inner


_pp = pprint.PrettyPrinter(
    sort_dicts=False,
    width=120,
    indent=4,
    stream=sys.stdout
)


def pretty_format(arg):
    return _pp.pformat(arg)


class LearningLogs:

    def __init__(self, base_dir: Path):
        self.path = base_dir / 'logs'
        self.path.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(str(self.path))
        with open(self.path / 'watch.sh', 'wt') as f:
            f.write(
                f"#! /bin/bash\n"
                f"source /home/$USER/miniconda3/etc/profile.d/conda.sh\n"
                f"conda activate thesis-core\n"
                f"tensorboard --logdir={str(self.path)}"
            )

    def log_scalar(self, tags: list[str], epoch: int, value: float):
        logger.info(f'{f"[{epoch}]" : <5} {pretty_format(tags) : <30} {value: .4}')
        self.writer.add_scalar('/'.join(tags), value, epoch)

    def log_comment(self, tags: list[str], epoch: int, value: str):
        logger.info(f'{f"[{epoch}]" : <5} {pretty_format(tags) : <30} {value}')

    def __del__(self):
        self.writer.close()
