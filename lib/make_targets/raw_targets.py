import numpy as np

from lib.make_targets.embedders import seqvec_embedder, esm_embedder, prottrans_embedder
from lib.util.pickle_util import load_pickle


def load_seqvec_raw():
    # NOTE it has 3 layers, but because of size issues only the first is used
    return {
        key: value[0]
        for key, value
        in load_pickle(seqvec_embedder.partial_path).items()
        if value is not None
    }


def load_esm_raw():
    return {
        key: value
        for key, value
        in load_pickle(esm_embedder.partial_path).items()
        if value is not None
    }


def load_prottrans_raw():
    return {
        key: value
        for key, value
        in load_pickle(prottrans_embedder.partial_path).items()
        if value is not None
    }


raw_target_repr_loaders = [
    load_seqvec_raw,
    load_esm_raw,
    load_prottrans_raw
]
