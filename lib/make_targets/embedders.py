from lib.util.anchor_util import TARGETS


class Embedder:
    all_ = []

    def __init__(self, repr_name: str, embedder_name: str, ):
        self.repr_name = repr_name
        self.embedder_name = embedder_name
        self.repr_path = TARGETS / f'{repr_name}.pt'
        self.partial_path = TARGETS / 'raw' / f'{repr_name}.partial'
        self.all_.append(self)


seqvec_embedder = Embedder(
    'target_seqvec',
    'SeqVecEmbedder'
)

esm_embedder = Embedder(
    'target_esm',
    'ESM1bEmbedder',
)

prottrans_embedder = Embedder(
    'target_prottrans',
    'ProtTransT5XLU50Embedder',
)
