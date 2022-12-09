import bio_embeddings.embed as embed

from lib.make_targets.embedders import Embedder
from lib.util.anchor_util import KIBA_PATH, SEQ_PATH
from lib.util.generate_util import batch_generate, safe_generate
from lib.util.pickle_util import load_pickle

if __name__ == '__main__':
    target_ids = load_pickle(KIBA_PATH).target_ids

    target_sequences = load_pickle(SEQ_PATH)

    for embedder in Embedder.all_:
        model = getattr(embed, embedder.embedder_name)(device=embedder.device, half_model=True)

        batch_generate(
            load_source_reprs=lambda: target_sequences,
            path=embedder.partial_path,
            batch_size=1
        )(
            lambda batch_sources: safe_generate(
                source_reprs=batch_sources,
                generator=lambda source_repr: model.embed(source_repr),
                name=embedder.repr_name,
                ids=target_ids,
                with_bar=False
            )
        )
