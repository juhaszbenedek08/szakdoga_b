import torch

from lib.make_drugs.raw_drugs import raw_drug_repr_loaders
from lib.make_dta.load_kiba import load_kiba
from lib.make_dta.minimal_dta import MinimalDTA
from lib.make_targets.raw_targets import raw_target_repr_loaders
from lib.util.anchor_util import FOLDING_PATH
from lib.util.dataset_util import CartesianList, cartesian_fuse, split, cartesian_split
from lib.util.device_util import cpu
from lib.util.load_util import load
from lib.util.random_util import get_generator


class Folding:
    def __init__(
            self,
            cartesian: CartesianList,
            generator: torch.Generator
    ):
        self.cartesian = cartesian
        self.no_test_cartesian = cartesian_split(cartesian, generator, 0.9)
        self.validate_drugs = split(self.no_test_drugs, generator, 0.2)
        self.validate_targets = split(self.no_test_targets, generator, 0.2)
        self.validate_pairs = split(self.no_test_cartesian, generator, 0.2)

    @property
    def drugs(self):
        return self.cartesian.lists[0]

    @property
    def targets(self):
        return self.cartesian.lists[1]

    @property
    def no_test_drugs(self):
        return self.no_test_cartesian.lists[0]

    @property
    def no_test_targets(self):
        return self.no_test_cartesian.lists[1]


FOLDING_SEED = 12335627395


@load(FOLDING_PATH)
def load_folding():
    kiba: MinimalDTA = load_kiba()

    return Folding(
        cartesian_fuse(
            [
                set(kiba.drug_ids.keys()),
                *(set(r().keys()) for r in raw_drug_repr_loaders)
            ],
            [
                set(kiba.target_ids.keys()),
                *(set(r().keys()) for r in raw_target_repr_loaders)
            ]
        ),
        get_generator(FOLDING_SEED, cpu)
    )
