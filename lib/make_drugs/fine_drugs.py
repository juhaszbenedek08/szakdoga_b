import numpy as np
import torch

from lib.make_drugs.raw_drugs import load_maccs_raw, load_rdkit_raw, load_morgan_raw
from lib.prepare_dataset.folding import Folding, load_folding
from lib.util.anchor_util import MACCS_PATH, RDKIT_PATH, MORGAN_PATH
from lib.util.device_util import drug_device, dtype
from lib.util.load_util import load
from lib.util.torch_util import load_torch, save_torch


class DrugTensorRepresentation:
    def __init__(self, mapping: dict[int, np.ndarray], folding: Folding):
        self._mapping = {key: i for i, key in enumerate(mapping.keys())}
        self._internal = torch.tensor(np.stack(tuple(mapping.values())), device=drug_device, dtype=dtype)

        no_test_drug_ids = set(folding.no_test_drugs)
        std, mean = torch.std_mean(
            torch.tensor(
                np.stack(
                    tuple(
                        value
                        for key, value
                        in mapping.items()
                        if key in no_test_drug_ids
                    )
                ),
                device=drug_device,
                dtype=dtype
            ),
            dim=0
        )
        eps = torch.tensor(1e-6, device=drug_device, dtype=dtype)

        self._internal = (self._internal - mean) / torch.sqrt(std + eps)

        self.repr_num = self._internal.size(1)

    def __getitem__(self, item: int) -> torch.Tensor:
        return self._internal[self._mapping[item]]

    def to(self, *args, **kwargs):
        self._internal = self._internal.to(*args, **kwargs)

    def keys(self):
        return self._mapping.keys()


@load(MACCS_PATH, load_torch, save_torch)
def load_maccs():
    return DrugTensorRepresentation(load_maccs_raw(), load_folding())


@load(RDKIT_PATH, load_torch, save_torch)
def load_rdkit():
    return DrugTensorRepresentation(load_rdkit_raw(), load_folding())


@load(MORGAN_PATH, load_torch, save_torch)
def load_morgan():
    return DrugTensorRepresentation(load_morgan_raw(), load_folding())


drug_repr_loaders = [
    load_maccs,
    load_rdkit,
    load_morgan
]
