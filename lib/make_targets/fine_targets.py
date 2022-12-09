import numpy as np
import torch

from lib.make_targets.embedders import prottrans_embedder, esm_embedder, seqvec_embedder
from lib.make_targets.raw_targets import load_seqvec_raw, load_esm_raw, load_prottrans_raw
from lib.prepare_dataset.folding import Folding, load_folding
from lib.util.device_util import target_device, dtype
from lib.util.load_util import load
from lib.util.torch_util import load_torch, save_torch


class TargetTensorRepresentation:
    def __init__(self, mapping: dict[int, np.ndarray], folding: Folding):
        targets = set(folding.targets)
        no_test_targets = set(folding.no_test_targets)

        mapping = {key: value for key, value in mapping.items() if key in targets}
        max_size = max(len(value) for value in mapping.values())

        order = folding.no_test_targets.copy()
        order.extend(key for key in targets if key not in no_test_targets)

        self._mapping = {key: i for i, key in enumerate(order)}
        self._internal = torch.tensor(
            np.stack(
                tuple(
                    np.pad(
                        mapping[key],
                        (
                            (0, max_size - len(mapping[key])),
                            (0, 0)
                        ),
                        mode='constant',
                        constant_values=np.nan
                    )
                    for key
                    in self._mapping.keys()
                )
            ),
            device=target_device,
            dtype=dtype
        )

        no_test_targets = self._internal[len(no_test_targets):]
        masked = torch.masked.masked_tensor(
            no_test_targets,
            torch.logical_not(
                no_test_targets.isnan()
            )
        )
        mean = masked.mean(dim=(0, 1))[None, None, :].to_tensor(np.nan)
        std = masked.std(dim=(0, 1))[None, None, :].to_tensor(np.nan)
        eps = torch.tensor(1e-6, device=target_device, dtype=dtype)

        self._internal = (self._internal - mean) / torch.sqrt(std + eps)
        self._internal = self._internal.nan_to_num(0.0)
        self._internal = torch.permute_copy(self._internal, (0, 2, 1))

        self.repr_num = self._internal.size(1)
        self.seq_length = self._internal.size(2)

    def __getitem__(self, item: int) -> torch.Tensor:
        return self._internal[self._mapping[item]]

    def to(self, *args, **kwargs):
        self._internal = self._internal.to(*args, **kwargs)

    def keys(self):
        return self._mapping.keys()


@load(seqvec_embedder.repr_path, load_torch, save_torch)
def load_seqvec():
    return TargetTensorRepresentation(load_seqvec_raw(), load_folding())


@load(esm_embedder.repr_path, load_torch, save_torch)
def load_esm():
    return TargetTensorRepresentation(load_esm_raw(), load_folding())


@load(prottrans_embedder.repr_path, load_torch, save_torch)
def load_prottrans():
    return TargetTensorRepresentation(load_prottrans_raw(), load_folding())


target_repr_loaders = [
    load_seqvec,
    load_esm,
    load_prottrans
]
