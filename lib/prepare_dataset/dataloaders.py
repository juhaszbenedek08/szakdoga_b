from math import log2, ceil
from statistics import mean
from typing import Union

import torch
from torch.nn.functional import interpolate
from torch.utils.data import DataLoader, Dataset, Sampler

from lib.util.dataset_util import MappingDataset
from lib.util.other_util import nearest_2_power


def autoencoder_dataloader(
        dataset: Dataset,
        sampler: Sampler,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype
):
    def helper(item):
        r = dataset[item].to(device, dtype=dtype)
        return r, r

    return DataLoader(
        dataset=MappingDataset(helper),
        sampler=sampler,
        batch_size=batch_size,
        drop_last=False
    )


def fused_dataset(*reprs, repr_num: Union[str, int] = 'max'):
    total_repr_num = sum(r.repr_num for r in reprs)
    if repr_num == 'max':
        repr_num = total_repr_num

        def helper(item):
            return torch.concatenate(tuple(r[item] for r in reprs))

    elif isinstance(repr_num, int):
        reduction_ratio = repr_num / total_repr_num
        repr_nums = [int(r.repr_num * reduction_ratio) for r in reprs]
        repr_nums[-1] = repr_num - sum(r.repr_num for r in reprs[:-1])

        def helper(item):
            return torch.concatenate(
                tuple(
                    interpolate(r[item][None, ...], size=size, mode='nearest-exact')[0]
                    for r, size in zip(reprs, repr_nums)
                ),
                dim=0
            )
    else:
        raise Exception()

    result = MappingDataset(helper)
    result.repr_num = repr_num
    return result


def fused_flattened_dataset(*reprs, repr_num: Union[str, int] = 'max'):
    total_repr_num = sum(r.repr_num * r.seq_length for r in reprs)
    if repr_num == 'max':
        repr_num = total_repr_num

        def helper(item):
            return torch.concatenate(tuple(r[item].flatten() for r in reprs))

    elif isinstance(repr_num, int):
        reduction_ratio = repr_num / total_repr_num
        repr_nums = [int(r.repr_num * r.seq_length * reduction_ratio) for r in reprs]
        repr_nums[-1] = repr_num - sum(r for r in repr_nums[:-1])

        def helper(item):
            return torch.concatenate(
                tuple(
                    interpolate(r[item].flatten()[None, None, ...], size=size, mode='nearest-exact')[0, 0]
                    for i, (r, size) in enumerate(zip(reprs, repr_nums))
                ),
                dim=0
            )
    else:
        raise Exception()

    result = MappingDataset(helper)
    result.repr_num = repr_num
    return result


def fused_aligned_dataset(*reprs, seq_length: Union[str, int] = 'max'):
    if seq_length == 'max':
        seq_length = 2 ** ceil(log2(max(r.seq_length for r in reprs)))
    elif seq_length == 'nearest':
        seq_length = nearest_2_power(int(mean(r.seq_length for r in reprs)))
    elif not isinstance(seq_length, int):
        raise Exception()

    def helper(item):
        return torch.concatenate(
            tuple(
                interpolate(r[item][None, ...], size=seq_length, mode='nearest-exact')[0]
                for r in reprs
            ),
            dim=0
        )

    result = MappingDataset(helper)
    result.repr_num = sum(r.repr_num for r in reprs)
    result.seq_length = seq_length
    return result


def predictor_dataset(
        *,
        drug_dataset,
        target_dataset,
        minimal_dta,
        device: torch.device,
        dtype: torch.dtype
):
    true = torch.tensor(1.0, device=device, dtype=dtype)
    false = torch.tensor(0.0, device=device, dtype=dtype)

    def helper(item):
        drug_id, target_id = item
        drug = drug_dataset[drug_id].to(device=device, dtype=dtype)
        target = target_dataset[target_id].to(device=device, dtype=dtype)
        affinity = true if minimal_dta[drug_id, target_id] else false
        return (drug, target), (drug, target, affinity)

    result = MappingDataset(helper)
    return result
