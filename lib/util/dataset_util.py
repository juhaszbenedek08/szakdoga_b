from dataclasses import dataclass
from functools import reduce
from itertools import product
from operator import mul
from typing import Callable, Any, Optional

import numpy as np
import torch
from torch import Generator, randperm, randint
from torch.utils.data import Sampler, Dataset


@dataclass
class MappingDataset(Dataset):
    mapping: Callable[[...], Any]

    def __getitem__(self, item):
        return self.mapping(item)


class CartesianList:
    def __init__(self, *lists: list):
        self.lists = lists
        self.total_length = reduce(mul, (len(l) for l in lists))

    def __getitem__(self, item):
        item = int(item)
        result = []
        for l in reversed(self.lists):
            item, rem = divmod(item, len(l))
            result.append(l[rem])
        return tuple(reversed(result))

    def __len__(self):
        return self.total_length

    def __iter__(self):
        return product(*self.lists)

    def as_cartesian_set(self):
        return CartesianSet(*(set(l) for l in self.lists))


class CartesianSet:
    def __init__(self, *sets: set):
        self.sets = sets
        self.total_length = reduce(mul, (len(s) for s in sets))

    def __contains__(self, item):
        return all(i in s for i, s in zip(item, self.sets))

    def __len__(self):
        return self.total_length

    def __iter__(self):
        return product(*self.sets)

    def as_cartesian_list(self):
        return CartesianSet(*(list(s) for s in self.sets))


class ExclusionRandomSampler(Sampler):
    def __init__(self,
                 total: list,
                 excluded: set,
                 generator: Generator,
                 limit: Optional[int] = None,
                 with_replacement: bool = False
                 ):
        super().__init__(data_source=None)
        self.total = total
        self.excluded = excluded
        self.generator = generator
        self.with_replacement = with_replacement
        self.limit = min((len(self.total) - len(self.excluded), limit or np.inf))

    def __iter__(self):
        if self.with_replacement:
            for i in torch.randint(0, len(self.total), (self.limit,), generator=self.generator):
                yield self.total[i]
        else:
            done = 0
            for i in randperm(len(self.total), generator=self.generator):
                item = self.total[i]
                if item not in self.excluded:
                    yield item
                    done += 1
                if done < self.limit:
                    return

    def __len__(self):
        return self.limit


class InclusionRandomSampler(Sampler):
    def __init__(
            self,
            included: list,
            generator: Generator,
            limit: Optional[int] = None,
            with_replacement: bool = False
    ):
        super().__init__(data_source=None)
        self.included = included
        self.generator = generator
        self.limit = min((len(self.included), limit or np.inf))
        self.with_replacement = with_replacement

    def __iter__(self):
        if self.with_replacement:
            for i in torch.randint(0, len(self.included), (self.limit,), generator=self.generator):
                yield self.included[i]
        else:
            for i in randperm(len(self.included), generator=self.generator):
                yield self.included[i]

    def __len__(self):
        return self.limit


def cartesian_split(
        cartesian: CartesianList,
        generator: torch.Generator,
        ratio: float
):
    result = []

    single_ratio = ratio ** (1 / len(cartesian.lists))
    for l in cartesian.lists:
        split = round(len(l) * single_ratio)
        order = randperm(len(l), generator=generator)
        result.append([l[index] for index in order[:split]])
    return CartesianList(*result)


def split(
        items: list,
        generator: torch.Generator,
        ratio: float
):
    order = randperm(len(items), generator=generator)
    split = int(len(items) * ratio)
    result = [items[index] for index in order[:split]]

    return result


def cartesian_fuse(*items: list[set]):
    def helper(set_list: list[set]):
        return list(
            reduce(
                lambda a, b: a.intersection(b),
                set_list[1:],
                set_list[0],
            )
        )

    return CartesianList(*(helper(item) for item in items))
