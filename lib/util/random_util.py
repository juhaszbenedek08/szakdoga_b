import torch

from lib.util.device_util import cuda


def get_generator(
        seed: int,
        device: torch.device
):
    result = torch.Generator(device)
    result.manual_seed(seed)
    return result


cuda_generator = torch.Generator(device=cuda)
