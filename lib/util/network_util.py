import math

import torch
from torch.nn import Module, init
from torch.nn.parameter import Parameter


def L1(x):
    return torch.sum(torch.abs(x.flatten()))


def L2(x):
    return torch.sum(torch.square(x))


class ElasticNetRegularized(Module):
    def __init__(self, module, l1_decay, l2_decay):
        super().__init__()
        self.module = module
        if l1_decay is None and l2_decay is None:
            return

        self.register_buffer('l1_decay', torch.tensor(0.0 if l1_decay is None else l1_decay))
        self.register_buffer('l2_decay', torch.tensor(0.0 if l2_decay is None else l2_decay))

        self.hook = self.module.register_full_backward_hook(self._weight_decay_hook)

    def _weight_decay_hook(self):
        for param in self.module.parameters():
            if param.grad is None or torch.all(param.grad == 0.0):
                param.grad = self.l1_decay * torch.sign(param.data) + self.l2_decay * param.data

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)


class RandomReductionLinear(Module):
    def __init__(
            self,
            input_width: int,
            subset_size: int,
            output_width: int,
            generator: torch.Generator,
            with_replacement: bool = False
    ):
        super().__init__()
        self.subset_size = subset_size

        if with_replacement:
            self.permutations = torch.concatenate(
                [
                    torch.randperm(input_width, dtype=torch.long, generator=generator)
                    for _
                    in range(math.ceil(output_width * subset_size / input_width))
                ]
            )[:output_width * subset_size].reshape(output_width, subset_size)
        else:
            self.permutations = torch.randint(
                0,
                input_width,
                (output_width, subset_size),
                dtype=torch.long,
                generator=generator
            )

        self.weight = Parameter(torch.empty((output_width, subset_size)))
        self.bias = Parameter(torch.empty(output_width))

        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        return torch.linalg.vecdot(
            x[..., self.permutations],
            self.weight
        ) + self.bias


class GetAttrConstructor(type):
    def __getitem__(cls, x):
        return cls(x)


class Selector(Module, metaclass=GetAttrConstructor):
    def __init__(self, args):
        super().__init__()
        self.args = args

    def forward(self, x):
        return x[self.args]


class Autoencoder(Module):
    def __init__(self, *, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        return self.decoder(self.encoder(x))


class Trivial(Module):
    def __init__(self, ):
        super().__init__()
        self.dummy = Parameter(torch.tensor(0.0))

    @staticmethod
    def forward(x):
        return x


class Concater(Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return torch.concatenate(x, dim=self.dim)


class StatelessModule(Module):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)

class AddBias(Module):
    def __init__(self):
        super().__init__()
        self.bias = Parameter()