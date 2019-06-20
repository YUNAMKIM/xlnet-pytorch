import torch
import torch.nn as nn


class HeadProjection(nn.Module):
    def __init__(self, config):
        super().__init__()
        kernel_weight = torch.rand([config.model.hidden_size, config.model.head_num, config.model.head_dim])
        self.kernel = nn.Parameter(kernel_weight)

    def forward(self, head_input) -> torch.Tensor:
        return torch.einsum('ibh,hnd->ibnd', head_input, self.kernel)


class HeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.q = HeadProjection(config)
        self.k = HeadProjection(config)
        self.v = HeadProjection(config)

    def forward(self, *inputs):
        return (model.forward(source) for source, model in zip(inputs, [self.q, self.k, self.k]))
