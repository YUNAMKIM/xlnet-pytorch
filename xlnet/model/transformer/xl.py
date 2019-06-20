import torch
import torch.nn as nn


class TransformerXLBias:
    def __init__(self, config):
        if config.model.untie_bias:
            r_w_bias = torch.rand([config.model.num_layers, config.model.head_num, config.model.head_dim])
            r_r_bias = torch.rand([config.model.num_layers, config.model.head_num, config.model.head_dim])
        else:
            r_w_bias = torch.rand([config.model.head_num, config.model.head_dim])
            r_r_bias = torch.rand([config.model.head_num, config.model.head_dim])

        self.r_w_bias = nn.Parameter(r_w_bias)
        self.r_r_bias = nn.Parameter(r_r_bias)


class TransformerXL(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.transformer = TransformerXLBias(config)

    def forward(self, *input):
        pass
