import torch.nn as nn
import torch

from xlnet.model.activation.gelu import get_activation


class PositionWisedFNN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layer_1 = nn.Linear(config.model.hidden_size, config.model.intermediate_size)
        self.layer_2 = nn.Linear(config.model.intermediate_size, config.model.hiden_size)

        self.activation = get_activation(config)
        self.dropout = nn.Dropout(config.model.dropout_prob)
        self.layer_norm = nn.LayerNorm(config.model.hidden_size)

    def forward(self, input_x: torch.Tensor) -> torch.Tensor:
        x = self.layer_1(input_x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.layer_2(x)
        x = self.dropout(x)
        x = self.layer_norm(x + input_x)
        return x
