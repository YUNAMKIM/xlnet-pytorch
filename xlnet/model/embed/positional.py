import torch.nn as nn
import torch


class PositionEmbedding(nn.Module):
    def forward(self, position_seq: torch.Tensor, inverse_frequency: torch.Tensor,
                batch_size: int = None) -> torch.Tensor:
        """
        Calculating Positional Embedding
        :param position_seq:
        :param inverse_frequency:
        :param batch_size:
        :return: [seq_len, batch_size, embed_dim]
        """
        x = torch.einsum('i,d->id', [position_seq, inverse_frequency])
        x = torch.cat([torch.sin(x), torch.cos(x)], -1)
        x = x.repeat([1, batch_size, 1]) if batch_size else x
        return x
