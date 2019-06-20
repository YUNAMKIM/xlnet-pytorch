import torch
import torch.nn as nn
import torch.nn.functional as fnn


class PostAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        kernel_weight = torch.tensor([config.model.hidden_size, config.model.head_num, config.model.head_dim])
        self.kernel = nn.Parameter(kernel_weight)
        self.dropout = nn.Dropout(config.model.dropout_prob)
        self.layer_norm = nn.LayerNorm(config.model.hidden_size)

    def forward(self, head_input, attn_vec, residual=True):
        attn_out = torch.einsum('ibnd,hnd->ibh', attn_vec, self.kernel)
        attn_out = self.dropout(attn_out)
        output = fnn.layer_norm(attn_out)
        return output
