import torch
import torch.nn as nn
import torch.nn.functional as fnn

from typing import Union


class AbsoluteAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dropout = nn.Dropout(config.model.dropout)

    def forward(self, q_head, k_head, v_head, attn_mask: torch.Tensor = None, scale: Union[int, torch.Tensor] = 1):
        attn_score = torch.einsum('ibnd,jbnd->ijbn', [q_head, k_head]) * scale

        if attn_mask is not None:
            attn_score = attn_score - 1e30 * attn_mask

        attn_prob = fnn.softmax(attn_score, dim=-1)
        attn_prob = self.dropout.forward(attn_prob)

        attn_vec = torch.einsum('ijbn,jbnd->ibnd', attn_prob, v_head)
        return attn_vec
