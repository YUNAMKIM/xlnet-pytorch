import torch

from xlnet.model.attention.core.head import HeadAttention, HeadProjection
from xlnet.model.attention.stream.relative import RelativeAttention
from xlnet.model.attention.core.post import PostAttention


class RelativeMultiHeadAttention(HeadAttention, RelativeAttention, PostAttention):
    def __init__(self, config):
        HeadAttention.__init__(self, config)
        PostAttention.__init__(self, config)
        RelativeAttention.__init__(self, config)

        self.config = config
        self.r = HeadProjection(config)

    def forward(self, h, r, r_w_bias, r_r_bias, seg_mat, r_s_bias, seg_embed,
                attn_mask, mems):
        """Multi-head attention with relative positional encoding."""

        scale = 1 / (self.config.model.head_dim ** 0.5)
        cat = torch.cat([mems, h], 0) if mems is not None and len(mems) > 1 else h

        # content heads
        q_head_h, k_head_h, v_head_h = HeadAttention.forward(self, h, cat, cat)

        # positional heads
        k_head_r = self.r.forward(r)

        # core attention ops
        attn_vec = RelativeAttention.forward(self, q_head_h, k_head_h, v_head_h,
                                             k_head_r, seg_embed, seg_mat, r_w_bias,
                                             r_r_bias, r_s_bias, attn_mask, scale=scale)

        output = PostAttention.forward(self, h, attn_vec)
        return output
