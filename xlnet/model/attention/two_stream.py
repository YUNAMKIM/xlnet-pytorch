import torch

from .core.head import HeadAttention, HeadProjection
from .core.post import PostAttention
from .stream.relative import RelativeAttention


class TwoStreamRelativeAttention(HeadAttention, RelativeAttention, PostAttention):
    def __init__(self, config):
        HeadAttention.__init__(self, config)
        RelativeAttention.__init__(self, config)
        PostAttention.__init__(self, config)

        self.config = config
        self.r = HeadProjection(config)

    def forward(self, h, g, r, mems, r_w_bias, r_r_bias, seg_mat, r_s_bias,
                seg_embed, attn_mask_h, attn_mask_g, target_mapping):
        scale = 1 / (self.config.model.head_dim ** 0.5)
        cat = torch.cat([mems, h], 0) if mems is not None and len(mems) > 1 else h

        # content-based query, key, value head
        q_head_h, k_head_h, v_head_h = HeadAttention.forward(self, h, cat, cat)

        # position-based key head
        k_head_r = self.r.forward(r)

        # -------h-stream-------
        # core attention ops
        attn_vec_h = RelativeAttention.forward(self, q_head_h, k_head_h, v_head_h, k_head_r,
                                               seg_embed, seg_mat, r_w_bias,
                                               r_r_bias, r_s_bias, attn_mask_h, scale)

        # post processing
        output_h = PostAttention.forward(self, h, attn_vec_h)

        # ------g-stream -----
        # query-stream query head
        q_head_g = self.q.forward(g)

        # core attention ops
        if target_mapping is not None:
            q_head_g = torch.einsum('mbnd,mlb->lbnd', q_head_g, target_mapping)

        attn_vec_g = RelativeAttention.forward(self, q_head_g, k_head_h, v_head_h, k_head_r, seg_embed, seg_mat,
                                               r_w_bias, r_r_bias, r_s_bias, attn_mask_g, scale)

        if target_mapping is not None:
            attn_vec_g = torch.einsum('lbnd,mlb->mbnd', attn_vec_g, target_mapping)

        # post processing
        output_g = self.post_attn(g, attn_vec_g)

        return output_h, output_g
