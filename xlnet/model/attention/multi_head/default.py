from xlnet.model.attention.stream.absolute import AbsoluteAttention
from xlnet.model.attention.core.post import PostAttention
from xlnet.model.attention.core.head import HeadAttention


class MultiHeadAttention(HeadAttention, PostAttention, AbsoluteAttention):
    def __init__(self, config):
        HeadAttention.__init__(self, config)
        PostAttention.__init__(self, config)
        AbsoluteAttention.__init__(self, config)

    def forward(self, q, k, v, attn_mask, scale, residual=True):
        q_head, k_head, v_head = HeadAttention.forward(self, q, k, v)
        attn_vec = AbsoluteAttention.forward(self, q_head, k, v, attn_mask, scale)
        output = PostAttention.forward(self, v, attn_vec, residual)
        return output
