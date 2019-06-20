import torch
import torch.nn as nn
import torch.nn.functional as fnn


class RelativeAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dropout = nn.Dropout(config.model.dropout_prob)

    def forward(self, q_head, k_head_h, v_head_h, k_head_r, seg_embed, seg_mat,
                r_w_bias, r_r_bias, r_s_bias, attn_mask=None, scale=1):
        # content based attention score
        ac = torch.einsum('ibnd,jbnd->ijbn', q_head + r_w_bias, k_head_h)

        # position based attention score
        bd = torch.einsum('ibnd,jbnd->ijbn', q_head + r_r_bias, k_head_r)
        bd = self.rel_shift(bd, klen=ac.size(1))

        # segment based attention score
        if seg_mat is not None:
            ef = torch.einsum('ibnd,snd->ibns', q_head + r_s_bias, seg_embed)
            ef = torch.einsum('ijbs,ibns->ijbn', seg_mat, ef)
        else:
            ef = 0

        # merge attention scores and perform masking
        attn_score = (ac + bd + ef) * scale
        if attn_mask is not None:
            # attn_score = attn_score * (1 - attn_mask) - 1e30 * attn_mask
            attn_score = attn_score - 1e30 * attn_mask

        # attention probability
        attn_prob = fnn.softmax(attn_score, 1)
        attn_prob = self.dropout(attn_prob)

        # attention output
        attn_vec = torch.einsum('ijbn,jbnd->ibnd', attn_prob, v_head_h)
        return attn_vec

    def rel_shift(self, x, klen=-1):
        """
        perform relative shift to form the relative attention score.

        origin codes
        x = tf.reshape(x, [x_size[1], x_size[0], x_size[2], x_size[3]])
        x = tf.slice(x, [1, 0, 0, 0], [-1, -1, -1, -1])
        x = tf.reshape(x, [x_size[0], x_size[1] - 1, x_size[2], x_size[3]])
        x = tf.slice(x, [0, 0, 0, 0], [-1, klen, -1, -1])
        """
        x = x.transpose(0, 1).narrow(0, 1, x.size(0) - 1)
        x = x.transpose(0, 1).narrow(1, 0, klen if klen > 0 else x.size(1) - klen)
        return x
