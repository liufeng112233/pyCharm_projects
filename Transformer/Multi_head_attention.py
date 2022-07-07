import torch.nn as nn
from Scaled_dot_product_attention import ScaledDotProductAttention


class MultiHeadAttention(nn.Module):

    def __init__(self, model_dim=512, num_heads=8, dropout=0.0):
        super(MultiHeadAttention, self).__init__()

        self.dim_per_head = model_dim / num_heads
        self.num_heads = num_heads
        self.linear_q = nn.Linear(model_dim, self.dim_per_head * num_heads)
        self.linear_k = nn.Linear(model_dim, self.dim_per_head * num_heads)
        self.linear_v = nn.Linear(model_dim, self.dim_per_head * num_heads)

        self.dot_product_attention = ScaledDotProductAttention(dropout)
        self.linear_final = nn.Linear(model_dim, model_dim)
        self.dropout = nn.Dropout(dropout)
        # multi-head attention之后需要做layer norm
        self.layer_num = nn.LayerNorm(model_dim)

    def forward(self, query, key, value, attn_mask=None):
        num_heads = self.num_heads
        dim_per_head = self.dim_per_head
        # 残差连接
        residual = query

        batch_size = key.size(0)

        # linear projection
        query = self.linear_q(query)  # [B, L, D]
        key = self.linear_k(key)  # [B, L, D]
        value = self.linear_v(value)  # [B, L, D]

        # split by head
        query = query.view(batch_size * num_heads, -1, dim_per_head)  # [B * 8, , D / 8]
        key = key.view(batch_size * num_heads, -1, dim_per_head)  #
        value = value.view(batch_size * num_heads, -1, dim_per_head)

        if attn_mask:
            attn_mask = attn_mask.repeat(num_heads, 1, 1)
        # scaled dot product attention
        scale = (key.size(-1) // num_heads) ** -0.5
        context, attention = self.dot_product_attention(
            query, key, value, scale, attn_mask
        )

        # concat heads
        context = context.view(batch_size, -1, dim_per_head * num_heads)

        # final linear projection
        output = self.linear_final(context)

        # dropout
        output = self.dropout(output)

        # add residual and norm layer
        output = self.layer_num(residual + output)

        return output, attention
