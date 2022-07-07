import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

class ScaledDotProductAttention(nn.Module):
    """
    Scaled dot-product attention mechanism.
    """

    def __init__(self, attention_dropout=0.0):
        super(ScaledDotProductAttention, self).__init__()
        self.dropout = nn.Dropout(attention_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, scale=None, attn_mask=None):
        """
        前向传播

        args:
            q: Queries张量，形状[B, L_q, D_q]
            k: keys张量， 形状[B, L_k, D_k]
            v: Values张量，形状[B, L_v, D_v]
            scale: 缩放因子，一个浮点标量
            attn_mask: Masking张量，形状[B, L_q, L_k]
        returns:
            上下文张量和attention张量
        """
        attention = torch.bmm(q, k.transpose(1, 2))
        if scale:
            attention = attention * scale
        if attn_mask:
            # 给需要mask的地方设置一个负无穷
            attention = attention.masked_fill_(attn_mask, -np.inf)
        # 计算softmax
        attention = self.softmax(attention)
        # 添加dropout
        attention = self.dropout(attention)
        # 和V做点积
        context = torch.bmm(attention, v)

        return context, attention