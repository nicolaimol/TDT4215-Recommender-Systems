import torch
from torch import nn


class ScaledDotProductAttention(nn.Module):
    def __init__(self, temperature, attn_dropout=0.1):
        super(ScaledDotProductAttention, self).__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = self.dropout(nn.Softmax(dim=-1)(attn))

        output = torch.matmul(attn, v)

        return output, attn


class MultiHeadAttention(nn.Module):
    def __init__(self, heads, dim_model, dim_key, dim_value, dropout=0.1):

        super(MultiHeadAttention, self).__init__()
        self.dim_key = dim_key
        self.dim_value = dim_value
        self.heads = heads

        self.to_q = nn.Linear(dim_model, heads * dim_key, bias=False)
        self.to_k = nn.Linear(dim_model, heads * dim_key, bias=False)
        self.to_v = nn.Linear(dim_model, heads * dim_value, bias=False)

        self.dropout = nn.Dropout(dropout)

        self.union = nn.Linear(heads * dim_value, dim_model, bias=False)

        self.attn = ScaledDotProductAttention(temperature=dim_key ** 0.5)

        self.layer_norm = nn.LayerNorm(dim_model, eps=1e-6)

    def forward(self, q, k, v, mask=None):
        dim_key, dim_value, heads = self.dim_key, self.dim_value, self.heads

        batch_size = q.size(0)
        lenght_q = q.size(1)
        lenght_k = k.size(1)
        lenght_l = v.size(1)

        residual = q

        q = self.layer_norm(q)

        q = self.to_q(q).view(batch_size, lenght_q, heads, dim_key)

        k = self.to_k(k).view(batch_size, lenght_k, heads, dim_key)

        v = self.to_v(v).view(batch_size, lenght_l, heads, dim_value)

        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(1)

        q, attn = self.attn(q, k, v, mask=mask)

        q = q.transpose(1, 2).contiguous().view(batch_size, lenght_q, -1)

        q = self.union(q)
        q = self.dropout(q)

        return q + residual, attn