from torch import nn

from BERT.PositionalEncoder import PositionalEncoder, PositionWiseFeedForward
from BERT.Attention import MultiHeadAttention


class Encoder(nn.Module):
    def __init__(self,
                 source_vocab_size: int,
                 embedding_dim: int,
                 layers: int,
                 heads: int,
                 dim_key: int,
                 dim_value: int,
                 dim_model: int,
                 dim_inner: int,
                 padding_idx: int,
                 dropout: float = 0.1,
                 num_pos: int = 128):
        super(Encoder, self).__init__()

        self.word_embedding = nn.Embedding(source_vocab_size, embedding_dim, padding_idx=padding_idx)

        self.positional_encoding = PositionalEncoder(embedding_dim, num_pos=num_pos)

        self.dropout = nn.Dropout(dropout)

        self.layer_stack = nn.ModuleList([
            EncoderLayer(dim_model,
                         dim_inner,
                         heads,
                         dim_key,
                         dim_value,
                         dropout=dropout) for _ in range(layers)
            ])

        self.layer_norm = nn.LayerNorm(dim_model, eps=1e-6)

    def forward(self, source_seq, source_mask, return_attns=False):
        enc_slf_attn_list = []

        # -- Forward
        enc_output = self.word_embedding(source_seq)
        enc_output = self.positional_encoding(enc_output)
        enc_output = self.dropout(enc_output)

        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(enc_output, source_mask)

            enc_slf_attn_list += [enc_slf_attn] if return_attns else []

        enc_output = self.layer_norm(enc_output)

        if return_attns:
            return enc_output, enc_slf_attn_list

        return enc_output


class EncoderLayer(nn.Module):

    def __init__(self,
                 dim_model,
                 dim_inner,
                 heads,
                 dim_key,
                 dim_value,
                 dropout=0.1):

        super(EncoderLayer, self).__init__()

        self.slf_attn = MultiHeadAttention(heads,
                                           dim_model,
                                           dim_key,
                                           dim_value,
                                           dropout=dropout)

        self.pos_ffn = PositionWiseFeedForward(dim_model,
                                               dim_inner,
                                               dropout=dropout)

    def forward(self, enc_input, slf_attn_mask=None):
        enc_output, enc_slf_attn = self.slf_attn(enc_input,
                                                 enc_input,
                                                 enc_input, mask=slf_attn_mask)

        enc_output = self.pos_ffn(enc_output)

        return enc_output, enc_slf_attn
