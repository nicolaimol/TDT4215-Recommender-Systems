from torch import nn

from BERT.PositionalEncoder import PositionalEncoder, PositionWiseFeedForward
from BERT.Attention import MultiHeadAttention


class Decoder(nn.Module):
    def __init__(self,
                 target_vocab_size: int,
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
        super(Decoder, self).__init__()

        self.word_embedding = nn.Embedding(target_vocab_size, embedding_dim, padding_idx=padding_idx)

        self.positional_encoding = PositionalEncoder(embedding_dim, num_pos=num_pos)

        self.dropout = nn.Dropout(dropout)

        self.layer_stack = nn.ModuleList([
            DecoderLayer(dim_model,
                            dim_inner,
                            heads,
                            dim_key,
                            dim_value,
                            dropout=dropout) for _ in range(layers)
            ])

        self.layer_norm = nn.LayerNorm(dim_model, eps=1e-6)

    def forward(self,
                target_seq,
                target_mask,
                enc_output,
                source_mask,
                return_attns=False):

        dec_slf_attn_list, dec_enc_attn_list = [], []

        # -- Forward

        dec_output = self.word_embedding(target_seq)
        dec_output = self.positional_encoding(dec_output)
        dec_output = self.dropout(dec_output)

        for dec_layer in self.layer_stack:

            dec_output, dec_slf_attn, dec_enc_attn = dec_layer(
                dec_output,
                enc_output,
                target_mask,
                source_mask)

            dec_slf_attn_list += [dec_slf_attn] if return_attns else []
            dec_enc_attn_list += [dec_enc_attn] if return_attns else []

        dec_output = self.layer_norm(dec_output)

        if return_attns:
            return dec_output, dec_slf_attn_list, dec_enc_attn_list

        return dec_output


class DecoderLayer(nn.Module):
    def __init__(self,
                 dim_model: int,
                 dim_inner: int,
                 heads: int,
                 dim_key: int,
                 dim_value: int,
                 dropout: float = 0.1):
        super(DecoderLayer, self).__init__()

        self.slf_attn = MultiHeadAttention(
            heads,
            dim_model,
            dim_key,
            dim_value,
            dropout=dropout)

        self.enc_attn = MultiHeadAttention(
            heads,
            dim_model,
            dim_key,
            dim_value,
            dropout=dropout)

        self.pos_ffn = PositionWiseFeedForward(
            dim_model,
            dim_inner,
            dropout=dropout)

    def forward(self,
                dec_input,
                enc_output,
                slf_attn_mask=None,
                dec_enc_attn_mask=None):

        dec_output, dec_slf_attn = self.slf_attn(
            dec_input,
            dec_input,
            dec_input,
            mask=slf_attn_mask)

        dec_output, dec_enc_attn = self.enc_attn(
            dec_output,
            enc_output,
            enc_output,
            mask=dec_enc_attn_mask)

        dec_output = self.pos_ffn(dec_output)

        return dec_output, dec_slf_attn, dec_enc_attn