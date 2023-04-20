from torch import nn
from BERT.Encoder import Encoder


class RecommendationTransformer(nn.Module):

    def __init__(self,
                 vocab_size: int,
                 heads: int = 4,
                 layers: int = 6,
                 embedding_dim: int = 256,
                 padding_idx: int = 0,
                 num_pos: int = 128):
        super(RecommendationTransformer, self).__init__()
        """Recommendation model initializer
            Args:
                vocab_size (int): Number of unique tokens/items
                heads (int, optional): Number of heads in the Multi-Head Self Attention Transformers (). Defaults to 4.
                layers (int, optional): Number of Layers. Defaults to 6.
                emb_dim (int, optional): Embedding Dimension. Defaults to 256.
                pad_id (int, optional): Token used to pad tensors. Defaults to 0.
                num_pos (int, optional): Positional Embedding, fixed sequence. Defaults to 128
            """

        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.num_pos = num_pos
        self.vocab_size = vocab_size

        self.encoder = Encoder(
            source_vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            layers=layers,
            heads=heads,
            dim_model=embedding_dim,
            dim_inner=embedding_dim * 4,
            dim_value=embedding_dim,
            dim_key=embedding_dim,
            padding_idx=padding_idx,
            num_pos=num_pos,
        )

        self.rec = nn.Linear(embedding_dim, vocab_size)

    def forward(self, source, source_mask):
        enc_op = self.encoder(source, source_mask)

        op = self.rec(enc_op)

        return op.permute(0, 2, 1)

