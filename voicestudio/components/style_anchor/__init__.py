from torch import nn


class DirectStyleAnchor(nn.Module):
    """Style Embedding Anchor using Direct Optimization"""
    def __init__(self, embedding_dim=512, pretrained_bos=None):
        super().__init__()

        if pretrained_bos is not None:
            self.style_anchor = nn.Parameter(pretrained_bos.clone())
        else:
            self.style_anchor = nn.Parameter(
                torch.randn(1, embedding_dim) * 0.02
            )

    def forward(self, token_embeddings):
        token_embeddings[:, 0] = self.style_anchor
        return token_embeddings


class EncoderStyleAnchor(nn.Module):
    """Style Embedding Anchor using In-direct Optimization with 2-Layer MLP (P-Tuning Concept)"""
    def __init__(self, embedding_dim=512):
        super().__init__()
        self.base = nn.Parameter(torch.randn(1, embedding_dim))

        self.encoder = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.Tanh(),
            nn.Linear(embedding_dim, embedding_dim)
        )
