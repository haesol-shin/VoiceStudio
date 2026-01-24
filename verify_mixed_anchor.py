
import torch
from voicestudio.components.style_anchor import MixedStyleAnchorEmbedding

def test_mixed_anchor():
    try:
        embedding = MixedStyleAnchorEmbedding(
            num_embeddings=100, 
            embedding_dim=16, 
            direct_anchor_token_id=1, 
            encoder_anchor_token_id=2
        )
        input_ids = torch.tensor([1, 2, 3])
        output = embedding(input_ids)
        print("MixedStyleAnchorEmbedding forward pass successful.")
    except Exception as e:
        print(f"MixedStyleAnchorEmbedding failed: {e}")

if __name__ == "__main__":
    test_mixed_anchor()
