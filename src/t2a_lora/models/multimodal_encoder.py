"""Multimodal encoder for text and audio conditions."""

import torch
import torch.nn as nn
from typing import Optional, Union, Dict, Any
from dataclasses import dataclass
from transformers import AutoModel, AutoTokenizer


@dataclass
class MultimodalEncoderConfig:
    """Configuration for MultimodalEncoder."""
    text_encoder_name: str = "klue/roberta-large"  # Korean language model
    audio_encoder_dim: int = 512   # Dimension of audio feature encoder
    fusion_dim: int = 768          # Dimension of fusion layer
    fusion_method: str = "concat"  # Fusion method: "concat", "add", "cross_attention"
    dropout: float = 0.1           # Dropout rate
    use_projection: bool = True    # Whether to use projection layers


class MultimodalEncoder(nn.Module):
    """
    Multimodal encoder that processes both text descriptions and audio vectors.
    
    Supports Korean and English text descriptions along with audio feature vectors
    to generate unified condition embeddings for the hypernetwork.
    """
    
    def __init__(self, config: MultimodalEncoderConfig):
        super().__init__()
        self.config = config
        
        # Text encoder (using pre-trained Korean language model)
        self.text_encoder = AutoModel.from_pretrained(config.text_encoder_name)
        self.tokenizer = AutoTokenizer.from_pretrained(config.text_encoder_name)
        
        # Freeze text encoder initially (can be unfrozen for fine-tuning)
        for param in self.text_encoder.parameters():
            param.requires_grad = False
            
        # Audio feature encoder
        self.audio_encoder = nn.Sequential(
            nn.Linear(config.audio_encoder_dim, config.fusion_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.fusion_dim, config.fusion_dim),
        )
        
        # Text projection layer
        text_hidden_size = self.text_encoder.config.hidden_size
        if config.use_projection:
            self.text_proj = nn.Linear(text_hidden_size, config.fusion_dim)
        else:
            self.text_proj = nn.Identity()
            config.fusion_dim = text_hidden_size
        
        # Fusion layer
        self.fusion_layer = self._build_fusion_layer()
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.LayerNorm(self._get_fusion_output_dim()),
            nn.Dropout(config.dropout),
            nn.Linear(self._get_fusion_output_dim(), config.fusion_dim),
        )
        
    def _build_fusion_layer(self) -> nn.Module:
        """Build fusion layer based on fusion method."""
        if self.config.fusion_method == "concat":
            return nn.Identity()
        elif self.config.fusion_method == "add":
            return nn.Identity()
        elif self.config.fusion_method == "cross_attention":
            return nn.MultiheadAttention(
                embed_dim=self.config.fusion_dim,
                num_heads=8,
                dropout=self.config.dropout,
                batch_first=True,
            )
        else:
            raise ValueError(f"Unknown fusion method: {self.config.fusion_method}")
    
    def _get_fusion_output_dim(self) -> int:
        """Get output dimension of fusion layer."""
        if self.config.fusion_method == "concat":
            return self.config.fusion_dim * 2
        else:
            return self.config.fusion_dim
    
    def encode_text(
        self, 
        text: Union[str, list[str]], 
        device: Optional[torch.device] = None
    ) -> torch.Tensor:
        """
        Encode text descriptions.
        
        Args:
            text: Text description(s) to encode
            device: Target device for computation
            
        Returns:
            Text embeddings [batch_size, fusion_dim]
        """
        if isinstance(text, str):
            text = [text]
            
        # Tokenize text
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )
        
        if device is not None:
            inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Encode text
        with torch.no_grad():
            outputs = self.text_encoder(**inputs)
            
        # Use [CLS] token embedding or mean pooling
        if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
            text_embedding = outputs.pooler_output
        else:
            # Mean pooling over sequence length
            text_embedding = outputs.last_hidden_state.mean(dim=1)
        
        # Project to fusion dimension
        text_embedding = self.text_proj(text_embedding)
        
        return text_embedding
    
    def encode_audio(self, audio_features: torch.Tensor) -> torch.Tensor:
        """
        Encode audio feature vectors.
        
        Args:
            audio_features: Audio feature vectors [batch_size, audio_dim]
            
        Returns:
            Audio embeddings [batch_size, fusion_dim]
        """
        return self.audio_encoder(audio_features)
    
    def fuse_modalities(
        self, 
        text_emb: torch.Tensor, 
        audio_emb: torch.Tensor
    ) -> torch.Tensor:
        """
        Fuse text and audio embeddings.
        
        Args:
            text_emb: Text embeddings [batch_size, fusion_dim]
            audio_emb: Audio embeddings [batch_size, fusion_dim]
            
        Returns:
            Fused embeddings [batch_size, fusion_output_dim]
        """
        if self.config.fusion_method == "concat":
            fused = torch.cat([text_emb, audio_emb], dim=-1)
        elif self.config.fusion_method == "add":
            fused = text_emb + audio_emb
        elif self.config.fusion_method == "cross_attention":
            # Use text as query, audio as key/value
            fused, _ = self.fusion_layer(
                text_emb.unsqueeze(1),  # Add sequence dimension
                audio_emb.unsqueeze(1),
                audio_emb.unsqueeze(1),
            )
            fused = fused.squeeze(1)  # Remove sequence dimension
        else:
            raise ValueError(f"Unknown fusion method: {self.config.fusion_method}")
            
        return fused
    
    def forward(
        self,
        text: Optional[Union[str, list[str]]] = None,
        audio_features: Optional[torch.Tensor] = None,
        text_embeddings: Optional[torch.Tensor] = None,
        audio_embeddings: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass of multimodal encoder.
        
        Args:
            text: Text descriptions (if not pre-computed)
            audio_features: Audio feature vectors (if not pre-computed)
            text_embeddings: Pre-computed text embeddings
            audio_embeddings: Pre-computed audio embeddings
            
        Returns:
            Condition embeddings [batch_size, fusion_dim]
        """
        # Encode text if not provided
        if text_embeddings is None:
            if text is None:
                raise ValueError("Either text or text_embeddings must be provided")
            text_embeddings = self.encode_text(text, device=next(self.parameters()).device)
        
        # Encode audio if not provided
        if audio_embeddings is None:
            if audio_features is None:
                raise ValueError("Either audio_features or audio_embeddings must be provided")
            audio_embeddings = self.encode_audio(audio_features)
        
        # Fuse modalities
        fused = self.fuse_modalities(text_embeddings, audio_embeddings)
        
        # Output projection
        condition_embedding = self.output_proj(fused)
        
        return condition_embedding
    
    def unfreeze_text_encoder(self):
        """Unfreeze text encoder for fine-tuning."""
        for param in self.text_encoder.parameters():
            param.requires_grad = True
    
    def freeze_text_encoder(self):
        """Freeze text encoder."""
        for param in self.text_encoder.parameters():
            param.requires_grad = False
