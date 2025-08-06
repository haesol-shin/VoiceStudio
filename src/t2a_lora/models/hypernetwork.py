"""Main T2A-LoRA generator model."""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass

from .hypernetwork import HyperNetwork, HyperNetworkConfig
from .multimodal_encoder import MultimodalEncoder, MultimodalEncoderConfig


@dataclass
class T2ALoRAConfig:
    """Configuration for T2A-LoRA generator."""
    multimodal_config: MultimodalEncoderConfig
    hypernetwork_config: HyperNetworkConfig
    target_model_config: Dict[str, Any]
    
    # Additional model settings
    use_cache: bool = True
    cache_size: int = 1000
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "T2ALoRAConfig":
        """Create config from dictionary."""
        return cls(
            multimodal_config=MultimodalEncoderConfig(**config_dict.get("multimodal", {})),
            hypernetwork_config=HyperNetworkConfig(**config_dict.get("hypernetwork", {})),
            target_model_config=config_dict.get("target_model", {}),
            use_cache=config_dict.get("use_cache", True),
            cache_size=config_dict.get("cache_size", 1000),
        )


class T2ALoRAGenerator(nn.Module):
    """
    T2A-LoRA Generator: Complete model for generating LoRA weights from text/audio conditions.
    
    This model combines the multimodal encoder and hypernetwork to generate
    LoRA weights that can be applied to target TTS models for voice adaptation.
    """
    
    def __init__(self, config: T2ALoRAConfig):
        super().__init__()
        self.config = config
        
        # Build multimodal encoder
        self.multimodal_encoder = MultimodalEncoder(config.multimodal_config)
        
        # Update hypernetwork config with encoder output dimension
        hypernetwork_config = config.hypernetwork_config
        hypernetwork_config.condition_dim = config.multimodal_config.fusion_dim
        
        # Build hypernetwork
        self.hypernetwork = HyperNetwork(
            hypernetwork_config,
            config.target_model_config
        )
        
        # Cache for storing generated LoRA weights
        if config.use_cache:
            self.cache = {}
            self.cache_keys = []
        
    def forward(
        self,
        text: Optional[Union[str, List[str]]] = None,
        audio_features: Optional[torch.Tensor] = None,
        text_embeddings: Optional[torch.Tensor] = None,
        audio_embeddings: Optional[torch.Tensor] = None,
        use_cache: bool = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Generate LoRA weights from text and/or audio conditions.
        
        Args:
            text: Text description(s) of desired voice characteristics
            audio_features: Audio feature vectors
            text_embeddings: Pre-computed text embeddings
            audio_embeddings: Pre-computed audio embeddings
            use_cache: Whether to use caching (overrides config setting)
            
        Returns:
            Dictionary of LoRA weights for target model layers
        """
        # Check cache if enabled
        if use_cache is None:
            use_cache = self.config.use_cache
            
        if use_cache:
            cache_key = self._get_cache_key(text, audio_features, text_embeddings, audio_embeddings)
            if cache_key in self.cache:
                return self.cache[cache_key]
        
        # Encode conditions
        condition_embedding = self.multimodal_encoder(
            text=text,
            audio_features=audio_features,
            text_embeddings=text_embeddings,
            audio_embeddings=audio_embeddings,
        )
        
        # Generate LoRA weights
        lora_weights = self.hypernetwork(condition_embedding)
        
        # Store in cache
        if use_cache:
            self._update_cache(cache_key, lora_weights)
        
        return lora_weights
    
    def generate_lora_adapters(
        self,
        text: Optional[Union[str, List[str]]] = None,
        audio_features: Optional[torch.Tensor] = None,
        lora_rank: Optional[int] = None,
        lora_alpha: Optional[float] = None,
        target_modules: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Generate LoRA adapters ready for integration with target models.
        
        Args:
            text: Text description(s) of desired voice characteristics
            audio_features: Audio feature vectors
            lora_rank: LoRA rank (overrides config)
            lora_alpha: LoRA alpha (overrides config)
            target_modules: Target modules (overrides config)
            
        Returns:
            Dictionary containing LoRA adapters and configuration
        """
        # Generate LoRA weights
        lora_weights = self.forward(text=text, audio_features=audio_features)
        
        # Prepare LoRA configuration
        lora_config = self.hypernetwork.get_lora_config()
        if lora_rank is not None:
            lora_config["r"] = lora_rank
        if lora_alpha is not None:
            lora_config["lora_alpha"] = lora_alpha
        if target_modules is not None:
            lora_config["target_modules"] = target_modules
        
        return {
            "lora_weights": lora_weights,
            "lora_config": lora_config,
            "condition_text": text,
            "condition_audio_shape": audio_features.shape if audio_features is not None else None,
        }
    
    def _get_cache_key(
        self,
        text: Optional[Union[str, List[str]]],
        audio_features: Optional[torch.Tensor],
        text_embeddings: Optional[torch.Tensor],
        audio_embeddings: Optional[torch.Tensor],
    ) -> str:
        """Generate cache key for given inputs."""
        key_parts = []
        
        if text is not None:
            if isinstance(text, str):
                key_parts.append(f"text:{hash(text)}")
            else:
                key_parts.append(f"text:{hash(tuple(text))}")
        
        if audio_features is not None:
            key_parts.append(f"audio:{hash(audio_features.data.tobytes())}")
            
        if text_embeddings is not None:
            key_parts.append(f"text_emb:{hash(text_embeddings.data.tobytes())}")
            
        if audio_embeddings is not None:
            key_parts.append(f"audio_emb:{hash(audio_embeddings.data.tobytes())}")
        
        return "|".join(key_parts)
    
    def _update_cache(self, key: str, value: Dict[str, torch.Tensor]):
        """Update cache with new key-value pair."""
        if len(self.cache_keys) >= self.config.cache_size:
            # Remove oldest entry
            oldest_key = self.cache_keys.pop(0)
            del self.cache[oldest_key]
        
        self.cache[key] = value
        self.cache_keys.append(key)
    
    def clear_cache(self):
        """Clear the cache."""
        self.cache.clear()
        self.cache_keys.clear()
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "cache_size": len(self.cache),
            "max_cache_size": self.config.cache_size,
            "cache_enabled": self.config.use_cache,
        }
    
    def save_pretrained(self, save_directory: str):
        """Save model and configuration."""
        import os
        import json
        
        os.makedirs(save_directory, exist_ok=True)
        
        # Save model state
        torch.save(self.state_dict(), os.path.join(save_directory, "pytorch_model.bin"))
        
        # Save configuration
        config_dict = {
            "multimodal": self.config.multimodal_config.__dict__,
            "hypernetwork": self.config.hypernetwork_config.__dict__,
            "target_model": self.config.target_model_config,
            "use_cache": self.config.use_cache,
            "cache_size": self.config.cache_size,
        }
        
        with open(os.path.join(save_directory, "config.json"), "w") as f:
            json.dump(config_dict, f, indent=2)
    
    @classmethod
    def from_pretrained(cls, model_directory: str) -> "T2ALoRAGenerator":
        """Load model from directory."""
        import os
        import json
        
        # Load configuration
        config_path = os.path.join(model_directory, "config.json")
        with open(config_path, "r") as f:
            config_dict = json.load(f)
        
        config = T2ALoRAConfig.from_dict(config_dict)
        
        # Create model
        model = cls(config)
        
        # Load state dict
        state_dict_path = os.path.join(model_directory, "pytorch_model.bin")
        state_dict = torch.load(state_dict_path, map_location="cpu")
        model.load_state_dict(state_dict)
        
        return model
