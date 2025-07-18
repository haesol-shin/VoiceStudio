#!/usr/bin/env python3
"""Basic usage example for T2A-LoRA."""

import torch
import numpy as np
from t2a_lora import T2ALoRAGenerator, T2ALoRAConfig
from t2a_lora.models import MultimodalEncoderConfig, HyperNetworkConfig


def main():
    """Basic usage example."""
    print("T2A-LoRA Basic Usage Example")
    print("=" * 40)
    
    # Create model configuration
    config = T2ALoRAConfig(
        multimodal_config=MultimodalEncoderConfig(
            text_encoder_name="klue/roberta-base",  # Smaller model for demo
            fusion_method="concat",
            fusion_dim=512,
        ),
        hypernetwork_config=HyperNetworkConfig(
            condition_dim=512,
            hidden_dim=256,
            num_layers=2,
        ),
        target_model_config={
            "lora_rank": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.1,
        }
    )
    
    print("Creating T2A-LoRA generator...")
    generator = T2ALoRAGenerator(config)
    generator.eval()
    
    # Example 1: Generate from Korean text description
    print("\n1. Korean Text Description:")
    korean_text = "젊은 여성의 따뜻하고 친근한 목소리로"
    print(f"Input: {korean_text}")
    
    with torch.no_grad():
        lora_adapters = generator.generate_lora_adapters(text=korean_text)
    
    print("Generated LoRA configuration:")
    for key, value in lora_adapters["lora_config"].items():
        print(f"  {key}: {value}")
    
    print("Generated weight shapes:")
    for name, weight in lora_adapters["lora_weights"].items():
        print(f"  {name}: {weight.shape}")
    
    # Example 2: Generate from English text description
    print("\n2. English Text Description:")
    english_text = "A deep, authoritative male voice with clear pronunciation"
    print(f"Input: {english_text}")
    
    with torch.no_grad():
        lora_adapters = generator.generate_lora_adapters(text=english_text)
    
    print("Generated LoRA configuration:")
    for key, value in lora_adapters["lora_config"].items():
        print(f"  {key}: {value}")
    
    # Example 3: Generate from dummy audio features
    print("\n3. Audio Features + Text Description:")
    # Create dummy audio features (in practice, extract from real audio)
    dummy_audio_features = torch.randn(1, 512)  # Batch size 1, feature dim 512
    text_with_audio = "Similar voice but with more emotion"
    print(f"Audio features shape: {dummy_audio_features.shape}")
    print(f"Text: {text_with_audio}")
    
    with torch.no_grad():
        lora_adapters = generator.generate_lora_adapters(
            text=text_with_audio,
            audio_features=dummy_audio_features
        )
    
    print("Generated LoRA configuration:")
    for key, value in lora_adapters["lora_config"].items():
        print(f"  {key}: {value}")
    
    # Example 4: Batch generation
    print("\n4. Batch Generation:")
    batch_texts = [
        "Professional documentary narrator",
        "Friendly customer service representative", 
        "Dramatic storyteller with rich emotions"
    ]
    
    print(f"Batch texts: {batch_texts}")
    
    with torch.no_grad():
        batch_lora_adapters = generator.generate_lora_adapters(text=batch_texts)
    
    print("Batch generation results:")
    for name, weights in batch_lora_adapters["lora_weights"].items():
        print(f"  {name}: {weights.shape} (batch_size={weights.shape[0]})")
    
    # Example 5: Cache demonstration
    print("\n5. Cache Performance:")
    import time
    
    # First generation (cache miss)
    start_time = time.time()
    with torch.no_grad():
        generator.generate_lora_adapters(text=korean_text)
    first_time = time.time() - start_time
    
    # Second generation (cache hit)
    start_time = time.time()
    with torch.no_grad():
        generator.generate_lora_adapters(text=korean_text)
    second_time = time.time() - start_time
    
    print(f"First generation: {first_time:.4f}s")
    print(f"Second generation (cached): {second_time:.4f}s")
    print(f"Speedup: {first_time/second_time:.2f}x")
    
    # Cache statistics
    cache_stats = generator.get_cache_stats()
    print(f"Cache stats: {cache_stats}")
    
    print("\n" + "=" * 40)
    print("Example completed successfully!")


if __name__ == "__main__":
    main()
