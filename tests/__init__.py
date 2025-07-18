"""Tests for T2A-LoRA models."""

import pytest
import torch
from t2a_lora.models import (
    HyperNetwork, HyperNetworkConfig,
    MultimodalEncoder, MultimodalEncoderConfig,
    T2ALoRAGenerator, T2ALoRAConfig
)


class TestHyperNetwork:
    """Test HyperNetwork implementation."""

    def test_hypernetwork_creation(self):
        """Test HyperNetwork creation."""
        config = HyperNetworkConfig(
            condition_dim=768,
            hidden_dim=256,
            num_layers=2,
        )
        target_config = {"lora_rank": 16}

        hypernetwork = HyperNetwork(config, target_config)
        assert hypernetwork is not None
        assert hypernetwork.config == config

    def test_hypernetwork_forward(self):
        """Test HyperNetwork forward pass."""
        config = HyperNetworkConfig(
            condition_dim=768,
            hidden_dim=256,
            num_layers=2,
        )
        target_config = {"lora_rank": 16}

        hypernetwork = HyperNetwork(config, target_config)

        # Test forward pass
        batch_size = 4
        condition_embedding = torch.randn(batch_size, config.condition_dim)

        lora_weights = hypernetwork(condition_embedding)

        assert isinstance(lora_weights, dict)
        assert len(lora_weights) > 0

        # Check weight shapes
        for name, weight in lora_weights.items():
            assert weight.shape[0] == batch_size
            if name.endswith("_A"):
                assert len(weight.shape) == 3  # (batch, rank, in_dim)
            elif name.endswith("_B"):
                assert len(weight.shape) == 3  # (batch, out_dim, rank)


class TestMultimodalEncoder:
    """Test MultimodalEncoder implementation."""

    def test_multimodal_encoder_creation(self):
        """Test MultimodalEncoder creation."""
        config = MultimodalEncoderConfig(
            text_encoder_name="klue/roberta-base",
            fusion_method="concat",
        )

        encoder = MultimodalEncoder(config)
        assert encoder is not None
        assert encoder.config == config

    def test_text_encoding(self):
        """Test text encoding."""
        config = MultimodalEncoderConfig(
            text_encoder_name="klue/roberta-base",
            fusion_method="concat",
        )

        encoder = MultimodalEncoder(config)

        # Test single text
        text = "젊은 여성의 따뜻한 목소리"
        text_emb = encoder.encode_text(text)

        assert text_emb.shape == (1, config.fusion_dim)

        # Test batch text
        batch_text = ["따뜻한 목소리", "차분한 목소리", "활기찬 목소리"]
        batch_text_emb = encoder.encode_text(batch_text)

        assert batch_text_emb.shape == (3, config.fusion_dim)

    def test_audio_encoding(self):
        """Test audio encoding."""
        config = MultimodalEncoderConfig(
            audio_encoder_dim=512,
            fusion_dim=768,
        )

        encoder = MultimodalEncoder(config)

        # Test audio features
        batch_size = 2
        audio_features = torch.randn(batch_size, config.audio_encoder_dim)
        audio_emb = encoder.encode_audio(audio_features)

        assert audio_emb.shape == (batch_size, config.fusion_dim)

    def test_fusion_methods(self):
        """Test different fusion methods."""
        fusion_methods = ["concat", "add", "cross_attention"]

        for method in fusion_methods:
            config = MultimodalEncoderConfig(
                text_encoder_name="klue/roberta-base",
                fusion_method=method,
                fusion_dim=512,
            )

            encoder = MultimodalEncoder(config)

            batch_size = 2
            text_emb = torch.randn(batch_size, config.fusion_dim)
            audio_emb = torch.randn(batch_size, config.fusion_dim)

            fused = encoder.fuse_modalities(text_emb, audio_emb)

            expected_dim = config.fusion_dim * 2 if method == "concat" else config.fusion_dim
            assert fused.shape == (batch_size, expected_dim)

    def test_multimodal_forward(self):
        """Test complete multimodal forward pass."""
        config = MultimodalEncoderConfig(
            text_encoder_name="klue/roberta-base",
            fusion_method="add",
            fusion_dim=512,
        )

        encoder = MultimodalEncoder(config)

        # Test with text and audio
        text = ["따뜻한 목소리", "차분한 목소리"]
        audio_features = torch.randn(2, config.audio_encoder_dim)

        condition_emb = encoder(text=text, audio_features=audio_features)

        assert condition_emb.shape == (2, config.fusion_dim)


class TestT2ALoRAGenerator:
    """Test T2ALoRAGenerator implementation."""

    def test_generator_creation(self):
        """Test T2ALoRAGenerator creation."""
        config = T2ALoRAConfig(
            multimodal_config=MultimodalEncoderConfig(
                text_encoder_name="klue/roberta-base",
                fusion_dim=512,
            ),
            hypernetwork_config=HyperNetworkConfig(
                condition_dim=512,
                hidden_dim=256,
            ),
            target_model_config={"lora_rank": 16},
        )

        generator = T2ALoRAGenerator(config)
        assert generator is not None
        assert generator.config == config

    def test_text_generation(self):
        """Test LoRA generation from text."""
        config = T2ALoRAConfig(
            multimodal_config=MultimodalEncoderConfig(
                text_encoder_name="klue/roberta-base",
                fusion_dim=512,
            ),
            hypernetwork_config=HyperNetworkConfig(
                condition_dim=512,
                hidden_dim=256,
            ),
            target_model_config={"lora_rank": 16},
        )

        generator = T2ALoRAGenerator(config)
        generator.eval()

        # Test single text
        with torch.no_grad():
            lora_weights = generator(text="따뜻한 목소리")

        assert isinstance(lora_weights, dict)
        assert len(lora_weights) > 0

        # Test batch text
        with torch.no_grad():
            batch_lora_weights = generator(text=["따뜻한 목소리", "차분한 목소리"])

        assert isinstance(batch_lora_weights, dict)
        for name, weight in batch_lora_weights.items():
            assert weight.shape[0] == 2  # batch size

    def test_audio_generation(self):
        """Test LoRA generation from audio features."""
        config = T2ALoRAConfig(
            multimodal_config=MultimodalEncoderConfig(
                text_encoder_name="klue/roberta-base",
                fusion_dim=512,
            ),
            hypernetwork_config=HyperNetworkConfig(
                condition_dim=512,
                hidden_dim=256,
            ),
            target_model_config={"lora_rank": 16},
        )

        generator = T2ALoRAGenerator(config)
        generator.eval()

        # Test audio features
        audio_features = torch.randn(1, 512)

        with torch.no_grad():
            lora_weights = generator(
                text="비슷한 목소리",
                audio_features=audio_features
            )

        assert isinstance(lora_weights, dict)
        assert len(lora_weights) > 0

    def test_lora_adapter_generation(self):
        """Test complete LoRA adapter generation."""
        config = T2ALoRAConfig(
            multimodal_config=MultimodalEncoderConfig(
                text_encoder_name="klue/roberta-base",
                fusion_dim=512,
            ),
            hypernetwork_config=HyperNetworkConfig(
                condition_dim=512,
                hidden_dim=256,
            ),
            target_model_config={"lora_rank": 16},
        )

        generator = T2ALoRAGenerator(config)
        generator.eval()

        with torch.no_grad():
            adapters = generator.generate_lora_adapters(
                text="전문적인 아나운서 목소리"
            )

        assert "lora_weights" in adapters
        assert "lora_config" in adapters
        assert "condition_text" in adapters

        # Check LoRA config
        lora_config = adapters["lora_config"]
        assert "r" in lora_config
        assert "lora_alpha" in lora_config
        assert "target_modules" in lora_config

    def test_caching(self):
        """Test caching functionality."""
        config = T2ALoRAConfig(
            multimodal_config=MultimodalEncoderConfig(
                text_encoder_name="klue/roberta-base",
                fusion_dim=512,
            ),
            hypernetwork_config=HyperNetworkConfig(
                condition_dim=512,
                hidden_dim=256,
            ),
            target_model_config={"lora_rank": 16},
            use_cache=True,
            cache_size=10,
        )

        generator = T2ALoRAGenerator(config)
        generator.eval()

        text = "테스트 목소리"

        # First generation
        with torch.no_grad():
            result1 = generator(text=text)

        # Second generation (should use cache)
        with torch.no_grad():
            result2 = generator(text=text)

        # Results should be identical (from cache)
        for key in result1.keys():
            assert torch.equal(result1[key], result2[key])

        # Check cache stats
        cache_stats = generator.get_cache_stats()
        assert cache_stats["cache_size"] == 1
        assert cache_stats["cache_enabled"] is True

    def test_save_load(self, tmp_path):
        """Test model saving and loading."""
        config = T2ALoRAConfig(
            multimodal_config=MultimodalEncoderConfig(
                text_encoder_name="klue/roberta-base",
                fusion_dim=512,
            ),
            hypernetwork_config=HyperNetworkConfig(
                condition_dim=512,
                hidden_dim=256,
            ),
            target_model_config={"lora_rank": 16},
        )

        # Create and save model
        generator = T2ALoRAGenerator(config)
        save_path = tmp_path / "test_model"
        generator.save_pretrained(str(save_path))

        # Load model
        loaded_generator = T2ALoRAGenerator.from_pretrained(str(save_path))

        # Test that loaded model works
        with torch.no_grad():
            original_result = generator(text="테스트")
            loaded_result = loaded_generator(text="테스트")

        # Results should be identical
        for key in original_result.keys():
            assert torch.equal(original_result[key], loaded_result[key])


@pytest.fixture
def sample_config():
    """Sample configuration for testing."""
    return T2ALoRAConfig(
        multimodal_config=MultimodalEncoderConfig(
            text_encoder_name="klue/roberta-base",
            fusion_dim=512,
        ),
        hypernetwork_config=HyperNetworkConfig(
            condition_dim=512,
            hidden_dim=256,
        ),
        target_model_config={"lora_rank": 16},
    )


def test_config_serialization(sample_config):
    """Test configuration serialization."""
    config_dict = sample_config.to_dict()
    assert isinstance(config_dict, dict)

    # Test deserialization
    new_config = T2ALoRAConfig.from_dict(config_dict)
    assert new_config.multimodal_config.fusion_dim == sample_config.multimodal_config.fusion_dim
    assert new_config.hypernetwork_config.hidden_dim == sample_config.hypernetwork_config.hidden_dim
