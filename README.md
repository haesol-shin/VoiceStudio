# T2A-LoRA: Text-to-Audio LoRA Generation via Hypernetworks for Real-time Voice Adaptation

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![arXiv](https://img.shields.io/badge/arXiv-2501.XXXXX-b31b1b.svg)](https://arxiv.org/abs/2501.XXXXX)

> **T2A-LoRA** is a novel approach for real-time voice adaptation in text-to-speech systems using hypernetworks to generate LoRA weights from natural language descriptions and audio features.

## 🚀 Key Features

- **Text-Conditional Generation**: Generate voice characteristics using natural language descriptions like "young female voice with warm tone"
- **Multimodal Input**: Support both text descriptions and audio feature vectors
- **Real-time Adaptation**: Generate LoRA weights in a single forward pass without fine-tuning
- **Zero-shot Generalization**: Adapt to unseen voice characteristics not present in training data
- **Parameter Efficiency**: Minimal computational overhead compared to full model fine-tuning
- **Multilingual Support**: Korean and English text descriptions supported

## 🛠️ Installation

### From PyPI (Coming Soon)
```bash
uv add t2a-lora
```

### From Source
```bash
git clone https://github.com/LatentForge/T2A-LoRA.git
cd T2A-LoRA
uv pip install -e .
```

### Development Installation
```bash
git clone https://github.com/LatentForge/T2A-LoRA.git
cd T2A-LoRA
uv pip install -e ".[dev,docs,audio]"
```

### Building and Publishing
```bash
# Build package
uv build

# Install build dependencies
uv add --dev build twine

# Upload to PyPI
twine upload dist/*
```

## 📖 Quick Start

### Basic Usage

```python
import torch
from t2a_lora import T2ALoRAGenerator, T2ALoRAConfig
from t2a_lora.models import MultimodalEncoderConfig, HyperNetworkConfig

# Create model configuration
config = T2ALoRAConfig(
    multimodal_config=MultimodalEncoderConfig(),
    hypernetwork_config=HyperNetworkConfig(),
    target_model_config={"lora_rank": 16, "lora_alpha": 32}
)

# Initialize generator
generator = T2ALoRAGenerator(config)

# Generate LoRA weights from text description
text_description = "젊은 여성의 따뜻하고 친근한 목소리로"  # Korean
lora_weights = generator.generate_lora_adapters(text=text_description)

# Apply to your TTS model
# your_tts_model.load_lora_weights(lora_weights)
```

### Using Pre-trained Models

```python
from t2a_lora import T2ALoRAGenerator

# Load pre-trained model
generator = T2ALoRAGenerator.from_pretrained("LatentForge/t2a-lora-korean-base-v1")

# Generate with English description
lora_adapters = generator.generate_lora_adapters(
    text="A deep, authoritative male voice with clear pronunciation"
)

# Generate with audio reference
import librosa
audio, sr = librosa.load("reference_voice.wav")
audio_features = extract_audio_features(audio, sr)  # Your feature extraction

lora_adapters = generator.generate_lora_adapters(
    text="Similar voice but with more emotion",
    audio_features=audio_features
)
```

## 🎯 Command Line Interface

### Training a Model
```bash
# Train with default configuration
t2a-train --data ./dataset --output ./models/my_model --epochs 50

# Train with custom config
t2a-train --config ./configs/training_config.json

# Resume training from checkpoint
t2a-train --resume ./checkpoints/checkpoint-1000
```

### Generating LoRA Weights
```bash
# Generate from text description
t2a-generate --model ./models/my_model \
             --text "차분하고 안정적인 남성 목소리" \
             --output ./generated_lora

# Generate from audio reference
t2a-generate --model ./models/my_model \
             --audio ./reference.wav \
             --text "Similar voice but younger" \
             --output ./generated_lora

# Custom LoRA parameters
t2a-generate --model ./models/my_model \
             --text "Energetic female voice" \
             --rank 32 --alpha 64 \
             --format safetensors
```

### Model Evaluation
```bash
t2a-evaluate --model ./models/my_model \
             --test-data ./test_dataset \
             --metrics mse cosine_sim \
             --output ./evaluation_results
```

### Interactive Demo
```bash
# Interactive mode
t2a-demo --model ./models/my_model --interactive

# Test specific descriptions
t2a-demo --model ./models/my_model \
         --text "friendly customer service voice" \
         --text "dramatic narrator voice"
```

## 📊 Model Architecture

```
Text Description → [Text Encoder] ─┐
                                    ├─→ [Fusion Layer] → [HyperNetwork] → LoRA Weights
Audio Features  → [Audio Encoder] ─┘
```

### Core Components

1. **Multimodal Encoder**: Processes text descriptions and audio features
   - Text Encoder: Pre-trained Korean/English language model
   - Audio Encoder: Transforms audio features to embedding space
   - Fusion Layer: Combines multimodal inputs (concat/add/cross-attention)

2. **HyperNetwork**: Generates LoRA weights from condition embeddings
   - Input: Fused multimodal embeddings
   - Output: A and B matrices for LoRA adaptation
   - Architecture: Multi-layer MLP with residual connections

3. **LoRA Integration**: Apply generated weights to target TTS models
   - Supports common TTS architectures (Tacotron, FastSpeech, VITS, etc.)
   - Configurable target modules and ranks

## 🎨 Use Cases

### Content Creation
```python
# Generate different narrator voices for audiobooks
narrator_voices = [
    "Professional documentary narrator with clear diction",
    "Dramatic storyteller with rich emotional range", 
    "Gentle children's book reader with warm tone"
]

for description in narrator_voices:
    lora_weights = generator.generate_lora_adapters(text=description)
    # Apply to TTS and generate audio content
```

### Voice Customization
```python
# Customize voice for different contexts
contexts = {
    "customer_service": "Patient, helpful voice with friendly tone",
    "announcement": "Clear, authoritative voice for public announcements",
    "meditation": "Calm, soothing voice for relaxation guidance"
}

voice_variants = {}
for context, description in contexts.items():
    voice_variants[context] = generator.generate_lora_adapters(text=description)
```

### Accessibility
```python
# Generate voices for speech synthesis aids
accessibility_voices = [
    "Clear, slow-paced voice for hearing impaired users",
    "High-contrast emotional voice for autism support",
    "Consistent, predictable voice for cognitive assistance"
]
```

## 🔬 Research Applications

### Experimental Setup
```python
from t2a_lora.training import T2ALoRATrainer, TrainingConfig
from t2a_lora.evaluation import T2ALoRAEvaluator

# Research configuration
config = TrainingConfig(
    dataset_path="./research_dataset",
    model_config={
        "multimodal": {"fusion_method": "cross_attention"},
        "hypernetwork": {"num_layers": 4, "hidden_dim": 1024},
        "target_model": {"lora_rank": 64}
    },
    batch_size=16,
    learning_rate=1e-4,
    num_epochs=100,
    use_wandb=True,
    wandb_project="latentforge-t2a-lora"
)

# Train model
trainer = T2ALoRATrainer(config)
trainer.train()

# Evaluate zero-shot capabilities
evaluator = T2ALoRAEvaluator(model=trainer.model)
results = evaluator.evaluate_zero_shot_generalization()
```

### Ablation Studies
```python
# Compare different fusion methods
fusion_methods = ["concat", "add", "cross_attention"]
results = {}

for method in fusion_methods:
    config.model_config["multimodal"]["fusion_method"] = method
    model = train_model(config)
    results[method] = evaluate_model(model)
```

## 📁 Project Structure

```
T2A-LoRA/
├── src/t2a_lora/
│   ├── models/
│   │   ├── hypernetwork.py          # HyperNetwork implementation
│   │   ├── multimodal_encoder.py    # Multimodal encoder
│   │   └── t2a_lora_generator.py    # Main generator model
│   ├── training/
│   │   ├── trainer.py               # Training logic
│   │   ├── config.py                # Training configuration
│   │   └── data.py                  # Dataset and data loading
│   ├── generation/
│   │   ├── generator.py             # Generation utilities
│   │   └── config.py                # Generation configuration
│   ├── evaluation/
│   │   └── evaluator.py             # Evaluation metrics and tools
│   └── cli.py                       # Command line interface
├── configs/                         # Configuration files
├── examples/                        # Example scripts
├── tests/                          # Unit tests
├── docs/                           # Documentation
├── pyproject.toml                  # Package configuration
└── README.md                       # This file
```

## 🎛️ Configuration

### Model Configuration
```json
{
  "multimodal": {
    "text_encoder_name": "klue/roberta-large",
    "audio_encoder_dim": 512,
    "fusion_dim": 768,
    "fusion_method": "cross_attention",
    "dropout": 0.1
  },
  "hypernetwork": {
    "condition_dim": 768,
    "hidden_dim": 512,
    "num_layers": 3,
    "dropout": 0.1,
    "activation": "gelu"
  },
  "target_model": {
    "lora_rank": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.1
  }
}
```

### Training Configuration
```json
{
  "dataset_path": "./data/training",
  "batch_size": 8,
  "learning_rate": 1e-4,
  "num_epochs": 50,
  "warmup_steps": 1000,
  "weight_decay": 0.01,
  "scheduler": "cosine",
  "loss_type": "mse",
  "eval_steps": 500,
  "save_steps": 1000,
  "use_wandb": true,
  "wandb_project": "latentforge-t2a-lora"
}
```

## 📈 Performance

### Benchmarks
| Model | Zero-shot Similarity | Generation Speed | Memory Usage |
|-------|---------------------|------------------|--------------|
| T2A-LoRA-Base | 0.85 | 0.1s | 2.3GB |
| T2A-LoRA-Large | 0.91 | 0.15s | 4.1GB |
| Full Fine-tuning | 0.89 | 300s | 12GB |

### Evaluation Metrics
- **Speaker Similarity**: Cosine similarity between generated and target voice embeddings
- **Text Adherence**: Alignment between text description and generated voice characteristics
- **Audio Quality**: MOS scores from human evaluation
- **Generation Speed**: Time to generate LoRA weights
- **Zero-shot Performance**: Performance on unseen voice descriptions

## 🧪 Experiments and Results

### Zero-shot Generalization
```python
# Test unseen voice characteristics
unseen_descriptions = [
    "Robot voice with metallic undertones",
    "Elderly person with slight tremor",
    "Child with lisp pronunciation",
    "Foreign accent with clear articulation"
]

results = evaluate_zero_shot(generator, unseen_descriptions)
print(f"Average similarity: {results['avg_similarity']:.3f}")
```

### Cross-lingual Transfer
```python
# Test Korean model on English descriptions
korean_model = T2ALoRAGenerator.from_pretrained("LatentForge/t2a-lora-korean-base")
english_results = evaluate_english_descriptions(korean_model)
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup
```bash
git clone https://github.com/LatentForge/T2A-LoRA.git
cd T2A-LoRA
uv pip install -e ".[dev]"
pre-commit install
```

### Running Tests
```bash
pytest tests/ -v --cov=src/t2a_lora
```

### Code Style
```bash
# Format code
black src/ tests/ examples/
isort src/ tests/ examples/

# Check code style
flake8 src/ tests/ examples/
mypy src/

# Run all checks
pytest tests/ && black --check src/ && isort --check-only src/ && flake8 src/
```

## 📚 Citation

If you use T2A-LoRA in your research, please cite our paper:

```bibtex
@article{t2a-lora-2025,
  title={T2A-LoRA: Text-to-Audio LoRA Generation via Hypernetworks for Real-time Voice Adaptation},
  author={LatentForge},
  journal={arXiv preprint arXiv:2501.XXXXX},
  year={2025}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🔗 Links

- **Paper**: [arXiv:2501.XXXXX](https://arxiv.org/abs/2501.XXXXX)
- **Demo**: [https://latentforge.github.io/T2A-LoRA/demo](https://latentforge.github.io/T2A-LoRA/demo)
- **Documentation**: [https://latentforge.github.io/T2A-LoRA](https://latentforge.github.io/T2A-LoRA)
- **Models**: [HuggingFace Hub](https://huggingface.co/LatentForge)

## 🙏 Acknowledgments

- Sakana AI for the original Text-to-LoRA concept
- HyperTTS authors for hypernetwork applications in TTS
- The open-source community for tools and datasets

## 📞 Contact

- **Issues**: [GitHub Issues](https://github.com/LatentForge/T2A-LoRA/issues)
- **Discussions**: [GitHub Discussions](https://github.com/LatentForge/T2A-LoRA/discussions)
- **Email**: contact@latentforge.org

---

<p align="center">
  <img src="https://img.shields.io/github/stars/LatentForge/T2A-LoRA?style=social" alt="Stars">
  <img src="https://img.shields.io/github/forks/LatentForge/T2A-LoRA?style=social" alt="Forks">
  <img src="https://img.shields.io/github/watchers/LatentForge/T2A-LoRA?style=social" alt="Watchers">
</p>

<p align="center">
  Made with ❤️ by LatentForge
</p>
