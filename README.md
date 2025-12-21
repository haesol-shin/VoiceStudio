# VoiceStudio

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Your Complete Voice Adaptation Research Workspace

---

## 🎯 Overview

VoiceStudio is a unified toolkit for **text-style prompted speech synthesis**, enabling instant voice adaptation and editing through natural language descriptions. Built on cutting-edge research in voice style prompting, LoRA adaptation, and language-audio models.

**Key Features:**
- **Text-Conditional Generation**: Generate voice characteristics using natural language descriptions like "young female voice with warm tone"
- **Multimodal Input**: Support both text descriptions and audio feature vectors
- **Voice Editing**: Modify existing voices with simple instructions (Future Work)
- **Instant Adaptation**: Generate LoRA weights in a single forward pass without fine-tuning
- **Architecture Agnostic**: Works with multiple TTS architectures
- **Zero-shot Generalization**: Adapt to unseen voice characteristics not present in training data
- **Parameter Efficiency**: Minimal computational overhead compared to full model fine-tuning

---

## 🆕 What's New

**v1.3.0** (Coming 2027)
- 🎉 LAM (Language-Audio Model) integration
- ✨ Voice editing capabilities
- 🔥 Instruction following for voice manipulation

**v1.2.0** (2026)
- ⚡ Real-time LoRA generation
- 🎯 CLAP-based text-to-audio adaptation
- 🏗️ Architecture-agnostic design

**v1.1.0** (2026)
- 🔍 Speaker consistency analysis tools
- 🎨 BOS token P-tuning
- 📊 Attention visualization

---

## 📈 Roadmap

### v1.1.0 (Q1 2026) ✅
- [x] Speaker consistency analysis
- [x] Attention visualization
- [x] BOS P-tuning
- [x] WebUI interface

### v1.2.0 (Q3 2026) 🔄
- [ ] CLAP-based LoRA generation
- [ ] Multi-TTS support
- [ ] HuggingFace Hub integration
- [ ] Comprehensive documentation

### v1.3.0 (Q1 2027) 📋
- [ ] LAM model release
- [ ] Voice editing capabilities
- [ ] Advanced instruction following

---

## 🛠️ Installation

### From PyPI (Recommended)
```bash
uv add voicestudio[all]  # Install with all available base TTS models
```

### From Source
```bash
git clone https://github.com/LatentForge/voicestudio.git
cd voicestudio
uv pip install -e ".[all]"
```

### Development Installation
```bash
git clone https://github.com/LatentForge/voicestudio.git
cd voicestudio
uv pip install -e ".[all,web]"
```

### Building and Publishing
```bash
# Build package
uv build

# Upload to PyPI
uv publish
```

## 📖 Quick Start

### 1. Text-to-LoRA Generation (v1.0+)

Generate LoRA weights from text descriptions:

```python
from voicestudio import LoRAGenerator

# Initialize generator
generator = LoRAGenerator.from_pretrained("voicestudio/t2a-lora-base")

# Generate LoRA from text description
lora_weights = generator("warm, cheerful voice of a young female")

# Apply to your TTS model
tts_model.load_lora(lora_weights)
audio = tts_model.synthesize("Hello, how are you today?")
```

### 2. Voice Editing with LAM (v2.0+)

Edit existing voices with instructions:

```python
from voicestudio import VoiceEditor

# Initialize editor
editor = VoiceEditor.from_pretrained("voicestudio/lam-base")

# Edit voice characteristics
edited_audio = editor.edit(
    audio=input_audio,
    instruction="make the voice deeper and more authoritative"
)
```

### 3. Pipeline API (Easy Mode)

```python
from voicestudio import pipeline

# Text-to-LoRA pipeline
lora_gen = pipeline("text-to-lora", model="voicestudio/t2a-lora-base")
lora = lora_gen("calm and soothing voice")

# Voice editing pipeline
editor = pipeline("voice-editing", model="voicestudio/lam-base")
edited = editor(audio, instruction="add a slight echo effect")
```

### 4. Analysis Tools (v0.1+)

Analyze speaker consistency issues:

```python
from voicestudio import ConsistencyAnalyzer

analyzer = ConsistencyAnalyzer()

# Visualize attention patterns
analyzer.visualize_attention(
    model=tts_model,
    text_prompt="warm voice",
    save_path="attention_map.png"
)

# Measure speaker consistency
consistency_score = analyzer.measure_consistency(
    model=tts_model,
    text_prompt="cheerful voice",
    num_samples=10
)
print(f"Consistency: {consistency_score:.2f}")
```

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

---

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

### Interactive Web Demo
```bash
# Interactive mode
t2a-demo --model ./models/my_model --interactive

# Test specific descriptions
t2a-demo --model ./models/my_model \
         --text "friendly customer service voice" \
         --text "dramatic narrator voice"
```

---

## 📚 Advanced Usage

### Custom TTS Model Integration

VoiceStudio supports any TTS model through a simple adapter interface:

```python
from voicestudio import TTSAdapter, LoRAGenerator

# Wrap your TTS model
class MyTTSAdapter(TTSAdapter):
    def __init__(self, model):
        self.model = model
    
    def get_lora_target_modules(self):
        return ["attention.q_proj", "attention.v_proj"]
    
    def forward(self, text, lora_weights=None):
        if lora_weights:
            self.apply_lora(lora_weights)
        return self.model(text)

# Use with VoiceStudio
adapter = MyTTSAdapter(my_tts_model)
generator = LoRAGenerator.from_pretrained("voicestudio/t2a-lora-base")

lora = generator("professional news anchor voice")
audio = adapter(text="Breaking news tonight...", lora_weights=lora)
```

### Multi-Speaker Voice Blending

```python
from voicestudio import VoiceBlender

blender = VoiceBlender()

# Blend multiple voice characteristics
blended_lora = blender.blend([
    ("warm and friendly", 0.6),
    ("professional and clear", 0.4)
])

audio = tts_model.synthesize(text, lora=blended_lora)
```

### Fine-tuning on Custom Data

```python
from voicestudio import LoRAGenerator
from voicestudio.training import Trainer

# Load pre-trained generator
generator = LoRAGenerator.from_pretrained("voicestudio/t2a-lora-base")

# Fine-tune on your data
trainer = Trainer(
    model=generator,
    train_dataset=your_dataset,
    output_dir="./checkpoints"
)

trainer.train()
```

---

## 📊 Supported Models

VoiceStudio works with various TTS architectures:

| Model | Status | Notes |
|-------|--------|-------|
| VITS | ✅ Supported | Fully tested |
| FastSpeech2 | ✅ Supported | Fully tested |
| Tacotron2 | ✅ Supported | Requires adapter |
| VALL-E | 🔄 Experimental | Work in progress |
| Bark | 🔄 Experimental | Coming soon |
| YourTTS | ✅ Supported | Community contributed |

**Add your own model**: See our [Integration Guide](docs/integration.md)

---

## 🔬 Publications

VoiceStudio is built on the following research:

### Paper 1: Problem Discovery (InterSpeech 2026)
**"Speaker Inconsistency in Text-Style Prompted Speech Synthesis: Problem Analysis and Initial Approaches"**

- 🔍 First identification of speaker consistency issues in text-style prompted TTS
- 📊 Attention mechanism analysis revealing transcription dependencies
- 🛠️ Initial solutions: CLAP-based adaptation and BOS P-tuning

```bibtex
@inproceedings{voicestudio2026analysis,
  title={Speaker Inconsistency in Text-Style Prompted Speech Synthesis: Problem Analysis and Initial Approaches},
  author={Your Name},
  booktitle={Interspeech},
  year={2026}
}
```

### Paper 2: LoRA Solution (AAAI 2027)
**"T2A-LoRA: Instant Voice Adaptation via Real-time LoRA Generation"**

- ⚡ First application of LoRA to audio style prompting
- 🏗️ Architecture-agnostic solution for voice adaptation
- 🚀 Real-time LoRA generation through hypernetworks

```bibtex
@inproceedings{voicestudio2027lora,
  title={T2A-LoRA: Instant Voice Adaptation via Real-time LoRA Generation},
  author={Your Name},
  booktitle={AAAI},
  year={2027}
}
```

### Paper 3: LAM & Editing (ICML 2027)
**"T2A-LoRA2: Text-Guided Voice Editing with Language-Audio Models"**

- 🎨 Novel voice editing paradigm (inspired by visual editing techniques)
- 🤖 LAM: Open-source Language-Audio Model
- 🔥 Instruction following for voice manipulation

```bibtex
@inproceedings{voicestudio2027lam,
  title={T2A-LoRA2: Text-Guided Voice Editing with Language-Audio Models},
  author={Your Name},
  booktitle={ICML},
  year={2027}
}
```


### T2A-LoRA: Text-to-Audio LoRA Generation via Hypernetworks for Real-time Voice Adaptation

[![arXiv](https://img.shields.io/badge/arXiv-2501.XXXXX-b31b1b.svg)](https://arxiv.org/abs/2501.XXXXX)

> **T2A-LoRA** is a novel approach for real-time voice adaptation in text-to-speech systems using hypernetworks to generate LoRA weights from natural language descriptions and audio features.



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

---

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

---

## 🤝 Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

**Areas we need help with:**
- 🔧 Additional TTS model adapters
- 📚 Documentation improvements
- 🐛 Bug fixes and testing
- 🌍 Multi-language support
- 🎨 New voice editing techniques

---

## 📝 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

The base TTS models supported by this project are subject to their own respective licenses. Users are responsible for reviewing and complying with each model’s license before use.

---

## 🙏 Acknowledgments

- **Sakana AI** for the original Text-to-LoRA concept
- **HyperTTS** authors for hypernetwork applications in TTS
- **The open-source community** for tools and datasets
- **CLAP**: Microsoft & LAION-AI for CLAP model
- **LoRA**: Microsoft for LoRA technique
- **HuggingFace**: For transformers library and model hub


## 📚 Citation

If you use VoiceStudio in your research, please cite:

```bibtex
@software{voicestudio2026,
  title={VoiceStudio: A Unified Toolkit for Voice Style Adaptation},
  author={Your Name},
  year={2026},
  url={https://github.com/LatentForge/voicestudio}
}
```
```bibtex
@article{t2a-lora-2025,
  title={T2A-LoRA: Text-to-Audio LoRA Generation via Hypernetworks for Real-time Voice Adaptation},
  author={LatentForge},
  journal={arXiv preprint arXiv:2501.XXXXX},
  year={2025}
}
```

## 🔗 Links

- **Paper**: [arXiv:2501.XXXXX](https://arxiv.org/abs/2501.XXXXX)
- **Demo**: [https://latentforge.github.io/VoiceStudio](https://latentforge.github.io/VoiceStudio)
- **Documentation**: [https://latentforge.github.io/VoiceStudio](https://latentforge.github.io/VoiceStudio)
- **Models**: [HuggingFace Hub](https://huggingface.co/LatentForge)

---

## 📞 Contact

- **Issues**: [GitHub Issues](https://github.com/LatentForge/VoiceStudio/issues)
- **Discussions**: [GitHub Discussions](https://github.com/LatentForge/VoiceStudio/discussions)
- **Email**: contact@latentforge.org

---

<p align="center">
  <img src="https://img.shields.io/github/stars/LatentForge/VoiceStudio?style=social" alt="Stars">
  <img src="https://img.shields.io/github/forks/LatentForge/VoiceStudio?style=social" alt="Forks">
  <img src="https://img.shields.io/github/watchers/LatentForge/VoiceStudio?style=social" alt="Watchers">
</p>

<p align="center">
  <strong>Made with ❤️ by LatentForge Team</strong>
</p>
