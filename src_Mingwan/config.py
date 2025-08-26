import os
from dataclasses import dataclass
from typing import Optional, List, Union
# 민관: 파이썬 최신 버전에서는, list 나 dict 같은 걸로, typing import 없이 해결할 수도 있다는 것 같음.


@dataclass #민관: 자동으로 __init__, __repr__ 등의 메서드를 생성해줌.
class TrainingConfig:
    """Training configuration"""
    # Model configurations
    tts_model_name: str = "nari-labs/Dia-1.6B-0626"
    clap_model_name: str = "laion/larger_clap_general"
    qwen_model_name: str = "Qwen/Qwen3-Embedding-0.6B"
    
    # Dataset configuration
    dataset_name: str = "tictap11/libritts_p_dataset_20250821_095157"
    
    # LoRA configuration
    lora_r: int = 16
    lora_alpha: int = 32 #민관: W_new = W_original + (alpha/r) * A * B 이렇게 사용됨.
    lora_dropout: float = 0.1
    # Target selection mode for LoRA modules:
    # - "auto": auto-discover from the loaded Dia model
    # - "manual": use user-provided target_modules as-is (validated at runtime)
    # - "smart" (default): intersect manual list with auto-discovered; if empty, use auto
    lora_target_mode: str = "smart"
    # Manual target list (used when lora_target_mode == "manual" or combined in "smart")
    target_modules: Optional[List[str]] = None
    
    # Hypernetwork configuration
    hypernetwork_hidden_dim: int = 512
    hypernetwork_num_layers: int = 3
    
    # Training parameters
    batch_size: int = 4
    learning_rate: float = 1e-4
    num_epochs: int = 10
    warmup_steps: int = 100
    save_steps: int = 500
    eval_steps: int = 100
    logging_steps: int = 50
    
    # Audio processing
    sample_rate: int = 48000 #민관: CLAP 이었나, DIA 였나, 에서 48000 을 사용함. 원본 샘플은 24 kHz 이기 때문에, 나중에 cast_colum 같은걸로 맞춰줄거임.
    n_mels: int = 80
    hop_length: int = 256
    win_length: int = 1024
    
    # Paths
    output_dir: str = "./outputs"
    cache_dir: str = "./cache"
    
    # Weights & Biases
    # 민관: 학습 진행 과정 추척 도구. 우선은 사용 안 하려고, 주석 처리 하겠음.
    use_wandb: bool = False
    wandb_project: str = "latentforge-tts"
    
    # Device
    device: str = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "auto"
    
    # 민관: dataclass 객체가 생성된 후 자동으로 실행되는 메서드
    def __post_init__(self):
        # Only auto-populate a reasonable manual default list if user chose manual mode and didn't provide one
        if self.lora_target_mode == "manual" and self.target_modules is None:
            from transformers import AutoConfig
            cfg = AutoConfig.from_pretrained(self.tts_model_name)
            num_decoder_layers = getattr(cfg.decoder_config, 'num_hidden_layers', 18)
            decoder_self_attn_layers = [
                f"decoder.layers.{i}.self_attention.{proj}"
                for i in range(num_decoder_layers)
                for proj in ["q_proj", "k_proj", "v_proj", "o_proj"]
            ]
            decoder_cross_attn_layers = [
                f"decoder.layers.{i}.cross_attention.{proj}"
                for i in range(num_decoder_layers)
                for proj in ["q_proj", "k_proj", "v_proj", "o_proj"]
            ]
            # Focus LoRA on attention projections by default
            self.target_modules = decoder_self_attn_layers + decoder_cross_attn_layers


@dataclass
class AudioConfig:
    """Audio processing configuration"""
    sample_rate: int = 48000
    n_mels: int = 80
    hop_length: int = 256
    win_length: int = 1024
    n_fft: int = 1024
    f_min: float = 0.0
    f_max: Optional[float] = None
    
    def __post_init__(self):
        if self.f_max is None:
            self.f_max = self.sample_rate // 2
