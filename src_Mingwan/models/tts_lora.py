import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any
from transformers import AutoModel, AutoTokenizer, AutoProcessor
import torch
import torch.nn as nn
from transformers import AutoModelForTextToWaveform, AutoProcessor
from peft import LoraConfig, get_peft_model
from peft.tuners.lora import LoraLayer

from src.models.hypernetwork import Hypernetwork

class TTSLoRA(nn.Module):
    def __init__(self, model_id, lora_r, lora_alpha, lora_dropout):
        super().__init__()
        
        self.lora_r = lora_r
        base_model = AutoModelForTextToWaveform.from_pretrained(model_id)
        self.processor = AutoProcessor.from_pretrained(model_id)
        
        # --- 1. 동적으로 LoRA 타겟 모듈과 필요 가중치 크기 계산 ---
        target_modules_names = []
        self.lora_layers_info = []
        total_lora_weights_size = 0

        for name, module in base_model.named_modules():
            if "decoder" in name and ("self_attn" in name or "cross_attn" in name) and isinstance(module, nn.Linear):
                target_modules_names.append(name)
                in_features = module.in_features
                out_features = module.out_features
                
                # lora_A (r, in_features), lora_B (out_features, r)
                lora_a_size = self.lora_r * in_features
                lora_b_size = out_features * self.lora_r
                
                self.lora_layers_info.append({
                    "name": name,
                    "lora_a_shape": (self.lora_r, in_features),
                    "lora_b_shape": (out_features, self.lora_r),
                    "lora_a_size": lora_a_size,
                    "lora_b_size": lora_b_size,
                })
                total_lora_weights_size += lora_a_size + lora_b_size

        # PEFT 설정
        config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=list(set(target_modules_names)), # 중복 제거
            lora_dropout=lora_dropout,
            bias="none",
        )
        
        self.peft_model = get_peft_model(base_model, config)
        
        # --- 2. HyperNetwork 초기화 ---
        # 계산된 전체 가중치 크기를 HyperNetwork에 전달
        self.hypernetwork = Hypernetwork(
            output_size=total_lora_weights_size,
            lora_r=lora_r 
        )

        # 기본 모델의 파라미터는 동결
        for name, param in self.peft_model.named_parameters():
            if 'lora_' not in name:
                param.requires_grad = False
        
        # 나중에 가중치를 업데이트할 LoRA 레이어에 대한 참조 저장
        self.lora_layer_references = []
        # 정보 목록과 순서를 맞추기 위해 lora_layers_info의 이름 순서대로 참조를 찾음
        model_modules = dict(self.peft_model.named_modules())
        for layer_info in self.lora_layers_info:
            module_name = layer_info["name"]
            if module_name in model_modules and isinstance(model_modules[module_name], LoraLayer):
                self.lora_layer_references.append(model_modules[module_name])


    def update_lora_weights(self, lora_weights):
        """HyperNetwork에서 생성된 가중치를 LoRA 레이어에 적용"""
        current_pos = 0
        # 저장된 LoRA 레이어 참조와 정보를 순회
        for lora_layer, layer_info in zip(self.lora_layer_references, self.lora_layers_info):
            # lora_A 가중치 추출 및 적용
            lora_a_size = layer_info["lora_a_size"]
            lora_a_shape = layer_info["lora_a_shape"]
            lora_a_w = lora_weights[current_pos : current_pos + lora_a_size].view(lora_a_shape)
            lora_layer.lora_A['default'].weight.data = lora_a_w
            current_pos += lora_a_size
            
            # lora_B 가중치 추출 및 적용
            lora_b_size = layer_info["lora_b_size"]
            lora_b_shape = layer_info["lora_b_shape"]
            lora_b_w = lora_weights[current_pos : current_pos + lora_b_size].view(lora_b_shape)
            lora_layer.lora_B['default'].weight.data = lora_b_w
            current_pos += lora_b_size


    def forward(self, style_prompts, content_prompts, labels=None, attention_mask=None):
        # 1. CLAP으로 style_prompts에서 임베딩 추출 및 HyperNetwork로 LoRA 가중치 생성
        lora_weights = self.hypernetwork(style_prompts)
        
        # 2. 생성된 lora_weights를 peft_model에 적용
        self.update_lora_weights(lora_weights)
        
        # 3. LoRA가 적용된 TTS 모델로 음성 생성
        if self.training and labels is not None:
            # During training with teacher-forcing
            outputs = self.peft_model(
                input_ids=content_prompts,
                attention_mask=attention_mask,
                speech_inputs=labels
            )
            loss = self.loss_fn(outputs.speech_values, labels)
            return {"loss": loss, "logits": outputs.speech_values}
        else:
            # During inference
            outputs = self.peft_model.generate(
                input_ids=content_prompts,
                attention_mask=attention_mask,
            )
            return {"speech_values": outputs}


    def loss_fn(self, generated_audio, target_audio):
        # 간단한 L1 loss 예시
        return nn.L1Loss()(generated_audio, target_audio)
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any
from transformers import AutoModel, AutoTokenizer, AutoProcessor
from peft import LoraConfig, get_peft_model, TaskType
from .hypernetwork import Hypernetwork, CLAPEncoder, QwenTextEncoder


class TTSWithLoRA(nn.Module):
    """
    TTS model with LoRA adaptation controlled by Hypernetwork
    """
    
    def __init__(
        self,
        tts_model_name: str = "nari-labs/Dia-1.6B-0626",
        clap_model_name: str = "laion/larger_clap_general",
        qwen_model_name: str = "Qwen/Qwen3-Embedding-0.6B",
        lora_config: LoraConfig = None,
        hypernetwork_config: Dict[str, Any] = None,
        use_qwen: bool = False,
        # New: explicit control for target selection
        target_mode: Optional[str] = None,  # "auto" | "manual" | "smart"
        manual_target_modules: Optional[List[str]] = None,
    ):
        super().__init__()
        
        # Load base TTS model
        self.tts_model = AutoModel.from_pretrained(tts_model_name)
        self.tts_processor = AutoProcessor.from_pretrained(tts_model_name)
        self.debug_once = True  # 디버그 프린트를 한 번만 실행하기 위한 플래그 (초기화 시점 조기 설정)
        
        # Initialize LoRA configuration
        if lora_config is None:
            lora_config = LoraConfig(
                task_type=TaskType.FEATURE_EXTRACTION,
                r=16,
                lora_alpha=32,
                lora_dropout=0.1,
                # target_modules 는 아래에서 실제 모델 구조를 스캔하여 자동 설정됨
                target_modules=None
            )
        
        # Target selection policy: manual / auto / smart
        # Determine target selection mode priority:
        # 1) explicit arg; 2) in lora_config attribute; 3) hypernetwork_config; 4) default "smart"
        mode_from_lora = getattr(lora_config, "lora_target_mode", None)
        mode_from_hn = None
        if hasattr(hypernetwork_config or {}, "get"):
            mode_from_hn = (hypernetwork_config or {}).get("lora_target_mode", None)
        target_mode = target_mode or mode_from_lora or mode_from_hn or "smart"

        discovered_targets = self._discover_attention_linear_modules(self.tts_model)
        if not discovered_targets:
            discovered_targets = self._discover_attention_linear_modules(self.tts_model, fallback_broad=True)
        if self.debug_once:
            print("[DEBUG] Discovered attention Linear modules for LoRA:")
            for n in discovered_targets[:20]:
                print("  ", n)
            if len(discovered_targets) > 20:
                print(f"  ... and {len(discovered_targets)-20} more")

        # Manual targets: prefer explicit arg, else from lora_config.target_modules
        user_targets = manual_target_modules if manual_target_modules is not None else getattr(lora_config, 'target_modules', None)
        # Normalize user_targets to list if a single string
        if isinstance(user_targets, str):
            user_targets = [user_targets]

        if target_mode == "manual":
            # Use user-provided targets as-is; if none provided, keep discovered for safety
            lora_config.target_modules = user_targets or discovered_targets
        elif target_mode == "auto":
            lora_config.target_modules = discovered_targets
        else:
            # smart (default): intersect; if empty or user not given, use discovered
            if user_targets:
                intersection = sorted(list(set(user_targets).intersection(set(discovered_targets))))
                lora_config.target_modules = intersection or discovered_targets
            else:
                lora_config.target_modules = discovered_targets
        
        self.lora_config = lora_config
        
        # Apply LoRA to TTS model (DiaModel)
        self.tts_model_with_lora = get_peft_model(self.tts_model, lora_config)
        
        # Initialize encoders
        if use_qwen:
            self.text_encoder = QwenTextEncoder(qwen_model_name)
            text_embedding_dim = 1536  # Qwen3-Embedding-0.6B dimension
        else:
            self.clap_encoder = CLAPEncoder(clap_model_name)
            text_embedding_dim = 512  # CLAP text embedding dimension
        
        self.audio_encoder = CLAPEncoder(clap_model_name)
        
        # Initialize Hypernetwork
        hypernetwork_config = hypernetwork_config or {}
        clap_audio_dim = 512  # CLAP audio embedding dimension
        
        # Calculate input dimension for hypernetwork
        if use_qwen:
            # When using Qwen for text + CLAP for audio
            hypernetwork_input_dim = clap_audio_dim + text_embedding_dim
        else:
            # When using CLAP for both audio and text
            hypernetwork_input_dim = clap_audio_dim + text_embedding_dim
        
        self.hypernetwork = Hypernetwork(
            tts_model=self.tts_model,
            input_dim=hypernetwork_input_dim,
            hidden_dim=hypernetwork_config.get('hidden_dim', 512),
            num_layers=hypernetwork_config.get('num_layers', 3),
            lora_config=lora_config,
            target_modules=lora_config.target_modules
        )
        
        self.use_qwen = use_qwen

    def _apply_lora_weights(self, lora_weights: Dict[str, Tuple[torch.Tensor, torch.Tensor]]):
        """Apply generated LoRA weights to the model's LoRA layers."""
        if self.debug_once:
            print("[DEBUG] --- Applying LoRA Weights ---")
            print(f"[DEBUG] LoRA weight keys: {list(lora_weights.keys())}")

        applied_count = 0
        # named_modules 를 사용하여 정확한 모듈 경로 이름을 그대로 사용
        for name, peft_module in self.tts_model_with_lora.named_modules():
            if hasattr(peft_module, 'lora_A') and ('default' in getattr(peft_module, 'lora_A', {})):
                # Try exact match first
                key = name
                weights = lora_weights.get(key)
                if weights is None:
                    # Try matching by decoder-suffix to ignore peft wrapper prefixes
                    if 'decoder.' in name:
                        suffix = name[name.find('decoder.'):]
                        weights = lora_weights.get(suffix)
                if weights is None:
                    # Also try without leading 'model.' in case base model had it
                    if '.model.' in name:
                        alt = name.split('.model.', 1)[1]
                        weights = lora_weights.get(alt)
                        if weights is None and 'decoder.' in alt:
                            alt_suffix = alt[alt.find('decoder.'):]
                            weights = lora_weights.get(alt_suffix)
                if weights is not None:
                    lora_A_w, lora_B_w = weights
                    peft_module.lora_A['default'].weight.data = lora_A_w
                    peft_module.lora_B['default'].weight.data = lora_B_w
                    applied_count += 1
                    if self.debug_once:
                        print(f"  -> SUCCESS: Applied weights to {name}")
                elif self.debug_once:
                    print(f"  -> WARNING: No weights found for {name} in generated weights dict.")

        if self.debug_once:
            print(f"[DEBUG] Total LoRA layers updated: {applied_count}")
            print("[DEBUG] --- Finished Applying LoRA Weights ---")


    def _get_target_module(self, module_name: str):
        """Get target module by name"""
        parts = module_name.split('.')
        module = self.tts_model_with_lora
        
        try:
            for part in parts:
                if '*' in part:
                    continue
                module = getattr(module, part)
            return module
        except AttributeError:
            return None
    
    def encode_speaker_characteristics(
        self,
        audio: Optional[torch.Tensor] = None,
        text: Optional[List[str]] = None,
        sample_rate: int = 22050
    ) -> torch.Tensor:
        """Encode speaker characteristics from audio and/or text"""
        embeddings = []
        if audio is not None:
            audio_embeddings = self.audio_encoder.encode_audio(audio, sample_rate)
            embeddings.append(audio_embeddings)
        if text is not None:
            if self.use_qwen:
                text_embeddings = self.text_encoder.encode_text(text)
            else:
                text_embeddings = self.clap_encoder.encode_text(text)
            embeddings.append(text_embeddings)
        
        if not embeddings:
            raise ValueError("At least one of audio or text must be provided")
        
        return embeddings[0] if len(embeddings) == 1 else torch.cat(embeddings, dim=-1)
    
    def forward(
        self,
        content_text: List[str],
        speaker_audio: Optional[torch.Tensor] = None,
        speaker_text: Optional[List[str]] = None,
        sample_rate: int = 48000,
        **tts_kwargs
    ) -> torch.Tensor:
        """Forward pass through the TTS model with LoRA adaptation."""
        batch_size = len(content_text)
        all_outputs = []
        device = next(self.tts_model_with_lora.parameters()).device
        
        text_inputs = self.tts_processor(
            text=content_text, return_tensors="pt", padding=True, truncation=True
        )
        text_inputs = {k: v.to(device) for k, v in text_inputs.items()}

        for i in range(batch_size):
            if self.debug_once:
                print(f"[DEBUG] --- Processing batch item {i} ---")

            single_speaker_audio = speaker_audio[i:i+1] if speaker_audio is not None else None
            single_speaker_text = [speaker_text[i]] if speaker_text is not None else None
            
            speaker_embeddings = self.encode_speaker_characteristics(
                audio=single_speaker_audio, text=single_speaker_text, sample_rate=sample_rate
            )
            if self.debug_once:
                print(f"[DEBUG] Speaker embedding shape: {speaker_embeddings.shape}")

            lora_weights = self.hypernetwork(speaker_embeddings)
            if self.debug_once:
                if lora_weights:
                    first_key = list(lora_weights.keys())[0]
                    lora_a_shape = lora_weights[first_key][0].shape
                    lora_b_shape = lora_weights[first_key][1].shape
                    print(f"[DEBUG] Generated LoRA weights shape (A, B for first module): {lora_a_shape}, {lora_b_shape}")
                else:
                    print("[DEBUG] Hypernetwork returned empty weights dict.")

            if speaker_embeddings.dim() > 1 and lora_weights and list(lora_weights.values())[0][0].dim() > 2:
                 lora_weights = {k: (v[0][0], v[1][0]) for k, v in lora_weights.items()}
                 if self.debug_once:
                    first_key = list(lora_weights.keys())[0]
                    lora_a_shape = lora_weights[first_key][0].shape
                    lora_b_shape = lora_weights[first_key][1].shape
                    print(f"[DEBUG] LoRA weights shape after un-batching: {lora_a_shape}, {lora_b_shape}")

            self._apply_lora_weights(lora_weights)

            single_text_inputs = {k: v[i:i+1] for k, v in text_inputs.items()}

            with torch.amp.autocast(device_type='cuda'):
                outputs = self.tts_model_with_lora(**single_text_inputs, **tts_kwargs)
            
            if hasattr(outputs, 'spectrogram'): output_audio = outputs.spectrogram
            elif hasattr(outputs, 'audio'): output_audio = outputs.audio
            elif hasattr(outputs, 'waveform'): output_audio = outputs.waveform
            else: output_audio = outputs.last_hidden_state
            
            if self.debug_once:
                print(f"[DEBUG] Raw model output shape: {output_audio.shape}")

            if output_audio.dim() == 4 and output_audio.shape[1] == 1:
                output_audio = output_audio.squeeze(1)

            all_outputs.append(output_audio)
        
        self.debug_once = False # 다음부터는 디버그 메시지 출력 안함

        max_len = max(out.shape[-1] for out in all_outputs)
        padded_outputs = [torch.nn.functional.pad(out, (0, max_len - out.shape[-1])) for out in all_outputs]
            
        return torch.cat(padded_outputs, dim=0)

    def _discover_attention_linear_modules(self, model: nn.Module, fallback_broad: bool = False) -> List[str]:
        """
        Discover attention projection Linear module names under decoder layers.
        - Prefer names containing 'self_attention' or 'cross_attention'. If fallback_broad=True, also accept any 'attn' or 'attention'.
        - Only include nn.Linear modules and typical proj names (q_proj, k_proj, v_proj, o_proj).
        """
        names: List[str] = []
        attn_keys = ["self_attention", "cross_attention"]
        if fallback_broad:
            attn_keys = ["attn", "attention"]  # broader match
        valid_suffixes = ["q_proj", "k_proj", "v_proj", "o_proj"]
        for name, module in model.named_modules():
            if "decoder" not in name:
                continue
            if not any(k in name for k in attn_keys):
                continue
            # Must be exact projection sub-module
            if not any(name.endswith(suf) for suf in valid_suffixes):
                continue
            if isinstance(module, nn.Linear):
                names.append(name)
        # Remove duplicates and sort for stability
        return sorted(list(set(names)))
    
    def generate_speech(
        self,
        content_text: str,
        speaker_audio: Optional[torch.Tensor] = None,
        speaker_text: Optional[str] = None,
        sample_rate: int = 48000,
        **generation_kwargs
    ) -> torch.Tensor:
        """Generate speech for a single input"""
        return self.forward(
            content_text=[content_text],
            speaker_audio=speaker_audio.unsqueeze(0) if speaker_audio is not None else None,
            speaker_text=[speaker_text] if speaker_text is not None else None,
            sample_rate=sample_rate,
            **generation_kwargs
        )[0]

