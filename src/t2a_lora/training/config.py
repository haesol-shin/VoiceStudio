"""Training configuration for T2A-LoRA."""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
import os


@dataclass
class TrainingConfig:
    """Configuration for T2A-LoRA training."""
    
    # Model config
    model_config: Dict[str, Any] = field(default_factory=dict)
    
    # Data config
    dataset_path: str = ""
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1
    max_samples: Optional[int] = None
    
    # Training hyperparameters
    batch_size: int = 8
    learning_rate: float = 1e-4
    num_epochs: int = 10
    warmup_steps: int = 1000
    weight_decay: float = 0.01
    gradient_clip_norm: float = 1.0
    
    # Optimizer config
    optimizer: str = "adamw"
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    
    # Scheduler config
    scheduler: str = "cosine"
    scheduler_kwargs: Dict[str, Any] = field(default_factory=dict)
    
    # Loss config
    loss_type: str = "mse"
    loss_weights: Dict[str, float] = field(default_factory=lambda: {"reconstruction": 1.0, "regularization": 0.1})
    
    # Logging and checkpointing
    output_dir: str = "./outputs"
    logging_dir: str = "./logs"
    logging_steps: int = 50
    save_steps: int = 1000
    eval_steps: int = 500
    save_total_limit: int = 3
    
    # Evaluation
    eval_strategy: str = "steps"
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False
    early_stopping_patience: int = 5
    
    # Hardware
    device: str = "auto"
    mixed_precision: bool = True
    dataloader_num_workers: int = 4
    pin_memory: bool = True
    
    # Distributed training
    distributed: bool = False
    local_rank: int = -1
    
    # Reproducibility
    seed: int = 42
    
    # Wandb logging
    use_wandb: bool = True
    wandb_project: str = "latentforge-t2a-lora"
    wandb_run_name: Optional[str] = None
    wandb_tags: List[str] = field(default_factory=list)
    
    # Resume training
    resume_from_checkpoint: Optional[str] = None
    
    # Data augmentation
    use_data_augmentation: bool = True
    augmentation_config: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Post-initialization processing."""
        # Create output directories
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.logging_dir, exist_ok=True)
        
        # Set device
        if self.device == "auto":
            import torch
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Validate config
        self._validate_config()
    
    def _validate_config(self):
        """Validate configuration parameters."""
        assert self.train_split + self.val_split + self.test_split == 1.0, \
            "Data splits must sum to 1.0"
        
        assert self.batch_size > 0, "Batch size must be positive"
        assert self.learning_rate > 0, "Learning rate must be positive"
        assert self.num_epochs > 0, "Number of epochs must be positive"
        
        assert self.eval_strategy in ["steps", "epoch"], \
            "Eval strategy must be 'steps' or 'epoch'"
        
        assert self.loss_type in ["mse", "l1", "huber"], \
            "Loss type must be one of: mse, l1, huber"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        from dataclasses import asdict
        return asdict(self)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "TrainingConfig":
        """Create config from dictionary."""
        return cls(**config_dict)
    
    def save(self, filepath: str):
        """Save config to file."""
        import json
        with open(filepath, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, filepath: str) -> "TrainingConfig":
        """Load config from file."""
        import json
        with open(filepath, "r") as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)
