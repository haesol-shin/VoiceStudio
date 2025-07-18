"""Command line interface for T2A-LoRA."""

import click
import torch
import json
from pathlib import Path
from typing import Optional, List

from .models import T2ALoRAGenerator, T2ALoRAConfig
from .generation import GenerationConfig


@click.group()
@click.version_option()
def main():
    """T2A-LoRA: Text-to-Audio LoRA Generation via Hypernetworks for Real-time Voice Adaptation."""
    pass


@main.command()
@click.option("--config", "-c", type=click.Path(exists=True), help="Path to training config file")
@click.option("--data", "-d", type=click.Path(exists=True), help="Path to training data")
@click.option("--output", "-o", type=click.Path(), default="./outputs", help="Output directory")
@click.option("--epochs", "-e", type=int, default=10, help="Number of training epochs")
@click.option("--batch-size", "-b", type=int, default=8, help="Batch size")
@click.option("--learning-rate", "-lr", type=float, default=1e-4, help="Learning rate")
@click.option("--device", type=str, default="auto", help="Device to use (cuda/cpu/auto)")
@click.option("--wandb/--no-wandb", default=True, help="Use Weights & Biases logging")
@click.option("--resume", type=click.Path(exists=True), help="Resume from checkpoint")
def train(
    config: Optional[str],
    data: Optional[str], 
    output: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    device: str,
    wandb: bool,
    resume: Optional[str],
):
    """Train T2A-LoRA model."""
    from .training import T2ALoRATrainer, TrainingConfig
    
    # Load or create training config
    if config:
        training_config = TrainingConfig.load(config)
    else:
        training_config = TrainingConfig(
            dataset_path=data or "",
            output_dir=output,
            num_epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            device=device,
            use_wandb=wandb,
            resume_from_checkpoint=resume,
        )
    
    # Create trainer
    trainer = T2ALoRATrainer(training_config)
    
    # Start training
    click.echo(f"Starting training with config: {training_config}")
    trainer.train()
    
    click.echo(f"Training completed. Model saved to {output}")


@main.command()
@click.option("--model", "-m", type=click.Path(exists=True), required=True, help="Path to trained model")
@click.option("--text", "-t", type=str, help="Text description of voice characteristics")
@click.option("--audio", "-a", type=click.Path(exists=True), help="Path to reference audio file")
@click.option("--output", "-o", type=click.Path(), default="./generated_lora", help="Output path for LoRA weights")
@click.option("--config", "-c", type=click.Path(exists=True), help="Path to generation config")
@click.option("--rank", "-r", type=int, default=16, help="LoRA rank")
@click.option("--alpha", type=float, default=32.0, help="LoRA alpha")
@click.option("--format", type=click.Choice(["safetensors", "pytorch", "json"]), default="safetensors", help="Output format")
def generate(
    model: str,
    text: Optional[str],
    audio: Optional[str],
    output: str,
    config: Optional[str],
    rank: int,
    alpha: float,
    format: str,
):
    """Generate LoRA weights from text/audio conditions."""
    
    if not text and not audio:
        raise click.UsageError("Either --text or --audio must be provided")
    
    # Load generation config
    if config:
        with open(config, "r") as f:
            gen_config = GenerationConfig.from_dict(json.load(f))
    else:
        gen_config = GenerationConfig(lora_rank=rank, lora_alpha=alpha)
    
    # Load model
    click.echo(f"Loading model from {model}...")
    generator = T2ALoRAGenerator.from_pretrained(model)
    generator.eval()
    
    # Prepare inputs
    audio_features = None
    if audio:
        click.echo(f"Processing audio file: {audio}")
        # TODO: Implement audio feature extraction
        # audio_features = extract_audio_features(audio)
    
    # Generate LoRA weights
    click.echo("Generating LoRA weights...")
    with torch.no_grad():
        lora_adapters = generator.generate_lora_adapters(
            text=text,
            audio_features=audio_features,
            lora_rank=gen_config.lora_rank,
            lora_alpha=gen_config.lora_alpha,
        )
    
    # Save output
    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)
    
    if format == "safetensors":
        from safetensors.torch import save_file
        save_file(lora_adapters["lora_weights"], output_path / "lora_weights.safetensors")
    elif format == "pytorch":
        torch.save(lora_adapters["lora_weights"], output_path / "lora_weights.pt")
    elif format == "json":
        # Convert tensors to lists for JSON serialization
        json_data = {
            "lora_config": lora_adapters["lora_config"],
            "condition_text": lora_adapters["condition_text"],
            "condition_audio_shape": lora_adapters["condition_audio_shape"],
        }
        with open(output_path / "lora_config.json", "w") as f:
            json.dump(json_data, f, indent=2)
    
    # Save config
    with open(output_path / "generation_config.json", "w") as f:
        json.dump(lora_adapters["lora_config"], f, indent=2)
    
    click.echo(f"LoRA weights saved to {output_path}")


@main.command()
@click.option("--model", "-m", type=click.Path(exists=True), required=True, help="Path to trained model")
@click.option("--test-data", "-d", type=click.Path(exists=True), help="Path to test data")
@click.option("--output", "-o", type=click.Path(), default="./evaluation", help="Output directory for results")
@click.option("--batch-size", "-b", type=int, default=8, help="Batch size for evaluation")
@click.option("--metrics", multiple=True, default=["mse", "cosine_sim"], help="Metrics to compute")
def evaluate(
    model: str,
    test_data: Optional[str],
    output: str,
    batch_size: int,
    metrics: List[str],
):
    """Evaluate T2A-LoRA model."""
    from .evaluation import T2ALoRAEvaluator
    
    # Load model
    click.echo(f"Loading model from {model}...")
    generator = T2ALoRAGenerator.from_pretrained(model)
    
    # Create evaluator
    evaluator = T2ALoRAEvaluator(
        model=generator,
        test_data_path=test_data,
        batch_size=batch_size,
        metrics=list(metrics),
    )
    
    # Run evaluation
    click.echo("Running evaluation...")
    results = evaluator.evaluate()
    
    # Save results
    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)
    
    with open(output_path / "evaluation_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    # Print results
    click.echo("\nEvaluation Results:")
    for metric, value in results.items():
        click.echo(f"  {metric}: {value:.4f}")


@main.command()
@click.option("--text", "-t", type=str, multiple=True, help="Text descriptions to test")
@click.option("--model", "-m", type=click.Path(exists=True), help="Path to trained model")
@click.option("--interactive/--no-interactive", default=True, help="Interactive mode")
def demo(text: List[str], model: Optional[str], interactive: bool):
    """Demo T2A-LoRA generation."""
    
    if model:
        generator = T2ALoRAGenerator.from_pretrained(model)
    else:
        click.echo("No model provided, using dummy generator for demo")
        # Create dummy config for demo
        from .models import T2ALoRAConfig, MultimodalEncoderConfig, HyperNetworkConfig
        config = T2ALoRAConfig(
            multimodal_config=MultimodalEncoderConfig(),
            hypernetwork_config=HyperNetworkConfig(),
            target_model_config={"lora_rank": 16},
        )
        generator = T2ALoRAGenerator(config)
    
    generator.eval()
    
    def generate_from_text(text_input: str):
        """Generate LoRA from text input."""
        with torch.no_grad():
            lora_adapters = generator.generate_lora_adapters(text=text_input)
        
        click.echo(f"Generated LoRA for: '{text_input}'")
        click.echo(f"LoRA config: {lora_adapters['lora_config']}")
        click.echo(f"Weight shapes: {[(k, v.shape) for k, v in lora_adapters['lora_weights'].items()]}")
        return lora_adapters
    
    # Process provided text inputs
    for text_input in text:
        generate_from_text(text_input)
        click.echo("-" * 50)
    
    # Interactive mode
    if interactive:
        click.echo("\nEntering interactive mode. Type 'quit' to exit.")
        while True:
            text_input = input("\nEnter text description: ").strip()
            if text_input.lower() in ["quit", "exit", "q"]:
                break
            if text_input:
                try:
                    generate_from_text(text_input)
                except Exception as e:
                    click.echo(f"Error: {e}")


@main.command()
@click.option("--model-dir", "-m", type=click.Path(exists=True), required=True, help="Path to model directory")
def info(model_dir: str):
    """Show model information."""
    
    # Load config
    config_path = Path(model_dir) / "config.json"
    if config_path.exists():
        with open(config_path, "r") as f:
            config = json.load(f)
        
        click.echo("Model Configuration:")
        click.echo(json.dumps(config, indent=2))
    else:
        click.echo("No config.json found in model directory")
    
    # Show model file info
    model_path = Path(model_dir) / "pytorch_model.bin"
    if model_path.exists():
        model_size = model_path.stat().st_size / (1024 * 1024)  # MB
        click.echo(f"\nModel file size: {model_size:.2f} MB")
    else:
        click.echo("No pytorch_model.bin found in model directory")


if __name__ == "__main__":
    main()
