"""Command-line interface for training experiments."""

import argparse
import yaml
from pathlib import Path

from ..training.trainer import CLTrainer


def train_command(args):
    """Run training experiment."""
    # Load config if provided
    if args.config:
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)
    else:
        config = {}

    # Create trainer with config
    trainer = CLTrainer(
        model_name=config.get("model_name", args.model_name),
        device=config.get("device", args.device),
        experiment_dir=config.get("experiment_dir", args.experiment_dir),
        experiment_name=config.get("experiment_name", args.experiment_name),
        seed=config.get("seed", args.seed),
        batch_size=config.get("batch_size", args.batch_size),
        learning_rate=config.get("learning_rate", args.learning_rate),
        epochs=config.get("epochs", args.epochs),
        lora_r=config.get("lora_r", args.lora_r),
        lora_alpha=config.get("lora_alpha", args.lora_alpha),
        lambda_ortho=config.get("lambda_ortho", args.lambda_ortho),
        lambda_ewc=config.get("lambda_ewc", args.lambda_ewc),
        replay_ratio=config.get("replay_ratio", args.replay_ratio),
        use_ewc=config.get("use_ewc", args.use_ewc),
        use_orthogonal=config.get("use_orthogonal", args.use_orthogonal),
        use_replay=config.get("use_replay", args.use_replay),
        use_lateral=config.get("use_lateral", args.use_lateral),
        checkpoint_every=config.get("checkpoint_every", args.checkpoint_every),
        keep_last_n_checkpoints=config.get("keep_last_n_checkpoints", args.keep_last_n_checkpoints),
    )

    # Train sequence
    trainer.train_sequence(resume=args.resume)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="Continual Learning Training")

    # Config file
    parser.add_argument(
        "--config",
        type=str,
        help="Path to YAML config file",
    )

    # Model config
    parser.add_argument(
        "--model-name",
        type=str,
        default="distilbert-base-uncased",
        help="Base model name",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device to use",
    )

    # Experiment config
    parser.add_argument(
        "--experiment-dir",
        type=str,
        default="experiments",
        help="Experiment directory",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="default",
        help="Experiment name",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )

    # Training config
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
        help="Learning rate",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of epochs per task",
    )

    # LoRA config
    parser.add_argument(
        "--lora-r",
        type=int,
        default=8,
        help="LoRA rank",
    )
    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=16,
        help="LoRA alpha",
    )

    # Regularization config
    parser.add_argument(
        "--lambda-ortho",
        type=float,
        default=0.1,
        help="Orthogonal regularization weight",
    )
    parser.add_argument(
        "--lambda-ewc",
        type=float,
        default=100.0,
        help="EWC regularization weight",
    )

    # Replay config
    parser.add_argument(
        "--replay-ratio",
        type=float,
        default=0.2,
        help="Ratio of replay samples in batch",
    )

    # Component flags
    parser.add_argument(
        "--use-ewc",
        action="store_true",
        default=True,
        help="Use EWC",
    )
    parser.add_argument(
        "--no-ewc",
        dest="use_ewc",
        action="store_false",
        help="Disable EWC",
    )
    parser.add_argument(
        "--use-orthogonal",
        action="store_true",
        default=True,
        help="Use orthogonal LoRA",
    )
    parser.add_argument(
        "--no-orthogonal",
        dest="use_orthogonal",
        action="store_false",
        help="Disable orthogonal LoRA",
    )
    parser.add_argument(
        "--use-replay",
        action="store_true",
        default=True,
        help="Use generative replay",
    )
    parser.add_argument(
        "--no-replay",
        dest="use_replay",
        action="store_false",
        help="Disable generative replay",
    )
    parser.add_argument(
        "--use-lateral",
        action="store_true",
        default=False,
        help="Use lateral connections",
    )
    
    # Checkpointing config
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=100,
        help="Save checkpoint every N steps (0 to disable)",
    )
    parser.add_argument(
        "--keep-last-n-checkpoints",
        type=int,
        default=3,
        help="Keep only last N checkpoints",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from latest checkpoint",
    )

    args = parser.parse_args()

    train_command(args)


if __name__ == "__main__":
    main()


