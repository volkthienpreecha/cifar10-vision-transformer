"""Train a from-scratch Vision Transformer on CIFAR-10."""

import argparse
import json
from pathlib import Path

import torch
from torch import nn

from .data import build_cifar10_loaders
from .engine import evaluate, resolve_device, train_epoch
from .model import vit_base, vit_small, vit_tiny


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["tiny", "small", "base"], default="tiny")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--data-dir", type=Path, default=Path("data/cifar10"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs/vit"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=2)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.batch_size <= 0 or args.epochs <= 0 or args.weight_decay < 0:
        raise SystemExit("batch size and epochs must be positive; weight decay must be nonnegative")
    learning_rate = args.learning_rate or 5e-4 * args.batch_size / 256
    if learning_rate <= 0:
        raise SystemExit("learning rate must be positive")
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    train_loader, validation_loader = build_cifar10_loaders(
        args.data_dir, args.batch_size, args.seed, args.num_workers
    )
    model = {"tiny": vit_tiny, "small": vit_small, "base": vit_base}[args.model]()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, betas=(0.9, 0.95), weight_decay=args.weight_decay
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    history = []
    best_accuracy = -1.0
    for epoch in range(1, args.epochs + 1):
        training = train_epoch(model, train_loader, criterion, optimizer, device)
        validation = evaluate(model, validation_loader, criterion, device)
        history.append({"epoch": epoch, "train": training, "validation": validation})
        print(json.dumps(history[-1]))
        if validation["accuracy"] > best_accuracy:
            best_accuracy = validation["accuracy"]
            torch.save(model.state_dict(), args.output_dir / "best_model.pth")
    (args.output_dir / "metrics.json").write_text(json.dumps(history, indent=2) + "\n")


if __name__ == "__main__":
    main()
