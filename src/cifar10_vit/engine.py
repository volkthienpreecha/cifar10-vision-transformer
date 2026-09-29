"""Training and evaluation helpers for CIFAR-10 classifiers."""

from collections.abc import Iterable

import torch
from torch import nn


def resolve_device(requested: str = "auto") -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if requested not in {"cpu", "cuda", "mps"}:
        raise ValueError("device must be auto, cpu, cuda, or mps")
    return torch.device(requested)


def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    if logits.ndim != 2 or targets.ndim != 1 or logits.shape[0] != targets.shape[0]:
        raise ValueError("logits and targets must have shapes (N, C) and (N,)")
    if targets.numel() == 0:
        raise ValueError("cannot compute accuracy for an empty batch")
    return float((logits.argmax(dim=1) == targets).float().mean().item())


def _run_epoch(
    model: nn.Module,
    loader: Iterable[tuple[torch.Tensor, torch.Tensor]],
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
) -> dict[str, float]:
    training = optimizer is not None
    model.to(device).train(training)
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            loss = criterion(logits, targets)
            if optimizer is not None:
                loss.backward()
                optimizer.step()
            batch_size = targets.shape[0]
            total_loss += float(loss.item()) * batch_size
            total_correct += int((logits.argmax(dim=1) == targets).sum().item())
            total_examples += batch_size
    if total_examples == 0:
        raise ValueError("cannot process an empty data loader")
    return {"loss": total_loss / total_examples, "accuracy": total_correct / total_examples}


def train_epoch(
    model: nn.Module,
    loader: Iterable[tuple[torch.Tensor, torch.Tensor]],
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> dict[str, float]:
    return _run_epoch(model, loader, criterion, device, optimizer)


def evaluate(
    model: nn.Module,
    loader: Iterable[tuple[torch.Tensor, torch.Tensor]],
    criterion: nn.Module,
    device: torch.device,
) -> dict[str, float]:
    return _run_epoch(model, loader, criterion, device, optimizer=None)
