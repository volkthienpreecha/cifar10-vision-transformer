"""Parser for historical Vision Transformer training logs."""

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class EpochMetrics:
    epoch: int
    train_loss: float
    train_accuracy: float
    val_loss: float
    val_accuracy: float


_METRIC_PATTERN = re.compile(
    r"^\[\s*(?P<epoch>\d+)\]\s+"
    r"train loss:\s*(?P<train_loss>\d+(?:\.\d+)?)\s*\|\s*"
    r"train accuracy:\s*(?P<train_accuracy>\d+(?:\.\d+)?)\s*\|\s*"
    r"val loss:\s*(?P<val_loss>\d+(?:\.\d+)?)\s*\|\s*"
    r"val accuracy:\s*(?P<val_accuracy>\d+(?:\.\d+)?)$"
)


def parse_training_history(text: str) -> list[EpochMetrics]:
    """Parse complete epoch lines and reject ambiguous partial records."""

    history = []
    seen_epochs: set[int] = set()
    for line in text.splitlines():
        match = _METRIC_PATTERN.fullmatch(line.strip())
        if match is None:
            if "train loss:" in line:
                raise ValueError(f"partial metric line: {line.strip()}")
            continue
        epoch = int(match.group("epoch"))
        if epoch in seen_epochs:
            raise ValueError(f"duplicate epoch: {epoch}")
        seen_epochs.add(epoch)
        history.append(
            EpochMetrics(
                epoch,
                float(match.group("train_loss")),
                float(match.group("train_accuracy")),
                float(match.group("val_loss")),
                float(match.group("val_accuracy")),
            )
        )
    return history
