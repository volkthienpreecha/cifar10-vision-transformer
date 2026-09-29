import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from cifar10_vit.engine import accuracy, evaluate, resolve_device, train_epoch


def test_accuracy_matches_hand_checked_logits() -> None:
    logits = torch.tensor([[0.1, 0.9], [0.8, 0.2], [0.4, 0.6]])
    targets = torch.tensor([1, 0, 0])
    assert accuracy(logits, targets) == pytest.approx(2 / 3)


def test_one_optimizer_step_and_evaluation_return_metrics() -> None:
    loader = DataLoader(
        TensorDataset(torch.randn(8, 4), torch.tensor([0, 1, 0, 1, 0, 1, 0, 1])),
        batch_size=4,
    )
    model = nn.Linear(4, 2)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    assert train_epoch(model, loader, criterion, optimizer, torch.device("cpu"))["loss"] >= 0
    assert 0 <= evaluate(model, loader, criterion, torch.device("cpu"))["accuracy"] <= 1


def test_evaluate_rejects_empty_loader() -> None:
    empty = DataLoader(TensorDataset(torch.empty(0, 4), torch.empty(0, dtype=torch.long)))
    with pytest.raises(ValueError, match="empty"):
        evaluate(nn.Linear(4, 2), empty, nn.CrossEntropyLoss(), torch.device("cpu"))


def test_resolve_device_accepts_cpu() -> None:
    assert resolve_device("cpu") == torch.device("cpu")


@pytest.mark.parametrize(
    ("cuda_available", "mps_available", "expected"),
    [(True, True, "cuda"), (False, True, "mps"), (False, False, "cpu")],
)
def test_auto_device_order(
    monkeypatch: pytest.MonkeyPatch,
    cuda_available: bool,
    mps_available: bool,
    expected: str,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda_available)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: mps_available)
    assert resolve_device("auto") == torch.device(expected)
