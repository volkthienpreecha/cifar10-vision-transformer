from pathlib import Path

import pytest

from cifar10_vit.data import build_cifar10_loaders, split_indices


def test_split_indices_are_deterministic_and_disjoint() -> None:
    first = split_indices(100, 60, 20, seed=11)
    second = split_indices(100, 60, 20, seed=11)
    assert first == second
    assert len(first[0]) == 60
    assert len(first[1]) == 20
    assert set(first[0]).isdisjoint(first[1])


@pytest.mark.parametrize("batch_size", [0, -3])
def test_loader_builder_rejects_invalid_batch_size_before_download(batch_size: int) -> None:
    with pytest.raises(ValueError, match="batch_size"):
        build_cifar10_loaders(Path("unused"), batch_size=batch_size, seed=1)
