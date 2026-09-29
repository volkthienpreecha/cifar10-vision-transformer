"""CIFAR-10 augmentation and deterministic split helpers."""

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset


def split_indices(
    dataset_size: int, train_size: int, validation_size: int, seed: int
) -> tuple[list[int], list[int]]:
    if min(dataset_size, train_size, validation_size) <= 0:
        raise ValueError("dataset and split sizes must be positive")
    if train_size + validation_size > dataset_size:
        raise ValueError("requested split is larger than the dataset")
    order = torch.randperm(dataset_size, generator=torch.Generator().manual_seed(seed)).tolist()
    return order[:train_size], order[train_size : train_size + validation_size]


class _TransformSubset(Dataset[tuple[torch.Tensor, int]]):
    def __init__(self, dataset: Dataset[Any], indices: list[int], transform: Any):
        self.dataset = dataset
        self.indices = indices
        self.transform = transform

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        image, target = self.dataset[self.indices[index]]
        return self.transform(image), int(target)


def build_cifar10_loaders(
    data_dir: Path,
    batch_size: int,
    seed: int,
    num_workers: int = 2,
) -> tuple[DataLoader[Any], DataLoader[Any]]:
    """Download CIFAR-10 and construct deterministic 40k/10k loaders."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if num_workers < 0:
        raise ValueError("num_workers must be nonnegative")

    from torchvision import datasets, transforms

    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    train_transform = transforms.Compose(
        [
            transforms.Resize(40),
            transforms.RandomCrop(32),
            transforms.RandomHorizontalFlip(),
            transforms.RandomAffine(degrees=0, translate=(0.2, 0.2), scale=(0.95, 1.05)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    validation_transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(mean, std)]
    )
    dataset = datasets.CIFAR10(data_dir, train=True, download=True)
    train_indices, validation_indices = split_indices(len(dataset), 40_000, 10_000, seed)
    train_dataset = _TransformSubset(dataset, train_indices, train_transform)
    validation_dataset = _TransformSubset(dataset, validation_indices, validation_transform)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        generator=torch.Generator().manual_seed(seed),
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    return train_loader, validation_loader
