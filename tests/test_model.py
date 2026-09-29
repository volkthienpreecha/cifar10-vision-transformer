import pytest
import torch

from cifar10_vit.model import MultiHeadSelfAttention, VisionTransformer


def test_attention_preserves_batch_and_token_axes() -> None:
    attention = MultiHeadSelfAttention(input_dim=12, embed_dim=24, num_heads=3)
    assert attention(torch.randn(2, 5, 12)).shape == (2, 5, 24)


def test_vit_returns_one_logit_vector_per_image() -> None:
    model = VisionTransformer(
        image_size=32,
        patch_size=4,
        num_layers=2,
        num_heads=4,
        embed_dim=32,
        mlp_hidden_dim=64,
        num_classes=10,
    )
    assert model(torch.randn(2, 3, 32, 32)).shape == (2, 10)


def test_attention_rejects_incompatible_head_dimension() -> None:
    with pytest.raises(ValueError, match="divisible"):
        MultiHeadSelfAttention(input_dim=12, embed_dim=25, num_heads=4)


def test_vit_rejects_nondivisible_patch_geometry() -> None:
    with pytest.raises(ValueError, match="divisible"):
        VisionTransformer(30, 4, 2, 4, 32, 64, 10)


def test_vit_rejects_runtime_image_size_mismatch() -> None:
    model = VisionTransformer(32, 4, 1, 4, 32, 64, 10)
    with pytest.raises(ValueError, match="32"):
        model(torch.randn(2, 3, 28, 28))


def test_runtime_padding_inherits_model_dtype() -> None:
    model = VisionTransformer(32, 4, 1, 3, 24, 48, 10).double()
    output = model(torch.randn(2, 3, 32, 32, dtype=torch.float64))
    assert output.dtype == torch.float64
