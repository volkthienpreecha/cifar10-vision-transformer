"""Vision Transformer layers implemented directly with PyTorch tensor operations."""

import math

import torch
from torch import nn


class MultiHeadSelfAttention(nn.Module):
    """Scaled dot-product self-attention with vectorized heads."""

    def __init__(self, input_dim: int, embed_dim: int, num_heads: int) -> None:
        super().__init__()
        if min(input_dim, embed_dim, num_heads) <= 0:
            raise ValueError("attention dimensions must be positive")
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.key = nn.Linear(input_dim, embed_dim)
        self.query = nn.Linear(input_dim, embed_dim)
        self.value = nn.Linear(input_dim, embed_dim)
        self.output = nn.Linear(embed_dim, embed_dim)

    def _split_heads(self, values: torch.Tensor) -> torch.Tensor:
        batch, tokens, _ = values.shape
        return values.reshape(batch, tokens, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 3:
            raise ValueError("attention inputs must have shape (batch, tokens, channels)")
        keys = self._split_heads(self.key(inputs))
        queries = self._split_heads(self.query(inputs))
        values = self._split_heads(self.value(inputs))
        weights = torch.softmax(queries @ keys.transpose(-2, -1) / math.sqrt(self.head_dim), dim=-1)
        attended = weights @ values
        batch, _, tokens, _ = attended.shape
        joined = attended.transpose(1, 2).reshape(batch, tokens, self.embed_dim)
        return self.output(joined)


class TransformerEncoderBlock(nn.Module):
    """Pre-normalized attention and MLP residual block."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_hidden_dim: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if mlp_hidden_dim <= 0 or not 0 <= dropout < 1:
            raise ValueError("mlp_hidden_dim must be positive and dropout must be in [0, 1)")
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attention = MultiHeadSelfAttention(embed_dim, embed_dim, num_heads)
        self.attention_dropout = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = inputs + self.attention_dropout(self.attention(self.norm1(inputs)))
        return hidden + self.mlp(self.norm2(hidden))


class VisionTransformer(nn.Module):
    """Classify fixed-size RGB images using learned patch and position embeddings."""

    def __init__(
        self,
        image_size: int,
        patch_size: int,
        num_layers: int,
        num_heads: int,
        embed_dim: int,
        mlp_hidden_dim: int,
        num_classes: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if min(
            image_size,
            patch_size,
            num_layers,
            num_heads,
            embed_dim,
            mlp_hidden_dim,
            num_classes,
        ) <= 0:
            raise ValueError("all model dimensions must be positive")
        if image_size % patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size")
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_heads = num_heads
        patches_per_side = image_size // patch_size
        self.num_patches = patches_per_side**2
        patch_dim = 3 * patch_size**2
        self.patch_embedding = nn.Linear(patch_dim, embed_dim)
        self.class_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.position_embedding = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        self.embedding_dropout = nn.Dropout(dropout)
        self.encoder = nn.ModuleList(
            TransformerEncoderBlock(embed_dim, num_heads, mlp_hidden_dim, dropout)
            for _ in range(num_layers)
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.classifier = nn.Linear(embed_dim, num_classes)

    def _patchify(self, images: torch.Tensor) -> torch.Tensor:
        batch = images.shape[0]
        side = self.image_size // self.patch_size
        patches = images.reshape(batch, 3, side, self.patch_size, side, self.patch_size)
        patches = patches.permute(0, 2, 4, 3, 5, 1)
        return patches.reshape(batch, self.num_patches, -1)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        expected = (3, self.image_size, self.image_size)
        if images.ndim != 4 or tuple(images.shape[1:]) != expected:
            raise ValueError(f"expected images with shape (N, 3, {self.image_size}, {self.image_size})")
        patch_tokens = self.patch_embedding(self._patchify(images))
        class_tokens = self.class_token.expand(images.shape[0], -1, -1)
        hidden = torch.cat((class_tokens, patch_tokens), dim=1) + self.position_embedding
        hidden = self.embedding_dropout(hidden)
        padding_length = (-hidden.shape[1]) % self.num_heads
        if padding_length:
            hidden = torch.cat(
                (hidden, hidden.new_zeros(hidden.shape[0], padding_length, hidden.shape[2])), dim=1
            )
        for layer in self.encoder:
            hidden = layer(hidden)
        return self.classifier(self.norm(hidden[:, 0]))


def vit_tiny(num_classes: int = 10, patch_size: int = 4, image_size: int = 32) -> VisionTransformer:
    return VisionTransformer(image_size, patch_size, 12, 3, 192, 768, num_classes)


def vit_small(num_classes: int = 10, patch_size: int = 4, image_size: int = 32) -> VisionTransformer:
    return VisionTransformer(image_size, patch_size, 12, 6, 384, 1536, num_classes)


def vit_base(num_classes: int = 10, patch_size: int = 4, image_size: int = 32) -> VisionTransformer:
    return VisionTransformer(image_size, patch_size, 12, 12, 768, 3072, num_classes)
