"""Static token pruning for Vision Transformers.

Prunes the lowest L2-norm patch tokens at a fixed layer during the forward
pass. The CLS token is always preserved at index 0.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class StaticPrunedViT(nn.Module):
    """DeiT wrapper that drops low-norm patch tokens after a chosen layer.

    Args:
        base_model: a timm VisionTransformer instance (already configured for
            the target number of classes).
        keep_ratio: fraction of patch tokens to retain, e.g. 0.25 keeps the
            top-25% by L2 norm.
        prune_layer: block index (1-based count) after which pruning is
            applied. prune_layer=6 means blocks 1-6 run first, then pruning,
            then blocks 7-12.
    """

    def __init__(self, base_model: nn.Module, keep_ratio: float, prune_layer: int = 6):
        """Initialise the pruned model wrapper."""
        super().__init__()
        self.base = base_model
        self.keep_ratio = keep_ratio
        self.prune_layer = prune_layer

    def _prune_tokens(self, x: torch.Tensor) -> torch.Tensor:
        """Drop patch tokens with the lowest L2 norm, keeping CLS intact.

        Args:
            x: token tensor of shape (B, N+1, D) where index 0 is the CLS token
               and indices 1..N are patch tokens.

        Returns:
            Pruned tensor of shape (B, k+1, D) where k = round(N * keep_ratio).
        """
        cls = x[:, :1, :]           # (B, 1, D)
        patches = x[:, 1:, :]       # (B, N, D)
        # int() forces a concrete Python int so tracing (fvcore) doesn't
        # receive a symbolic Tensor that can't be passed to round().
        N = int(patches.shape[1])

        k = max(1, round(N * self.keep_ratio))

        # Score each patch token by its L2 norm across the embedding dimension
        scores = patches.norm(dim=-1)                          # (B, N)
        _, topk_idx = scores.topk(k, dim=-1, sorted=False)    # (B, k)
        topk_idx, _ = topk_idx.sort(dim=-1)                   # preserve order

        # Gather the kept tokens
        idx_exp = topk_idx.unsqueeze(-1).expand(-1, -1, patches.shape[-1])
        kept = patches.gather(1, idx_exp)                      # (B, k, D)

        return torch.cat([cls, kept], dim=1)                   # (B, k+1, D)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the pruned forward pass.

        Executes the first prune_layer transformer blocks, prunes patch tokens,
        then completes the remaining blocks before the classification head.

        Args:
            x: input image tensor of shape (B, C, H, W).

        Returns:
            Logit tensor of shape (B, num_classes).
        """
        base = self.base

        # Patch embedding
        x = base.patch_embed(x)

        # CLS token concatenation and positional embedding (timm 1.x API)
        x = base._pos_embed(x)

        # Regularisation layers (Identity for standard DeiT)
        x = base.patch_drop(x)
        x = base.norm_pre(x)

        # Blocks before pruning point
        for blk in base.blocks[: self.prune_layer]:
            x = blk(x)

        # Static token pruning
        x = self._prune_tokens(x)

        # Remaining blocks
        for blk in base.blocks[self.prune_layer :]:
            x = blk(x)

        x = base.norm(x)
        return base.forward_head(x)

    def forward_with_confidence(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the forward pass and return both logits and per-image confidence.

        Confidence is defined as the maximum softmax probability, used by
        cascade inference (Step 5) to decide whether to accept or escalate.

        Args:
            x: input image tensor of shape (B, C, H, W).

        Returns:
            Tuple of (logits, confidence) where both have shape (B,) for
            confidence and (B, num_classes) for logits.
        """
        logits = self.forward(x)
        confidence = F.softmax(logits, dim=-1).max(dim=-1).values
        return logits, confidence

    def tokens_info(self, image_size: int, patch_size: int = 16) -> tuple[int, int]:
        """Return (tokens_total, tokens_kept) for reporting in metrics.json.

        Args:
            image_size: spatial resolution of input images (assumed square).
            patch_size: patch size used by the model.

        Returns:
            Tuple of (total patch tokens, kept patch tokens).
        """
        tokens_total = (image_size // patch_size) ** 2
        tokens_kept = max(1, round(tokens_total * self.keep_ratio))
        return tokens_total, tokens_kept
