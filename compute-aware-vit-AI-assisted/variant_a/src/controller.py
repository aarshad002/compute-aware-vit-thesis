"""Confidence-based dynamic controller for adaptive token pruning.

After the first prune_layer transformer blocks, a lightweight controller head
estimates per-image confidence from the CLS token. At inference, that
confidence is compared against two configurable thresholds to select a token
budget of 25%, 50%, or 75%.

During training the budget is fixed (train_keep_ratio) and the controller head
is trained with an auxiliary classification loss so it learns to distinguish
easy from hard images at the midpoint of the network.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DynamicControllerViT(nn.Module):
    """DeiT wrapper with a mid-network confidence controller.

    Args:
        base_model: a timm VisionTransformer instance.
        num_classes: number of output classes.
        prune_layer: index of the block after which routing is performed.
            Blocks 0..(prune_layer-1) are always run; blocks prune_layer..end
            see only the kept tokens.
        train_keep_ratio: fixed token budget used during training (0.50 by
            default). The controller is trained as an auxiliary head.
    """

    BUDGET_RATIOS = (0.25, 0.50, 0.75)

    def __init__(
        self,
        base_model: nn.Module,
        num_classes: int,
        prune_layer: int = 6,
        train_keep_ratio: float = 0.50,
    ):
        """Initialise controller model and auxiliary head."""
        super().__init__()
        self.base = base_model
        self.num_classes = num_classes
        self.prune_layer = prune_layer
        self.train_keep_ratio = train_keep_ratio

        embed_dim = base_model.embed_dim
        # Lightweight controller head applied to the CLS token at layer prune_layer
        self.controller_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, num_classes),
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run_shared_blocks(self, x: torch.Tensor) -> torch.Tensor:
        """Run patch embed, positional encoding, and first prune_layer blocks.

        Args:
            x: raw image tensor (B, C, H, W).

        Returns:
            Intermediate token tensor (B, N+1, D) before pruning.
        """
        base = self.base
        x = base.patch_embed(x)
        x = base._pos_embed(x)
        x = base.patch_drop(x)
        x = base.norm_pre(x)
        for blk in base.blocks[: self.prune_layer]:
            x = blk(x)
        return x

    def _prune_tokens(self, x: torch.Tensor, keep_ratio: float) -> torch.Tensor:
        """Drop lowest L2-norm patch tokens; CLS token at index 0 is preserved.

        Args:
            x: token tensor (B, N+1, D).
            keep_ratio: fraction of patch tokens to keep.

        Returns:
            Pruned tensor (B, k+1, D).
        """
        cls = x[:, :1, :]
        patches = x[:, 1:, :]
        N = int(patches.shape[1])
        k = max(1, round(N * keep_ratio))
        scores = patches.norm(dim=-1)
        _, topk_idx = scores.topk(k, dim=-1, sorted=False)
        topk_idx, _ = topk_idx.sort(dim=-1)
        idx_exp = topk_idx.unsqueeze(-1).expand(-1, -1, patches.shape[-1])
        kept = patches.gather(1, idx_exp)
        return torch.cat([cls, kept], dim=1)

    def _run_tail(self, x: torch.Tensor) -> torch.Tensor:
        """Run blocks from prune_layer onward, norm, and classification head.

        Args:
            x: pruned token tensor (B, k+1, D).

        Returns:
            Logit tensor (B, num_classes).
        """
        for blk in self.base.blocks[self.prune_layer :]:
            x = blk(x)
        x = self.base.norm(x)
        return self.base.forward_head(x)

    # ------------------------------------------------------------------
    # Forward passes
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Training forward: fixed budget + auxiliary controller loss.

        Uses train_keep_ratio for token pruning so gradients flow through
        both the main classification path and the controller head jointly.

        Args:
            x: input image tensor (B, C, H, W).

        Returns:
            Tuple of (main_logits, ctrl_logits), each (B, num_classes).
            main_logits: from the full pruned-and-classified path.
            ctrl_logits: from the controller head on the mid-network CLS token.
        """
        h = self._run_shared_blocks(x)                    # (B, N+1, D)
        ctrl_logits = self.controller_head(h[:, 0])       # (B, num_classes)
        h_pruned = self._prune_tokens(h, self.train_keep_ratio)
        main_logits = self._run_tail(h_pruned)
        return main_logits, ctrl_logits

    @torch.no_grad()
    def forward_dynamic(
        self,
        x: torch.Tensor,
        high_thresh: float,
        low_thresh: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Inference forward: per-image routing based on controller confidence.

        Routing rules (high confidence = easy image = small budget):
            conf >= high_thresh  →  25% token budget
            conf >= low_thresh   →  50% token budget
            conf <  low_thresh   →  75% token budget

        Images with the same budget are processed as a mini-batch for
        efficiency; results are gathered back into original order.

        Args:
            x: input image tensor (B, C, H, W).
            high_thresh: confidence threshold above which 25% budget is used.
            low_thresh: confidence threshold above which 50% budget is used.

        Returns:
            Tuple of (logits, budgets) where logits is (B, num_classes) and
            budgets is a (B,) float tensor with values in {0.25, 0.50, 0.75}.
        """
        B = x.shape[0]
        h = self._run_shared_blocks(x)                        # (B, N+1, D)

        ctrl_logits = self.controller_head(h[:, 0])           # (B, num_classes)
        ctrl_conf = F.softmax(ctrl_logits, dim=-1).max(dim=-1).values  # (B,)

        # Assign each image a keep_ratio
        budgets = torch.where(
            ctrl_conf >= high_thresh,
            torch.full((B,), 0.25, device=x.device),
            torch.where(
                ctrl_conf >= low_thresh,
                torch.full((B,), 0.50, device=x.device),
                torch.full((B,), 0.75, device=x.device),
            ),
        )

        # Process each budget group as a batch
        final_logits = torch.zeros(B, self.num_classes, device=x.device)
        for ratio in self.BUDGET_RATIOS:
            mask = budgets == ratio
            if not mask.any():
                continue
            h_group = self._prune_tokens(h[mask], ratio)
            final_logits[mask] = self._run_tail(h_group)

        return final_logits, budgets
