"""Adaptive Token Sampling (ATS, Fayyaz et al., ECCV 2022) on DeiT-Tiny.

ATS is a parameter-free, training-free token reduction method. At each ATS stage it
scores patch tokens by (CLS-attention x value-norm), turns the scores into a per-image
probability distribution, and draws K_max samples by inverse-transform sampling on a
fixed grid in [0, 1]. Duplicate sampled indices collapse, so the realised number of
kept tokens K' is variable per image — this is the source of ATS's adaptivity. The CLS
token is always kept.

This module wraps timm's ``deit_tiny_patch16_224`` with the direct-attribution pattern
used elsewhere in this repo (see vit_static.py / controller.py / vit_multibudget.py), so
the dense checkpoint loads with strict=True after stripping its ``model.`` prefix.

Implementation choice (documented per the task): because K' varies per image, the token
counts diverge after the first ATS stage. Rather than padding + masking, the forward runs
the shared prefix (blocks before the first ATS stage) batched, then runs the remaining
blocks per image (batch dimension looped). This is exact (no padding artifacts) and is
fine for the eval-only, no-gradient setting in which ATS is used here.
"""

from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import timm


class ATSAttention(nn.Module):
    """timm Attention with Adaptive Token Sampling on the attention output.

    Reuses the submodules of the original attention (``qkv``/``proj``/...) by reference
    so the parameter keys are byte-identical to the wrapped block — the dense checkpoint
    therefore loads with strict=True.
    """

    def __init__(
        self, orig_attn: nn.Module, k_max: int, use_value_norm: bool = True,
    ) -> None:
        """Wrap an existing timm Attention module with ATS sampling.

        Args:
            orig_attn: The original timm ``Attention`` instance to wrap.
            k_max: Upper bound on patch tokens sampled at this stage.
            use_value_norm: Multiply CLS-attention scores by the token value norm.
        """
        super().__init__()
        self.num_heads = orig_attn.num_heads
        self.head_dim = orig_attn.head_dim
        self.scale = orig_attn.scale
        # Reuse submodules by reference -> exact weights, identical state_dict keys.
        self.qkv = orig_attn.qkv
        self.q_norm = orig_attn.q_norm
        self.k_norm = orig_attn.k_norm
        self.attn_drop = orig_attn.attn_drop
        if hasattr(orig_attn, 'norm'):
            self.norm = orig_attn.norm
        self.proj = orig_attn.proj
        self.proj_drop = orig_attn.proj_drop

        self.k_max = k_max
        self.use_value_norm = use_value_norm

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """Compute attention explicitly and adaptively sample surviving tokens.

        Args:
            x: Pre-normalised token sequence (1, N, C). Operates per image (B=1).

        Returns:
            (output, kept_indices, kprime) where output is (1, K'+1, C), kept_indices
            is a (K'+1,) long tensor into the input token axis (index 0 = CLS), and
            kprime is the number of unique sampled patch tokens (excludes CLS).
        """
        # Concrete ints so the fvcore JIT trace bakes per-image shapes (and so the
        # shape-derived clamp bound below stays a Python int, not a CPU tensor).
        B, N, C = int(x.shape[0]), int(x.shape[1]), int(x.shape[2])
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)                                   # (B, H, N, N)

        # ATS scores: CLS-attention to patch tokens, optionally x value-norm, summed heads.
        cls_attn = attn[:, :, 0, 1:]                                  # (B, H, N-1)
        if self.use_value_norm:
            v_norm = v[:, :, 1:, :].norm(dim=-1)                      # (B, H, N-1)
            scores = (cls_attn * v_norm).sum(dim=1)                   # (B, N-1)
        else:
            scores = cls_attn.sum(dim=1)                              # (B, N-1)

        # Per-image inverse-transform sampling on a fixed grid (B == 1 here).
        prob = scores / (scores.sum(dim=-1, keepdim=True) + 1e-12)
        cdf = prob.cumsum(dim=-1)[0]                                  # (N-1,)
        grid = (2 * torch.arange(self.k_max, device=x.device) + 1) / (2 * self.k_max)
        idx = torch.searchsorted(cdf, grid).clamp(max=N - 2)         # (k_max,)
        unique = torch.unique(idx)                                    # sorted, (K',)
        kprime = int(unique.numel())
        kept = torch.cat([torch.zeros(1, dtype=torch.long, device=x.device),
                          unique + 1])                                # CLS + patches

        # Sampled attention output O = A^s . V for the kept query tokens.
        a_s = attn[0][:, kept, :]                                     # (H, K'+1, N)
        o = a_s @ v[0]                                                # (H, K'+1, head_dim)
        o = o.transpose(0, 1).reshape(kept.numel(), C).unsqueeze(0)   # (1, K'+1, C)
        o = self.proj_drop(self.proj(o))
        return o, kept, kprime


class ATSBlock(nn.Module):
    """timm Block whose attention sub-block adaptively samples tokens.

    Submodules are reused by reference from the wrapped block, preserving exact
    parameter keys. The residual at the ATS stage is gathered at the sampled indices
    so it matches the reduced attention output; the MLP sub-block then runs normally.
    """

    def __init__(
        self, orig_block: nn.Module, k_max: int, use_value_norm: bool,
        kprime_log_ref: Dict[int, List[int]], block_idx: int,
    ) -> None:
        """Wrap an existing timm Block with ATS.

        Args:
            orig_block: The original timm ``Block`` to wrap.
            k_max: Upper bound on sampled patch tokens at this stage.
            use_value_norm: Whether to weight scores by value norm.
            kprime_log_ref: Shared dict (block_idx -> list of K') the parent reads back.
            block_idx: This block's index, used as the log key.
        """
        super().__init__()
        self.norm1 = orig_block.norm1
        self.attn = ATSAttention(orig_block.attn, k_max, use_value_norm)
        self.ls1 = orig_block.ls1
        self.drop_path1 = orig_block.drop_path1
        self.norm2 = orig_block.norm2
        self.mlp = orig_block.mlp
        self.ls2 = orig_block.ls2
        self.drop_path2 = orig_block.drop_path2
        self._kprime_log = kprime_log_ref
        self._block_idx = block_idx

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run attention with ATS, gather the residual, then the MLP sub-block.

        Args:
            x: Token sequence (1, N, C).

        Returns:
            Reduced token sequence (1, K'+1, C).
        """
        o, kept, kprime = self.attn(self.norm1(x))
        x_sampled = x.index_select(1, kept)                          # residual at kept idx
        x = x_sampled + self.drop_path1(self.ls1(o))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        self._kprime_log[self._block_idx].append(kprime)
        return x


class VitATS(nn.Module):
    """DeiT-Tiny with Adaptive Token Sampling at a set of transformer blocks."""

    def __init__(
        self,
        num_classes: int = 100,
        pretrained: bool = False,
        K_max: int = 196,
        ats_stages: Tuple[int, ...] = (2, 3, 4, 5, 6, 7, 8, 9, 10),
        use_value_norm: bool = True,
    ) -> None:
        """Initialize the ATS model.

        Args:
            num_classes: Number of output classes.
            pretrained: Load ImageNet-pretrained timm weights (kept False; weights come
                from the dense checkpoint via ``load_dense_checkpoint``).
            K_max: Upper bound on tokens kept per ATS stage (realised K' is usually lower).
            ats_stages: 0-indexed block indices at which to apply ATS.
            use_value_norm: Include the value norm in the ATS score (paper default).
        """
        super().__init__()
        base = timm.create_model(
            'deit_tiny_patch16_224', pretrained=pretrained, num_classes=num_classes)
        self.patch_embed = base.patch_embed
        self.cls_token = base.cls_token
        self.pos_embed = base.pos_embed
        self.pos_drop = base.pos_drop
        self.blocks = base.blocks
        self.norm = base.norm
        self.head = base.head

        self.ats_stages = tuple(ats_stages)
        self.K_max = K_max
        self.use_value_norm = use_value_norm
        self.model_name = 'vit_ats'

        self._kprime_log: Dict[int, List[int]] = {i: [] for i in self.ats_stages}
        for i in self.ats_stages:
            self.blocks[i] = ATSBlock(
                self.blocks[i], K_max, use_value_norm, self._kprime_log, i)

    def reset_kprime_log(self) -> None:
        """Clear per-stage K' tracking from a previous forward (in place)."""
        for i in self.ats_stages:
            self._kprime_log[i].clear()

    def get_kprime_log(self) -> List[torch.Tensor]:
        """Per-stage K' for each image of the last forward.

        Returns:
            One (B,) long tensor per ATS stage, in ``ats_stages`` order.
        """
        return [torch.tensor(self._kprime_log[i], dtype=torch.long) for i in self.ats_stages]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standard classification forward.

        The shared prefix (blocks before the first ATS stage) runs batched; the
        remaining blocks run per image because token counts diverge after sampling.

        Args:
            x: Input images (B, 3, 224, 224).

        Returns:
            Logits (B, num_classes).
        """
        self.reset_kprime_log()
        B = x.shape[0]
        x = self.patch_embed(x)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls, x), dim=1) + self.pos_embed
        x = self.pos_drop(x)

        first_ats = min(self.ats_stages) if self.ats_stages else len(self.blocks)
        for i in range(first_ats):
            x = self.blocks[i](x)                                    # batched prefix

        logits = []
        for b in range(B):
            xb = x[b:b + 1]
            for i in range(first_ats, len(self.blocks)):
                xb = self.blocks[i](xb)                              # per image
            xb = self.norm(xb)
            logits.append(self.head(xb[:, 0]))
        return torch.cat(logits, dim=0)

    def load_dense_checkpoint(self, path: str) -> None:
        """Load a VitDense checkpoint, stripping its ``model.`` key prefix.

        Args:
            path: Path to ``best_model.pt`` saved from VitDense.
        """
        sd = torch.load(path, map_location='cpu')
        stripped: Dict[str, torch.Tensor] = {}
        renamed = 0
        for k, v in sd.items():
            if k.startswith('model.'):
                stripped[k[len('model.'):]] = v
                renamed += 1
            else:
                stripped[k] = v
        print(f"  stripped 'model.' prefix from {renamed}/{len(sd)} keys")
        self.load_state_dict(stripped, strict=True)
        print("  load_state_dict(strict=True) succeeded")

    def get_num_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
