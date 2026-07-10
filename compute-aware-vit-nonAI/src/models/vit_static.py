import timm
import torch
import torch.nn as nn


class StaticPrunedViT(nn.Module):
    def __init__(
        self,
        model_name: str,
        num_classes: int,
        pretrained: bool = True,
        prune_layer: int = 6,
        keep_tokens: int = 128,
        score_method: str = "l2",
    ):
        super().__init__()

        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=num_classes,
        )

        self.prune_layer = prune_layer
        self.keep_tokens = keep_tokens
        self.score_method = score_method

    def score_tokens(self, patch_tokens: torch.Tensor) -> torch.Tensor:
        # patch_tokens: [B, N, C]
        if self.score_method == "l2":
            scores = torch.norm(patch_tokens, dim=-1)   # [B, N]
        else:
            raise ValueError(f"Unsupported score_method: {self.score_method}")
        return scores

    def prune_patch_tokens(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 1+N, C], token 0 is CLS
        cls_token = x[:, :1, :]      # [B, 1, C]
        patch_tokens = x[:, 1:, :]   # [B, N, C]

        scores = self.score_tokens(patch_tokens)  # [B, N]

        k = min(self.keep_tokens, patch_tokens.shape[1])

        topk_indices = torch.topk(scores, k=k, dim=1).indices
        topk_indices, _ = torch.sort(topk_indices, dim=1)

        gather_index = topk_indices.unsqueeze(-1).expand(-1, -1, patch_tokens.shape[-1])
        kept_patch_tokens = torch.gather(patch_tokens, dim=1, index=gather_index)

        x_pruned = torch.cat([cls_token, kept_patch_tokens], dim=1)  # [B, 1+k, C]
        return x_pruned

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        model = self.backbone

        x = model.patch_embed(x)

        cls_token = model.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)

        x = x + model.pos_embed
        x = model.pos_drop(x)

        for i, block in enumerate(model.blocks):
            x = block(x)

            if i + 1 == self.prune_layer:
                x = self.prune_patch_tokens(x)

        x = model.norm(x)
        cls_out = x[:, 0]
        x = model.head(cls_out)

        return x


def build_static_model(config):
    model_name = config["model"]["name"]
    num_classes = config["model"]["num_classes"]
    pretrained = config["model"].get("pretrained", True)

    prune_cfg = config["pruning"]

    model = StaticPrunedViT(
        model_name=model_name,
        num_classes=num_classes,
        pretrained=pretrained,
        prune_layer=prune_cfg.get("prune_layer", 6),
        keep_tokens=prune_cfg.get("keep_tokens", 128),
        score_method=prune_cfg.get("score_method", "l2"),
    )

    return model