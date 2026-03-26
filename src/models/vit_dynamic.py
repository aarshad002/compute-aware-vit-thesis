import timm
import torch
import torch.nn as nn


class DynamicPrunedViT(nn.Module):
    def __init__(self, config):
        super().__init__()

        model_cfg = config["model"]
        pruning_cfg = config.get("pruning", {})
        controller_cfg = config.get("controller", {})

        self.model_name = model_cfg["name"]
        self.num_classes = model_cfg["num_classes"]
        self.pretrained = model_cfg.get("pretrained", True)

        self.prune_layer = pruning_cfg.get("prune_layer", 6)
        self.score_method = pruning_cfg.get("score_method", "l2")

        self.budget_options = controller_cfg.get(
            "budget_options", [0.25, 0.50, 0.75, 1.0]
        )

        self.backbone = timm.create_model(
            self.model_name,
            pretrained=self.pretrained,
            num_classes=self.num_classes,
        )

    def compute_token_scores(self, patch_tokens):
        """
        patch_tokens: Tensor of shape (B, N, D)
        returns: Tensor of shape (B, N)
        """
        if self.score_method == "l2":
            return torch.norm(patch_tokens, dim=-1)

        else:
            raise ValueError(f"Unsupported score method: {self.score_method}")

    def forward(self, x):
        """
        Temporary forward:
        for now just run dense backbone normally.
        Later we will replace this with:
        - run up to prune_layer
        - compute scores
        - choose budget
        - prune tokens
        - continue forward
        """
        return self.backbone(x)


def build_dynamic_model(config):
    return DynamicPrunedViT(config)