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
        self.keep_ratio = pruning_cfg.get("keep_ratio", 0.5)
        
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

    def select_topk_tokens(self, patch_tokens, token_scores, keep_ratio):
        """
        patch_tokens: (B, N, D)
        token_scores: (B, N)
        keep_ratio: float, e.g. 0.5

        returns:
            selected_tokens: (B, K, D)
            selected_scores: (B, K)
            selected_indices: (B, K)
        """
        B, N, D = patch_tokens.shape
        K = max(1, int(N * keep_ratio))

        selected_scores, selected_indices = torch.topk(
            token_scores, k=K, dim=1, largest=True, sorted=True
        )

        gather_indices = selected_indices.unsqueeze(-1).expand(-1, -1, D)
        selected_tokens = torch.gather(patch_tokens, dim=1, index=gather_indices)

        return selected_tokens, selected_scores, selected_indices
        
    
    def forward(self, x):
        # Patch embedding
        x = self.backbone.patch_embed(x)   # (B, N, D)
        B, N, D = x.shape

        # Add CLS token
        cls_tokens = self.backbone.cls_token.expand(B, -1, -1)   # (B, 1, D)
        x = torch.cat((cls_tokens, x), dim=1)                    # (B, 1+N, D)

        # Add positional embeddings
        x = x + self.backbone.pos_embed
        x = self.backbone.pos_drop(x)

        # Run transformer blocks up to prune_layer
        for i, blk in enumerate(self.backbone.blocks):
            x = blk(x)
            if i + 1 == self.prune_layer:
                break

        # Split CLS and patch tokens
        cls_token = x[:, :1, :]        # (B, 1, D)
        patch_tokens = x[:, 1:, :]     # (B, N, D)

        # Compute token scores
        token_scores = self.compute_token_scores(patch_tokens)   # (B, N)

        # Temporary fixed ratio for testing
        keep_ratio = self.keep_ratio

        selected_tokens, selected_scores, selected_indices = self.select_topk_tokens(
            patch_tokens, token_scores, keep_ratio
        )

        # Rebuild reduced sequence: CLS + selected patch tokens
        x = torch.cat((cls_token, selected_tokens), dim=1)   # (B, 1+K, D)

        # Continue remaining transformer blocks
        for i, blk in enumerate(self.backbone.blocks):
            if i + 1 > self.prune_layer:
                x = blk(x)

        # Final norm
        x = self.backbone.norm(x)

        # CLS token for classification
        cls_output = x[:, 0]   # (B, D)

        # Classifier head
        logits = self.backbone.head(cls_output)   # (B, num_classes)

        return logits

def build_dynamic_model(config):
    return DynamicPrunedViT(config)