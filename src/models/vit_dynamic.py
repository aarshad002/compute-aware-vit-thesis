import timm
import torch
import torch.nn as nn

class BudgetController(nn.Module):
    def __init__(self, input_dim=8, hidden_dim=32, num_budgets=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_budgets)
        )

    def forward(self, x):
        return self.net(x)


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
        self.controller_enabled = controller_cfg.get("enabled", False)

        controller_hidden_dim = controller_cfg.get("hidden_dim", 32)

        self.controller = BudgetController(
            input_dim=8,
            hidden_dim=controller_hidden_dim,
            num_budgets=len(self.budget_options)
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

    def predict_keep_ratio(self, controller_features):
        budget_logits = self.controller(controller_features)   # (B, num_budgets)
        budget_probs = torch.softmax(budget_logits, dim=1)     # (B, num_budgets)

        budget_indices = torch.argmax(budget_probs, dim=1)     # (B,)

        if controller_features.size(0) != 1:
            raise ValueError(
                "Current controller-enabled dynamic prototype supports only batch_size=1."
            )

        chosen_keep_ratio = self.budget_options[budget_indices[0].item()]
        expected_keep_ratio = (
            budget_probs
            * torch.tensor(
                self.budget_options,
                device=controller_features.device,
                dtype=budget_probs.dtype,
            ).unsqueeze(0)
        ).sum(dim=1)   # (B,)

        return chosen_keep_ratio, expected_keep_ratio, budget_logits, budget_probs, budget_indices
    
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
    
    
    def compute_controller_features(self, token_scores):
        """
        token_scores: (B, N)

        returns:
            features: (B, F)
        """
        mean_score = token_scores.mean(dim=1, keepdim=True)
        std_score = token_scores.std(dim=1, keepdim=True)
        max_score = token_scores.max(dim=1, keepdim=True).values
        min_score = token_scores.min(dim=1, keepdim=True).values

        top2_scores, _ = torch.topk(token_scores, k=2, dim=1)
        top1 = top2_scores[:, 0:1]
        top2 = top2_scores[:, 1:2]
        margin = top1 - top2

        probs = torch.softmax(token_scores, dim=1)
        entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=1, keepdim=True)

        features = torch.cat(
            [mean_score, std_score, max_score, min_score, top1, top2, margin, entropy],
            dim=1,
        )

        return features   
        
    def forward(self, x, return_debug=False, return_controller_info=False):
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

        # Compute controller features (for future use in adaptive pruning)
        controller_features = self.compute_controller_features(token_scores)
        if self.controller_enabled:
            chosen_keep_ratio, expected_keep_ratio, budget_logits, budget_probs, budget_indices = \
                self.predict_keep_ratio(controller_features)
            keep_ratio = chosen_keep_ratio
        else:
            keep_ratio = self.keep_ratio
            expected_keep_ratio = torch.tensor([self.keep_ratio], device=x.device, dtype=x.dtype)
            budget_logits = None
            budget_probs = None
            budget_indices = None
        #print("controller_features shape:", controller_features.shape)
        #print("budget_logits shape:", budget_logits.shape)
        #print("budget_indices shape:", budget_indices.shape)
        #print("predicted keep_ratio:", keep_ratio)
        
        

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

        if return_debug:
            return {
                "logits": logits,
                "keep_ratio": keep_ratio,
                "expected_keep_ratio": expected_keep_ratio,
                "budget_logits": budget_logits,
                "budget_probs": budget_probs,
                "budget_indices": budget_indices,
                "token_scores": token_scores,
                "controller_features": controller_features,
            }

        if return_controller_info:
            return {
                "logits": logits,
                "keep_ratio": keep_ratio,
                "expected_keep_ratio": expected_keep_ratio,
                "budget_logits": budget_logits,
                "budget_probs": budget_probs,
                "budget_indices": budget_indices,
            }

def build_dynamic_model(config):
    return DynamicPrunedViT(config)