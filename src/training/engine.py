import torch
import torch.nn as nn

def train_one_epoch(model, loader, criterion, optimizer, device, controller_loss_weight=0.01):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    budget_counts = None
    expected_keep_ratio_sum = 0.0
    expected_keep_ratio_count = 0

    for batch in loader:
        if len(batch) == 3:
            images, labels, _ = batch
        else:
            images, labels = batch

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        if getattr(model, "controller_enabled", False):
            outputs = model(images, return_controller_info=True)
            logits          = outputs["logits"]
            expected_keep_ratio = outputs["expected_keep_ratio"]
            budget_indices  = outputs["budget_indices"]
            budget_logits   = outputs["budget_logits"]

            if budget_counts is None:
                num_budgets = budget_logits.shape[1]
                budget_counts = [0] * num_budgets

            for idx in budget_indices.detach().cpu().tolist():
                budget_counts[idx] += 1

            expected_keep_ratio_sum  += expected_keep_ratio.mean().item()
            expected_keep_ratio_count += 1

            cls_loss       = criterion(logits, labels)
            budget_penalty = expected_keep_ratio.mean()

            # FIX 1: use the argument, not a hardcoded 0.01
            # FIX 2: budget_penalty is now differentiable after Gumbel fix
            #         in vit_dynamic.py — gradient flows to controller
            loss = cls_loss + controller_loss_weight * budget_penalty

        else:
            logits = model(images)
            loss   = criterion(logits, labels)

        loss.backward()

        # FIX 3: clip gradients — critical with batch_size=1
        # noisy single-sample gradients can cause huge destructive updates
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds   = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total   += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc  = correct / total

    avg_expected_keep_ratio = (
        expected_keep_ratio_sum / expected_keep_ratio_count
        if expected_keep_ratio_count > 0 else None
    )

    return epoch_loss, epoch_acc, budget_counts, avg_expected_keep_ratio


@torch.no_grad()
def validate_one_epoch(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total   = 0

    budget_counts = None
    expected_keep_ratio_sum  = 0.0
    expected_keep_ratio_count = 0

    for batch in loader:
        if len(batch) == 3:
            images, labels, _ = batch
        else:
            images, labels = batch

        images = images.to(device)
        labels = labels.to(device)

        if getattr(model, "controller_enabled", False):
            outputs = model(images, return_controller_info=True)
            logits          = outputs["logits"]
            loss            = criterion(logits, labels)
            expected_keep_ratio = outputs["expected_keep_ratio"]
            budget_indices  = outputs["budget_indices"]
            budget_logits   = outputs["budget_logits"]

            if budget_counts is None:
                num_budgets = budget_logits.shape[1]
                budget_counts = [0] * num_budgets

            for idx in budget_indices.detach().cpu().tolist():
                budget_counts[idx] += 1

            expected_keep_ratio_sum  += expected_keep_ratio.mean().item()
            expected_keep_ratio_count += 1

        else:
            logits = model(images)
            loss   = criterion(logits, labels)

        running_loss += loss.item() * images.size(0)
        preds   = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total   += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc  = correct / total

    avg_expected_keep_ratio = (
        expected_keep_ratio_sum / expected_keep_ratio_count
        if expected_keep_ratio_count > 0 else None
    )

    return epoch_loss, epoch_acc, budget_counts, avg_expected_keep_ratio


def train_controller_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total   = 0
    budget_counts = None

    for batch in loader:
        images, labels, indices, budget_targets = batch
        images         = images.to(device)
        budget_targets = budget_targets.to(device)

        optimizer.zero_grad()

        outputs       = model.forward_controller_only(images)
        budget_logits = outputs["budget_logits"]

        loss = criterion(budget_logits, budget_targets)
        loss.backward()

        # FIX 3 applies here too — supervised path also uses batch_size=32
        # so clipping is less critical but still good practice
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds   = budget_logits.argmax(dim=1)
        correct += (preds == budget_targets).sum().item()
        total   += budget_targets.size(0)

        if budget_counts is None:
            num_budgets = budget_logits.shape[1]
            budget_counts = [0] * num_budgets

        for idx in preds.detach().cpu().tolist():
            budget_counts[idx] += 1

    return running_loss / total, correct / total, budget_counts


@torch.no_grad()
def validate_controller_one_epoch(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total   = 0
    budget_counts = None

    for batch in loader:
        # FIX 4: labels unpacked but not used — harmless, left as-is for consistency
        images, labels, indices, budget_targets = batch
        images         = images.to(device)
        budget_targets = budget_targets.to(device)

        outputs       = model.forward_controller_only(images)
        budget_logits = outputs["budget_logits"]

        loss = criterion(budget_logits, budget_targets)
        running_loss += loss.item() * images.size(0)

        preds   = budget_logits.argmax(dim=1)
        correct += (preds == budget_targets).sum().item()
        total   += budget_targets.size(0)

        if budget_counts is None:
            num_budgets = budget_logits.shape[1]
            budget_counts = [0] * num_budgets

        for idx in preds.detach().cpu().tolist():
            budget_counts[idx] += 1

    return running_loss / total, correct / total, budget_counts