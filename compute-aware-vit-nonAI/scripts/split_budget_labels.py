import json
import random
from pathlib import Path

random.seed(42)

ROOT = Path(__file__).resolve().parents[1]

with open(ROOT / "data" / "budget_labels_val.json") as f:
    labels = json.load(f)

# labels keys are strings — convert to list of (idx, label) pairs
items = [(int(k), int(v)) for k, v in labels.items()]
random.shuffle(items)

split = int(0.8 * len(items))
train_items = items[:split]
val_items   = items[split:]

train_labels = {str(k): v for k, v in train_items}
val_labels   = {str(k): v for k, v in val_items}

with open(ROOT / "data" / "budget_labels_ctrl_train.json", "w") as f:
    json.dump(train_labels, f, indent=2)

with open(ROOT / "data" / "budget_labels_ctrl_val.json", "w") as f:
    json.dump(val_labels, f, indent=2)

from collections import Counter
print(f"Controller train: {len(train_labels)} samples")
print(f"  Distribution: {dict(sorted(Counter(train_labels.values()).items()))}")
print(f"Controller val:   {len(val_labels)} samples")
print(f"  Distribution: {dict(sorted(Counter(val_labels.values()).items()))}")