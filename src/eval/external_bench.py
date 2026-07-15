"""
Frozen train/dev/test split for external benchmarks (M3.1).

The split is a measurement contract: M3.1 reports zero-shot numbers on the
full query set AND on the held-out test split; M3.2 fine-tuning may train
on `train`, tune on `dev`, and compares against M3.1 ONLY on `test`.
Regenerating the split invalidates that comparison — which is why
scripts/download_zalo.py refuses to overwrite an existing splits.json.
"""
import json
import random
from typing import Dict, List

SPLITS_FILE = "data/benchmarks/zalo/splits.json"
DEFAULT_SEED = 20260715


def make_splits(
    question_ids: List[str],
    seed: int = DEFAULT_SEED,
    ratios=(0.8, 0.1, 0.1),
) -> Dict:
    """Deterministic shuffle-then-slice. Input order must not matter, so ids
    are sorted before shuffling with the seeded RNG."""
    ids = sorted(question_ids)
    random.Random(seed).shuffle(ids)
    n = len(ids)
    n_train = int(n * ratios[0])
    n_dev = int(n * ratios[1])
    return {
        "seed": seed,
        "ratios": list(ratios),
        "train": ids[:n_train],
        "dev": ids[n_train:n_train + n_dev],
        "test": ids[n_train + n_dev:],
    }


def load_splits(path: str = SPLITS_FILE) -> Dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)
