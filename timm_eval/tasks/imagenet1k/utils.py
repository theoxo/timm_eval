from typing import Dict

import torch


def process_results(logits: torch.Tensor, label: int) -> Dict[str, float]:
    _, top_five_logit_idxs = torch.topk(logits, 5)
    results = {
        "top1": float(label == top_five_logit_idxs[0]),
        "top5": float(label in top_five_logit_idxs),
    }
    return results


def error_unimplemented_path() -> None:
    raise ValueError("Missing path!")
