from __future__ import annotations

from typing import Dict, List

import numpy as np
import torch


def integrated_gradients_for_action(
    actor,
    state: np.ndarray,
    action: int,
    steps: int = 32,
    baseline: np.ndarray | None = None,
    device: str = "cpu",
) -> np.ndarray:
    """Integrated-gradients style attribution for a chosen action logit."""
    actor.eval()

    if baseline is None:
        baseline = np.zeros_like(state, dtype=np.float32)

    x = torch.tensor(state, dtype=torch.float32, device=device)
    b = torch.tensor(baseline, dtype=torch.float32, device=device)
    delta = x - b

    acc_grad = torch.zeros_like(x)

    for k in range(1, steps + 1):
        alpha = float(k) / float(steps)
        xk = (b + alpha * delta).clone().detach().requires_grad_(True)
        logits = actor(xk.unsqueeze(0)).squeeze(0)
        score = logits[action]
        score.backward()
        acc_grad += xk.grad.detach()

    avg_grad = acc_grad / float(steps)
    attribution = (delta * avg_grad).detach().cpu().numpy()
    return attribution.astype(np.float32)


def summarize_attributions(
    actor,
    state_samples: List[np.ndarray],
    actions: List[int],
    feature_names: List[str],
    device: str = "cpu",
) -> Dict[str, float]:
    totals = np.zeros(len(feature_names), dtype=np.float64)

    for s, a in zip(state_samples, actions):
        attr = integrated_gradients_for_action(actor, s, a, device=device)
        totals += np.abs(attr)

    if len(state_samples) > 0:
        totals /= float(len(state_samples))

    return {name: float(value) for name, value in zip(feature_names, totals)}
