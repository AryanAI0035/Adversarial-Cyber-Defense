import torch


class CyberGameEnv:
    """Feature-space perturbations; these do not establish valid network packets."""

    def __init__(self, epsilon=0.1):
        if epsilon < 0:
            raise ValueError("epsilon must be non-negative")
        self.epsilon = epsilon

    def apply_attack(self, original_logs: torch.Tensor, perturbations: torch.Tensor) -> torch.Tensor:
        # Held-out values can exceed training min/max. Preserve their original
        # range rather than moving them to [0, 1] by more than epsilon.
        delta = perturbations.clamp(-1.0, 1.0) * self.epsilon
        lower = torch.minimum(original_logs, torch.zeros_like(original_logs))
        upper = torch.maximum(original_logs, torch.ones_like(original_logs))
        return torch.minimum(torch.maximum(original_logs + delta, lower), upper)
