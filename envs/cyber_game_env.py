import torch

# ==============================================================================
# THEORY: Cyber Game Environment
# ------------------------------------------------------------------------------
# In a Reinforcement Learning setup, we usually have an "Environment" (like a 
# Mario level) where the "Agent" takes actions.
# 
# Since we are using pure PyTorch to optimize our Attacker directly via gradients 
# (which is mathematically much faster than RL for continuous zero-sum games), 
# our "Environment" is simply the physics of how a packet can be changed.
#
# The rules of the game:
# 1. The Attacker outputs a perturbation vector.
# 2. We multiply it by EPSILON (the attacker's budget).
# 3. We add it to the original network log.
# 4. We clamp the result between 0 and 1 so it stays within valid normalized 
#    bounds (a packet can't have negative bytes!).
# ==============================================================================

class CyberGameEnv:
    def __init__(self, epsilon=0.1):
        """
        Args:
            epsilon: The maximum allowed perturbation budget. 
                     0.1 means the attacker can change any feature by up to 10%.
        """
        self.epsilon = epsilon
        
    def apply_attack(self, original_logs: torch.Tensor, perturbations: torch.Tensor) -> torch.Tensor:
        """
        Applies the attacker's mathematically bounded perturbation to the logs.
        """
        # Scale the [-1, 1] Tanh output by the epsilon budget
        scaled_perturbations = perturbations * self.epsilon
        
        # Add the attack to the real network traffic
        attacked_logs = original_logs + scaled_perturbations
        
        # Clamp between 0 and 1 (since our data loader Min-Max normalizes to [0, 1])
        # This ensures the attacker doesn't create mathematically impossible network packets.
        attacked_logs = torch.clamp(attacked_logs, 0.0, 1.0)
        
        return attacked_logs
