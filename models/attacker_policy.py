import torch
import torch.nn as nn

# ==============================================================================
# THEORY: The Attacker Network (A_phi)
# ------------------------------------------------------------------------------
# In Game Theory, the Attacker is trying to MAXIMIZE the classification error.
# It takes a malicious network packet that was caught by the Defender, and it 
# learns how to subtly alter the packet's features so it "sneaks past" the Defender.
#
# But wait! A hacker can't just change a packet infinitely, otherwise the connection 
# drops and the hack fails. They have a "Budget". 
# 
# We enforce this mathematically using the Tanh() activation function. Tanh squashes 
# the neural network's output to exactly [-1, 1]. We then multiply this by an 
# Epsilon (epsilon) value to strictly bound the maximum perturbation.
# ==============================================================================

class AttackerPolicy(nn.Module):
    def __init__(self, input_dim: int):
        super(AttackerPolicy, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            
            nn.Linear(128, 64),
            nn.ReLU(),
            
            # The output layer matches the input dimension! 
            # It generates a custom perturbation for EVERY single feature.
            nn.Linear(64, input_dim),
            
            # Tanh bounds the perturbations strictly between -1 and 1
            nn.Tanh() 
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns the raw normalized perturbations in the range [-1, 1].
        The environment will scale this by epsilon.
        """
        return self.network(x)

if __name__ == "__main__":
    # Smoke test
    model = AttackerPolicy(input_dim=38)
    dummy_traffic = torch.randn(10, 38)
    perturbations = model(dummy_traffic)
    print("Attacker Output Shape:", perturbations.shape)
    print("Min val:", perturbations.min().item(), "| Max val:", perturbations.max().item())
