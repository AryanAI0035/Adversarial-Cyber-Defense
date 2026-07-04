import torch
import torch.nn as nn

# ==============================================================================
# THEORY: The Defender Network (D_theta)
# ------------------------------------------------------------------------------
# In Game Theory, the Defender is trying to MINIMIZE the classification error.
# It receives a network packet (a vector of 38 numbers) and must output a single 
# probability between 0 and 1.
# - Output near 0 = "This is safe, normal traffic."
# - Output near 1 = "RED ALERT! This is a malicious intrusion."
# 
# We use a standard deep neural network (Multi-Layer Perceptron) with ReLU 
# activations. The final layer is a Sigmoid function, which perfectly maps the 
# output to a probability [0, 1].
# ==============================================================================

class DefenderNet(nn.Module):
    def __init__(self, input_dim: int):
        super(DefenderNet, self).__init__()
        
        # Deep Residual-style structure without the skip connections to keep it simple
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2), # Dropout prevents overfitting!
            
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(64, 32),
            nn.ReLU(),
            
            nn.Linear(32, 1),
            nn.Sigmoid() # Outputs probability of being Malicious (1)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

if __name__ == "__main__":
    # Smoke test
    model = DefenderNet(input_dim=38)
    dummy_traffic = torch.randn(10, 38)
    print("Defender Output Shape:", model(dummy_traffic).shape)
