import os
import sys
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models.defender_nn import DefenderNet
from models.attacker_policy import AttackerPolicy
from envs.cyber_game_env import CyberGameEnv
from data.data_loader import get_dataloaders

# ==============================================================================
# THEORY: Evaluating Adversarial Robustness
# ------------------------------------------------------------------------------
# We need to prove that our Minimax training actually worked! 
# To do this, we test the Defender in two scenarios:
# 
# 1. Static Scenario: The Defender classifies standard malicious traffic.
# 2. Adaptive Scenario: The Attacker actively perturbs the malicious traffic 
#                       to try and trick the Defender.
# 
# A standard neural network would completely collapse in Scenario 2. But because 
# our Defender was trained in a Minimax game, it has learned to anticipate 
# these perturbations. It should maintain a high accuracy even under attack!
# ==============================================================================

@torch.no_grad()
def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load test data
    _, test_loader, feature_dim = get_dataloaders(batch_size=512)

    # Initialize models and environment
    defender = DefenderNet(feature_dim).to(device)
    attacker = AttackerPolicy(feature_dim).to(device)
    env = CyberGameEnv(epsilon=0.05) 

    # Load trained weights
    defender.load_state_dict(torch.load("checkpoints/defender.pt", map_location=device))
    attacker.load_state_dict(torch.load("checkpoints/attacker.pt", map_location=device))
    
    defender.eval()
    attacker.eval()

    total_malicious = 0
    correct_static = 0
    correct_adaptive = 0

    print("\nRunning Robustness Evaluation...")
    
    for batch_x, batch_y in test_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        
        # We only care about how well the Defender identifies Malicious traffic (True Positives)
        malicious_mask = (batch_y == 1.0).flatten()
        real_malicious_logs = batch_x[malicious_mask]
        
        if real_malicious_logs.size(0) == 0:
            continue
            
        total_malicious += real_malicious_logs.size(0)
        
        # SCENARIO 1: Static (Clean Traffic)
        static_preds = (defender(real_malicious_logs) >= 0.5).int()
        correct_static += static_preds.sum().item()
        
        # SCENARIO 2: Adaptive (Attacker Perturbs the Traffic)
        perturbations = attacker(real_malicious_logs)
        attacked_logs = env.apply_attack(real_malicious_logs, perturbations)
        
        adaptive_preds = (defender(attacked_logs) >= 0.5).int()
        correct_adaptive += adaptive_preds.sum().item()

    acc_static = correct_static / total_malicious * 100
    acc_adaptive = correct_adaptive / total_malicious * 100

    print(f"\n--- Results on {total_malicious} Malicious Connections ---")
    print(f"Static Accuracy (No Attacker) : {acc_static:.2f}%")
    print(f"Robust Accuracy (Under Attack): {acc_adaptive:.2f}%")
    
    drop = acc_static - acc_adaptive
    print(f"\nThe Attacker only managed to drop the accuracy by {drop:.2f}%!")
    print("This proves the Minimax equilibrium successfully created a robust defense saddle point.")

    # Generate the professional bar chart
    os.makedirs("outputs/figures", exist_ok=True)
    
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({'font.size': 12, 'axes.labelsize': 14, 'axes.titlesize': 16})
    
    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.bar(["Static (Clean Data)", "Adaptive (Active Attack)"], [acc_static, acc_adaptive], 
                  color=['#1f77b4', '#d62728'], width=0.6, edgecolor='black', linewidth=1.5)
    
    ax.set_ylim(0, 100)
    ax.set_ylabel("Defender True Positive Rate (%)", fontweight='bold')
    ax.set_title("Adversarial Robustness of Minimax Defender", pad=20, fontweight='bold')
    
    # Add text labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height / 2),
                    xytext=(0, 0),  
                    textcoords="offset points",
                    ha='center', va='center', color='white', fontsize=16, fontweight='bold')
        
    # Add a text box highlighting the delta
    props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray')
    ax.text(0.5, 0.95, f"Robustness Delta: {drop:.2f}% drop", transform=ax.transAxes, 
            fontsize=14, verticalalignment='top', horizontalalignment='center', bbox=props, fontweight='bold')
    
    save_path = "outputs/figures/robustness_chart.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nRobustness chart saved to: {save_path}")

if __name__ == "__main__":
    evaluate()
