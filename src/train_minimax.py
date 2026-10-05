import os
import sys
import json
import torch
import torch.nn as nn
import torch.optim as optim
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models.defender_nn import DefenderNet
from models.attacker_policy import AttackerPolicy
from envs.cyber_game_env import CyberGameEnv
from data.data_loader import get_dataloaders

# Alternating attacker/defender training. A fixed epoch budget does not prove equilibrium.

def print_header(feat_dim):
    print("="*75)
    print("  ADVERSARIAL MINIMAX FRAMEWORK FOR AUTONOMOUS CYBER DEFENSE")
    print("="*75)
    print(f"  System Parameters:")
    print(f"  - Feature Space    : {feat_dim} Dimensions (NSL-KDD)")
    print(f"  - Topology         : Learned attacker / binary defender")
    print(f"  - Training Budget  : 15 epochs; no equilibrium guarantee")
    print(f"  - Attack Budget    : \u03b5 = 0.05 (L-inf Norm Bound)")
    print("="*75)
    print("  Starting Alternating Optimization Loop...")
    print(f"  {'Epoch':<8} | {'Attacker BCE Loss':<18} | {'Defender BCE Loss':<18} | {'Time (s)':<10}")
    print("-" * 75)

def train_minimax_step(
    batch_features: torch.Tensor, 
    batch_labels: torch.Tensor, 
    attacker: nn.Module, 
    defender: nn.Module, 
    env: CyberGameEnv,
    opt_attacker: optim.Optimizer, 
    opt_defender: optim.Optimizer, 
    criterion: nn.Module
):
    malicious_mask = (batch_labels == 1.0).flatten()
    real_malicious_logs = batch_features[malicious_mask]
    
    if real_malicious_logs.size(0) == 0:
        for param in defender.parameters():
            param.requires_grad = True
        opt_defender.zero_grad()
        loss = criterion(defender(batch_features), batch_labels)
        loss.backward()
        opt_defender.step()
        return 0.0, loss.item()

    batch_size = real_malicious_logs.size(0)
    target_malicious = torch.ones(batch_size, 1, device=batch_features.device)
    target_evaded = torch.zeros(batch_size, 1, device=batch_features.device) 

    # --------------------------------------------------------------------------
    # PHASE 1: Optimize Attacker (Maximize Defender's Error)
    # --------------------------------------------------------------------------
    for param in defender.parameters(): param.requires_grad = False
    for param in attacker.parameters(): param.requires_grad = True

    opt_attacker.zero_grad()
    perturbations = attacker(real_malicious_logs)
    attacked_logs = env.apply_attack(real_malicious_logs, perturbations)
    defender_output = defender(attacked_logs)
    attacker_loss = criterion(defender_output, target_evaded)
    attacker_loss.backward()
    opt_attacker.step()

    # --------------------------------------------------------------------------
    # PHASE 2: Optimize Defender (Minimize Error)
    # --------------------------------------------------------------------------
    for param in defender.parameters(): param.requires_grad = True
    for param in attacker.parameters(): param.requires_grad = False

    opt_defender.zero_grad()
    loss_standard = criterion(defender(batch_features), batch_labels)
    
    with torch.no_grad():
        worst_case_perturbations = attacker(real_malicious_logs)
        worst_case_logs = env.apply_attack(real_malicious_logs, worst_case_perturbations)
        
    outputs_perturbed = defender(worst_case_logs)
    loss_robust = criterion(outputs_perturbed, target_malicious)
    
    defender_loss = (loss_standard + loss_robust) / 2
    defender_loss.backward()
    opt_defender.step()

    return attacker_loss.item(), defender_loss.item()


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    train_loader, test_loader, feature_dim = get_dataloaders(batch_size=512)

    defender = DefenderNet(feature_dim).to(device)
    attacker = AttackerPolicy(feature_dim).to(device)
    env = CyberGameEnv(epsilon=0.05) 

    opt_defender = optim.AdamW(defender.parameters(), lr=1e-3, weight_decay=1e-4)
    opt_attacker = optim.AdamW(attacker.parameters(), lr=2e-3, weight_decay=1e-4) 
    criterion = nn.BCELoss()

    EPOCHS = 15
    print_header(feature_dim)
    
    history = {"attacker_loss": [], "defender_loss": []}

    for epoch in range(1, EPOCHS + 1):
        epoch_start = time.time()
        defender.train()
        attacker.train()
        
        running_a_loss = 0.0
        running_d_loss = 0.0
        batches = 0

        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            a_loss, d_loss = train_minimax_step(
                batch_x, batch_y, attacker, defender, env, opt_attacker, opt_defender, criterion
            )
            running_a_loss += a_loss
            running_d_loss += d_loss
            batches += 1

        avg_a = running_a_loss / batches
        avg_d = running_d_loss / batches
        epoch_time = time.time() - epoch_start
        
        history["attacker_loss"].append(avg_a)
        history["defender_loss"].append(avg_d)

        print(f"  {epoch:<8d} | {avg_a:<18.4f} | {avg_d:<18.4f} | {epoch_time:<10.2f}")

    # Save models and history
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)
    
    torch.save(defender.state_dict(), "checkpoints/defender.pt")
    torch.save(attacker.state_dict(), "checkpoints/attacker.pt")
    
    with open("outputs/training_history.json", "w") as f:
        json.dump(history, f)
        
    print("="*75)
    print("  Training budget complete. Models saved to checkpoints/")
    print("="*75)

if __name__ == "__main__":
    train()
