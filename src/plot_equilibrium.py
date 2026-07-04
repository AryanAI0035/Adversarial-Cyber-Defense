import os
import sys
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models.attacker_policy import AttackerPolicy
from data.data_loader import get_dataloaders

# Setup sleek, professional visualization styling
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'legend.fontsize': 12,
    'figure.titlesize': 18
})

def plot_minimax_convergence():
    history_file = "outputs/training_history.json"
    if not os.path.exists(history_file):
        print(f"Error: {history_file} not found. Please run train_minimax.py first.")
        return

    with open(history_file, "r") as f:
        history = json.loads(f.read())
        
    epochs = range(1, len(history["attacker_loss"]) + 1)
    
    fig, ax1 = plt.subplots(figsize=(10, 6))

    color = '#1f77b4'
    ax1.set_xlabel('Training Epoch')
    ax1.set_ylabel('Attacker Regret (Loss)', color=color, fontweight='bold')
    line1 = ax1.plot(epochs, history["attacker_loss"], color=color, linewidth=2.5, marker='o', label='Attacker (Maximize)')
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  
    color = '#ff7f0e'
    ax2.set_ylabel('Defender Risk (Loss)', color=color, fontweight='bold')
    line2 = ax2.plot(epochs, history["defender_loss"], color=color, linewidth=2.5, marker='s', label='Defender (Minimize)')
    ax2.tick_params(axis='y', labelcolor=color)

    # Add a title and equilibrium annotation
    plt.title("Minimax Zero-Sum Game Optimization Dynamics", pad=20)
    
    # Legend
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='center right', frameon=True, shadow=True)

    fig.tight_layout()  
    
    os.makedirs("outputs/figures", exist_ok=True)
    save_path = "outputs/figures/equilibrium_convergence.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Convergence plot saved to: {save_path}")

def plot_radar_chart():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, test_loader, feature_dim = get_dataloaders(batch_size=512)
    
    attacker = AttackerPolicy(feature_dim).to(device)
    if not os.path.exists("checkpoints/attacker.pt"):
        print("Error: checkpoints/attacker.pt not found. Run training first.")
        return
    attacker.load_state_dict(torch.load("checkpoints/attacker.pt", map_location=device))
    attacker.eval()

    # Get one batch of malicious traffic
    sample_traffic = None
    for batch_x, batch_y in test_loader:
        malicious_mask = (batch_y == 1.0).flatten()
        if malicious_mask.sum() > 0:
            sample_traffic = batch_x[malicious_mask].to(device)
            break
            
    if sample_traffic is None: return

    # Generate perturbations
    with torch.no_grad():
        perturbations = attacker(sample_traffic).cpu().numpy()
        
    # We will plot the absolute mean perturbation for 8 key features 
    # (Since plotting all 38 is too cluttered)
    # Using specific indices for standard NSL-KDD continuous features 
    # 0: duration, 4: src_bytes, 5: dst_bytes, 22: count, 23: srv_count, etc.
    key_indices = [0, 4, 5, 22, 23, 31, 32, 33]
    key_labels = ['Duration', 'Src Bytes', 'Dst Bytes', 'Count', 'Srv Count', 'Dst Host Count', 'Dst Srv Count', 'Same Srv Rate']
    
    # Get absolute mean perturbation for these features across the batch
    avg_perturbations = np.abs(perturbations[:, key_indices].mean(axis=0))

    # Math to setup radar chart
    N = len(key_labels)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    
    avg_perturbations = np.concatenate((avg_perturbations, [avg_perturbations[0]]))

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    
    plt.xticks(angles[:-1], key_labels, size=12, fontweight='bold')
    
    ax.plot(angles, avg_perturbations, linewidth=2, linestyle='solid', color='#d62728')
    ax.fill(angles, avg_perturbations, '#d62728', alpha=0.25)
    
    plt.title("Attacker Policy: Feature Perturbation Strategy", size=18, y=1.1)
    
    save_path = "outputs/figures/feature_perturbation_radar.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Radar chart saved to: {save_path}")

if __name__ == "__main__":
    plot_minimax_convergence()
    plot_radar_chart()
