import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from envs.cyber_game_env import CyberGameEnv
from src.train_minimax import train_minimax_step


def test_attack_budget_including_out_of_training_range():
    x = torch.tensor([[-1.5, 0.0, 0.5, 1.0, 2.5]])
    for direction in [-3.0, 0.0, 3.0]:
        attacked = CyberGameEnv(0.05).apply_attack(x, torch.full_like(x, direction))
        assert torch.all((attacked - x).abs() <= 0.050001)
        assert torch.all((attacked[:, 1:4] >= 0) & (attacked[:, 1:4] <= 1))
        if direction == 0:
            assert torch.equal(attacked, x)


def test_benign_only_batch_restores_defender_gradients():
    defender = torch.nn.Sequential(torch.nn.Linear(2, 1), torch.nn.Sigmoid())
    attacker = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Tanh())
    for parameter in defender.parameters():
        parameter.requires_grad = False
    before = [p.detach().clone() for p in defender.parameters()]
    _, loss = train_minimax_step(torch.ones(3, 2), torch.zeros(3, 1), attacker, defender,
                                CyberGameEnv(0.05), torch.optim.AdamW(attacker.parameters()),
                                torch.optim.AdamW(defender.parameters()), torch.nn.BCELoss())
    assert loss > 0
    assert all(p.requires_grad for p in defender.parameters())
    assert any(not torch.equal(old, new) for old, new in zip(before, defender.parameters()))
