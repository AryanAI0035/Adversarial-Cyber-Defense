# Adversarial Cyber Defense — Attacker–Defender Training Framework

A PyTorch experiment for binary intrusion detection on NSL-KDD. A learned attacker perturbs numeric features, while a defender trains on clean and perturbed examples using alternating AdamW updates.

## Implementation

- Standard NSL-KDD split: 125,973 training rows and 22,544 test rows.
- 38 numeric features after removing protocol, service and flag. Some retained features are discrete; arbitrary perturbations are not guaranteed to represent valid network traffic.
- Min/max scaling fit on training rows only. Held-out values may exceed the training range.
- Attacker updates target the benign label on malicious inputs; defender updates combine clean classification loss and perturbed malicious classification loss.
- A fixed 15-epoch training budget, saved checkpoints and recorded loss histories.

## Evaluation and limitations

`src/evaluate_robustness.py` reports **attack recall**, not overall accuracy: the fraction of malicious test connections detected at threshold 0.5. It compares clean rows with perturbations from the saved, co-trained attacker. This is not an independent worst-case attack evaluation.

No Nash equilibrium, zero-day robustness, 2% degradation result or production security guarantee has been established. A rise in recall under this attacker does not prove security. Loss curves document training, not equilibrium.

The corrected attack function limits each feature change to epsilon (0.05 in training/evaluation), including held-out values outside [0, 1]. Previously, clipping such values to [0, 1] could violate that budget. Checkpoints have not been retrained; evaluate again with the corrected function. The three local figures have been regenerated with corrected metric labels and feature indices; copies from older versions should not be reused.

The corrected evaluation on the existing checkpoints detects 7927/12833 clean malicious rows (61.77%) and 8997/12833 perturbed rows (70.11%). Zero rows exceed epsilon within a 1e-6 tolerance. Exact counts and checkpoint hashes are recorded in `outputs/corrected_evaluation.json`.

## Run

From this directory:

```bash
pip install -r requirements.txt
python src/train_minimax.py
python src/evaluate_robustness.py
python src/plot_equilibrium.py
```

Training downloads NSL-KDD if absent and overwrites the saved checkpoints. Evaluation reads existing checkpoints and writes a recall chart; plotting writes training-loss and perturbation figures.

## Regression checks

```bash
python -m pytest tests -q
```
