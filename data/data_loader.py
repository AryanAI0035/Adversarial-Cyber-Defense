import os
import urllib.request
import tarfile
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# ==============================================================================
# THEORY: The NSL-KDD Dataset
# ------------------------------------------------------------------------------
# To build a realistic Cyber Defense system, we need real network traffic data.
# The NSL-KDD dataset is one of the most famous datasets in cybersecurity research. 
# It contains millions of network connection records, labeled as either "normal" 
# or as specific types of cyber attacks (like DoS, Probing, U2R, R2L).
#
# Our script automatically downloads the dataset if it doesn't exist, extracts 
# the 38 purely continuous mathematical features (like 'duration', 'src_bytes', 
# 'dst_bytes', etc.), normalizes them, and drops the 3 categorical columns. 
# 
# Why drop the categorical columns? 
# Because our Attacker AI uses gradients to "perturb" the numbers slightly. 
# You can mathematically add 0.05 to 'src_bytes', but you can't mathematically 
# add 0.05 to a string like 'tcp' or 'udp'!
# ==============================================================================

NSL_KDD_URL = "http://205.174.165.80/CICDataset/NSL-KDD/Dataset/NSL-KDD.zip"
# Note: The official URL is sometimes down. Another reliable mirror for the tar.gz:
FALLBACK_URL = "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain%2B.txt"
TEST_URL = "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTest%2B.txt"

# NSL-KDD has 41 features. Columns 1, 2, 3 are categorical (protocol, service, flag).
# Column 41 is the label, and 42 is the difficulty score (which we ignore).
CATEGORICAL_COLS = [1, 2, 3]

def download_dataset(raw_dir: str):
    """Downloads the NSL-KDD dataset if not already present."""
    os.makedirs(raw_dir, exist_ok=True)
    train_path = os.path.join(raw_dir, "KDDTrain+.txt")
    test_path = os.path.join(raw_dir, "KDDTest+.txt")

    if not os.path.exists(train_path):
        print(f"Downloading NSL-KDD Train dataset to {train_path}...")
        urllib.request.urlretrieve(FALLBACK_URL, train_path)
    
    if not os.path.exists(test_path):
        print(f"Downloading NSL-KDD Test dataset to {test_path}...")
        urllib.request.urlretrieve(TEST_URL, test_path)

    return train_path, test_path

def load_and_preprocess(filepath: str, train_mins=None, train_maxs=None):
    """Loads CSV, drops categorical cols, normalizes continuous cols, and encodes labels."""
    # Read the CSV (it has no header)
    df = pd.read_csv(filepath, header=None)
    
    # Extract labels (column 41). Anything that isn't 'normal' is an attack (1).
    labels = (df[41] != 'normal').astype(int).values
    
    # Drop categorical columns and the difficulty score (column 42)
    df = df.drop(columns=CATEGORICAL_COLS + [41, 42])
    
    features = df.values.astype(np.float32)

    # Min-Max Normalization
    if train_mins is None:
        train_mins = np.min(features, axis=0)
        train_maxs = np.max(features, axis=0)
    
    ranges = train_maxs - train_mins
    ranges[ranges == 0] = 1.0  # Avoid division by zero
    features = (features - train_mins) / ranges
    
    return features, labels, train_mins, train_maxs

# ==============================================================================
# CLASS THEORY: PyTorch Dataset
# ------------------------------------------------------------------------------
# We wrap our sanitized data in a standard PyTorch Dataset object so the 
# DataLoader can efficiently pass "batches" of network logs to our GPUs.
# ==============================================================================
class CyberDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

def get_dataloaders(raw_dir="data/raw", batch_size=256):
    train_path, test_path = download_dataset(raw_dir)

    print("Processing Training Data...")
    train_feat, train_labels, mins, maxs = load_and_preprocess(train_path)
    
    print("Processing Testing Data...")
    test_feat, test_labels, _, _ = load_and_preprocess(test_path, train_mins=mins, train_maxs=maxs)

    train_loader = DataLoader(CyberDataset(train_feat, train_labels), batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(CyberDataset(test_feat, test_labels), batch_size=batch_size, shuffle=False)

    print(f"Data Loaders Ready! (Features: {train_feat.shape[1]})")
    print(f"Train samples: {len(train_labels)} | Test samples: {len(test_labels)}")
    
    return train_loader, test_loader, train_feat.shape[1]

if __name__ == "__main__":
    get_dataloaders()
