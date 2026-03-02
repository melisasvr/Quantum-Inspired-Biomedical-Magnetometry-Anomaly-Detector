"""
src/detect_classical.py
-----------------------
Classical anomaly detection on MEG feature vectors.

Methods:
  1. Isolation Forest  (scikit-learn)
  2. Autoencoder       (PyTorch, lightweight MLP)

Usage (CLI):
    python src/detect_classical.py \
        --input data/processed \
        --method isolation_forest \
        --contamination 0.05 \
        --output results/classical/
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Isolation Forest
# ---------------------------------------------------------------------------

def run_isolation_forest(features: np.ndarray,
                          contamination: float = 0.05,
                          random_state: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """
    Fit an Isolation Forest and return anomaly labels + raw scores.

    Parameters
    ----------
    features      : (n_epochs, n_features)
    contamination : expected fraction of anomalies
    random_state  : RNG seed

    Returns
    -------
    labels : (n_epochs,) — 1 = anomaly, 0 = normal
    scores : (n_epochs,) — higher = more anomalous (inverted decision fn)
    """
    scaler = StandardScaler()
    X = scaler.fit_transform(features)

    clf = IsolationForest(contamination=contamination, random_state=random_state, n_jobs=-1)
    clf.fit(X)

    raw = clf.decision_function(X)   # more negative → more anomalous
    scores = -raw                    # flip so higher = more anomalous
    labels = (clf.predict(X) == -1).astype(int)

    print(f"[IsolationForest] Detected {labels.sum()} anomalies "
          f"({100*labels.mean():.1f}%) out of {len(labels)} epochs")
    return labels, scores


# ---------------------------------------------------------------------------
# Autoencoder
# ---------------------------------------------------------------------------

def build_autoencoder(input_dim: int, latent_dim: int = 16):
    """
    Lightweight MLP autoencoder using PyTorch.
    """
    import torch
    import torch.nn as nn

    class Autoencoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, latent_dim),
            )
            self.decoder = nn.Sequential(
                nn.Linear(latent_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 128),
                nn.ReLU(),
                nn.Linear(128, input_dim),
            )

        def forward(self, x):
            return self.decoder(self.encoder(x))

    return Autoencoder()


def run_autoencoder(features: np.ndarray,
                    contamination: float = 0.05,
                    epochs_train: int = 50,
                    latent_dim: int = 16,
                    lr: float = 1e-3) -> tuple[np.ndarray, np.ndarray]:
    """
    Train an autoencoder and detect anomalies by reconstruction error.

    Returns
    -------
    labels : (n_epochs,) — 1 = anomaly
    scores : (n_epochs,) — reconstruction MSE (higher = more anomalous)
    """
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    scaler = StandardScaler()
    X = scaler.fit_transform(features).astype(np.float32)

    tensor_X = torch.tensor(X)
    loader = DataLoader(TensorDataset(tensor_X), batch_size=64, shuffle=True)

    model = build_autoencoder(X.shape[1], latent_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    model.train()
    for epoch in range(epochs_train):
        total_loss = 0.0
        for (batch,) in loader:
            recon = model(batch)
            loss = criterion(recon, batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(batch)
        if (epoch + 1) % 10 == 0:
            print(f"  [AE] Epoch {epoch+1:3d}/{epochs_train}  loss={total_loss/len(X):.6f}")

    model.eval()
    with torch.no_grad():
        recon = model(tensor_X)
        scores = ((tensor_X - recon) ** 2).mean(dim=1).numpy()

    threshold = np.percentile(scores, 100 * (1 - contamination))
    labels = (scores >= threshold).astype(int)

    print(f"[Autoencoder] Detected {labels.sum()} anomalies "
          f"({100*labels.mean():.1f}%) | threshold={threshold:.6f}")
    return labels, scores


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def detect_classical(input_dir: str, method: str = "isolation_forest",
                      contamination: float = 0.05, output_dir: str = "results/classical") -> dict:
    """
    Full classical detection pipeline.
    Loads features from input_dir, runs detection, saves results.
    """
    os.makedirs(output_dir, exist_ok=True)

    features = np.load(os.path.join(input_dir, "features.npy"))
    print(f"[classical] Loaded features: {features.shape}")

    if method == "isolation_forest":
        labels, scores = run_isolation_forest(features, contamination)
    elif method == "autoencoder":
        labels, scores = run_autoencoder(features, contamination)
    else:
        raise ValueError(f"Unknown method: {method}. Choose 'isolation_forest' or 'autoencoder'.")

    # Save results
    np.save(os.path.join(output_dir, "labels.npy"), labels)
    np.save(os.path.join(output_dir, "scores.npy"), scores)

    gt_path = os.path.join(input_dir, "ground_truth.npy")
    metrics = {}
    if os.path.exists(gt_path):
        from sklearn.metrics import roc_auc_score, average_precision_score
        ground_truth = np.load(gt_path)
        # Convert sample-level GT to epoch-level GT
        # An epoch is anomalous if any GT event falls within it
        epochs = np.load(os.path.join(input_dir, "epochs.npy"))
        n_epochs = len(epochs)
        sfreq = 1000.0
        epoch_len = epochs.shape[2]
        step = int(epoch_len * 0.5)
        gt_labels = np.zeros(n_epochs, dtype=int)
        for i in range(n_epochs):
            start = i * step
            end = start + epoch_len
            if any(start <= t < end for t in ground_truth):
                gt_labels[i] = 1

        if gt_labels.sum() > 0:
            auc = roc_auc_score(gt_labels, scores)
            ap = average_precision_score(gt_labels, scores)
            metrics = {"roc_auc": float(auc), "avg_precision": float(ap)}
            print(f"[classical] ROC-AUC={auc:.3f}  AvgPrecision={ap:.3f}")
            np.save(os.path.join(output_dir, "gt_labels_epoch.npy"), gt_labels)

    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump({"method": method, "contamination": contamination, **metrics}, f, indent=2)

    print(f"[classical] Results saved → {output_dir}")
    return {"labels": labels, "scores": scores, "metrics": metrics}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/processed")
    parser.add_argument("--method", default="isolation_forest",
                        choices=["isolation_forest", "autoencoder"])
    parser.add_argument("--contamination", type=float, default=0.05)
    parser.add_argument("--output", default="results/classical")
    args = parser.parse_args()
    detect_classical(args.input, args.method, args.contamination, args.output)