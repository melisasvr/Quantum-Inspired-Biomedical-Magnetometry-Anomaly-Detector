"""
src/detect_quantum.py
---------------------
Quantum-enhanced MEG anomaly detection via NV-center magnetometer simulation.

Core idea
---------
A real NV-center magnetometer's sensitivity is limited by photon shot noise.
Squeezed light states can push noise below the shot-noise limit in one
quadrature at the cost of increased noise in the conjugate quadrature.

This module:
  1. Simulates a squeezed-state NV readout noise model using QuTiP.
  2. Applies the effective noise reduction as a per-channel weight in the
     anomaly scoring step — amplifying sensitivity to sub-threshold neural
     events that the classical detector misses.
  3. Runs a weighted Isolation Forest and returns enhanced anomaly scores.

Usage (CLI):
    python src/detect_quantum.py \
        --input data/processed \
        --squeezing 0.8 \
        --output results/quantum/
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, field

import numpy as np

# QuTiP is imported lazily so the module can be imported even without qutip
# (unit tests mock it out).
try:
    import qutip as qt
    QUTIP_AVAILABLE = True
except ImportError:  # pragma: no cover
    QUTIP_AVAILABLE = False


# ---------------------------------------------------------------------------
# Quantum magnetometer noise model
# ---------------------------------------------------------------------------

@dataclass
class NVMagnetometerParams:
    """Physical parameters of an NV-diamond magnetometer readout."""
    hilbert_dim: int = 60          # Fock-space truncation
    coherent_amplitude: float = 2.0  # |alpha| — mean photon amplitude
    squeezing_r: float = 0.8        # squeezing parameter r (0 = no squeezing)
    detection_efficiency: float = 0.95  # photon collection efficiency η
    dark_count_rate: float = 0.01       # normalized dark counts


@dataclass
class QuantumNoiseResult:
    """Output of the quantum noise simulation."""
    squeezing_r: float
    noise_reduction_dB: float        # dB below shot noise in squeezed quadrature
    sensitivity_gain: float          # linear sensitivity improvement factor
    squeezed_variance: float         # Var(X_squeezed)
    unsqueezed_variance: float       # Var(X_unsqueezed) for reference
    wigner_xvec: np.ndarray = field(repr=False, default_factory=lambda: np.array([]))
    wigner_W: np.ndarray = field(repr=False, default_factory=lambda: np.array([]))


def simulate_nv_squeezed_state(params: NVMagnetometerParams,
                                compute_wigner: bool = True) -> QuantumNoiseResult:
    """
    Simulate squeezed-coherent NV magnetometer readout using QuTiP.

    The squeezed coherent state is:
        |ψ⟩ = D(α) S(r) |0⟩

    where D(α) is the displacement operator and S(r) is the squeeze operator.

    Sensitivity gain is computed as the ratio of shot-noise variance to
    squeezed-quadrature variance, corrected for detection efficiency.

    Parameters
    ----------
    params         : NVMagnetometerParams
    compute_wigner : if True, compute Wigner function (slower, for plots)

    Returns
    -------
    QuantumNoiseResult
    """
    if not QUTIP_AVAILABLE:
        raise ImportError("qutip is required. Install with: pip install qutip")

    N = params.hilbert_dim
    alpha = params.coherent_amplitude
    r = params.squeezing_r
    eta = params.detection_efficiency

    # --- Build squeezed coherent state |α, r⟩ = D(α)S(r)|0⟩ ---
    psi = qt.displace(N, alpha) * qt.squeeze(N, r) * qt.basis(N, 0)

    # --- Quadrature operators ---
    # X = (a + a†) / sqrt(2)   — "amplitude" quadrature
    # P = (a - a†) / (i*sqrt(2)) — "phase" quadrature
    a = qt.destroy(N)
    X = (a + a.dag()) / np.sqrt(2)

    # Variance in squeezed quadrature
    X_expect = qt.expect(X, psi)
    X2_expect = qt.expect(X * X, psi)
    squeezed_var = float(np.real(X2_expect - X_expect ** 2))

    # Shot-noise (coherent state) variance = 0.5 in these units
    shot_noise_var = 0.5

    # Noise reduction (dB below shot noise)
    noise_reduction_dB = -10.0 * np.log10(squeezed_var / shot_noise_var)

    # Detection-efficiency-corrected sensitivity gain
    # η*Var_squeezed + (1-η)*Var_shot vs η*Var_shot + (1-η)*Var_shot
    effective_var = eta * squeezed_var + (1 - eta) * shot_noise_var
    effective_shot = shot_noise_var  # reference
    sensitivity_gain = float(np.sqrt(effective_shot / effective_var))

    print(f"[quantum] Squeezing r={r:.2f} | "
          f"Noise reduction: {noise_reduction_dB:.2f} dB | "
          f"Sensitivity gain: {sensitivity_gain:.3f}×")

    # --- Wigner function (optional, for visualisation) ---
    wigner_xvec = np.array([])
    wigner_W = np.array([])
    if compute_wigner:
        wigner_xvec = np.linspace(-5, 5, 150)
        wigner_W = np.array(qt.wigner(psi, wigner_xvec, wigner_xvec))

    return QuantumNoiseResult(
        squeezing_r=r,
        noise_reduction_dB=noise_reduction_dB,
        sensitivity_gain=sensitivity_gain,
        squeezed_variance=squeezed_var,
        unsqueezed_variance=shot_noise_var,
        wigner_xvec=wigner_xvec,
        wigner_W=wigner_W,
    )


# ---------------------------------------------------------------------------
# Quantum-enhanced anomaly scoring
# ---------------------------------------------------------------------------

def quantum_enhanced_scores(features: np.ndarray,
                              sensitivity_gain: float,
                              contamination: float = 0.05,
                              random_state: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply quantum sensitivity gain to feature space before anomaly detection.

    Strategy:
      - Channels with higher signal variance benefit most from squeezing
        (noise floor reduction unmasks sub-threshold anomalies).
      - We compute per-feature weights proportional to signal SNR and scale
        features by sensitivity_gain before Isolation Forest scoring.

    Parameters
    ----------
    features         : (n_epochs, n_features) — raw feature matrix
    sensitivity_gain : linear sensitivity improvement from quantum simulation
    contamination    : expected anomaly fraction

    Returns
    -------
    labels : (n_epochs,) — 1 = anomaly, 0 = normal
    scores : (n_epochs,) — higher = more anomalous
    """
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X = scaler.fit_transform(features)

    # Feature-wise weighting: amplify features whose variance is relatively low
    # (these are the channels where the quantum noise floor reduction matters most)
    feature_snr = np.abs(X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
    # Normalise weights so mean = 1, then boost by sensitivity_gain
    weights = sensitivity_gain * (feature_snr / (feature_snr.mean() + 1e-8))
    weights = np.clip(weights, 0.5, sensitivity_gain * 2)

    X_enhanced = X * weights[np.newaxis, :]

    clf = IsolationForest(contamination=contamination, random_state=random_state, n_jobs=-1)
    clf.fit(X_enhanced)

    raw = clf.decision_function(X_enhanced)
    scores = -raw
    labels = (clf.predict(X_enhanced) == -1).astype(int)

    print(f"[quantum] Enhanced detection: {labels.sum()} anomalies "
          f"({100*labels.mean():.1f}%) out of {len(labels)} epochs")
    return labels, scores


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def detect_quantum(input_dir: str, squeezing: float = 0.8,
                    contamination: float = 0.05,
                    output_dir: str = "results/quantum") -> dict:
    """Full quantum-enhanced detection pipeline."""
    os.makedirs(output_dir, exist_ok=True)

    features = np.load(os.path.join(input_dir, "features.npy"))
    print(f"[quantum] Loaded features: {features.shape}")

    # 1. Simulate NV magnetometer with squeezing
    params = NVMagnetometerParams(squeezing_r=squeezing)
    qresult = simulate_nv_squeezed_state(params, compute_wigner=True)

    # Save Wigner data for visualisation
    np.save(os.path.join(output_dir, "wigner_xvec.npy"), qresult.wigner_xvec)
    np.save(os.path.join(output_dir, "wigner_W.npy"), qresult.wigner_W)

    # 2. Quantum-enhanced anomaly scoring
    labels, scores = quantum_enhanced_scores(
        features, qresult.sensitivity_gain, contamination)

    # 3. Save results
    np.save(os.path.join(output_dir, "labels.npy"), labels)
    np.save(os.path.join(output_dir, "scores.npy"), scores)

    gt_path = os.path.join(input_dir, "ground_truth.npy")
    metrics = {
        "squeezing_r": squeezing,
        "noise_reduction_dB": qresult.noise_reduction_dB,
        "sensitivity_gain": qresult.sensitivity_gain,
    }

    if os.path.exists(gt_path):
        from sklearn.metrics import roc_auc_score, average_precision_score
        gt_epoch_path = os.path.join(input_dir, "..", "classical", "gt_labels_epoch.npy")
        # Try to reuse epoch-level GT computed by classical step
        if os.path.exists(gt_epoch_path):
            gt_labels = np.load(gt_epoch_path)
        else:
            ground_truth = np.load(gt_path)
            epochs = np.load(os.path.join(input_dir, "epochs.npy"))
            epoch_len = epochs.shape[2]
            step = int(epoch_len * 0.5)
            gt_labels = np.zeros(len(epochs), dtype=int)
            for i in range(len(epochs)):
                start = i * step
                end = start + epoch_len
                if any(start <= t < end for t in ground_truth):
                    gt_labels[i] = 1

        if gt_labels.sum() > 0:
            auc = roc_auc_score(gt_labels, scores)
            ap = average_precision_score(gt_labels, scores)
            metrics.update({"roc_auc": float(auc), "avg_precision": float(ap)})
            print(f"[quantum] ROC-AUC={auc:.3f}  AvgPrecision={ap:.3f}")

    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"[quantum] Results saved → {output_dir}")
    return {"labels": labels, "scores": scores, "metrics": metrics, "qresult": qresult}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/processed")
    parser.add_argument("--squeezing", type=float, default=0.8)
    parser.add_argument("--contamination", type=float, default=0.05)
    parser.add_argument("--output", default="results/quantum")
    args = parser.parse_args()
    detect_quantum(args.input, args.squeezing, args.contamination, args.output)