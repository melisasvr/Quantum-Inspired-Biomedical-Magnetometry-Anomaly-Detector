"""
src/visualize.py
----------------
Generate all plots:
  1. 2D brain anomaly heatmap (topomap-style sensor grid)
  2. Temporal anomaly time-series with flagged events
  3. Classical vs. Quantum ROC / sensitivity comparison
  4. Wigner function + squeezing noise-reduction curve
  5. Edge filter compression overview

Usage (CLI):
    python src/visualize.py \
        --classical results/classical \
        --quantum   results/quantum \
        --processed data/processed \
        --output    plots/
"""

from __future__ import annotations

import argparse
import os

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import seaborn as sns
from matplotlib.gridspec import GridSpec
from scipy.interpolate import griddata


# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------

PALETTE = {
    "classical": "#E07B3F",   # warm orange
    "quantum":   "#3B82F6",   # electric blue
    "anomaly":   "#EF4444",   # red
    "normal":    "#22C55E",   # green
    "bg":        "#0F172A",   # dark navy
    "surface":   "#1E293B",
    "text":      "#F1F5F9",
    "grid":      "#334155",
}

plt.rcParams.update({
    "figure.facecolor":  PALETTE["bg"],
    "axes.facecolor":    PALETTE["surface"],
    "axes.edgecolor":    PALETTE["grid"],
    "axes.labelcolor":   PALETTE["text"],
    "xtick.color":       PALETTE["text"],
    "ytick.color":       PALETTE["text"],
    "text.color":        PALETTE["text"],
    "grid.color":        PALETTE["grid"],
    "grid.linewidth":    0.5,
    "axes.grid":         True,
    "font.family":       "monospace",
    "axes.titlepad":     10,
})


def _save(fig: plt.Figure, path: str, dpi: int = 150) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"[viz] Saved → {path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 1. Brain anomaly heatmap
# ---------------------------------------------------------------------------

def plot_brain_heatmap(scores: np.ndarray, n_channels: int,
                        title: str = "Anomaly Heatmap",
                        output_path: str = "plots/heatmap.png") -> None:
    """
    Render a 2D heatmap of per-channel mean anomaly scores on a
    pseudo-topomap (circular head outline, sensors on a grid).

    Parameters
    ----------
    scores     : (n_epochs, n_channels) or (n_channels,) mean scores per channel
    n_channels : number of MEG sensor channels
    """
    if scores.ndim == 2:
        # If per-epoch features were split across channels, reduce
        ch_scores = scores.mean(axis=0)[:n_channels]
    else:
        ch_scores = scores[:n_channels]

    # Place sensors in a ring + centre grid pattern (approximate MEG array)
    rng = np.random.default_rng(0)
    n = len(ch_scores)

    # Create pseudo-topomap sensor positions on concentric rings
    positions = []
    rings = [0, 8, 16, 32, n - 56]  # rough ring sizes
    rings = [max(0, r) for r in rings]
    for ring_idx, ring_n in enumerate(rings):
        if ring_n <= 0:
            continue
        r = ring_idx * 0.22
        for j in range(ring_n):
            angle = 2 * np.pi * j / ring_n
            positions.append((r * np.cos(angle), r * np.sin(angle)))
        if len(positions) >= n:
            break

    # Fall back to random if not enough
    while len(positions) < n:
        positions.append(rng.uniform(-0.8, 0.8, 2).tolist())
    positions = np.array(positions[:n])

    # Interpolate onto regular grid for smooth heatmap
    grid_x, grid_y = np.mgrid[-1:1:200j, -1:1:200j]
    grid_z = griddata(positions, ch_scores, (grid_x, grid_y), method="cubic", fill_value=0)

    # Mask outside head circle
    mask = grid_x ** 2 + grid_y ** 2 > 0.95
    grid_z[mask] = np.nan

    fig, ax = plt.subplots(figsize=(7, 7), facecolor=PALETTE["bg"])
    ax.set_facecolor(PALETTE["bg"])

    cmap = plt.cm.inferno
    im = ax.imshow(grid_z.T, extent=[-1, 1, -1, 1], origin="lower",
                   cmap=cmap, interpolation="bicubic", vmin=0)

    # Sensor dots
    sc = ax.scatter(positions[:, 0], positions[:, 1],
                    c=ch_scores, cmap=cmap, s=40, edgecolors="white",
                    linewidths=0.5, zorder=3)

    # Head outline
    head_circle = plt.Circle((0, 0), 0.95, fill=False,
                               color="white", linewidth=1.5, zorder=4)
    ax.add_patch(head_circle)
    # Nose
    ax.annotate("▲", (0, 0.95), ha="center", va="bottom",
                 fontsize=10, color="white", zorder=5)

    cbar = fig.colorbar(sc, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Mean Anomaly Score", color=PALETTE["text"])
    cbar.ax.yaxis.set_tick_params(color=PALETTE["text"])

    ax.set_title(title, fontsize=14, color=PALETTE["text"], fontweight="bold")
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)
    ax.set_aspect("equal")
    ax.axis("off")

    _save(fig, output_path)


# ---------------------------------------------------------------------------
# 2. Temporal anomaly plot
# ---------------------------------------------------------------------------

def plot_temporal_anomalies(raw_data: np.ndarray, sfreq: float,
                              classical_scores: np.ndarray,
                              quantum_scores: np.ndarray,
                              ground_truth: np.ndarray | None,
                              output_path: str = "plots/temporal.png") -> None:
    """
    Plot: (top) mean MEG signal, (bottom) classical vs quantum anomaly scores.
    """
    t_raw = np.arange(raw_data.shape[1]) / sfreq
    mean_signal = raw_data.mean(axis=0)

    n_epochs = len(classical_scores)
    t_epochs = np.linspace(0, t_raw[-1], n_epochs)

    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=False,
                              gridspec_kw={"height_ratios": [2, 1.5, 1.5]})

    # --- Panel 1: Raw MEG ---
    ax = axes[0]
    ax.plot(t_raw[:10000], mean_signal[:10000] * 1e13, color="#64748B",
            lw=0.6, label="Mean MEG (fT)")
    if ground_truth is not None:
        gt_t = ground_truth / sfreq
        gt_t = gt_t[gt_t < t_raw[10000]]
        ax.vlines(gt_t, ymin=mean_signal[:10000].min() * 1e13,
                  ymax=mean_signal[:10000].max() * 1e13,
                  color=PALETTE["anomaly"], lw=1, alpha=0.6,
                  label="True anomaly")
    ax.set_ylabel("Amplitude (fT)", fontsize=10)
    ax.set_title("Mean MEG Signal (first 10 s)", fontsize=12)
    ax.legend(fontsize=8, loc="upper right")

    # --- Panel 2: Classical scores ---
    ax = axes[1]
    ax.plot(t_epochs, classical_scores, color=PALETTE["classical"], lw=1.2, label="Classical")
    thr_c = np.percentile(classical_scores, 90)
    ax.axhline(thr_c, color=PALETTE["classical"], lw=0.8, ls="--", alpha=0.6)
    ax.fill_between(t_epochs, classical_scores, thr_c,
                     where=classical_scores > thr_c, alpha=0.35,
                     color=PALETTE["classical"])
    ax.set_ylabel("Anomaly Score", fontsize=10)
    ax.set_title("Classical Detector", fontsize=12)
    ax.legend(fontsize=8)

    # --- Panel 3: Quantum scores ---
    ax = axes[2]
    ax.plot(t_epochs, quantum_scores, color=PALETTE["quantum"], lw=1.2, label="Quantum-Enhanced")
    thr_q = np.percentile(quantum_scores, 90)
    ax.axhline(thr_q, color=PALETTE["quantum"], lw=0.8, ls="--", alpha=0.6)
    ax.fill_between(t_epochs, quantum_scores, thr_q,
                     where=quantum_scores > thr_q, alpha=0.35,
                     color=PALETTE["quantum"])
    ax.set_ylabel("Anomaly Score", fontsize=10)
    ax.set_xlabel("Time (s)", fontsize=10)
    ax.set_title("Quantum-Enhanced Detector", fontsize=12)
    ax.legend(fontsize=8)

    fig.suptitle("Anomaly Detection Timeline", fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    _save(fig, output_path)


# ---------------------------------------------------------------------------
# 3. ROC / sensitivity comparison
# ---------------------------------------------------------------------------

def plot_roc_comparison(classical_scores: np.ndarray,
                         quantum_scores: np.ndarray,
                         gt_labels: np.ndarray,
                         output_path: str = "plots/roc_comparison.png") -> None:
    """
    Plot ROC curves for classical and quantum detectors, plus sensitivity curves.
    """
    from sklearn.metrics import roc_curve, auc, precision_recall_curve

    fig = plt.figure(figsize=(13, 5.5), facecolor=PALETTE["bg"])
    gs = GridSpec(1, 2, figure=fig, wspace=0.35)

    # --- ROC ---
    ax_roc = fig.add_subplot(gs[0])
    for label, scores, color in [
        ("Classical", classical_scores, PALETTE["classical"]),
        ("Quantum-Enhanced", quantum_scores, PALETTE["quantum"]),
    ]:
        fpr, tpr, _ = roc_curve(gt_labels, scores)
        roc_auc = auc(fpr, tpr)
        ax_roc.plot(fpr, tpr, color=color, lw=2,
                    label=f"{label} (AUC={roc_auc:.3f})")
        ax_roc.fill_between(fpr, tpr, alpha=0.07, color=color)

    ax_roc.plot([0, 1], [0, 1], color=PALETTE["grid"], ls="--", lw=1)
    ax_roc.set_xlabel("False Positive Rate", fontsize=11)
    ax_roc.set_ylabel("True Positive Rate", fontsize=11)
    ax_roc.set_title("ROC Curves", fontsize=13, fontweight="bold")
    ax_roc.legend(fontsize=9)

    # --- Precision-Recall ---
    ax_pr = fig.add_subplot(gs[1])
    for label, scores, color in [
        ("Classical", classical_scores, PALETTE["classical"]),
        ("Quantum-Enhanced", quantum_scores, PALETTE["quantum"]),
    ]:
        prec, rec, _ = precision_recall_curve(gt_labels, scores)
        pr_auc = auc(rec, prec)
        ax_pr.plot(rec, prec, color=color, lw=2,
                   label=f"{label} (AP={pr_auc:.3f})")
        ax_pr.fill_between(rec, prec, alpha=0.07, color=color)

    ax_pr.set_xlabel("Recall (Sensitivity)", fontsize=11)
    ax_pr.set_ylabel("Precision", fontsize=11)
    ax_pr.set_title("Precision–Recall Curves", fontsize=13, fontweight="bold")
    ax_pr.legend(fontsize=9)

    fig.suptitle("Classical vs. Quantum-Enhanced Anomaly Detection",
                  fontsize=14, fontweight="bold")
    _save(fig, output_path)


# ---------------------------------------------------------------------------
# 4. Wigner function + squeezing curve
# ---------------------------------------------------------------------------

def plot_quantum_state(wigner_xvec: np.ndarray, wigner_W: np.ndarray,
                        squeezing_r: float,
                        output_path: str = "plots/quantum_state.png") -> None:
    """
    Side-by-side: Wigner function of squeezed state | noise reduction vs. r.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    # --- Wigner function ---
    ax = axes[0]
    vmax = np.abs(wigner_W).max()
    im = ax.contourf(wigner_xvec, wigner_xvec, wigner_W,
                      levels=80, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.contour(wigner_xvec, wigner_xvec, wigner_W,
                levels=12, colors="white", linewidths=0.3, alpha=0.3)
    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label("W(x, p)", color=PALETTE["text"])
    ax.set_xlabel("X quadrature", fontsize=11)
    ax.set_ylabel("P quadrature", fontsize=11)
    ax.set_title(f"Wigner Function — Squeezed State (r={squeezing_r:.2f})",
                  fontsize=12, fontweight="bold")

    # Annotate squeezing direction
    ax.annotate("← squeezed", xy=(-2.0, 0.1), fontsize=8,
                 color="white", alpha=0.8)

    # --- Noise reduction vs r ---
    ax2 = axes[1]
    r_vals = np.linspace(0, 1.8, 300)
    noise_dB = -10.0 * np.log10(np.exp(-2 * r_vals))  # Var_squeezed = e^{-2r}/2

    ax2.plot(r_vals, noise_dB, color=PALETTE["quantum"], lw=2.5)
    ax2.fill_between(r_vals, noise_dB, alpha=0.15, color=PALETTE["quantum"])
    ax2.axvline(squeezing_r, color=PALETTE["anomaly"], lw=1.5, ls="--",
                 label=f"r = {squeezing_r:.2f}  ({-10*np.log10(np.exp(-2*squeezing_r)):.1f} dB)")

    # Reference levels
    for db_ref, label in [(3, "3 dB"), (6, "6 dB"), (10, "10 dB")]:
        ax2.axhline(db_ref, color=PALETTE["grid"], lw=0.7, ls=":", alpha=0.8)
        ax2.text(0.05, db_ref + 0.1, label, fontsize=7.5, color=PALETTE["text"], alpha=0.7)

    ax2.set_xlabel("Squeezing parameter r", fontsize=11)
    ax2.set_ylabel("Noise reduction (dB below shot noise)", fontsize=11)
    ax2.set_title("Sensitivity vs. Squeezing", fontsize=12, fontweight="bold")
    ax2.legend(fontsize=9)

    fig.suptitle("NV-Center Quantum Magnetometer State", fontsize=14, fontweight="bold")
    fig.tight_layout()
    _save(fig, output_path)


# ---------------------------------------------------------------------------
# 5. Edge filter overview
# ---------------------------------------------------------------------------

def plot_edge_filter(scores: np.ndarray,
                      flags: np.ndarray,
                      thresholds: np.ndarray,
                      events: list[tuple[int, int]],
                      compression_ratio: float,
                      output_path: str = "plots/edge_filter.png") -> None:
    """Visualise the edge filtering result."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)

    t = np.arange(len(scores))

    # --- Score + threshold + flags ---
    ax = axes[0]
    ax.plot(t, scores, color=PALETTE["quantum"], lw=0.8, alpha=0.8, label="Anomaly score")
    ax.plot(t, thresholds, color="#F59E0B", lw=1.2, ls="--", alpha=0.9, label="Local threshold")
    ax.fill_between(t, scores, where=flags.astype(bool),
                     color=PALETTE["anomaly"], alpha=0.5, label="Flagged")
    ax.set_ylabel("Score", fontsize=10)
    ax.set_title("Edge Filter: Anomaly Score + Adaptive Threshold", fontsize=12, fontweight="bold")
    ax.legend(fontsize=8, loc="upper right")

    # --- Binary flags + event markers ---
    ax2 = axes[1]
    ax2.fill_between(t, flags, color=PALETTE["anomaly"], alpha=0.7, step="mid", label="Flagged epoch")
    for start, end in events:
        ax2.axvspan(start, end, alpha=0.2, color=PALETTE["quantum"])

    ax2.set_yticks([0, 1])
    ax2.set_yticklabels(["Normal", "Anomaly"])
    ax2.set_xlabel("Epoch index", fontsize=10)
    ax2.set_title(
        f"Flagged Epochs | Compression: {100*compression_ratio:.1f}% data suppressed | "
        f"{len(events)} events detected",
        fontsize=11, fontweight="bold",
    )
    ax2.legend(fontsize=8)

    fig.tight_layout()
    _save(fig, output_path)


# ---------------------------------------------------------------------------
# Master visualisation runner
# ---------------------------------------------------------------------------

def visualize_all(classical_dir: str, quantum_dir: str,
                   processed_dir: str, output_dir: str = "plots") -> None:
    os.makedirs(output_dir, exist_ok=True)

    # Load data
    classical_scores = np.load(os.path.join(classical_dir, "scores.npy"))
    classical_labels = np.load(os.path.join(classical_dir, "labels.npy"))
    quantum_scores   = np.load(os.path.join(quantum_dir, "scores.npy"))
    quantum_labels   = np.load(os.path.join(quantum_dir, "labels.npy"))

    epochs    = np.load(os.path.join(processed_dir, "epochs.npy"))
    raw       = np.load(os.path.join(processed_dir, "raw_filtered.npy"))
    n_channels = raw.shape[0]

    gt_path = os.path.join(processed_dir, "ground_truth.npy")
    ground_truth = np.load(gt_path) if os.path.exists(gt_path) else None

    gt_epoch_path = os.path.join(classical_dir, "gt_labels_epoch.npy")
    gt_labels_epoch = np.load(gt_epoch_path) if os.path.exists(gt_epoch_path) else None

    # Per-channel scores: assign each epoch's score to all its channels
    # (simple broadcast for heatmap)
    ch_classical = classical_scores.mean() * np.ones(n_channels)
    ch_quantum   = quantum_scores.mean()   * np.ones(n_channels)
    # Add some spatial variation via channel-wise feature variance
    feat = np.load(os.path.join(processed_dir, "features.npy"))
    feat_std = feat.std(axis=0)
    if len(feat_std) >= n_channels:
        ch_classical = feat_std[:n_channels] * (classical_scores.mean() / feat_std[:n_channels].mean())
        ch_quantum   = feat_std[:n_channels] * (quantum_scores.mean()   / feat_std[:n_channels].mean())

    # 1. Brain heatmaps
    plot_brain_heatmap(ch_classical, n_channels,
                        title="Classical — Brain Anomaly Heatmap",
                        output_path=os.path.join(output_dir, "heatmap_classical.png"))
    plot_brain_heatmap(ch_quantum, n_channels,
                        title="Quantum-Enhanced — Brain Anomaly Heatmap",
                        output_path=os.path.join(output_dir, "heatmap_quantum.png"))

    # 2. Temporal
    plot_temporal_anomalies(raw, sfreq=1000.0,
                             classical_scores=classical_scores,
                             quantum_scores=quantum_scores,
                             ground_truth=ground_truth,
                             output_path=os.path.join(output_dir, "temporal_anomalies.png"))

    # 3. ROC comparison (only if GT available)
    if gt_labels_epoch is not None and gt_labels_epoch.sum() > 0:
        plot_roc_comparison(classical_scores, quantum_scores, gt_labels_epoch,
                             output_path=os.path.join(output_dir, "roc_comparison.png"))
    else:
        print("[viz] Skipping ROC plot — no ground-truth epoch labels found.")

    # 4. Wigner / quantum state
    wigner_xvec_path = os.path.join(quantum_dir, "wigner_xvec.npy")
    wigner_W_path    = os.path.join(quantum_dir, "wigner_W.npy")
    if os.path.exists(wigner_xvec_path):
        import json
        with open(os.path.join(quantum_dir, "metrics.json")) as f:
            q_metrics = json.load(f)
        plot_quantum_state(
            np.load(wigner_xvec_path),
            np.load(wigner_W_path),
            squeezing_r=q_metrics.get("squeezing_r", 0.8),
            output_path=os.path.join(output_dir, "quantum_state.png"),
        )

    # 5. Edge filter
    edge_flags_path = os.path.join(quantum_dir, "edge_flags.npy")
    edge_thr_path   = os.path.join(quantum_dir, "edge_thresholds.npy")
    if os.path.exists(edge_flags_path):
        import json
        with open(os.path.join(quantum_dir, "edge_filter_summary.json")) as f:
            ef = json.load(f)
        plot_edge_filter(
            quantum_scores,
            np.load(edge_flags_path),
            np.load(edge_thr_path),
            [(s, e) for s, e in ef["events"]],
            ef["compression_ratio"],
            output_path=os.path.join(output_dir, "edge_filter.png"),
        )

    print(f"[viz] All plots saved → {output_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--classical", default="results/classical")
    parser.add_argument("--quantum",   default="results/quantum")
    parser.add_argument("--processed", default="data/processed")
    parser.add_argument("--output",    default="plots")
    args = parser.parse_args()
    visualize_all(args.classical, args.quantum, args.processed, args.output)