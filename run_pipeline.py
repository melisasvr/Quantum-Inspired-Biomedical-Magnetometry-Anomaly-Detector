"""
run_pipeline.py
---------------
End-to-end pipeline runner for the Quantum-Inspired Biomedical Magnetometry
Anomaly Detector.

Steps:
  1. Generate / download MEG data
  2. Preprocess (filter, epoch, extract features)
  3. Classical anomaly detection (Isolation Forest or Autoencoder)
  4. Quantum-enhanced anomaly detection (QuTiP squeezed states)
  5. Edge filtering (local adaptive thresholding)
  6. Visualisation (heatmaps, ROC, temporal, Wigner)

Usage:
    python run_pipeline.py
    python run_pipeline.py --subject 01 --squeezing 1.0 --method autoencoder
    python run_pipeline.py --synthetic          # force synthetic data
"""

from __future__ import annotations

import argparse
import os
import sys
import time

# ---------------------------------------------------------------------------
# Ensure scripts/ and src/ are importable as plain folders (no package needed)
# ---------------------------------------------------------------------------
_ROOT = os.path.dirname(os.path.abspath(__file__))
for _p in (_ROOT, os.path.join(_ROOT, "scripts"), os.path.join(_ROOT, "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Quantum MEG Anomaly Detector — Full Pipeline"
    )
    parser.add_argument("--dataset", default="ds003483",
                        help="OpenNeuro dataset accession ID")
    parser.add_argument("--subject", default="01", help="Subject ID")
    parser.add_argument("--squeezing", type=float, default=0.8,
                        help="NV-center squeezing parameter r (0 = classical limit)")
    parser.add_argument("--method", default="isolation_forest",
                        choices=["isolation_forest", "autoencoder"],
                        help="Classical baseline method")
    parser.add_argument("--contamination", type=float, default=0.05,
                        help="Expected anomaly fraction (0–0.5)")
    parser.add_argument("--threshold_percentile", type=float, default=92.0,
                        help="Edge filter threshold percentile")
    parser.add_argument("--synthetic", action="store_true",
                        help="Use synthetic data (no download required)")
    parser.add_argument("--data_dir", default="data",
                        help="Root data directory")
    parser.add_argument("--output_dir", default="results",
                        help="Root results directory")
    parser.add_argument("--plots_dir", default="plots",
                        help="Output directory for plots")
    parser.add_argument("--no_plots", action="store_true",
                        help="Skip visualisation step")
    args = parser.parse_args()

    bids_root = os.path.join(args.data_dir, args.dataset)
    processed_dir = os.path.join(args.data_dir, "processed")
    classical_dir = os.path.join(args.output_dir, "classical")
    quantum_dir   = os.path.join(args.output_dir, "quantum")

    t0 = time.time()
    print("=" * 60)
    print("  Quantum-Inspired MEG Anomaly Detector")
    print("=" * 60)

    # ── Step 1: Data ─────────────────────────────────────────────
    print("\n[1/6] Data acquisition")
    from download_data import download_dataset, generate_synthetic_bids

    meg_npy = os.path.join(bids_root, f"sub-{args.subject}", "meg",
                            f"sub-{args.subject}_task-rest_meg.npy")
    if args.synthetic or not os.path.exists(meg_npy):
        generate_synthetic_bids(bids_root)
    else:
        print(f"[pipeline] Data already exists at {bids_root}, skipping download.")

    # ── Step 2: Preprocess ───────────────────────────────────────
    print("\n[2/6] Preprocessing")
    from preprocess import preprocess_pipeline
    proc = preprocess_pipeline(bids_root, args.subject, processed_dir)

    # ── Step 3: Classical detection ──────────────────────────────
    print(f"\n[3/6] Classical detection ({args.method})")
    from detect_classical import detect_classical
    classical_result = detect_classical(
        processed_dir, args.method, args.contamination, classical_dir)

    # ── Step 4: Quantum-enhanced detection ───────────────────────
    print(f"\n[4/6] Quantum-enhanced detection (r={args.squeezing})")
    from detect_quantum import detect_quantum
    quantum_result = detect_quantum(
        processed_dir, args.squeezing, args.contamination, quantum_dir)

    # ── Step 5: Edge filtering ───────────────────────────────────
    print("\n[5/6] Edge filtering")
    from edge_filter import run_edge_filter
    import os as _os
    scores_path = _os.path.join(quantum_dir, "scores.npy")
    run_edge_filter(scores_path, args.threshold_percentile, quantum_dir)

    # ── Step 6: Visualisation ────────────────────────────────────
    if not args.no_plots:
        print("\n[6/6] Generating plots")
        from visualize import visualize_all
        visualize_all(classical_dir, quantum_dir, processed_dir, args.plots_dir)
    else:
        print("\n[6/6] Visualisation skipped (--no_plots)")

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"  Pipeline complete in {elapsed:.1f}s")
    print(f"  Results → {args.output_dir}/")
    print(f"  Plots   → {args.plots_dir}/")
    print("=" * 60)

    # Print summary
    c_metrics = classical_result.get("metrics", {})
    q_metrics = quantum_result.get("metrics", {})
    print("\nDetection Summary")
    print("-" * 40)
    if "roc_auc" in c_metrics:
        print(f"  Classical ROC-AUC  : {c_metrics['roc_auc']:.3f}")
    if "roc_auc" in q_metrics:
        print(f"  Quantum   ROC-AUC  : {q_metrics['roc_auc']:.3f}")
        delta = q_metrics["roc_auc"] - c_metrics.get("roc_auc", q_metrics["roc_auc"])
        print(f"  Improvement        : +{delta:.3f}")
    if "noise_reduction_dB" in q_metrics:
        print(f"  Noise reduction    : {q_metrics['noise_reduction_dB']:.2f} dB")
    if "sensitivity_gain" in q_metrics:
        print(f"  Sensitivity gain   : {q_metrics['sensitivity_gain']:.3f}×")


if __name__ == "__main__":
    main()