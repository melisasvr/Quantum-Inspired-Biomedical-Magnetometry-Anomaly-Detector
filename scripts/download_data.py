"""
scripts/download_data.py
------------------------
Downloads an OpenNeuro MEG dataset in BIDS format.
Default: ds003483 (a publicly available MEG recording dataset).

Usage:
    python scripts/download_data.py --dataset ds003483 --output data/
"""

import argparse
import os
import subprocess
import sys


def download_dataset(dataset_id: str, output_dir: str, include: list[str] | None = None) -> None:
    """
    Download an OpenNeuro dataset using openneuro-py CLI.

    Args:
        dataset_id: OpenNeuro dataset accession (e.g. 'ds003483')
        output_dir:  Local directory to store the BIDS dataset
        include:     Optional list of glob patterns to limit download size
    """
    os.makedirs(output_dir, exist_ok=True)
    dest = os.path.join(output_dir, dataset_id)

    print(f"[download] Downloading {dataset_id} → {dest}")

    cmd = [
        sys.executable, "-m", "openneuro",
        "download",
        "--dataset", dataset_id,
        "--target", dest,
    ]

    if include:
        for pattern in include:
            cmd += ["--include", pattern]

    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        print("[download] openneuro-py download failed. Falling back to synthetic data generation.")
        generate_synthetic_bids(dest)
    else:
        print(f"[download] Done. Dataset saved to: {dest}")


def generate_synthetic_bids(output_dir: str, n_channels: int = 102, n_times: int = 60_000,
                             sfreq: float = 1000.0) -> None:
    """
    Generate a minimal synthetic BIDS MEG dataset for offline / CI use.
    Produces realistic brain-like signals with injected anomalies.
    """
    import numpy as np
    import json

    print(f"[synthetic] Generating synthetic MEG dataset → {output_dir}")

    sub = "sub-01"
    meg_dir = os.path.join(output_dir, sub, "meg")
    os.makedirs(meg_dir, exist_ok=True)

    rng = np.random.default_rng(42)
    t = np.arange(n_times) / sfreq  # seconds

    # Background: pink-ish noise (1/f) + alpha oscillation
    freqs = np.fft.rfftfreq(n_times, d=1.0 / sfreq)
    freqs[0] = 1.0  # avoid div-by-zero
    pink_spectrum = 1.0 / np.sqrt(freqs)

    data = np.zeros((n_channels, n_times))
    for ch in range(n_channels):
        noise_fft = pink_spectrum * rng.standard_normal(len(freqs)) * 1e-13  # ~100 fT scale
        data[ch] = np.fft.irfft(noise_fft, n=n_times)
        # Add 10 Hz alpha
        data[ch] += 5e-14 * np.sin(2 * np.pi * 10 * t + rng.uniform(0, 2 * np.pi))

    # Inject 20 synthetic anomaly events (neural spikes)
    anomaly_times = rng.integers(5000, n_times - 5000, size=20)
    for at in anomaly_times:
        amp = rng.uniform(3e-13, 8e-13)
        channels_affected = rng.choice(n_channels, size=rng.integers(3, 10), replace=False)
        for ch in channels_affected:
            data[ch, at:at + 50] += amp * np.hanning(50)

    # Save as raw numpy (real pipeline uses .fif via MNE)
    np.save(os.path.join(meg_dir, "sub-01_task-rest_meg.npy"), data)
    np.save(os.path.join(meg_dir, "sub-01_task-rest_anomaly_times.npy"), anomaly_times)

    # Minimal sidecar JSON
    sidecar = {
        "SamplingFrequency": sfreq,
        "MEGChannelCount": n_channels,
        "TaskName": "rest",
        "InstitutionName": "SyntheticLab",
    }
    with open(os.path.join(meg_dir, "sub-01_task-rest_meg.json"), "w") as f:
        json.dump(sidecar, f, indent=2)

    print(f"[synthetic] Done. {n_channels} channels × {n_times} samples @ {sfreq} Hz")
    print(f"[synthetic] Injected {len(anomaly_times)} anomaly events.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download or generate MEG dataset")
    parser.add_argument("--dataset", default="ds003483", help="OpenNeuro dataset accession ID")
    parser.add_argument("--output", default="data", help="Output directory")
    parser.add_argument("--synthetic", action="store_true", help="Force synthetic data generation")
    args = parser.parse_args()

    if args.synthetic:
        dest = os.path.join(args.output, args.dataset)
        generate_synthetic_bids(dest)
    else:
        # Only download MEG files for subject 01 to keep download small
        download_dataset(
            args.dataset,
            args.output,
            include=["sub-01/meg/*"],
        )