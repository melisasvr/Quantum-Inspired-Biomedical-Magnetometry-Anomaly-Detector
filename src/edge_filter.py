"""
src/edge_filter.py
------------------
Edge-like local anomaly filtering.

Mimics what a wearable MEG device would do: instead of streaming all raw data,
only flag and transmit epochs where anomaly scores exceed a local adaptive
threshold. This reduces "data transmission" by 80–95% in typical settings.

Also provides:
  - hysteresis thresholding to avoid spurious rapid toggling
  - event merging (nearby anomaly windows → single event)
  - compression ratio reporting

Usage (CLI):
    python src/edge_filter.py \
        --scores results/quantum/scores.npy \
        --threshold_percentile 92 \
        --output results/quantum/
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np


def adaptive_threshold(scores: np.ndarray,
                        window_size: int = 50,
                        k: float = 2.5) -> np.ndarray:
    """
    Compute a local adaptive threshold: μ(scores) + k * σ(scores)
    over a rolling window.

    Parameters
    ----------
    scores      : (n_epochs,) anomaly scores
    window_size : rolling window size in epochs
    k           : number of standard deviations above local mean

    Returns
    -------
    thresholds : (n_epochs,) local threshold values
    """
    n = len(scores)
    thresholds = np.zeros(n)
    half = window_size // 2

    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half)
        window = scores[lo:hi]
        thresholds[i] = window.mean() + k * window.std()

    return thresholds


def hysteresis_threshold(scores: np.ndarray,
                          high_pct: float = 92.0,
                          low_pct: float = 80.0) -> np.ndarray:
    """
    Schmitt-trigger style hysteresis thresholding to avoid rapid toggling.

    An epoch is flagged ON when score > high_threshold,
    and returns to OFF when score < low_threshold.

    Parameters
    ----------
    scores   : (n_epochs,)
    high_pct : percentile for rising edge
    low_pct  : percentile for falling edge

    Returns
    -------
    flags : (n_epochs,) int — 1 = anomaly, 0 = normal
    """
    high_thr = np.percentile(scores, high_pct)
    low_thr = np.percentile(scores, low_pct)

    flags = np.zeros(len(scores), dtype=int)
    state = False
    for i, s in enumerate(scores):
        if not state and s >= high_thr:
            state = True
        elif state and s < low_thr:
            state = False
        flags[i] = int(state)

    return flags


def merge_events(flags: np.ndarray,
                  min_gap: int = 3) -> list[tuple[int, int]]:
    """
    Merge adjacent flagged epochs separated by fewer than min_gap clean epochs.

    Returns
    -------
    List of (start, end) epoch index pairs for each merged anomaly event.
    """
    events = []
    in_event = False
    start = 0

    i = 0
    while i < len(flags):
        if flags[i] == 1 and not in_event:
            in_event = True
            start = i
        elif flags[i] == 0 and in_event:
            # Check if the gap is too small (merge)
            gap_end = i
            while gap_end < len(flags) and gap_end - i < min_gap:
                if flags[gap_end] == 1:
                    break
                gap_end += 1
            if gap_end < len(flags) and flags[gap_end] == 1:
                # Small gap → continue event
                i = gap_end
                continue
            else:
                events.append((start, i - 1))
                in_event = False
        i += 1

    if in_event:
        events.append((start, len(flags) - 1))

    return events


def edge_filter(scores: np.ndarray,
                 threshold_percentile: float = 92.0,
                 hysteresis: bool = True,
                 adaptive: bool = True,
                 min_gap: int = 3) -> dict:
    """
    Full edge filtering pipeline.

    Parameters
    ----------
    scores               : (n_epochs,) raw anomaly scores
    threshold_percentile : global percentile for hard threshold
    hysteresis           : use Schmitt-trigger hysteresis
    adaptive             : overlay adaptive local threshold
    min_gap              : minimum clean-epoch gap for event separation

    Returns
    -------
    dict with keys:
      flags            : (n_epochs,) binary flags
      events           : list of (start, end) tuples
      compression_ratio: fraction of epochs NOT transmitted
      thresholds       : (n_epochs,) adaptive threshold (if used)
    """
    n = len(scores)

    if hysteresis:
        flags = hysteresis_threshold(scores,
                                      high_pct=threshold_percentile,
                                      low_pct=threshold_percentile - 12)
    else:
        thr = np.percentile(scores, threshold_percentile)
        flags = (scores >= thr).astype(int)

    if adaptive:
        local_thr = adaptive_threshold(scores)
        # AND with adaptive: only keep flags where score also exceeds local thr
        flags = flags & (scores >= local_thr).astype(int)
        thresholds = local_thr
    else:
        thresholds = np.full(n, np.percentile(scores, threshold_percentile))

    events = merge_events(flags, min_gap=min_gap)
    n_flagged = flags.sum()
    compression_ratio = 1.0 - n_flagged / n

    print(f"[edge_filter] Flagged {n_flagged}/{n} epochs ({100*n_flagged/n:.1f}%)")
    print(f"[edge_filter] Detected {len(events)} merged anomaly events")
    print(f"[edge_filter] Data compression ratio: {100*compression_ratio:.1f}%")

    return {
        "flags": flags,
        "events": events,
        "compression_ratio": compression_ratio,
        "thresholds": thresholds,
        "n_flagged": n_flagged,
        "n_events": len(events),
    }


def run_edge_filter(scores_path: str,
                     threshold_percentile: float = 92.0,
                     output_dir: str = "results/quantum") -> dict:
    """Load scores, run edge filter, save results."""
    os.makedirs(output_dir, exist_ok=True)

    scores = np.load(scores_path)
    result = edge_filter(scores, threshold_percentile)

    np.save(os.path.join(output_dir, "edge_flags.npy"), result["flags"])
    np.save(os.path.join(output_dir, "edge_thresholds.npy"), result["thresholds"])

    summary = {
        "n_epochs": len(scores),
        "n_flagged": int(result["n_flagged"]),
        "n_events": result["n_events"],
        "compression_ratio": float(result["compression_ratio"]),
        "events": [(int(s), int(e)) for s, e in result["events"]],
    }
    with open(os.path.join(output_dir, "edge_filter_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[edge_filter] Results saved → {output_dir}")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", required=True, help="Path to scores.npy")
    parser.add_argument("--threshold_percentile", type=float, default=92.0)
    parser.add_argument("--output", default="results/quantum")
    args = parser.parse_args()
    run_edge_filter(args.scores, args.threshold_percentile, args.output)