#!/usr/bin/env python3
"""
MindInk / ZuCo Step 8
SNR / Signal-Quality Evaluation

Compares:
    1. no_denoising
    2. iclabel_hard_reject

Important:
This is a baseline signal-quality evaluation, not a claim of "true"
neural signal vs noise because no clean EEG ground truth is available.

Metrics:
    - RMS
    - variance
    - peak-to-peak amplitude
    - spectral band power
    - 50 Hz power (diagnostic; Step 3 already notch-filtered)
    - relative band-power ratios
    - SNR proxy

SNR proxy:
    Signal power = 1-45 Hz power
    Noise power  = broadband residual outside the 1-45 Hz band,
                   within the available 1-45 Hz bandwidth, therefore
                   not directly separable after Step 3.

Because Step 3 low-passed at 45 Hz, a conventional EEG SNR based on
higher-frequency noise cannot be recovered. Therefore this script
reports a conservative quality/SNR proxy using robust signal statistics
and spectral power ratios rather than pretending that an exact SNR
is available.

Outputs:
    metrics/<subject>/sentence_metrics.csv
    metrics/<subject>/summary.json
    metrics/<subject>/metrics.npy
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

try:
    from scipy.signal import welch
except ImportError as exc:
    raise RuntimeError(
        "Step 8 requires scipy. Install it with: pip install scipy"
    ) from exc


FS = 250.0
N_CHANNELS = 104   # raw route
EXPECTED_SENTENCES = 349

BASELINES = (
    "no_denoising",
    "iclabel_hard_reject",
)

EPS = 1e-12

# EEG frequency bands.
BANDS = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 45.0),
}


def load_sentence(path: Path) -> np.ndarray:
    x = np.load(path, allow_pickle=False)

    if x.ndim != 2:
        raise RuntimeError(f"{path}: expected 2-D array, got {x.shape}")

    if x.shape[0] != N_CHANNELS:
        raise RuntimeError(
            f"{path}: expected {N_CHANNELS} channels, got {x.shape}"
        )

    if not np.isfinite(x).all():
        raise RuntimeError(f"{path}: contains NaN/Inf")

    return x.astype(np.float64, copy=False)


def robust_rms(x: np.ndarray) -> float:
    """
    RMS after per-channel demeaning.
    """
    y = x - np.mean(x, axis=1, keepdims=True)
    return float(np.sqrt(np.mean(y * y)))


def peak_to_peak(x: np.ndarray) -> float:
    return float(np.mean(np.ptp(x, axis=1)))


def channel_variance(x: np.ndarray) -> float:
    return float(np.mean(np.var(x, axis=1)))


def spectral_metrics(x: np.ndarray, fs: float) -> dict:
    """
    Average Welch PSD across channels.

    Step 3 already band-limited the signal to 0.5/1-45 Hz and applied
    a 50 Hz notch. Therefore the 45-100 Hz ICLabel range and conventional
    high-frequency noise SNR are unavailable here.
    """
    n_samples = x.shape[1]

    if n_samples < 16:
        return {
            "total_power": np.nan,
            "delta_power": np.nan,
            "theta_power": np.nan,
            "alpha_power": np.nan,
            "beta_power": np.nan,
            "gamma_power": np.nan,
            "alpha_relative": np.nan,
            "beta_relative": np.nan,
            "gamma_relative": np.nan,
            "low_frequency_power": np.nan,
            "high_frequency_power": np.nan,
            "alpha_prominence_db": np.nan,
            "delta_contamination_frac": np.nan,
            "low_high_ratio_db": np.nan,
            "spectral_snr_proxy_db": np.nan,
            "power_50hz": np.nan,
        }

    nperseg = min(1024, n_samples)

    freqs, psd = welch(
        x,
        fs=fs,
        axis=1,
        nperseg=nperseg,
        noverlap=nperseg // 2,
        detrend="constant",
        scaling="density",
    )

    mean_psd = np.mean(psd, axis=0)

    def band_power(low: float, high: float) -> float:
        mask = (freqs >= low) & (freqs < high)
        if not np.any(mask):
            return np.nan
        return float(np.trapezoid(mean_psd[mask], freqs[mask]))

    total = band_power(1.0, 45.0)   # keep neural band for relative powers
    delta = band_power(*BANDS["delta"])
    theta = band_power(*BANDS["theta"])
    alpha = band_power(*BANDS["alpha"])
    beta = band_power(*BANDS["beta"])
    gamma = band_power(*BANDS["gamma"])

    low = band_power(1.0, 13.0)
    high = band_power(13.0, 45.0)

    # -------------------------------------------------------------------------
    # QUALITY PROXY (raw route, corrected).
    #
    # The OLD proxy 10*log10(low/high) was CONFOUNDED: eye-blink and drift energy
    # is <4 Hz (in the numerator), so blink-contaminated data scored as HIGHER
    # "SNR". That rewards exactly the artifact we remove. On real ZuCo there is no
    # clean reference, so we cannot report a true SNR here regardless.
    #
    # Instead we report an ARTIFACT-SENSITIVE quality proxy that a good denoiser
    # should INCREASE: alpha-band prominence = alpha power relative to the broadband
    # neural baseline (theta+alpha+beta). Blink/drift removal RAISES this ratio,
    # because it strips low-frequency non-neural power without touching alpha.
    # This is still a proxy, not clean-reference SNR — for a true quantitative SNR
    # number, use the semi-synthetic controlled-corruption test (step9b), which has
    # ground truth.
    baseline = band_power(4.0, 30.0)   # theta+alpha+beta broadband neural baseline
    if baseline > EPS and alpha > EPS:
        alpha_prominence_db = float(10.0 * np.log10(alpha / baseline))
    else:
        alpha_prominence_db = np.nan

    # Also report a delta-contamination index: high delta fraction indicates
    # residual ocular/drift artifact. A good denoiser should DECREASE this.
    delta_frac = float(delta / total) if total > EPS else np.nan

    # Keep the old ratio too, but RENAMED honestly so nobody reads it as SNR.
    if high > EPS and low > EPS:
        low_high_ratio_db = float(10.0 * np.log10(low / high))
    else:
        low_high_ratio_db = np.nan

    def relative(power):
        return float(power / total) if total > EPS else np.nan

    # Step 3 notch-filtered at 50 Hz, so this is expected to be absent.
    p50 = band_power(49.0, 51.0)

    return {
        "total_power": total,
        "delta_power": delta,
        "theta_power": theta,
        "alpha_power": alpha,
        "beta_power": beta,
        "gamma_power": gamma,
        "alpha_relative": relative(alpha),
        "beta_relative": relative(beta),
        "gamma_relative": relative(gamma),
        "low_frequency_power": low,
        "high_frequency_power": high,
        "alpha_prominence_db": alpha_prominence_db,
        "delta_contamination_frac": delta_frac,
        "low_high_ratio_db": low_high_ratio_db,
        "spectral_snr_proxy_db": alpha_prominence_db,  # back-compat alias -> corrected metric
        "power_50hz": p50,
    }


def compute_metrics(x: np.ndarray) -> dict:
    metrics = {
        "rms": robust_rms(x),
        "variance": channel_variance(x),
        "peak_to_peak": peak_to_peak(x),
    }
    metrics.update(spectral_metrics(x, FS))
    return metrics


def load_sentence_indices(root: Path) -> list[int]:
    indices = []

    for p in root.glob("sentence_*"):
        if not p.is_dir():
            continue

        try:
            idx = int(p.name.split("_")[1])
        except (IndexError, ValueError):
            continue

        indices.append(idx)

    return sorted(indices)


def evaluate_subject(
    subject: str,
    baselines_root: Path,
    output_root: Path,
    max_sentences: int | None,
):
    subject_root = baselines_root / subject

    if not subject_root.exists():
        raise RuntimeError(f"Missing baseline directory: {subject_root}")

    baseline_dirs = {}
    for baseline in BASELINES:
        p = subject_root / baseline
        if not p.exists():
            raise RuntimeError(f"Missing baseline: {p}")
        baseline_dirs[baseline] = p

    indices = load_sentence_indices(baseline_dirs["no_denoising"])

    if not indices:
        raise RuntimeError("No sentence directories found.")

    if max_sentences is not None:
        indices = indices[:max_sentences]

    print("=" * 80)
    print("MindInk / ZuCo Step 8")
    print("SNR / Signal-Quality Evaluation")
    print("=" * 80)
    print(f"Subject/session : {subject}")
    print(f"Sampling rate   : {FS} Hz")
    print(f"Channels        : {N_CHANNELS}")
    print()
    print("Baselines:")
    print("  no_denoising")
    print("  iclabel_hard_reject")
    print()
    print("Metric note:")
    print("  Raw route: 0.5-100 Hz band. No clean reference on real ZuCo, so we")
    print("  report alpha_prominence_db = 10*log10(alpha/(theta+alpha+beta)); a good")
    print("  denoiser INCREASES it. True SNR: use the step9b synthetic-corruption test.")
    print()

    records = []

    for pos, idx in enumerate(indices, start=1):
        sentence_name = f"sentence_{idx:03d}"

        print(f"[{pos}/{len(indices)}] Sentence {idx:03d}")

        row = {
            "subject": subject,
            "sentence_index": idx,
        }

        for baseline in BASELINES:
            path = (
                baseline_dirs[baseline]
                / sentence_name
                / "sensor_continuous.npy"
            )

            if not path.exists():
                raise RuntimeError(f"Missing continuous EEG: {path}")

            x = load_sentence(path)

            print(
                f"  {baseline:22s}: "
                f"shape={x.shape}"
            )

            metrics = compute_metrics(x)

            for key, value in metrics.items():
                row[f"{baseline}_{key}"] = value

        # Direct comparison: ICLabel hard reject minus no-denoising.
        for metric in (
            "rms",
            "variance",
            "peak_to_peak",
            "total_power",
            "delta_power",
            "theta_power",
            "alpha_power",
            "beta_power",
            "gamma_power",
            "alpha_relative",
            "beta_relative",
            "gamma_relative",
            "low_frequency_power",
            "high_frequency_power",
            "spectral_snr_proxy_db",
            "alpha_prominence_db",
            "delta_contamination_frac",
            "low_high_ratio_db",
            "power_50hz",
        ):
            a = row[f"no_denoising_{metric}"]
            b = row[f"iclabel_hard_reject_{metric}"]

            row[f"delta_{metric}"] = (
                b - a
                if np.isfinite(a) and np.isfinite(b)
                else np.nan
            )

        records.append(row)

    output_dir = output_root / subject
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "sentence_metrics.csv"

    fieldnames = list(records[0].keys())

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    # Save a numeric matrix for downstream analysis.
    metric_fields = [
        k for k in fieldnames
        if k not in ("subject", "sentence_index")
    ]

    matrix = np.array(
        [
            [
                r[k] if np.isfinite(r[k]) else np.nan
                for k in metric_fields
            ]
            for r in records
        ],
        dtype=np.float64,
    )

    np.save(output_dir / "metrics.npy", matrix)

    # Summary statistics.
    summary = {
        "subject": subject,
        "sentences_evaluated": len(records),
        "sampling_rate_hz": FS,
        "channels": N_CHANNELS,
        "baselines": list(BASELINES),
        "metric_fields": metric_fields,
        "note": (
            "Step 3 low-pass filtered at 45 Hz, so exact high-frequency "
            "EEG SNR cannot be recovered. spectral_snr_proxy_db is a "
            "within-band spectral concentration proxy."
        ),
        "summary": {},
    }

    for baseline in BASELINES:
        summary["summary"][baseline] = {}

        for metric in (
            "rms",
            "variance",
            "peak_to_peak",
            "total_power",
            "delta_power",
            "theta_power",
            "alpha_power",
            "beta_power",
            "gamma_power",
            "alpha_relative",
            "beta_relative",
            "gamma_relative",
            "low_frequency_power",
            "high_frequency_power",
            "spectral_snr_proxy_db",
            "alpha_prominence_db",
            "delta_contamination_frac",
            "low_high_ratio_db",
            "power_50hz",
        ):
            values = np.array(
                [
                    r[f"{baseline}_{metric}"]
                    for r in records
                ],
                dtype=float,
            )

            finite = values[np.isfinite(values)]

            if finite.size:
                summary["summary"][baseline][metric] = {
                    "mean": float(np.mean(finite)),
                    "std": float(np.std(finite)),
                    "median": float(np.median(finite)),
                    "min": float(np.min(finite)),
                    "max": float(np.max(finite)),
                    "finite_count": int(finite.size),
                }
            else:
                summary["summary"][baseline][metric] = {
                    "mean": None,
                    "std": None,
                    "median": None,
                    "min": None,
                    "max": None,
                    "finite_count": 0,
                }

    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print()
    print("=" * 80)
    print("STEP 8 — VALIDATION")
    print("=" * 80)
    print(f"Sentences evaluated : {len(records)}")
    print(f"CSV                 : {csv_path}")
    print(f"Metrics matrix      : {output_dir / 'metrics.npy'}")
    print(f"Summary              : {summary_path}")

    for baseline in BASELINES:
        values = np.array(
            [
                r[f"{baseline}_spectral_snr_proxy_db"]
                for r in records
            ],
            dtype=float,
        )
        finite = values[np.isfinite(values)]

        if finite.size:
            print(
                f"{baseline:22s}: "
                f"proxy SNR mean={np.mean(finite):.4f} dB, "
                f"median={np.median(finite):.4f} dB"
            )

    print()
    print("Step 8 outputs:")
    print("  sentence-level signal-quality metrics")
    print("  baseline comparison metrics")
    print("  spectral SNR proxy")
    print()
    print("IMPORTANT:")
    print("  This is a signal-quality evaluation, not clean-ground-truth SNR.")
    print()
    print("STEP 8 COMPLETE")


def main():
    parser = argparse.ArgumentParser(
        description="MindInk Step 8 SNR/signal-quality evaluation"
    )
    parser.add_argument(
        "--subject",
        required=True,
        help="Subject/session ID, e.g. YAG",
    )
    parser.add_argument(
        "--baselines",
        default="baselines",
        help="Step-7 baseline root",
    )
    parser.add_argument(
        "--output",
        default="metrics",
        help="Step-8 output root",
    )
    parser.add_argument(
        "--max-sentences",
        type=int,
        default=None,
        help="Process only first N sentences",
    )

    args = parser.parse_args()

    if args.max_sentences is not None and args.max_sentences <= 0:
        raise SystemExit("--max-sentences must be positive")

    evaluate_subject(
        subject=args.subject,
        baselines_root=Path(args.baselines),
        output_root=Path(args.output),
        max_sentences=args.max_sentences,
    )


if __name__ == "__main__":
    main()
