"""
MindInk / ZuCo Step 6
=====================

Component feature-vector construction.

Input
-----
    ica/<SUBJECT>/
        mixing_matrix.npy
        metadata.json

    segments/<SUBJECT>/
        sentence_XXX/
            words/
                word_YYY/
                    fixation_ZZZ_sources.npy

Features per ICA component
--------------------------
1. Relative PSD / band-power features:
       delta, theta, alpha, beta, gamma
2. ICLabel posterior probabilities:
       brain, muscle, eye, heart, line_noise, channel_noise, other
3. Kurtosis
4. Sample entropy
5. Autocorrelation statistics
6. Topography statistics from mixing column A[:, k]

Important
---------
- Component index is NEVER used as a feature.
- ICA component ordering is NOT assumed to be comparable across subjects.
- The ICA mixing matrix is frozen; this step only extracts features.
- Features are computed from the complete available component source
  time series for the subject/session, not from individual word segments.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import mne
from scipy import signal, stats

try:
    from mne_icalabel.iclabel import (
        get_iclabel_features,
        run_iclabel as run_iclabel_network,
    )
except ImportError as exc:
    raise ImportError(
        "mne_icalabel is required for Step 6. "
        "Install it with: pip install mne-icalabel"
    ) from exc


# ============================================================================
# Configuration
# ============================================================================

DEFAULT_SUBJECT = "YAG"

ICA_ROOT = "ica"
SEGMENTS_ROOT = "segments"
OUTPUT_ROOT = "features"

SFREQ = 250.0
# Raw route: the .fif from step3_raw_preprocess carries the real channel set
# (104 E-channels; 105 if RECOVER_CZ was enabled) WITH names + montage already set.
# N_CHANNELS is verified against the loaded raw at runtime, not hard-assumed.
N_CHANNELS = 104
N_COMPONENTS = 98

ICLABEL_L_FREQ = 1.0
ICLABEL_H_FREQ = 45.0

# ---------------------------------------------------------------------------
# ZuCo 2.0 retained 105-channel scalp layout
# ---------------------------------------------------------------------------
# The Step-3/Step-4 FIF files currently carry generic names EEG001..EEG105.
# Their order is the retained ZuCo scalp-channel order below.  This list is
# documented in the ZuCo EEG-to-text literature and is used here only to
# restore channel identity/geometry; it does NOT change the EEG samples or
# the frozen ICA decomposition.
ZUCO_CHANNEL_NAMES = [
    "E2", "E3", "E4", "E5", "E6", "E7", "E9", "E10", "E11",
    "E12", "E13", "E15", "E16", "E18", "E19", "E20", "E22",
    "E23", "E24", "E26", "E27", "E28", "E29", "E30", "E31",
    "E33", "E34", "E35", "E36", "E37", "E38", "E39", "E40",
    "E41", "E42", "E43", "E44", "E45", "E46", "E47", "E50",
    "E51", "E52", "E53", "E54", "E55", "E57", "E58", "E59",
    "E60", "E61", "E62", "E64", "E65", "E66", "E67", "E69",
    "E70", "E71", "E72", "E74", "E75", "E76", "E77", "E78",
    "E79", "E80", "E82", "E83", "E84", "E85", "E86", "E87",
    "E89", "E90", "E91", "E92", "E93", "E95", "E96", "E97",
    "E98", "E100", "E101", "E102", "E103", "E104", "E105",
    "E106", "E108", "E109", "E110", "E111", "E112", "E114",
    "E115", "E116", "E117", "E118", "E120", "E121", "E122",
    "E123", "E124", "Cz"
]

# RAW ROUTE: ZUCO_CHANNEL_NAMES is retained only for reference. The raw .fif
# already carries real channel names + montage, so no renaming happens and this
# list is NOT required to equal N_CHANNELS. (Length check intentionally removed.)

ZUCO_MONTAGE_NAME = "GSN-HydroCel-128"
ICLABEL_L_FREQ = 1.0
ICLABEL_H_FREQ = 100.0

# EEG bands required for the feature vector.
BANDS = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 45.0),
}

# Autocorrelation lags in samples at 250 Hz.
AUTOCORR_LAGS = (1, 5, 10, 25, 50)

# Number of samples used for ICLabel inference.
# ICLabel works on an MNE Raw object and needs channel information.
MAX_ICLABEL_SECONDS = 120.0

# Minimum samples required for robust time-domain features.
MIN_FEATURE_SAMPLES = 100


# ============================================================================
# Utilities
# ============================================================================

def finite_1d(x):
    """Return finite values from a 1-D array."""
    x = np.asarray(x, dtype=np.float64).ravel()
    return x[np.isfinite(x)]


def safe_mean(x):
    x = finite_1d(x)
    return float(np.mean(x)) if x.size else 0.0


def safe_std(x):
    x = finite_1d(x)
    return float(np.std(x)) if x.size else 0.0


def safe_rms(x):
    x = finite_1d(x)
    return float(np.sqrt(np.mean(x * x))) if x.size else 0.0


def safe_kurtosis(x):
    x = finite_1d(x)
    if x.size < 4:
        return 0.0
    value = stats.kurtosis(
        x,
        fisher=True,
        bias=False
    )
    return float(value) if np.isfinite(value) else 0.0


def robust_standardize(x):
    """
    Robustly standardize a component only for numerical feature
    calculations. This does NOT modify the saved EEG.
    """
    x = finite_1d(x)

    if x.size == 0:
        return x

    median = np.median(x)
    mad = np.median(np.abs(x - median))

    if mad > 0:
        return (x - median) / (1.4826 * mad)

    std = np.std(x)

    if std > 0:
        return (x - np.mean(x)) / std

    return x - np.mean(x)


# ============================================================================
# Spectral features
# ============================================================================

def compute_psd_features(x, sfreq):
    """
    Compute normalized log band powers.

    Returns:
        dict containing total log power and relative log powers.
    """

    x = finite_1d(x)

    if x.size < 16:
        result = {
            "total_log_power": 0.0
        }

        for band in BANDS:
            result[f"{band}_relative_power"] = 0.0
            result[f"{band}_log_power"] = 0.0

        return result

    nperseg = min(
        x.size,
        int(round(sfreq * 4.0))
    )

    freqs, psd = signal.welch(
        x,
        fs=sfreq,
        nperseg=nperseg,
        noverlap=nperseg // 2,
        detrend="constant"
    )

    psd = np.maximum(
        np.asarray(psd, dtype=np.float64),
        np.finfo(np.float64).tiny
    )

    total_mask = (
        (freqs >= 1.0)
        & (freqs <= 45.0)
    )

    total_power = np.trapezoid(
        psd[total_mask],
        freqs[total_mask]
    )

    total_power = max(
        float(total_power),
        np.finfo(np.float64).tiny
    )

    result = {
        "total_log_power":
            float(np.log10(total_power))
    }

    for name, (low, high) in BANDS.items():

        mask = (
            (freqs >= low)
            & (freqs <= high)
        )

        if np.count_nonzero(mask) >= 2:
            power = np.trapezoid(
                psd[mask],
                freqs[mask]
            )
        else:
            power = 0.0

        power = max(
            float(power),
            np.finfo(np.float64).tiny
        )

        result[f"{name}_log_power"] = float(
            np.log10(power)
        )

        result[f"{name}_relative_power"] = float(
            power / total_power
        )

    return result


# ============================================================================
# Entropy
# ============================================================================

def sample_entropy(x, m=2, r_ratio=0.2):
    """
    Lightweight sample entropy implementation.

    SampEn = -log(A/B), where:
        B = matching m-length templates
        A = matching (m+1)-length templates

    The tolerance is r_ratio * std(x).
    """

    x = finite_1d(x)

    if x.size < (m + 2):
        return 0.0

    # Downsample very long recordings for manageable O(N^2) complexity.
    max_points = 5000

    if x.size > max_points:
        idx = np.linspace(
            0,
            x.size - 1,
            max_points
        ).astype(int)
        x = x[idx]

    x = robust_standardize(x)

    if x.size < (m + 2):
        return 0.0

    r = r_ratio * np.std(x)

    if r <= 0 or not np.isfinite(r):
        return 0.0

    def count_matches(mm):
        templates = np.lib.stride_tricks.sliding_window_view(
            x,
            mm
        )

        n = templates.shape[0]

        if n < 2:
            return 0

        count = 0

        for i in range(n - 1):
            distances = np.max(
                np.abs(
                    templates[i + 1:] - templates[i]
                ),
                axis=1
            )
            count += int(
                np.count_nonzero(distances <= r)
            )

        return count

    b = count_matches(m)
    a = count_matches(m + 1)

    if a == 0 or b == 0:
        return 0.0

    value = -np.log(
        float(a) / float(b)
    )

    return float(value) if np.isfinite(value) else 0.0


# ============================================================================
# Autocorrelation
# ============================================================================

def autocorrelation_features(x, lags=AUTOCORR_LAGS):
    x = finite_1d(x)

    if x.size < 2:
        return {
            f"autocorr_lag_{lag}": 0.0
            for lag in lags
        }

    x = x - np.mean(x)

    denom = np.dot(x, x)

    result = {}

    for lag in lags:

        if lag >= x.size or denom <= 0:
            result[f"autocorr_lag_{lag}"] = 0.0
            continue

        value = np.dot(
            x[:-lag],
            x[lag:]
        ) / denom

        result[f"autocorr_lag_{lag}"] = (
            float(value)
            if np.isfinite(value)
            else 0.0
        )

    return result


# ============================================================================
# Topography features
# ============================================================================

def topography_features(topography):
    """
    Statistics of mixing column A[:, k].

    The component index is deliberately absent.
    """

    a = finite_1d(topography)

    if a.size == 0:
        return {
            "topo_mean": 0.0,
            "topo_std": 0.0,
            "topo_rms": 0.0,
            "topo_abs_mean": 0.0,
            "topo_abs_max": 0.0,
            "topo_kurtosis": 0.0,
            "topo_skewness": 0.0,
            "topo_energy": 0.0,
            "topo_l2_norm": 0.0,
            "topo_peak_to_rms": 0.0,
        }

    rms = safe_rms(a)

    return {
        "topo_mean": safe_mean(a),
        "topo_std": safe_std(a),
        "topo_rms": rms,
        "topo_abs_mean":
            float(np.mean(np.abs(a))),
        "topo_abs_max":
            float(np.max(np.abs(a))),
        "topo_kurtosis":
            safe_kurtosis(a),
        "topo_skewness":
            float(stats.skew(a, bias=False))
            if a.size >= 3
            else 0.0,
        "topo_energy":
            float(np.sum(a * a)),
        "topo_l2_norm":
            float(np.linalg.norm(a)),
        "topo_peak_to_rms":
            float(np.max(np.abs(a)) / rms)
            if rms > 0
            else 0.0,
    }


# ============================================================================
# Load ICA sources
# ============================================================================

def load_all_component_sources(
    segments_subject,
    n_components=N_COMPONENTS
):
    """
    Load and concatenate all sentence-level component source data.

    We intentionally use the continuous ICA source files from Step 4,
    rather than concatenating word/fixation segments. This avoids
    duplicating overlapping/gapped fixation intervals.
    """

    source_files = sorted(
        (
            Path(segments_subject)
            / "sentence_000"
        ).parent.glob(
            "sentence_*/words/word_*/*_sources.npy"
        )
    )

    # Prefer the Step-4 continuous sources if available.
    step4_candidate_roots = []

    # segments/YAG is not assumed to contain continuous source files,
    # so this function is replaced at runtime by explicit Step-4 loading.
    _ = source_files

    raise RuntimeError(
        "Internal loader placeholder. "
        "Use load_step4_continuous_sources() below."
    )


def load_step4_continuous_sources(
    ica_subject,
    n_components=N_COMPONENTS
):
    """
    Load sentence-level continuous ICA source files from Step 4.

    Expected:
        ica/<subject>/sources/sentence_XXX_sources.npy

    Each file:
        components x samples
    """

    sources_dir = (
        Path(ica_subject)
        / "sources"
    )

    if not sources_dir.exists():
        raise FileNotFoundError(
            f"Step 4 sources directory not found: "
            f"{sources_dir}"
        )

    files = sorted(
        sources_dir.glob(
            "sentence_*_sources.npy"
        )
    )

    if not files:
        raise FileNotFoundError(
            f"No Step 4 source files found in {sources_dir}"
        )

    arrays = []

    for path in files:

        arr = np.load(
            path,
            mmap_mode="r"
        )

        if arr.ndim != 2:
            raise RuntimeError(
                f"{path}: expected 2-D array, "
                f"got {arr.shape}"
            )

        if arr.shape[0] != n_components:
            raise RuntimeError(
                f"{path}: expected "
                f"{n_components} components, "
                f"got {arr.shape[0]}"
            )

        arrays.append(
            np.asarray(arr, dtype=np.float64)
        )

    return np.concatenate(
        arrays,
        axis=1
    ), files


# ============================================================================
# ICLabel
# ============================================================================

def _make_zuco_montage():
    """Build the 105-channel ZuCo montage from GSN-HydroCel-128 positions."""

    hydrocel = mne.channels.make_standard_montage(
        ZUCO_MONTAGE_NAME
    )

    hydrocel_pos = hydrocel.get_positions()["ch_pos"]

    ch_pos = {}
    missing = []

    for name in ZUCO_CHANNEL_NAMES:
        if name == "Cz":
            continue
        if name not in hydrocel_pos:
            missing.append(name)
        else:
            ch_pos[name] = hydrocel_pos[name]

    if missing:
        raise RuntimeError(
            "The GSN-HydroCel-128 montage is missing required ZuCo "
            f"electrodes: {missing}"
        )

    # ZuCo's published 105-channel list contains Cz as the final entry.
    # MNE's GSN-HydroCel-128 montage names its 128 electrodes E1..E128
    # and does not expose a separate 'Cz'.  Use the standard 10-20 Cz
    # coordinate only to supply the geometry required by ICLabel.
    standard_1020 = mne.channels.make_standard_montage("standard_1020")
    standard_pos = standard_1020.get_positions()["ch_pos"]

    if "Cz" not in standard_pos:
        raise RuntimeError("standard_1020 montage does not contain Cz.")

    ch_pos["Cz"] = standard_pos["Cz"]

    montage = mne.channels.make_dig_montage(
        ch_pos=ch_pos,
        coord_frame="head"
    )

    return montage


def apply_zuco_channel_geometry(analysis_raw, ica):
    """
    RAW ROUTE: trust the .fif's own geometry.

    step3_raw_preprocess already set real channel names (E2, E3, ... , optionally
    Cz) AND a montage built from the file's own electrode coordinates. So there is
    nothing to rename or rebuild here. We only VERIFY that a montage is present and
    that the ICA and Raw share the same channel set, then return the names.

    This deliberately replaces the old preprocessed-route surgery (renaming generic
    EEG001.. placeholders and rebuilding a HydroCel montage), which does not apply
    when the geometry is already correct on disk.
    """
    n = analysis_raw.info["nchan"]
    if len(ica.ch_names) != n:
        raise RuntimeError(
            "Frozen ICA channel count does not match the sensor data: "
            f"ICA={len(ica.ch_names)}, Raw={n}."
        )

    if list(ica.ch_names) != list(analysis_raw.ch_names):
        # Same count but different names/order would misalign topographies.
        raise RuntimeError(
            "ICA and Raw channel names/order differ. On the raw route both come "
            "from the same Step-3/Step-4 files and must match exactly."
        )

    if analysis_raw.get_montage() is None:
        raise RuntimeError(
            "Raw has no montage. step3_raw_preprocess should have attached one from "
            "the file's electrode coordinates; re-run Step 3."
        )

    # Ensure the ICA object also exposes an ICLabel-usable montage. get_iclabel_features
    # builds topographies from the ICA's channel geometry, so give it the standard
    # HydroCel positions by name (E2..E124 match GSN-HydroCel-128 exactly).
    if getattr(ica, "info", None) is not None:
        try:
            std = mne.channels.make_standard_montage("GSN-HydroCel-128")
            ica.info.set_montage(std, on_missing="warn", match_case=True,
                                 verbose=False)
        except Exception as _e:
            print(f"  WARN: could not set standard montage on ICA info: {_e}")

    return list(analysis_raw.ch_names)


# ============================================================================
# PATCH for step6_features.py  ->  replace your existing prepare_iclabel_raw()
# with the version below. The ONLY substantive change is the added average
# reference (CAR). ICLabel's topography features assume CAR; without it, scalp
# maps are distorted and confident classes (esp. EYE) collapse into "other".
# The 1 Hz high-pass you already had is kept.
# ============================================================================

def prepare_iclabel_raw(analysis_raw):
    """
    Create the temporary Raw used only for ICLabel inference.

    ICLabel REQUIRES:
      * common average reference (CAR)  <-- this was missing; it is the fix
      * 1 Hz high-pass (100 Hz low-pass ideally; ours is capped at 45 Hz by Step 3)

    The saved EEG and frozen ICA are never modified — this is a temporary copy.
    Note: 45-100 Hz is still unrecoverable (Step 3 low-pass), so muscle/line
    classes stay unreliable. CAR only rescues the LOW-frequency classes (eye),
    which is exactly what was collapsing into "other".
    """
    iclabel_raw = analysis_raw.copy()

    # --- Attach a STANDARD HydroCel montage BY NAME for ICLabel geometry -------
    # The raw .fif's montage came from EEGLAB XYZ coordinates whose frame/scale
    # ICLabel's 2D topo projection cannot handle (it collapses -> "array must be
    # at least 2-d"). Our channels are named E2..E124, which match MNE's
    # GSN-HydroCel-128 standard montage exactly, so we can supply correct,
    # ICLabel-validated 3D positions by name. This changes ONLY electrode geometry
    # on this temporary copy; data samples and the frozen ICA are untouched.
    try:
        std = mne.channels.make_standard_montage("GSN-HydroCel-128")
        iclabel_raw.set_montage(std, on_missing="warn", match_case=True,
                                verbose=False)
    except Exception as _e:
        print(f"  WARN: could not attach standard HydroCel montage: {_e}")

    # --- THE FIX: average reference, as ICLabel was trained to expect ---------
    # projection=False applies it directly to the data (not as a deferred proj),
    # which is what get_iclabel_features needs to see.
    iclabel_raw.set_eeg_reference(
        "average",
        projection=False,
        verbose=False,
    )

    # --- high-pass at 1 Hz if not already (unchanged from your version) -------
    if iclabel_raw.info["highpass"] < ICLABEL_L_FREQ:
        iclabel_raw.filter(
            l_freq=ICLABEL_L_FREQ,
            h_freq=None,
            picks="eeg",
            method="fir",
            phase="zero",
            verbose=False,
        )

    print(
        "  ICLabel reference           : average (CAR) applied"
    )
    print(
        "  ICLabel available bandwidth : "
        f"{ICLABEL_L_FREQ:.1f}-{iclabel_raw.info['lowpass']:.1f} Hz"
    )
    print(
        "  ICLabel note                : 45-100 Hz still unrecoverable "
        "(Step 3 low-pass); muscle/line remain unreliable."
    )

    return iclabel_raw


def run_iclabel(ica, analysis_raw):
    """
    Run the ICLabel neural network and retain the FULL 7-class posterior.

    ``mne_icalabel.label_components()`` intentionally returns only the
    winning-class probability per component (shape ``(n_components,)``).
    Step 6 requires the complete seven-class posterior vector, so we use the
    lower-level ICLabel API: ``get_iclabel_features()`` followed by
    ``run_iclabel()``.

    Returns
    -------
    labels : list of str
        Winning ICLabel class for each component.
    probabilities : ndarray
        Full posterior matrix with shape ``(n_components, 7)``.
    """

    if analysis_raw.get_montage() is None:
        raise RuntimeError(
            "Cannot run ICLabel: analysis Raw has no montage."
        )
    # RAW ROUTE: channel names are the real E-labels from the .fif, not the fixed
    # ZUCO_CHANNEL_NAMES list, so the old exact-order guard is intentionally dropped.

    iclabel_raw = prepare_iclabel_raw(analysis_raw)

    try:
        # MNE-ICLabel returns its internal features in the layout:
        #   topo      : (32, 32, 1, n_components)
        #   psd       : (1, 100, 1, n_components)
        #   autocorr  : (1, 100, 1, n_components)
        topo, psd, autocorr = get_iclabel_features(
            iclabel_raw,
            ica
        )
        print("  ICLabel feature shapes:")
        print(f"    topo     : {topo.shape}")
        print(f"    psd      : {psd.shape}")
        print(f"    autocorr : {autocorr.shape}")

        probabilities = run_iclabel_network(
            topo,
            psd,
            autocorr,
            backend="torch"
        )
        # The lower-level network API expects:
        #   images    : (n_components, 1, 32, 32)
        #   psds      : (n_components, 1, 1, 100)
        #   autocorr  : (n_components, 1, 1, 100)
       

    except Exception as exc:
        raise RuntimeError(
            "ICLabel inference failed after restoring the ZuCo "
            "HydroCel montage. Check that the frozen ICA and the 105-channel "
            "Raw have identical channel order."
        ) from exc

    probabilities = np.asarray(
        probabilities,
        dtype=np.float64
    )

    if probabilities.ndim != 2:
        raise RuntimeError(
            f"Unexpected ICLabel probability shape: {probabilities.shape}"
        )

    if probabilities.shape != (N_COMPONENTS, 7):
        raise RuntimeError(
            "Unexpected ICLabel posterior shape: "
            f"{probabilities.shape}; expected {(N_COMPONENTS, 7)}."
        )

    iclabel_names = [
        "brain",
        "muscle",
        "eye",
        "heart",
        "line_noise",
        "channel_noise",
        "other",
    ]

    labels = [
        iclabel_names[int(np.argmax(row))]
        for row in probabilities
    ]

    # Numerical sanity check: each row should be a probability distribution.
    row_sums = probabilities.sum(axis=1)
    if not np.all(np.isfinite(probabilities)):
        raise RuntimeError(
            "ICLabel returned NaN/Inf probabilities."
        )

    if not np.allclose(row_sums, 1.0, atol=1e-5):
        raise RuntimeError(
            "ICLabel posterior rows do not sum to 1. "
            f"Range: [{row_sums.min():.6f}, {row_sums.max():.6f}]"
        )

    return labels, probabilities


# ============================================================================
# Build feature vector
# ============================================================================

def build_component_features(
    sources,
    mixing,
    sfreq,
    iclabel_probabilities
):
    """
    Build one feature vector per ICA component.

    No component index is included.
    """

    n_components = sources.shape[0]

    if mixing.shape[1] != n_components:
        raise RuntimeError(
            "Mixing matrix/component source mismatch: "
            f"A={mixing.shape}, "
            f"sources={sources.shape}"
        )

    if iclabel_probabilities.shape[0] != n_components:
        raise RuntimeError(
            "ICLabel/component mismatch: "
            f"ICLabel={iclabel_probabilities.shape}, "
            f"sources={sources.shape}"
        )

    rows = []
    feature_names = []

    for k in range(n_components):

        x = np.asarray(
            sources[k],
            dtype=np.float64
        )

        # ------------------------------------------------------------
        # Basic time-domain features
        # ------------------------------------------------------------

        x_finite = finite_1d(x)

        if x_finite.size < MIN_FEATURE_SAMPLES:
            raise RuntimeError(
                f"Component {k} has only "
                f"{x_finite.size} finite samples; "
                "insufficient for Step 6."
            )

        features = {}

        # ------------------------------------------------------------
        # PSD / band powers
        # ------------------------------------------------------------

        features.update(
            compute_psd_features(
                x_finite,
                sfreq
            )
        )

        # ------------------------------------------------------------
        # ICLabel posterior probabilities
        # ------------------------------------------------------------

        iclabel_names = [
            "brain",
            "muscle",
            "eye",
            "heart",
            "line_noise",
            "channel_noise",
            "other",
        ]

        if iclabel_probabilities.shape[1] != 7:
            raise RuntimeError(
                "Expected 7 ICLabel classes, got "
                f"{iclabel_probabilities.shape[1]}"
            )

        for j, name in enumerate(iclabel_names):
            features[
                f"iclabel_{name}_prob"
            ] = float(
                iclabel_probabilities[k, j]
            )

        # ------------------------------------------------------------
        # Kurtosis
        # ------------------------------------------------------------

        features["kurtosis"] = safe_kurtosis(
            x_finite
        )

        # ------------------------------------------------------------
        # Sample entropy
        # ------------------------------------------------------------

        features["sample_entropy"] = sample_entropy(
            x_finite
        )

        # ------------------------------------------------------------
        # Autocorrelation
        # ------------------------------------------------------------

        features.update(
            autocorrelation_features(
                x_finite
            )
        )

        # ------------------------------------------------------------
        # Topography
        # ------------------------------------------------------------

        features.update(
            topography_features(
                mixing[:, k]
            )
        )

        rows.append(features)

        if not feature_names:
            feature_names = list(
                features.keys()
            )

    matrix = np.asarray(
        [
            [row[name] for name in feature_names]
            for row in rows
        ],
        dtype=np.float64
    )

    return matrix, feature_names


# ============================================================================
# Main Step 6
# ============================================================================

def run_step6(
    subject_id=DEFAULT_SUBJECT,
    ica_root=ICA_ROOT,
    segments_root=SEGMENTS_ROOT,
    output_root=OUTPUT_ROOT
):

    print("=" * 80)
    print("MindInk / ZuCo Step 6")
    print("ICA Component Feature-Vector Construction")
    print("=" * 80)

    ica_subject = (
        Path(ica_root)
        / subject_id
    )

    segments_subject = (
        Path(segments_root)
        / subject_id
    )

    output_subject = (
        Path(output_root)
        / subject_id
    )

    output_subject.mkdir(
        parents=True,
        exist_ok=True
    )

    # ------------------------------------------------------------
    # Validate Step 4 metadata
    # ------------------------------------------------------------

    metadata_path = (
        ica_subject
        / "metadata.json"
    )

    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Step 4 metadata not found: "
            f"{metadata_path}"
        )

    with open(
        metadata_path,
        "r",
        encoding="utf-8"
    ) as fp:
        ica_metadata = json.load(fp)

    # Raw Step 4 writes "n_components"; preprocessed Step 4 wrote "n_components_fitted".
    fitted = int(
        ica_metadata.get("n_components",
                         ica_metadata.get("n_components_fitted", -1))
    )

    if fitted != N_COMPONENTS:
        raise RuntimeError(
            f"Expected {N_COMPONENTS} ICA components, "
            f"but Step 4 reports {fitted}."
        )

    # ------------------------------------------------------------
    # Load frozen ICA model
    # ------------------------------------------------------------

    ica_model_path = (
        ica_subject
        / "ica_model-ica.fif"
    )

    if not ica_model_path.exists():
        # Backward compatibility with an older saved name.
        fallback = (
            ica_subject
            / "ica_model.fif"
        )

        if fallback.exists():
            ica_model_path = fallback
        else:
            raise FileNotFoundError(
                f"ICA model not found under {ica_subject}"
            )

    ica = mne.preprocessing.read_ica(
        ica_model_path,
        verbose=False
    )

    if ica.n_components_ != N_COMPONENTS:
        raise RuntimeError(
            f"Loaded ICA has "
            f"{ica.n_components_} components, "
            f"expected {N_COMPONENTS}."
        )

    # ------------------------------------------------------------
    # Obtain the SENSOR-SPACE ICA mixing matrix A
    # ------------------------------------------------------------
    # Step 4 also saves mixing_matrix.npy, but that saved array is the
    # reduced/internal 98 x 98 representation produced by MNE for this
    # rank-deficient ICA. Step 6 topography features require the actual
    # sensor-space component maps: 105 channels x 98 components.
    #
    # Therefore use the frozen MNE ICA object to recover A. This does NOT
    # refit ICA and does NOT change the ICA decomposition.
    # ------------------------------------------------------------

    # RAW ROUTE: Step 4 saves the true sensor-space A directly as mixing_sensor.npy
    # (n_channels x n_components). Prefer it; fall back to ica.get_components().
    # Sanity-assert they agree if both are available.
    n_ch = ica.get_components().shape[0]
    mixing = np.asarray(ica.get_components(), dtype=np.float64)

    sensor_path = ica_subject / "mixing_sensor.npy"
    if sensor_path.exists():
        saved_sensor = np.load(sensor_path)
        print(f"  Step-4 mixing_sensor.npy    : {saved_sensor.shape}")
        if saved_sensor.shape != mixing.shape:
            raise RuntimeError(
                f"mixing_sensor.npy {saved_sensor.shape} != ica.get_components() "
                f"{mixing.shape}."
            )
        if not np.allclose(saved_sensor, mixing, atol=1e-6):
            print("  WARN: mixing_sensor.npy differs from get_components(); "
                  "using get_components().")
        else:
            mixing = np.asarray(saved_sensor, dtype=np.float64)

    if mixing.shape != (n_ch, N_COMPONENTS):
        raise RuntimeError(
            f"Expected sensor-space mixing ({n_ch}, {N_COMPONENTS}), got {mixing.shape}"
        )

    print(
        f"  Sensor-space mixing matrix A : "
        f"{mixing.shape}"
    )

    # ------------------------------------------------------------
    # Load continuous Step-4 component sources
    # ------------------------------------------------------------

    print("\nLoading continuous ICA sources...")

    sources, source_files = (
        load_step4_continuous_sources(
            ica_subject
        )
    )

    print(
        f"  Source files : {len(source_files)}"
    )

    print(
        f"  Sources      : {sources.shape}"
    )

    # ------------------------------------------------------------
    # Load one representative processed Raw for ICLabel.
    #
    # ICLabel is run using the actual frozen ICA model and the
    # corresponding sensor-space processed EEG, not the pseudo-channel
    # component Raw.
    # ------------------------------------------------------------

    first_sentence = sorted(
        segments_subject.glob(
            "sentence_*/"
        )
    )

    if not first_sentence:
        raise FileNotFoundError(
            f"No sentence directories found in "
            f"{segments_subject}"
        )

    # RAW ROUTE: one continuous analysis FIF per subject (not per-sentence).
    processed_path = (
        Path("processed_raw")
        / subject_id
        / f"{subject_id}_analysis_raw.fif"
    )

    if not processed_path.exists():
        raise FileNotFoundError(
            "Step 6 (raw route) expects the Step-3 continuous Raw at:\n"
            f"  {processed_path}"
        )

    print("\nLoading processed sensor EEG for ICLabel...")

    analysis_raw = mne.io.read_raw_fif(
        processed_path,
        preload=True,
        verbose=False
    )

    if not np.isclose(
        analysis_raw.info["sfreq"],
        SFREQ
    ):
        raise RuntimeError(
            f"Expected {SFREQ} Hz, "
            f"got {analysis_raw.info['sfreq']} Hz."
        )

    # Raw route channel count is whatever Step 3 produced (104, or 105 with Cz).
    if analysis_raw.info["nchan"] != ica.get_components().shape[0]:
        raise RuntimeError(
            f"Raw channels ({analysis_raw.info['nchan']}) != ICA channels "
            f"({ica.get_components().shape[0]})."
        )

    # ------------------------------------------------------------
    # Restore ZuCo sensor identity + HydroCel geometry.
    # ------------------------------------------------------------

    original_channel_names = apply_zuco_channel_geometry(
        analysis_raw,
        ica
    )

    print("\nZuCo channel geometry:")
    print(f"  Original names : {original_channel_names[:3]} ... {original_channel_names[-3:]}")
    print(f"  Restored names : {analysis_raw.ch_names[:3]} ... {analysis_raw.ch_names[-3:]}")
    print(f"  Montage        : from .fif electrode coordinates (raw route)")
    print(f"  Channel count  : {len(analysis_raw.ch_names)}")

    # Use the full continuous source matrix for the numerical features.
    # ICLabel itself is run on a temporary filtered copy of the processed
    # sensor EEG.  The saved Step-3 data and frozen ICA are never modified.
    print("\nRunning ICLabel...")

    labels, probabilities = run_iclabel(
        ica,
        analysis_raw
    )

    print(
        f"  Components classified : "
        f"{probabilities.shape[0]}"
    )

    print(
        f"  Posterior classes      : "
        f"{probabilities.shape[1]}"
    )

    # ------------------------------------------------------------
    # Build feature matrix
    # ------------------------------------------------------------

    print("\nBuilding component feature vectors...")

    feature_matrix, feature_names = (
        build_component_features(
            sources,
            mixing,
            SFREQ,
            probabilities
        )
    )

    print(
        f"  Feature matrix : "
        f"{feature_matrix.shape}"
    )

    # ------------------------------------------------------------
    # Validate no NaN / Inf
    # ------------------------------------------------------------

    if not np.all(
        np.isfinite(feature_matrix)
    ):
        bad = np.argwhere(
            ~np.isfinite(feature_matrix)
        )

        raise RuntimeError(
            "Feature matrix contains NaN/Inf. "
            f"First bad locations: {bad[:10].tolist()}"
        )

    # ------------------------------------------------------------
    # Save matrix
    # ------------------------------------------------------------

    matrix_path = (
        output_subject
        / "component_features.npy"
    )

    np.save(
        matrix_path,
        feature_matrix
    )

    # ------------------------------------------------------------
    # Save feature names
    # ------------------------------------------------------------

    names_path = (
        output_subject
        / "feature_names.json"
    )

    with open(
        names_path,
        "w",
        encoding="utf-8"
    ) as fp:
        json.dump(
            feature_names,
            fp,
            indent=2
        )

    # ------------------------------------------------------------
    # Save ICLabel results
    # ------------------------------------------------------------

    iclabel_path = (
        output_subject
        / "iclabel.json"
    )

    iclabel_names = [
        "brain",
        "muscle",
        "eye",
        "heart",
        "line_noise",
        "channel_noise",
        "other",
    ]

    iclabel_records = []

    for k in range(N_COMPONENTS):

        record = {
            "label":
                str(labels[k])
            if labels is not None
            else None,
            "posterior": {
                iclabel_names[j]:
                    float(probabilities[k, j])
                for j in range(7)
            }
        }

        iclabel_records.append(record)

    with open(
        iclabel_path,
        "w",
        encoding="utf-8"
    ) as fp:
        json.dump(
            iclabel_records,
            fp,
            indent=2
        )

    # ------------------------------------------------------------
    # Save metadata
    # ------------------------------------------------------------

    metadata = {
        "step": "6",
        "subject_id": subject_id,
        "sampling_rate": SFREQ,
        "n_channels": N_CHANNELS,
        "n_components": N_COMPONENTS,
        "feature_matrix_shape":
            list(feature_matrix.shape),
        "feature_names":
            feature_names,
        "feature_groups": {
            "spectral": [
                name
                for name in feature_names
                if (
                    "power" in name
                    or name == "total_log_power"
                )
            ],
            "iclabel": [
                name
                for name in feature_names
                if name.startswith("iclabel_")
            ],
            "kurtosis": [
                "kurtosis"
            ],
            "entropy": [
                "sample_entropy"
            ],
            "autocorrelation": [
                name
                for name in feature_names
                if name.startswith("autocorr_")
            ],
            "topography": [
                name
                for name in feature_names
                if name.startswith("topo_")
            ]
        },
        "component_index_used_as_feature":
            False,
        "component_order_assumed_cross_subject":
            False,
        "ica_model":
            str(ica_model_path),
        "mixing_matrix":
            str(ica_subject / "mixing_sensor.npy"),
        "source_files":
            [str(p) for p in source_files],
        "channel_geometry": {
            "channel_names": original_channel_names,
            "montage": "from_fif_coordinates_raw_route",
            "mapping_mode": "trusted_from_step3_fif",
            "data_samples_modified": False
        },
        "iclabel": {
            "package": "mne_icalabel",
            "posterior_classes":
                iclabel_names,
            "temporary_filter_hz": [
                ICLABEL_L_FREQ,
                ICLABEL_H_FREQ
            ],
            "inference_raw_is_temporary_copy": True
        }
    }

    metadata_path_out = (
        output_subject
        / "metadata.json"
    )

    with open(
        metadata_path_out,
        "w",
        encoding="utf-8"
    ) as fp:
        json.dump(
            metadata,
            fp,
            indent=2
        )

    # ------------------------------------------------------------
    # Final validation
    # ------------------------------------------------------------

    print("\n" + "=" * 80)
    print("STEP 6 — VALIDATION")
    print("=" * 80)

    print(
        f"Components          : {feature_matrix.shape[0]}"
    )

    print(
        f"Features/component  : {feature_matrix.shape[1]}"
    )

    print(
        f"Feature matrix      : {matrix_path}"
    )

    print(
        f"Finite values       : "
        f"{np.isfinite(feature_matrix).all()}"
    )

    print(
        "Component index     : NOT USED"
    )

    print(
        "Cross-subject order : NOT ASSUMED"
    )

    print("\nFeature groups:")

    for group, names in metadata[
        "feature_groups"
    ].items():
        print(
            f"  {group:18s}: {len(names)}"
        )

    print("\nSTEP 6 COMPLETE")


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=(
            "MindInk Step 6 — "
            "ICA component feature-vector construction"
        )
    )

    parser.add_argument(
        "--subject",
        default=DEFAULT_SUBJECT
    )

    parser.add_argument(
        "--ica",
        default=ICA_ROOT
    )

    parser.add_argument(
        "--segments",
        default=SEGMENTS_ROOT
    )

    parser.add_argument(
        "--output",
        default=OUTPUT_ROOT
    )

    args = parser.parse_args()

    run_step6(
        subject_id=args.subject,
        ica_root=args.ica,
        segments_root=args.segments,
        output_root=args.output
    )
