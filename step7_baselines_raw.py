"""
step7_baselines.py

MindInk / ZuCo Step 7 — Classical Baselines
============================================

Workplan requirement
--------------------
Step 7 is a comparison baseline, NOT the proposed contribution.

Implemented baselines:

1. no-denoising
   - Pass the Step-3 0.5–45 Hz analysis EEG through unchanged.

2. ICLabel hard-reject
   - Read the frozen Step-6 ICLabel posterior.
   - Reject components whose artifact posterior is >= threshold
     for eye, muscle, or line_noise.
   - Zero those ICA sources.
   - Back-project the remaining frozen ICA sources with the
     sensor-space mixing matrix A:
          X_clean = A @ S_clean

Important
---------
- ICA is never refit.
- Step-4 sources and mixing matrix are frozen.
- No component index is used to decide rejection.
- ZuCo word.rawEEG is not used.
- No claim of novelty is made for ICLabel rejection.
- The threshold is a classical baseline hyperparameter and is
  exposed through --threshold.

Input
-----
processed/<subject>/
    sentence_XXX_analysis_raw.fif

ica/<subject>/
    mixing_matrix.npy
    metadata.json
    sources/sentence_XXX_sources.npy

features/<subject>/
    iclabel.json

segments/<subject>/
    sentence_XXX/metadata.json

Output
------
baselines/<subject>/
    manifest.json

    no_denoising/
        sentence_XXX/
            sensor_continuous.npy
            words/
                word_XXX_sensor.npy
                ...

    iclabel_hard_reject/
        sentence_XXX/
            sensor_continuous.npy
            words/
                word_XXX_sensor.npy
                ...

The word files use the exact processed sample indices already
validated in Step 5B.
"""

from pathlib import Path
import argparse
import json

import numpy as np
import mne


# ============================================================================
# Fixed pipeline configuration
# ============================================================================

DEFAULT_SUBJECT = "YAG"

N_CHANNELS = 104   # raw route (104 E-channels; 105 if RECOVER_CZ)
N_COMPONENTS = 98
SFREQ = 250.0

ARTIFACT_CLASSES = (
    "eye",
    "muscle",
    "line_noise",
)

DEFAULT_THRESHOLD = 0.50


# ============================================================================
# Generic helpers
# ============================================================================

def load_json(path):
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Required JSON not found: {path}")

    with open(path, "r", encoding="utf-8") as fp:
        return json.load(fp)


def save_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as fp:
        json.dump(obj, fp, indent=2)


def require_shape(name, array, expected):
    if tuple(array.shape) != tuple(expected):
        raise RuntimeError(
            f"{name}: expected shape {expected}, got {array.shape}"
        )


# ============================================================================
# Step-4 / Step-6 loading
# ============================================================================

def load_step4_ica(ica_subject):
    """
    Load the frozen Step-4 sensor-space mixing matrix.

    Step 4 also saved a reduced/internal mixing_matrix.npy of shape
    (98, 98). That artifact is retained for provenance, but Step 7
    needs the true sensor-space A = 105 x 98.

    We therefore reconstruct A from the frozen MNE ICA model.
    """

    ica_subject = Path(ica_subject)

    metadata = load_json(ica_subject / "metadata.json")

    fitted = int(metadata.get("n_components", metadata.get("n_components_fitted", -1)))

    if fitted != N_COMPONENTS:
        raise RuntimeError(
            f"Step 4 reports {fitted} ICA components; "
            f"expected fixed {N_COMPONENTS}."
        )

    model_path = ica_subject / "ica_model-ica.fif"

    if not model_path.exists():
        fallback = ica_subject / "ica_model.fif"
        if fallback.exists():
            model_path = fallback
        else:
            raise FileNotFoundError(
                f"Frozen ICA model not found under {ica_subject}"
            )

    ica = mne.preprocessing.read_ica(
        model_path,
        verbose=False
    )

    if ica.n_components_ != N_COMPONENTS:
        raise RuntimeError(
            f"Loaded ICA has {ica.n_components_} components; "
            f"expected {N_COMPONENTS}."
        )

    mixing = np.asarray(
        ica.get_components(),
        dtype=np.float64
    )

    require_shape(
        "Sensor-space mixing matrix A",
        mixing,
        (mixing.shape[0], N_COMPONENTS)
    )

    return ica, mixing, metadata


def load_sentence_sources(ica_subject, sentence_index):
    path = (
        Path(ica_subject)
        / "sources"
        / f"sentence_{sentence_index:03d}_sources.npy"
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Step-4 source file not found: {path}"
        )

    sources = np.asarray(
        np.load(path),
        dtype=np.float64
    )

    if sources.ndim != 2:
        raise RuntimeError(
            f"{path}: expected 2-D source matrix, got {sources.shape}"
        )

    if sources.shape[0] != N_COMPONENTS:
        raise RuntimeError(
            f"ICA sources sentence {sentence_index:03d}: "
            f"expected {N_COMPONENTS} components, got {sources.shape}"
        )

    return sources


# ============================================================================
# ICLabel
# ============================================================================

def load_iclabel_probabilities(features_subject):
    """
    Load Step-6 ICLabel posterior probabilities.

    Expected:
        98 records
        7 classes/component
    """

    path = Path(features_subject) / "iclabel.json"
    records = load_json(path)

    if len(records) != N_COMPONENTS:
        raise RuntimeError(
            f"ICLabel records: expected {N_COMPONENTS}, got {len(records)}"
        )

    class_names = [
        "brain",
        "muscle",
        "eye",
        "heart",
        "line_noise",
        "channel_noise",
        "other",
    ]

    probabilities = np.zeros(
        (N_COMPONENTS, len(class_names)),
        dtype=np.float64
    )

    labels = []

    for k, record in enumerate(records):
        posterior = record.get("posterior")

        if not isinstance(posterior, dict):
            raise RuntimeError(
                f"ICLabel record {k} has no posterior dictionary."
            )

        for j, name in enumerate(class_names):
            if name not in posterior:
                raise RuntimeError(
                    f"ICLabel record {k} missing class '{name}'."
                )

            probabilities[k, j] = float(posterior[name])

        labels.append(record.get("label"))

    if not np.all(np.isfinite(probabilities)):
        raise RuntimeError("ICLabel posterior contains NaN/Inf.")

    row_sums = probabilities.sum(axis=1)

    if not np.allclose(row_sums, 1.0, atol=1e-5):
        raise RuntimeError(
            "ICLabel posterior rows do not sum to 1. "
            f"Range=[{row_sums.min():.6f}, {row_sums.max():.6f}]"
        )

    return class_names, probabilities, labels


def determine_rejected_components(
    class_names,
    probabilities,
    threshold
):
    """
    Classical hard ICLabel rejection.

    A component is rejected if ANY of:
        P(eye)
        P(muscle)
        P(line_noise)

    is >= threshold.

    The component index is only the storage row; it is not used as
    a feature or as a semantic identity.
    """

    if not (0.0 < threshold <= 1.0):
        raise ValueError(
            f"ICLabel threshold must be in (0, 1], got {threshold}"
        )

    artifact_indices = [
        class_names.index(name)
        for name in ARTIFACT_CLASSES
    ]

    artifact_probabilities = probabilities[:, artifact_indices]

    rejected = np.any(
        artifact_probabilities >= threshold,
        axis=1
    )

    winning_artifact_probability = np.max(
        artifact_probabilities,
        axis=1
    )

    winning_artifact_class = np.asarray(
        ARTIFACT_CLASSES,
        dtype=object
    )[
        np.argmax(
            artifact_probabilities,
            axis=1
        )
    ]

    return (
        rejected,
        winning_artifact_probability,
        winning_artifact_class,
    )


# ============================================================================
# Sensor reconstruction
# ============================================================================

def back_project(mixing, sources):
    """
    Back-project frozen ICA sources into sensor space.

        X = A @ S

    A: 105 x 98
    S: 98 x N
    X: 105 x N
    """

    if sources.shape[0] != N_COMPONENTS:
        raise RuntimeError(
            f"Expected {N_COMPONENTS} source components, "
            f"got {sources.shape[0]}"
        )

    sensor = mixing @ sources

    require_shape(
        "Back-projected sensor signal",
        sensor,
        (N_CHANNELS, sources.shape[1])
    )

    return np.asarray(sensor, dtype=np.float64)


# ============================================================================
# Step-5B word slicing
# ============================================================================

def load_sentence_metadata(segments_subject, sentence_index):
    path = (
        Path(segments_subject)
        / f"sentence_{sentence_index:03d}"
        / "metadata.json"
    )

    return load_json(path)


def save_word_segments(
    sensor,
    sentence_metadata,
    output_sentence_dir
):
    """
    Save word-level baseline EEG using the already validated
    processed sample ranges from Step 5B.

    For words with multiple fixation records, the complete
    continuous interval from the first/final timing boundaries
    is represented once as word_XXX_sensor.npy.

    Words with no fixation are retained in metadata but no EEG
    array is created.
    """

    words_dir = (
        Path(output_sentence_dir)
        / "words"
    )
    words_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    saved = []

    # Step 5B stores one record per fixation.
    # Group valid records by word index.
    by_word = {}

    for record in sentence_metadata.get("words", []):
        if record.get("segment_status") != "valid":
            continue

        word_index = int(record["word_index"])

        by_word.setdefault(
            word_index,
            []
        ).append(record)

    for word_index, records in sorted(by_word.items()):

        starts = [
            int(x["processed_start_sample"])
            for x in records
        ]

        stops = [
            int(x["processed_stop_sample"])
            for x in records
        ]

        start = min(starts)
        stop = max(stops)

        if stop <= start:
            continue

        if start < 0 or stop > sensor.shape[1]:
            raise RuntimeError(
                f"Word {word_index}: sample range "
                f"[{start}, {stop}) exceeds sensor signal "
                f"length {sensor.shape[1]}."
            )

        word_sensor = sensor[:, start:stop]

        path = (
            words_dir
            / f"word_{word_index:03d}_sensor.npy"
        )

        np.save(path, word_sensor)

        saved.append({
            "word_index": word_index,
            "text": records[0].get("text", ""),
            "processed_start_sample": start,
            "processed_stop_sample": stop,
            "n_samples": int(stop - start),
            "file": str(path),
            "shape": list(word_sensor.shape),
        })

    return saved


# ============================================================================
# One baseline/sentence
# ============================================================================

def process_sentence(
    sentence_index,
    processed_subject,
    ica_subject,
    features_subject,
    segments_subject,
    no_denoising_dir,
    hard_reject_dir,
    mixing,
    rejected,
    threshold,
    artifact_probabilities,
    artifact_classes
):
    print(
        f"\nSentence {sentence_index:03d}"
    )

    # RAW ROUTE: reconstruct the no-denoising sensor signal directly from the frozen
    # ICA basis as X = A @ S. This needs no per-sentence .fif (the raw route has one
    # continuous file) and guarantees sample-count consistency with the sources.
    sources = load_sentence_sources(
        ica_subject,
        sentence_index
    )

    original_sensor = back_project(mixing, sources)   # (n_ch, N)

    require_shape(
        f"Original sensor EEG sentence {sentence_index:03d}",
        original_sensor,
        (mixing.shape[0], original_sensor.shape[1])
    )

    # ------------------------------------------------------------
    # No-denoising baseline
    # ------------------------------------------------------------

    no_denoising = original_sensor.copy()

    # ------------------------------------------------------------
    # ICLabel hard-reject baseline
    # ------------------------------------------------------------

    cleaned_sources = sources.copy()

    rejected_indices = np.flatnonzero(rejected)

    if rejected_indices.size:
        cleaned_sources[rejected_indices, :] = 0.0

    hard_reject = back_project(
        mixing,
        cleaned_sources
    )

    # ------------------------------------------------------------
    # Output directories
    # ------------------------------------------------------------

    no_sentence_dir = (
        Path(no_denoising_dir)
        / f"sentence_{sentence_index:03d}"
    )

    hard_sentence_dir = (
        Path(hard_reject_dir)
        / f"sentence_{sentence_index:03d}"
    )

    no_sentence_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    hard_sentence_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    no_cont_path = (
        no_sentence_dir
        / "sensor_continuous.npy"
    )

    hard_cont_path = (
        hard_sentence_dir
        / "sensor_continuous.npy"
    )

    np.save(no_cont_path, no_denoising)
    np.save(hard_cont_path, hard_reject)

    # RAW ROUTE: word-level segment files are already produced by step5_raw_segment.
    # Step 7 only needs the sentence-level continuous baselines above (Step 8 compares
    # those). We skip the Step-5B word-metadata re-slicing here.
    no_words = []
    hard_words = []

    record = {
        "sentence_index": sentence_index,
        "n_samples": int(original_sensor.shape[1]),
        "n_components": N_COMPONENTS,
        "n_rejected_components": int(rejected_indices.size),
        "rejected_components": rejected_indices.tolist(),
        "rejection_threshold": float(threshold),
        "rejection_classes": list(artifact_classes),
        "max_artifact_probability": float(
            artifact_probabilities.max()
        ),
        "no_denoising": {
            "type": "passthrough",
            "file": str(no_cont_path),
            "shape": list(no_denoising.shape),
            "n_word_outputs": len(no_words),
        },
        "iclabel_hard_reject": {
            "type": "hard_component_rejection",
            "file": str(hard_cont_path),
            "shape": list(hard_reject.shape),
            "n_word_outputs": len(hard_words),
        },
    }

    print(
        f"  Samples             : {original_sensor.shape[1]}"
    )
    print(
        f"  Rejected components : {rejected_indices.size}/{N_COMPONENTS}"
    )

    if rejected_indices.size:
        print(
            "  Rejected indices    : "
            + ", ".join(
                str(int(x))
                for x in rejected_indices
            )
        )

    return record


# ============================================================================
# Main
# ============================================================================

def run_step7(
    subject_id=DEFAULT_SUBJECT,
    processed_root="processed_raw",
    ica_root="ica",
    features_root="features",
    segments_root="segments",
    output_root="baselines",
    threshold=DEFAULT_THRESHOLD,
    max_sentences=None,
):
    print("=" * 80)
    print("MindInk / ZuCo Step 7")
    print("Classical Baselines")
    print("=" * 80)

    print(f"Subject/session : {subject_id}")
    print(f"ICLabel threshold: {threshold:.3f}")
    print(
        "Rejected classes : "
        + ", ".join(ARTIFACT_CLASSES)
    )

    processed_subject = (
        Path(processed_root)
        / subject_id
    )
    ica_subject = (
        Path(ica_root)
        / subject_id
    )
    features_subject = (
        Path(features_root)
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

    no_denoising_dir = (
        output_subject
        / "no_denoising"
    )
    hard_reject_dir = (
        output_subject
        / "iclabel_hard_reject"
    )

    no_denoising_dir.mkdir(
        parents=True,
        exist_ok=True
    )
    hard_reject_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # ------------------------------------------------------------
    # Validate required inputs
    # ------------------------------------------------------------

    for directory, label in [
        (ica_subject, "Step-4 ICA"),
        (features_subject, "Step-6 features"),
    ]:
        if not directory.exists():
            raise FileNotFoundError(
                f"{label} directory not found: {directory}"
            )
    # RAW ROUTE: Step 7 reconstructs the no-denoising sensor as A@sources, so it no
    # longer needs the Step-3 processed dir or the Step-5B segments dir to exist.

    # ------------------------------------------------------------
    # Frozen ICA and sensor-space A
    # ------------------------------------------------------------

    print("\nLoading frozen ICA...")

    _, mixing, ica_metadata = load_step4_ica(
        ica_subject
    )

    print(
        f"  Sensor-space mixing A : {mixing.shape}"
    )

    # ------------------------------------------------------------
    # Step-6 ICLabel posterior
    # ------------------------------------------------------------

    print("\nLoading Step-6 ICLabel posterior...")

    class_names, probabilities, labels = (
        load_iclabel_probabilities(
            features_subject
        )
    )

    print(
        f"  Components : {probabilities.shape[0]}"
    )
    print(
        f"  Classes    : {probabilities.shape[1]}"
    )

    (
        rejected,
        artifact_probabilities,
        artifact_classes
    ) = determine_rejected_components(
        class_names,
        probabilities,
        threshold
    )

    rejected_indices = np.flatnonzero(rejected)

    print(
        f"  Rejected   : "
        f"{rejected_indices.size}/{N_COMPONENTS}"
    )

    # ------------------------------------------------------------
    # Determine available sentences from Step 4 sources
    # ------------------------------------------------------------

    source_files = sorted(
        (
            Path(ica_subject)
            / "sources"
        ).glob(
            "sentence_*_sources.npy"
        )
    )

    if not source_files:
        raise FileNotFoundError(
            "No Step-4 continuous source files found."
        )

    sentence_indices = sorted(
        int(path.stem.split("_")[1])
        for path in source_files
    )

    if max_sentences is not None:
        sentence_indices = sentence_indices[:max_sentences]

    print(
        f"\nProcessing sentences: {len(sentence_indices)}"
    )

    records = []

    for count, sentence_index in enumerate(
        sentence_indices,
        start=1
    ):
        print(
            f"\n[{count}/{len(sentence_indices)}]"
        )

        records.append(
            process_sentence(
                sentence_index=sentence_index,
                processed_subject=processed_subject,
                ica_subject=ica_subject,
                features_subject=features_subject,
                segments_subject=segments_subject,
                no_denoising_dir=no_denoising_dir,
                hard_reject_dir=hard_reject_dir,
                mixing=mixing,
                rejected=rejected,
                threshold=threshold,
                artifact_probabilities=artifact_probabilities,
                artifact_classes=ARTIFACT_CLASSES,
            )
        )

    # ------------------------------------------------------------
    # Dataset manifest
    # ------------------------------------------------------------

    manifest = {
        "step": "7",
        "subject_id": subject_id,
        "description": "Classical baseline comparison pipelines",
        "sampling_rate": SFREQ,
        "n_channels": N_CHANNELS,
        "n_components": N_COMPONENTS,
        "baseline_methods": [
            "no_denoising",
            "iclabel_hard_reject",
        ],
        "iclabel": {
            "classes_used_for_rejection": list(ARTIFACT_CLASSES),
            "threshold": float(threshold),
            "posterior_source": "features/<subject>/iclabel.json",
            "n_rejected_components": int(rejected_indices.size),
            "rejected_components": rejected_indices.tolist(),
        },
        "frozen_ica": True,
        "ica_refit": False,
        "word_rawEEG_used": False,
        "segmentation_source": "Step-5B continuous-EEG sample indices",
        "novelty_claim": False,
        "sentences_processed": len(records),
        "sentences": records,
        "notes": [
            "No-denoising is a passthrough of the Step-3 analysis EEG.",
            "ICLabel hard-reject zeros components classified as eye, muscle, "
            "or line_noise above the configured posterior threshold.",
            "Rejected components are back-projected using the frozen "
            "sensor-space mixing matrix A.",
            "These are classical comparison baselines, not the proposed "
            "differentiable gate.",
        ],
    }

    manifest_path = (
        output_subject
        / "manifest.json"
    )

    save_json(
        manifest_path,
        manifest
    )

    # ------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------

    if len(records) != len(sentence_indices):
        raise RuntimeError(
            "Step 7 record count does not match sentence count."
        )

    for record in records:
        if record["no_denoising"]["shape"][0] != N_CHANNELS:
            raise RuntimeError(
                "No-denoising output has incorrect channel count."
            )

        if record["iclabel_hard_reject"]["shape"][0] != N_CHANNELS:
            raise RuntimeError(
                "ICLabel hard-reject output has incorrect channel count."
            )

    print("\n" + "=" * 80)
    print("STEP 7 COMPLETE")
    print("=" * 80)

    print(
        f"""
Subject/session:
    {subject_id}

Sentences processed:
    {len(records)}

Sampling rate:
    {SFREQ} Hz

Channels:
    {N_CHANNELS}

Frozen ICA components:
    {N_COMPONENTS}

Baselines:
    no-denoising
    ICLabel hard-reject

ICLabel artifact classes:
    eye
    muscle
    line_noise

ICLabel threshold:
    {threshold:.3f}

Rejected components:
    {rejected_indices.size}/{N_COMPONENTS}

ICA refitting:
    No

ZuCo word.rawEEG:
    Not used

Manifest:
    {manifest_path}
"""
    )


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="MindInk Step 7 — classical EEG denoising baselines"
    )

    parser.add_argument(
        "--subject",
        default=DEFAULT_SUBJECT,
        help="ZuCo subject/session ID."
    )

    parser.add_argument(
        "--processed",
        default="processed_raw",
        help="Step-3 processed root."
    )

    parser.add_argument(
        "--ica",
        default="ica",
        help="Step-4 ICA root."
    )

    parser.add_argument(
        "--features",
        default="features",
        help="Step-6 feature root."
    )

    parser.add_argument(
        "--segments",
        default="segments",
        help="Step-5B segmentation root."
    )

    parser.add_argument(
        "--output",
        default="baselines",
        help="Step-7 output root."
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help=(
            "ICLabel posterior threshold for hard rejection. "
            "A component is rejected when P(eye), P(muscle), or "
            "P(line_noise) >= threshold. Default: 0.50."
        )
    )

    parser.add_argument(
        "--max-sentences",
        type=int,
        default=None,
        help="Process only the first N sentences for testing."
    )

    args = parser.parse_args()

    run_step7(
        subject_id=args.subject,
        processed_root=args.processed,
        ica_root=args.ica,
        features_root=args.features,
        segments_root=args.segments,
        output_root=args.output,
        threshold=args.threshold,
        max_sentences=args.max_sentences,
    )
