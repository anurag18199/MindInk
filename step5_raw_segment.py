#!/usr/bin/env python3
"""
step5_raw_segment.py — Step 5 (RAW route) segmentation.

Your original step5B_segment.py read the preprocessed .mat via h5py and did complex
per-sentence clock CALIBRATION to align eye-tracking to EEG. On the raw route that
work is already DONE: step2_raw_load synced EEG<->ET via shared triggers, and
step3_raw_preprocess wrote every sentence span + fixation onset/offset (as EEG sample
indices at 250 Hz) into processed_raw/<subj>/metadata.json. So Step 5 here just SLICES.

It reproduces your Step-5B OUTPUT LAYOUT exactly, so Step 6/7/9 need no changes:

  segments/<subj>/
      sentence_XXX/
          sensor_continuous.npy                     (channels x time, whole sentence)
          words/
              word_YYY/
                  fixation_ZZZ_sensor.npy            (channels x fix_samples)
                  fixation_ZZZ_sources.npy           (98 x fix_samples)

Inputs:
  processed_raw/<subj>/<subj>_analysis_raw.fif   (continuous sensor EEG, 250 Hz)
  processed_raw/<subj>/metadata.json             (sentence spans + fixations)
  ica/<subj>/sources/sentence_XXX_sources.npy    (from step4_ica_raw)

Notes:
  * One fixation == one "word" segment here (raw ET gives fixation-level events).
    If you later group fixations into words, add a word_index map; the layout supports it.
  * Fixations shorter than MIN_FIX_SAMPLES or outside the sentence span are skipped
    and logged (status counters), never silently dropped.
"""

from pathlib import Path
import argparse
import json
import numpy as np

N_COMPONENTS = 98
MIN_FIX_SAMPLES = 2          # drop degenerate 1-sample fixations


def load_meta(processed_dir, subject):
    with open(processed_dir / subject / "metadata.json") as f:
        return json.load(f)


def load_sentence_sources(ica_root, subject, idx):
    p = Path(ica_root) / subject / "sources" / f"sentence_{idx:03d}_sources.npy"
    if not p.exists():
        return None
    S = np.load(p)
    if S.ndim != 2 or S.shape[0] != N_COMPONENTS:
        raise RuntimeError(f"{p}: expected (98,T), got {S.shape}")
    return S


def segment_subject(processed_root, ica_root, segments_root, subject, use_mne=True):
    processed_dir = Path(processed_root)
    meta = load_meta(processed_dir, subject)
    out_root = Path(segments_root) / subject
    out_root.mkdir(parents=True, exist_ok=True)

    # continuous sensor EEG
    if use_mne:
        import mne
        raw = mne.io.read_raw_fif(processed_dir / subject / f"{subject}_analysis_raw.fif",
                                  preload=True, verbose=False)
        sensor_all = raw.get_data()          # (channels, T)
    else:
        sensor_all = None                    # numpy-test mode: skip sensor slicing

    counters = {"sentences": 0, "fix_valid": 0, "fix_short": 0,
                "fix_out_of_span": 0, "fix_no_sources": 0}
    manifest = []

    for s in meta["sentences"]:
        idx = s["index"]
        a, b = int(s["onset"]), int(s["offset"])
        if b <= a:
            continue
        S = load_sentence_sources(ica_root, subject, idx)
        if S is None:
            counters["fix_no_sources"] += len(s.get("fixations", []))
            continue

        sent_dir = out_root / f"sentence_{idx:03d}"
        sent_dir.mkdir(parents=True, exist_ok=True)

        # sentence-level continuous sensor (channels x time), sliced to span
        if sensor_all is not None:
            sensor_sent = sensor_all[:, a:b].astype(np.float64)
            np.save(sent_dir / "sensor_continuous.npy", sensor_sent)

        words_dir = sent_dir / "words"
        # sources for this sentence are span-local: index 0 == sample a
        Tspan = S.shape[1]

        for w, fx in enumerate(s.get("fixations", [])):
            on, off = int(fx[0]), int(fx[1])          # absolute EEG samples (250 Hz)
            # convert to sentence-local indices (sources & sliced sensor start at a)
            ls, le = on - a, off - a
            if le - ls < MIN_FIX_SAMPLES:
                counters["fix_short"] += 1
                continue
            if ls < 0 or le > Tspan:
                counters["fix_out_of_span"] += 1
                continue

            word_dir = words_dir / f"word_{w:03d}"
            word_dir.mkdir(parents=True, exist_ok=True)

            # ICA source segment
            src_seg = S[:, ls:le].astype(np.float64)
            np.save(word_dir / f"fixation_{0:03d}_sources.npy", src_seg)
            # sensor segment
            if sensor_all is not None:
                sen_seg = sensor_all[:, on:off].astype(np.float64)
                np.save(word_dir / f"fixation_{0:03d}_sensor.npy", sen_seg)

            counters["fix_valid"] += 1
            manifest.append({"sentence": idx, "word": w,
                             "on": on, "off": off, "n_samples": le - ls})

        counters["sentences"] += 1

    with open(out_root / "segment_manifest.json", "w") as f:
        json.dump({"subject": subject, "counters": counters,
                   "n_segments": len(manifest)}, f, indent=2)

    print(f"[step5-raw] {subject}: sentences={counters['sentences']}  "
          f"valid_fixations={counters['fix_valid']}  short={counters['fix_short']}  "
          f"out_of_span={counters['fix_out_of_span']}  no_sources={counters['fix_no_sources']}")
    print(f"[step5-raw] wrote -> {out_root}")
    return counters


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Step 5 (raw route) segmentation")
    ap.add_argument("--processed", default="processed_raw")
    ap.add_argument("--ica", default="ica")
    ap.add_argument("--output", default="segments")
    ap.add_argument("--subject", default="YAG")
    a = ap.parse_args()
    segment_subject(a.processed, a.ica, a.output, a.subject, use_mne=True)