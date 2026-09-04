#!/usr/bin/env python3
"""
step4_ica_raw.py — Step 4 (RAW route). Fit ONE ICA per subject on the continuous
raw-route .fif produced by step3_raw_preprocess.py.

DIFFERENCES vs your original step4_ica.py:
  * INPUT is ONE continuous file (processed_raw/<subj>/<subj>_ica_raw.fif), not
    many per-sentence sentence_XXX_ica_raw.fif files. Raw Step 3 already
    concatenated blocks, so there is a single continuous recording to fit on.
  * We ALSO save the SENSOR-SPACE mixing A = ica.get_components() (n_ch x n_comp)
    as mixing_sensor.npy, so downstream never has to rediscover the (98,98) vs
    (105,98) issue. The raw internal mixing_matrix_ is still saved for provenance.
  * Sources are written per SENTENCE (sentence_XXX_sources.npy) by slicing the
    continuous sources with the sentence spans in metadata.json — so Step 5/6/9
    keep the SAME per-sentence sources layout they already expect.
  * N_CHANNELS defaults to 104 (raw route, Cz excluded). Set to 105 if you enabled
    RECOVER_CZ in Step 3.

Output layout (unchanged names, so Step 6/9 still work):
  ica/<subj>/
      ica_model-ica.fif
      unmixing_matrix.npy
      mixing_matrix.npy         (raw internal, may be reduced)
      mixing_sensor.npy         (NEW: sensor-space A = get_components(), n_ch x n_comp)
      sources/sentence_XXX_sources.npy
      metadata.json
"""

from pathlib import Path
import argparse
import json
import numpy as np
import mne
from mne.preprocessing import ICA

N_CHANNELS = 104          # raw route: 104 E-channels (or 105 if RECOVER_CZ=True)
N_COMPONENTS = 98
SAMPLING_RATE = 250.0
ICA_METHOD = "infomax"
EXTENDED = True
RANDOM_STATE = 42
MAX_ITER = 1000


def load_metadata(processed_dir):
    with open(processed_dir / "metadata.json", "r") as f:
        return json.load(f)


def fit_ica(ica_raw):
    print("\n" + "=" * 72)
    print("FITTING ONE ICA (raw route)")
    print("=" * 72)
    print(f"  samples={ica_raw.n_times}  sfreq={ica_raw.info['sfreq']}  "
          f"channels={ica_raw.info['nchan']}  components={N_COMPONENTS}")
    ica = ICA(n_components=N_COMPONENTS, method=ICA_METHOD,
              fit_params={"extended": EXTENDED}, random_state=RANDOM_STATE,
              max_iter=MAX_ITER)
    ica.fit(ica_raw, verbose=True)
    if ica.n_components_ != N_COMPONENTS:
        raise RuntimeError(f"ICA fitted {ica.n_components_} != {N_COMPONENTS} "
                           "components; fixed dimensionality required.")
    print(f"  fitted {ica.n_components_} components.")
    return ica


def save_model(ica, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    ica.save(out_dir / "ica_model-ica.fif", overwrite=True, verbose=False)
    np.save(out_dir / "unmixing_matrix.npy",
            np.asarray(ica.unmixing_matrix_, dtype=np.float64))
    np.save(out_dir / "mixing_matrix.npy",
            np.asarray(ica.mixing_matrix_, dtype=np.float64))
    # THE important one: sensor-space A (n_channels x n_components).
    A_sensor = np.asarray(ica.get_components(), dtype=np.float64)
    np.save(out_dir / "mixing_sensor.npy", A_sensor)
    print(f"  saved model + unmixing{ica.unmixing_matrix_.shape} + "
          f"mixing{ica.mixing_matrix_.shape} + mixing_sensor{A_sensor.shape}")
    return A_sensor


def save_sources_per_sentence(ica, analysis_raw, meta, sources_dir):
    """
    Get continuous sources once, then slice per sentence using metadata spans.
    Keeps the per-sentence sources layout that Step 5/6/9 already expect.
    """
    sources_dir.mkdir(parents=True, exist_ok=True)
    S_all = ica.get_sources(analysis_raw).get_data()   # (n_comp, T)
    T = S_all.shape[1]
    manifest = []
    for s in meta["sentences"]:
        idx = s["index"]
        a = max(0, int(s["onset"]))
        b = min(T, int(s["offset"]))
        if b <= a:
            print(f"  WARN sentence {idx}: empty span [{a},{b}], skipping")
            continue
        seg = S_all[:, a:b].astype(np.float64)
        np.save(sources_dir / f"sentence_{idx:03d}_sources.npy", seg)
        manifest.append({"sentence_index": idx, "n_components": seg.shape[0],
                         "n_samples": seg.shape[1]})
    print(f"  wrote {len(manifest)} per-sentence source files")
    return manifest


def run(processed_root, ica_root, subject):
    processed_dir = Path(processed_root) / subject
    out_dir = Path(ica_root) / subject
    sources_dir = out_dir / "sources"

    meta = load_metadata(processed_dir)
    ica_raw = mne.io.read_raw_fif(processed_dir / f"{subject}_ica_raw.fif",
                                  preload=True, verbose=False)
    analysis_raw = mne.io.read_raw_fif(processed_dir / f"{subject}_analysis_raw.fif",
                                       preload=True, verbose=False)

    if ica_raw.info["nchan"] != N_CHANNELS:
        print(f"  NOTE: channels={ica_raw.info['nchan']} (config N_CHANNELS={N_CHANNELS}). "
              "Set N_CHANNELS to match if this is intentional (e.g. 105 with Cz).")

    ica = fit_ica(ica_raw)
    A_sensor = save_model(ica, out_dir)
    manifest = save_sources_per_sentence(ica, analysis_raw, meta, sources_dir)

    with open(out_dir / "metadata.json", "w") as f:
        json.dump({"subject": subject, "step": 4, "route": "raw",
                   "n_channels": int(ica_raw.info["nchan"]),
                   "n_components": int(ica.n_components_),
                   "sampling_rate": SAMPLING_RATE,
                   "mixing_sensor_shape": list(A_sensor.shape),
                   "n_sentences": len(manifest),
                   "source_manifest": manifest,
                   "notes": ["Raw route: one continuous fit; sensor-space A saved as "
                             "mixing_sensor.npy; sources sliced per sentence from "
                             "metadata spans."]}, f, indent=2)
    print(f"\n[step4-raw] done -> {out_dir}")
    print(f"  ICA components={ica.n_components_}, sensor A={A_sensor.shape}, "
          f"sentences={len(manifest)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Step 4 (raw route) ICA")
    ap.add_argument("--processed", default="processed_raw")
    ap.add_argument("--output", default="ica")
    ap.add_argument("--subject", default="YAG")
    a = ap.parse_args()
    run(a.processed, a.output, a.subject)
