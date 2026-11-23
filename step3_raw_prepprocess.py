#!/usr/bin/env python3
"""
step3_raw_preprocess.py — Step 3 for the RAW ZuCo route.

Consumes RawBlock objects from step2_raw_load and produces, per subject:
  * one CONTINUOUS analysis Raw   (0.5-100 Hz, notch 50, avg-ref, 250 Hz)
  * one CONTINUOUS ICA Raw        (1-100 Hz,  notch 50, avg-ref, 250 Hz)
  * blocks concatenated in order (NR1, NR2, ...), with sentence/fixation/blink
    sample indices REMAPPED into the concatenated timeline.

KEY DIFFERENCES vs the preprocessed-route Step 3:
  * INPUT is raw 128ch EEGLAB data, not the sentenceData struct.
  * LOW-PASS is 100 Hz (was 45) — raw ZuCo carries 0.1-100 Hz, so ICLabel's
    muscle/line/eye classes are finally usable. (250 Hz resample => Nyquist 125,
    which safely passes 100 Hz.)
  * 128 -> 105 channel selection is done HERE, from the file's own coordinates,
    replacing the guessed ZUCO_CHANNEL_NAMES that broke ICLabel before.
  * Filtering is done on the CONCATENATED continuous signal (one filter pass per
    subject), not per short sentence — no more per-sentence edge transients.
  * BLINK intervals (ground truth) are carried through for Step 8 validation.

MNE is required for the filtering/montage/resample. The channel-selection,
concatenation, and index-remapping logic is pure numpy and is unit-tested in
__main__ against a real block WITHOUT needing MNE.

RAW EEG + ET files
       ↓
   Step 2 loader
       ↓
 RawBlock objects
       ↓
 Remove unwanted channels
       ↓
 Concatenate NR1 + NR2 + NR3
       ↓
 One continuous EEG
       ↓
 ┌─────────────────────┐
 │                     │
 ↓                     ↓
Analysis branch       ICA branch
0.5–100 Hz            1–100 Hz
50 Hz notch           50 Hz notch
Avg reference         Avg reference
250 Hz                250 Hz
 │                     │
 └──────────┬──────────┘
            ↓
     Remap annotations
            ↓
          Save
"""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional
import json
import numpy as np

from step2_raw_load import load_raw_block, RawBlock, Sentence, Fixation

# ---- config ---------------------------------------------------------------
LOW_CUTOFF = 0.5
HIGH_CUTOFF = 100.0          # <- was 45; raw data supports up to ~100 Hz
ICA_HIGH_PASS = 1.0
NOTCH_FREQ = 50.0
OUTPUT_SFREQ = 250.0

# ZuCo's EXACT kept set, read from the official preprocessed chanlocs105.mat.
# 105 kept = 104 of E1..E128 PLUS 'Cz' (the recording reference, recovered after
# average-referencing). The raw files are referenced at Cz, so they contain 128
# channels E1..E128 and NO Cz.
#
# The 24 electrodes DISCARDED from the raw 128 (9 EOG + 14 neck/face + E129 slot):
DISCARD_CHANNELS = [
    "E1", "E8", "E14", "E17", "E21", "E25", "E32", "E48", "E49", "E56",
    "E63", "E68", "E73", "E81", "E88", "E94", "E99", "E107", "E113",
    "E119", "E125", "E126", "E127", "E128",
]
# After discarding these 24 from the raw 128 you get the 104 E-channels ZuCo kept.
# 'Cz' (the 105th) is the reference; see RECOVER_CZ below.

# If True, re-add Cz as a recovered data channel after average-referencing, to match
# ZuCo's 105 exactly. If False, keep 104 and document the reference-electrode exclusion
# (recommended default — simpler, scientifically equivalent for ICA/gating).
RECOVER_CZ = False


# ---------------------------------------------------------------------------
# Pure-numpy core (unit-testable without MNE)
# ---------------------------------------------------------------------------
def select_channels(data, ch_names, ch_pos, discard):
    """Drop `discard` channels. Returns filtered data, names, positions, kept idx."""
    keep = [i for i, n in enumerate(ch_names) if n not in set(discard)]
    return (data[keep, :],
            [ch_names[i] for i in keep],
            ch_pos[keep, :],
            keep)


def concat_blocks_numpy(blocks: List[RawBlock]):
    """
    Concatenate block data along time and remap every sample index (sentence spans,
    fixation on/off, blink intervals) by each block's cumulative time offset.
    Returns (data, sentences, blink_intervals, offsets) — all in the concatenated
    timeline, still at the ORIGINAL sfreq (resampling happens later, in MNE).
    """
    if not blocks:
        raise ValueError("No blocks to concatenate.")
    nch = blocks[0].data.shape[0]
    for b in blocks:
        if b.data.shape[0] != nch:
            raise ValueError("Channel count differs across blocks.")

    datas, sentences, blinks, offsets = [], [], [], []
    running = 0
    sent_counter = 0
    for b in blocks:
        offsets.append(running)
        datas.append(b.data)
        for s in b.sentences:
            fixs = [Fixation(f.onset_sample + running, f.offset_sample + running,
                             f.duration_ms, f.x, f.y) for f in s.fixations]
            sentences.append(Sentence(sent_counter,
                                      s.onset_sample + running,
                                      s.offset_sample + running, fixs))
            sent_counter += 1
        if b.blink_intervals.size:
            blinks.append(b.blink_intervals + running)
        running += b.data.shape[1]

    data = np.concatenate(datas, axis=1)
    blink_intervals = (np.concatenate(blinks, axis=0)
                       if blinks else np.zeros((0, 2), int))
    return data, sentences, blink_intervals, offsets


def remap_indices_after_resample(sentences, blink_intervals, factor):
    """Scale all sample indices by `factor` (= new_sfreq/old_sfreq) after resampling."""
    def sc(x):
        return int(round(x * factor))
    new_sent = []
    for s in sentences:
        fixs = [Fixation(sc(f.onset_sample), sc(f.offset_sample),
                         f.duration_ms, f.x, f.y) for f in s.fixations]
        new_sent.append(Sentence(s.index, sc(s.onset_sample),
                                 sc(s.offset_sample), fixs))
    new_blinks = np.round(blink_intervals * factor).astype(int)
    return new_sent, new_blinks


# ---------------------------------------------------------------------------
# MNE section (needs mne; guarded so the numpy core can be tested without it)
# ---------------------------------------------------------------------------
def build_mne_raw(data, ch_names, ch_pos, sfreq):
    import mne
    info = mne.create_info(ch_names, sfreq=sfreq, ch_types="eeg")
    raw = mne.io.RawArray(data, info, verbose="error")      #Converts the NumPy EEG matrix into an MNE Raw object
    # Montage straight from the file's real electrode coordinates.
    pos = {n: ch_pos[i] / 1000.0 if np.isfinite(ch_pos[i]).all() else [0, 0, 0]
           for i, n in enumerate(ch_names)}  # EEGLAB XYZ are in ~mm; MNE wants m
    montage = mne.channels.make_dig_montage(ch_pos=pos, coord_frame="head") #creates the electrode montage.
    raw.set_montage(montage, on_missing="warn", verbose="error")    #attaches electrode positions to the MNE Raw object.
    return raw


def filter_branch(raw, hp, lp, notch, out_sfreq, recover_cz=False):
    import mne
    r = raw.copy()
    r.filter(l_freq=hp, h_freq=lp, method="fir", phase="zero", verbose="error")
    r.notch_filter(freqs=[notch], method="fir", verbose="error")
    # Optionally recover Cz (recording reference) as a data channel to match ZuCo's
    # 105 exactly: add it flat, then average-referencing populates it as -sum(others).
    if recover_cz and "Cz" not in r.ch_names:
        r = mne.add_reference_channels(r, "Cz", verbose="error")
    r.set_eeg_reference("average", projection=False, verbose="error")       #This makes the data average-referenced.
    if r.info["sfreq"] != out_sfreq:
        r.resample(out_sfreq, npad="auto", verbose="error")
    return r


def preprocess_subject_raw(blocks, output_dir, subject):
    # 1) channel selection on each block (from the file's own coordinates)
    sel = []
    for b in blocks:
        d, names, pos, _ = select_channels(b.data, b.ch_names, b.ch_pos,
                                           DISCARD_CHANNELS)
        sel.append(RawBlock(b.subject, b.block, d, b.sfreq, names, pos,
                            b.sentences, b.blink_intervals))
    print(f"[step3-raw] kept {sel[0].data.shape[0]} of {blocks[0].data.shape[0]} channels")

    # 2) concatenate blocks + remap indices (numpy)
    data, sentences, blinks, offsets = concat_blocks_numpy(sel)
    in_sfreq = sel[0].sfreq
    print(f"[step3-raw] concatenated {len(sel)} blocks -> {data.shape} @ {in_sfreq}Hz, "
          f"{len(sentences)} sentences, {len(blinks)} blinks")

    # 3) build MNE Raw, filter both branches
    raw = build_mne_raw(data, sel[0].ch_names, sel[0].ch_pos, in_sfreq)
    analysis = filter_branch(raw, LOW_CUTOFF, HIGH_CUTOFF, NOTCH_FREQ, OUTPUT_SFREQ,
                             recover_cz=RECOVER_CZ)
    ica = filter_branch(raw, ICA_HIGH_PASS, HIGH_CUTOFF, NOTCH_FREQ, OUTPUT_SFREQ,
                        recover_cz=RECOVER_CZ)

    # 4) remap indices for the new sample rate
    factor = OUTPUT_SFREQ / in_sfreq
    sentences_rs, blinks_rs = remap_indices_after_resample(sentences, blinks, factor)

    # 5) save
    out = Path(output_dir) / subject
    out.mkdir(parents=True, exist_ok=True)
    analysis.save(out / f"{subject}_analysis_raw.fif", overwrite=True, verbose="error")
    ica.save(out / f"{subject}_ica_raw.fif", overwrite=True, verbose="error")
    np.save(out / "blink_intervals.npy", blinks_rs)
    meta = {
        "subject": subject, "route": "raw",
        "n_channels": int(analysis.info["nchan"]),
        "sfreq": OUTPUT_SFREQ, "analysis_band": [LOW_CUTOFF, HIGH_CUTOFF],
        "ica_band": [ICA_HIGH_PASS, HIGH_CUTOFF], "notch": NOTCH_FREQ,
        "n_sentences": len(sentences_rs), "n_blinks": int(len(blinks_rs)),
        "blocks": [b.block for b in blocks],
        "sentences": [{"index": s.index, "onset": s.onset_sample,
                       "offset": s.offset_sample,
                       "n_fixations": len(s.fixations),
                       "fixations": [[f.onset_sample, f.offset_sample, f.duration_ms]
                                     for f in s.fixations]} for s in sentences_rs],
    }
    with open(out / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[step3-raw] saved analysis+ica raw, blink_intervals, metadata -> {out}")
    return analysis, ica, sentences_rs, blinks_rs


# ---------------------------------------------------------------------------
# CLI: load blocks then preprocess
# ---------------------------------------------------------------------------
def find_block_files(raw_dir, subject, block):
    """Locate the EEG and ET .mat files for one subject+block.

    Looks in raw_dir/, then raw_dir/<subject>/, then falls back to a recursive
    search by filename pattern.
    Input : raw_dir (str/Path), subject (e.g. 'YAG'), block (e.g. 'NR1').
    Output: (eeg_path, et_path) as Paths, or (None, None) if not found.
    """
    raw_dir = Path(raw_dir)
    # Most likely locations: flat in raw_dir, or inside a per-subject folder.
    for base in (raw_dir, raw_dir / subject):
        eeg = base / f"{subject}_{block}_EEG.mat"
        et  = base / f"{subject}_{block}_ET.mat"
        if eeg.exists() and et.exists():
            return eeg, et
    # Fallback: search anywhere under raw_dir (handles unexpected prefixes/nesting).
    eeg_hits = list(raw_dir.rglob(f"*{subject}_{block}_EEG.mat"))
    et_hits  = list(raw_dir.rglob(f"*{subject}_{block}_ET.mat"))
    if eeg_hits and et_hits:
        return eeg_hits[0], et_hits[0]
    return None, None



def process_one_subject(raw_dir, subject, block_names, output_dir):
    """Load, preprocess, and export one subject end-to-end.

    Input : raw_dir (root), subject (e.g. 'YAG'), block_names (e.g. ['NR1'..'NR7']),
            output_dir (where results go).
    Output: path to the exported zuco_step3_<subject>.npz, or None if no blocks
            could be loaded for this subject (so the batch loop can skip it).
    """
    blocks = []
    for blk in block_names:
        eeg, et = find_block_files(raw_dir, subject, blk)   # robust lookup from before
        if eeg is None:
            print(f"  [{subject}] skip {blk}: file not found")
            continue
        print(f"  [{subject}] found {blk}: {eeg}")
        blocks.append(load_raw_block(eeg, et, subject, blk))

    if not blocks:
        print(f"  [{subject}] no blocks loaded — skipping")
        return None

    # Heavy lifting (filter both branches, remap indices, save fif/json).
    analysis, ica, sentences_rs, blinks_rs = preprocess_subject_raw(
        blocks, output_dir, subject)

    # Export the analysis EEG as the .npz the baseline denoisers consume.
    eeg_data = analysis.get_data().astype("float32")        # (n_channels, n_times)
    out_npz = Path(output_dir) / subject / f"zuco_step3_{subject}.npz"
    np.savez(out_npz,
             eeg=eeg_data,
             sfreq=np.float32(OUTPUT_SFREQ),
             ch_names=np.array(analysis.ch_names))
    print(f"  [{subject}] exported baseline input -> {out_npz}")
    return out_npz


def discover_subjects(raw_dir, pattern="Y*"):
    """Find subject folders under raw_dir matching `pattern`.

    Input : raw_dir (root), glob pattern for subject folders.
    Output: sorted list of subject names (folder names only, directories only),
            e.g. ['YAC', 'YAG', 'YAK', ...].
    """
    return sorted(p.name for p in Path(raw_dir).glob(pattern) if p.is_dir() and any(p.glob("*_EEG.mat")))


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Step 3 (RAW route) preprocessing")
    # --subject runs one; --all-subjects runs every YA* folder found under --raw-dir.
    ap.add_argument("--subject", default=None,
                    help="single subject (e.g. YAG). Ignored if --all-subjects is set.")
    ap.add_argument("--all-subjects", action="store_true",
                    help="process every YA* folder under --raw-dir")
    ap.add_argument("--raw-dir", required=True,
                    help="root folder containing per-subject folders (YAG/, YAC/, ...)")
    ap.add_argument("--blocks", nargs="+",
                    default=["NR1", "NR2", "NR3", "NR4", "NR5", "NR6", "NR7"])
    ap.add_argument("--output", default="processed_raw")
    ap.add_argument("--subject-glob", default="Y*", help="glob pattern for subject folders (default YA*)")
    a = ap.parse_args()

    # Decide which subjects to run.
    if a.all_subjects:
        subjects = discover_subjects(a.raw_dir, a.subject_glob)
        if not subjects:
            raise SystemExit(f"No YA* subject folders found under {a.raw_dir}")
        print(f"[batch] found {len(subjects)} subjects: {subjects}")
    elif a.subject:
        subjects = [a.subject]
    else:
        raise SystemExit("Pass either --subject YAG or --all-subjects")

    # Process each; collect successes/failures so a bad subject doesn't stop the batch.
    done, failed = [], []
    for subj in subjects:
        print(f"\n===== {subj} =====")
        try:
            out = process_one_subject(a.raw_dir, subj, a.blocks, a.output)
            (done if out else failed).append(subj)
        except Exception as e:
            print(f"  [{subj}] ERROR: {e}")
            failed.append(subj)

    print(f"\n[batch] done: {done}")
    if failed:
        print(f"[batch] failed/empty: {failed}")

if __name__ == "__main__":
    main()
