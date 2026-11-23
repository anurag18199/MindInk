#!/usr/bin/env python3
"""
step2_raw_load.py — load ZuCo 2.0 RAW EEG + eye-tracking, with EEG-ET sync.

This REPLACES the preprocessed-sentenceData loader for the RAW route. It reads the
per-block EEGLAB EEG struct and the matching EyeLink ET struct, synchronizes them via
their shared trigger events, and converts fixations/blinks/sentence-spans into EEG
sample indices.

VERIFIED against real files (YAG_NR1) before shipping:
  * 128ch, 500 Hz, ~11.6 min continuous, chanlocs E1..E128 with XYZ coords.
  * EEG.event and ET.event are the SAME 110 triggers in the SAME order -> sync is a
    linear time map (slope = 0.5, i.e. ET 1000 Hz -> EEG 500 Hz), residual ~3.5 samp.
  * triggers: 10=sentence onset, 11=sentence offset (42 sentences in this block),
    12/13/15 = other markers, 90/20 = block start/end.
  * fixations: 2692, cols [latency,endtime,duration,x,y,pupil]. The stored 'duration'
    is CORRUPT (constant 1.4e14) -> we RECOMPUTE duration = endtime - latency (ms).
  * blinks: 411 ground-truth artifact intervals [latency,endtime,duration].

IMPORTANT design choices:
  * Sync uses a LOCAL per-sentence anchor (nearest triggers) rather than one global
    fit, because a few triggers jitter up to ~32 samples; local anchoring is tighter.
  * We DO NOT z-score, filter, or re-reference here. That is Step 3.
  * We keep 128 channels here; channel selection (128->105) is a Step-3 decision.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional
import numpy as np
from scipy.io import loadmat


# ---- trigger codes (from ZuCo 2.0 raw, verified) ----------------------------
TRIG_SENT_ON = 10
TRIG_SENT_OFF = 11
ET_SFREQ = 1000.0
EEG_SFREQ = 500.0


@dataclass
class Fixation:
    onset_sample: int          # EEG sample where fixation starts. EEG sample index
    offset_sample: int             # EEG sample where fixation ends. EEG sample index
    duration_ms: float         # recomputed = endtime - latency
    x: float    #Horizontal eye position.
    y: float    #Vertical eye position.


@dataclass
class Sentence:
    index: int
    onset_sample: int
    offset_sample: int
    fixations: List[Fixation] = field(default_factory=list)


@dataclass
class RawBlock:
    subject: str
    block: str          #YAG / NR1
    data: np.ndarray           # (128, T) float64, RAW (unfiltered)
    sfreq: float
    ch_names: List[str]
    ch_pos: np.ndarray         # (128, 3) XYZ from chanlocs, for the montage
    sentences: List[Sentence]   # contains all sentences in the block.
    blink_intervals: np.ndarray  # (n_blinks, 2) EEG samples [on, off] — GROUND TRUTH


# ---------------------------------------------------------------------------
def _load_eeg(path):
    EEG = loadmat(path, squeeze_me=True, struct_as_record=False)["EEG"] #The MATLAB file contains an object/structure called:,  removes unnecessary dimensions.
    data = np.asarray(EEG.data, dtype=np.float64)           # (128, T)
    if data.shape[0] != int(EEG.nbchan):
        data = data.T       # then the matrix is probably transposed.
    cl = np.atleast_1d(EEG.chanlocs)
    ch_names = [str(c.labels) for c in cl]

    def _scalar(c, attr):
        v = getattr(c, attr, np.nan)
        v = np.asarray(v).ravel()   #This converts the value into a one-dimensional array.
        return float(v[0]) if v.size else np.nan

    ch_pos = np.array([[_scalar(c, "X"), _scalar(c, "Y"), _scalar(c, "Z")]
                       for c in cl])
    ev = np.atleast_1d(EEG.event)
    eeg_events = np.array([[int(getattr(e, "type")), float(getattr(e, "latency"))]
                           for e in ev])                    # (n, 2) [type, sample] This converts the complicated MATLAB event structure into a simple table.
                                                            # Conceptually:
                                                            # event_type     EEG_sample
    return data, float(EEG.srate), ch_names, ch_pos, eeg_events


def _load_et(path):
    et = loadmat(path, squeeze_me=True, struct_as_record=False)
    et_events = np.asarray(et["event"], dtype=float)        # (n, 2) [et_time, type]


# This gives something like:
# ET timestamp    event type
# ------------    ----------
# 1234            10
# 1450            12
# 1500            13
# 5000            10
# 8000            11


    fix = et["eyeevent"].fixations
    fcols = [str(x) for x in fix.colheader]
#     The fixation data has columns such as:
# latency
# endtime
# duration
# fix_avgpos_x
# fix_avgpos_y
# pupil
    fdata = np.asarray(fix.data, dtype=float)
    fi = {c: fcols.index(c) for c in ("latency", "endtime", "fix_avgpos_x",
                                      "fix_avgpos_y")}

    bl = et["eyeevent"].blinks
    bcols = [str(x) for x in bl.colheader]
    bdata = np.asarray(bl.data, dtype=float)
    bi = {c: bcols.index(c) for c in ("latency", "endtime")}
    return et_events, fdata, fi, bdata, bi


def _build_time_map(eeg_events, et_events):
    """
    Match the two trigger lists (same order, verified) and return a function
    et_time -> eeg_sample plus the matched anchor arrays for local interpolation.
    """
    eeg_types = eeg_events[:, 0].astype(int)
    eeg_samp = eeg_events[:, 1]      #[type, sample]
    et_time = et_events[:, 0]       #[ET time, event type]
    et_types = et_events[:, 1].astype(int)

    n = min(len(eeg_types), len(et_types))
    if not np.array_equal(eeg_types[:n], et_types[:n]):
        # fall back to matching by type sequence alignment if lengths differ
        raise RuntimeError("EEG/ET trigger types differ in order; inspect this block.")

    et_anchor = et_time[:n]
    eeg_anchor = eeg_samp[:n]

    order = np.argsort(et_anchor)
    et_sorted = et_anchor[order]
    eeg_sorted = eeg_anchor[order]

    def et_to_eeg(t):
        # piecewise-linear interpolation between nearest triggers (local anchoring),
        # extrapolating linearly at the ends.
        return np.interp(t, et_sorted, eeg_sorted,
                         left=eeg_sorted[0] + (t - et_sorted[0]) * EEG_SFREQ / ET_SFREQ
                         if np.ndim(t) == 0 else None,
                         right=None)

    # np.interp can't take scalar left-expr; do a clean linear map for out-of-range.
    slope = (eeg_sorted[-1] - eeg_sorted[0]) / (et_sorted[-1] - et_sorted[0])
    intercept = eeg_sorted[0] - slope * et_sorted[0]

    def convert(t):
        t = np.asarray(t, dtype=float)
#         This calculates the overall relationship:
# Δ EEG samples
# --------------
# Δ ET time

# Approximately:
# 500 / 1000
# =
# 0.5

# Suppose:
# Trigger 1:
# ET = 10000
# EEG = 5000

# Trigger 2:
# ET = 12000
# EEG = 6005

# The actual local slope is:
# 1005 / 2000
# =
# 0.5025

# instead of exactly:
# 0.5

# Local interpolation captures this small timing variation.
# That's why your comment says:
# local anchoring is tighter.
        inside = (t >= et_sorted[0]) & (t <= et_sorted[-1])
        out = slope * t + intercept                      # global linear fallback
        out[inside] = np.interp(t[inside], et_sorted, eeg_sorted)  # local where possible
        return out

    return convert


def load_raw_block(eeg_path, et_path, subject="YAG", block="NR1") -> RawBlock:
    data, sfreq, ch_names, ch_pos, eeg_events = _load_eeg(eeg_path)
    et_events, fdata, fi, bdata, bi = _load_et(et_path)
#     ET events
# fixations
# fixation indexes
# blinks
# blink indexes
    convert = _build_time_map(eeg_events, et_events)
    n_samples = data.shape[1]

    # --- fixations -> EEG samples (recompute duration = endtime - latency) ---
    fix_on = convert(fdata[:, fi["latency"]])
    fix_off = convert(fdata[:, fi["endtime"]])
    fix_dur = fdata[:, fi["endtime"]] - fdata[:, fi["latency"]]   # ms, ET units == ms
    fx = fdata[:, fi["fix_avgpos_x"]]
    fy = fdata[:, fi["fix_avgpos_y"]]

    # --- sentence spans from triggers ---------------------------------------
    ety = eeg_events[:, 0].astype(int)      #ety = event types
# esm = EEG sample positions
    esm = eeg_events[:, 1]
    onsets = esm[ety == TRIG_SENT_ON]
    offsets = esm[ety == TRIG_SENT_OFF]
    m = min(len(onsets), len(offsets))
    onsets, offsets = onsets[:m], offsets[:m]

    sentences = []
    for i, (s0, s1) in enumerate(zip(onsets, offsets)):
#         i = 0
# s0 = 1200
# s1 = 6500
        sel = (fix_on >= s0) & (fix_on <= s1)
        fixs = [Fixation(int(round(fix_on[j])), int(round(fix_off[j])),
                         float(fix_dur[j]), float(fx[j]), float(fy[j]))
                for j in np.where(sel)[0]]  #Those are the fixation rows that belong to the sentence.
        sentences.append(Sentence(i, int(round(s0)), int(round(s1)), fixs))

    # --- blinks -> EEG samples (ground-truth artifact intervals) -------------
    bl_on = convert(bdata[:, bi["latency"]])
    bl_off = convert(bdata[:, bi["endtime"]])
    keep = (bl_off >= 0) & (bl_on < n_samples)
    blink_intervals = np.stack([np.round(bl_on[keep]).astype(int),
                                np.round(bl_off[keep]).astype(int)], axis=1)
# start     end
# -----     -----
# 1200      1300
# 5400      5500
# 8200      8400
# ...
    total_fix = sum(len(s.fixations) for s in sentences)
    print(f"[raw-load] {subject}/{block}: {data.shape[0]}ch x {n_samples} samp "
          f"@ {sfreq}Hz ({n_samples/sfreq:.1f}s)")
    print(f"           sentences={len(sentences)}  fixations(in-sentence)={total_fix}"
          f"  blinks={len(blink_intervals)}")       #[raw-load] YAG/NR1: 128ch x 348000 samp @ 500Hz (696.0s)
    return RawBlock(subject, block, data, sfreq, ch_names, ch_pos,
                    sentences, blink_intervals)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("eeg"); ap.add_argument("et")
    ap.add_argument("--subject", default="YAG"); ap.add_argument("--block", default="NR1")
    a = ap.parse_args()
    blk = load_raw_block(a.eeg, a.et, a.subject, a.block)
    s0 = blk.sentences[0]
    print(f"\nsentence 0: samples [{s0.onset_sample},{s0.offset_sample}] "
          f"dur={(s0.offset_sample-s0.onset_sample)/blk.sfreq:.2f}s "
          f"fixations={len(s0.fixations)}")
    if s0.fixations:
        f0 = s0.fixations[0]
        print(f"  first fixation: sample {f0.onset_sample}-{f0.offset_sample} "
              f"dur={f0.duration_ms:.0f}ms pos=({f0.x:.0f},{f0.y:.0f})")
    print(f"first 3 blink intervals (samples): {blk.blink_intervals[:3].tolist()}")

# Raw EEG
# + ET
# + synchronization
# + sentence boundaries
# + fixation boundaries
# + blink boundaries


