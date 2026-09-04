#!/usr/bin/env python3
"""
run_all_subjects.py — MindInk raw-route ORCHESTRATOR for all 18 ZuCo subjects.

Runs the full raw pipeline (Step 3 -> Step 9) for every subject, in one command,
by importing each step module and calling its run_* function in-process. Designed to
be robust for an overnight batch and to lay clean groundwork for the future decoder.

WHAT IT DOES PER SUBJECT
  Step 3  step3_raw_preprocess.preprocess_subject_raw   -> processed_raw/<subj>/
  Step 4  step4_ica_raw.run                             -> ica/<subj>/
  Step 5  step5_raw_segment.segment_subject             -> segments/<subj>/
  Step 6  step6_raw_features.run_step6                  -> features/<subj>/
  Step 7  step7_baselines_raw.run_step7                 -> baselines/<subj>/
  Step 8  step8_snr_raw.evaluate_subject                -> metrics/<subj>/
  Step 9  step9_raw_gating.run_step9                    -> gates/<subj>/

KEY BEHAVIOURS
  * Auto-discovers each subject's NR blocks (subjects have different block counts),
    by globbing <raw-root>/<subj>/<subj>_NR*_EEG.mat.
  * RESUMABLE: if a step's expected output already exists, it is skipped (unless
    --force). A crash on subject 12 does not redo subjects 1-11.
  * FAULT-ISOLATED: an exception in one subject/step is logged and the orchestrator
    continues with the next subject; a final summary reports pass/fail per step.
  * FUTURE-DECODER-READY: writes runs/manifest.json — a single index listing, per
    subject, the paths to its gate, sources, features, segments and baselines. The
    decoder stage can load this one file to find every subject's artifacts, and to
    build train/val/test splits BY SUBJECT (needed for subject-invariance).

DIRECTORY LAYOUT PRODUCED (all under --work-dir, default = current dir)
  processed_raw/<subj>/   ica/<subj>/   segments/<subj>/   features/<subj>/
  baselines/<subj>/       metrics/<subj>/  gates/<subj>/
  runs/manifest.json      runs/orchestrator.log

USAGE
  python run_all_subjects.py --raw-root "datasets-NR"
  python run_all_subjects.py --raw-root "datasets-NR" --subjects YAG YAC
  python run_all_subjects.py --raw-root "datasets-NR" --from-step 6 --force
  python run_all_subjects.py --raw-root "datasets-NR" --skip-step9   # e.g. run gating later on GPU

Expected raw layout:  <raw-root>/<SUBJ>/<SUBJ>_NR<k>_EEG.mat  and  _ET.mat
"""

from __future__ import annotations
import argparse
import json
import time
import traceback
from datetime import datetime
from pathlib import Path

# Step modules (raw route)
import step3_raw_prepprocess as s3
import step4_ica_raw as s4
import step5_raw_segment as s5
import step6_features as s6
import step7_baselines_raw as s7
import step8_snr_raw as s8
import step9_gating_fixed as s9

from step2_raw_load import load_raw_block


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------
def discover_subjects(raw_root: Path):
    """Every sub-folder of raw_root that contains at least one *_NR*_EEG.mat."""
    subs = []
    for d in sorted(p for p in raw_root.iterdir() if p.is_dir()):
        if list(d.glob(f"{d.name}_NR*_EEG.mat")):
            subs.append(d.name)
    return subs


def discover_blocks(raw_root: Path, subject: str):
    """Ordered list of NR block tags present for a subject, e.g. ['NR1','NR2',...]."""
    d = raw_root / subject
    tags = []
    for eeg in sorted(d.glob(f"{subject}_NR*_EEG.mat")):
        tag = eeg.name.replace(f"{subject}_", "").replace("_EEG.mat", "")
        et = d / f"{subject}_{tag}_ET.mat"
        if et.exists():
            tags.append(tag)
        else:
            log(f"    [warn] {subject} {tag}: EEG present but ET missing -> skipped")
    # sort NR1, NR2, ... numerically
    def keyf(t):
        num = "".join(ch for ch in t if ch.isdigit())
        return int(num) if num else 0
    return sorted(tags, key=keyf)


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
_LOG_FH = None
def log(msg):
    line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    if _LOG_FH:
        _LOG_FH.write(line + "\n"); _LOG_FH.flush()


# ---------------------------------------------------------------------------
# "Is this step already done?" checks (for resumability)
# ---------------------------------------------------------------------------
def done_step3(work, subj):
    return (work/"processed_raw"/subj/f"{subj}_ica_raw.fif").exists() and \
           (work/"processed_raw"/subj/f"{subj}_analysis_raw.fif").exists()
def done_step4(work, subj):
    return (work/"ica"/subj/"mixing_sensor.npy").exists() and \
           (work/"ica"/subj/"sources").exists()
def done_step5(work, subj):
    return (work/"segments"/subj/"segment_manifest.json").exists()
def done_step6(work, subj):
    return (work/"features"/subj/"component_features.npy").exists()
def done_step7(work, subj):
    return (work/"baselines"/subj/"manifest.json").exists()
def done_step8(work, subj):
    return (work/"metrics"/subj/"summary.json").exists()
def done_step9(work, subj):
    return (work/"gates"/subj/"gate_values.npy").exists()


# ---------------------------------------------------------------------------
# Per-subject pipeline
# ---------------------------------------------------------------------------
def process_subject(subject, raw_root, work, args, status):
    st = {}
    def mark(step, ok, note=""):
        st[step] = {"ok": ok, "note": note}

    # ---- Step 3: load blocks + preprocess ---------------------------------
    if args.from_step <= 3 and not (done_step3(work, subject) and not args.force):
        blocks_tags = discover_blocks(raw_root, subject)
        if not blocks_tags:
            raise RuntimeError("no NR blocks with matching EEG+ET found")
        log(f"    Step3: loading {len(blocks_tags)} blocks {blocks_tags}")
        blocks = []
        for tag in blocks_tags:
            eeg = raw_root/subject/f"{subject}_{tag}_EEG.mat"
            et  = raw_root/subject/f"{subject}_{tag}_ET.mat"
            blocks.append(load_raw_block(str(eeg), str(et), subject, tag))
        s3.preprocess_subject_raw(blocks, str(work/"processed_raw"), subject)
        mark(3, True, f"{len(blocks_tags)} blocks")
    else:
        mark(3, done_step3(work, subject), "skipped/exists")

    # ---- Step 4: ICA ------------------------------------------------------
    if args.from_step <= 4 and not (done_step4(work, subject) and not args.force):
        log("    Step4: fitting ICA (this is the slow one, ~5-10 min)")
        s4.run(str(work/"processed_raw"), str(work/"ica"), subject)
        mark(4, True)
    else:
        mark(4, done_step4(work, subject), "skipped/exists")

    # ---- Step 5: segmentation --------------------------------------------
    if args.from_step <= 5 and not (done_step5(work, subject) and not args.force):
        log("    Step5: segmenting")
        s5.segment_subject(str(work/"processed_raw"), str(work/"ica"),
                           str(work/"segments"), subject, use_mne=True)
        mark(5, True)
    else:
        mark(5, done_step5(work, subject), "skipped/exists")

    # ---- Step 6: features + ICLabel --------------------------------------
    if args.from_step <= 6 and not (done_step6(work, subject) and not args.force):
        log("    Step6: features + ICLabel")
        s6.run_step6(subject_id=subject, ica_root=str(work/"ica"),
                     segments_root=str(work/"segments"), output_root=str(work/"features"))
        mark(6, True)
    else:
        mark(6, done_step6(work, subject), "skipped/exists")

    # ---- Step 7: baselines -----------------------------------------------
    if args.from_step <= 7 and not (done_step7(work, subject) and not args.force):
        log("    Step7: classical baselines")
        s7.run_step7(subject_id=subject, processed_root=str(work/"processed_raw"),
                     ica_root=str(work/"ica"), features_root=str(work/"features"),
                     segments_root=str(work/"segments"), output_root=str(work/"baselines"),
                     threshold=args.threshold)
        mark(7, True)
    else:
        mark(7, done_step7(work, subject), "skipped/exists")

    # ---- Step 8: SNR / quality -------------------------------------------
    if args.from_step <= 8 and not (done_step8(work, subject) and not args.force):
        log("    Step8: signal-quality evaluation")
        s8.evaluate_subject(subject=subject, baselines_root=work/"baselines",
                            output_root=work/"metrics", max_sentences=None)
        mark(8, True)
    else:
        mark(8, done_step8(work, subject), "skipped/exists")

    # ---- Step 9: gate -----------------------------------------------------
    if not args.skip_step9 and args.from_step <= 9 and not (done_step9(work, subject) and not args.force):
        log(f"    Step9: training gate ({args.epochs} epochs)")
        gate_args = argparse.Namespace(
            subject=subject, ica=str(work/"ica"), features=str(work/"features"),
            baselines=str(work/"baselines"), output=str(work/"gates"), mixing=None,
            epochs=args.epochs, lr=1e-3, lambda_artifact=0.5, lambda_snr=1.0,
            lambda_sparsity=0.01, snrs=[-5.0,0.0,5.0,10.0], k_artifact=5,
            artifact_index=None, batch=16, max_sentences=None, seed=42, device=args.device)
        s9.run_step9(gate_args)
        mark(9, True)
    else:
        mark(9, done_step9(work, subject) if not args.skip_step9 else False,
             "skipped" )

    status[subject] = st
    return st


# ---------------------------------------------------------------------------
# Cross-subject manifest (for the future decoder / subject-invariance)
# ---------------------------------------------------------------------------
def build_manifest(work, subjects, status):
    entries = {}
    for subj in subjects:
        base = {
            "processed_analysis_fif": str(work/"processed_raw"/subj/f"{subj}_analysis_raw.fif"),
            "ica_dir":       str(work/"ica"/subj),
            "mixing_sensor": str(work/"ica"/subj/"mixing_sensor.npy"),
            "sources_dir":   str(work/"ica"/subj/"sources"),
            "features":      str(work/"features"/subj/"component_features.npy"),
            "iclabel":       str(work/"features"/subj/"iclabel.json"),
            "segments_dir":  str(work/"segments"/subj),
            "baselines_dir": str(work/"baselines"/subj),
            "gate_values":   str(work/"gates"/subj/"gate_values.npy"),
            "gated_sentences_dir": str(work/"gates"/subj/"sentences"),
            "blink_intervals": str(work/"processed_raw"/subj/"blink_intervals.npy"),
            "step_status":   status.get(subj, {}),
        }
        # only keep the entry if the core artifacts exist
        base["complete"] = all(Path(base[k]).exists() for k in
                               ("mixing_sensor","features","segments_dir"))
        entries[subj] = base

    complete = [s for s in subjects if entries[s]["complete"]]
    manifest = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "n_subjects_total": len(subjects),
        "n_subjects_complete": len(complete),
        "subjects_complete": complete,
        "subjects": entries,
        # a suggested leave-one-subject-out split scaffold for the decoder stage
        "loso_folds": [{"test": s, "train": [o for o in complete if o != s]}
                       for s in complete],
        "notes": [
            "This manifest is the single entry point for the decoder / subject-invariance stage.",
            "Load per-subject artifacts via these paths; build splits BY SUBJECT using loso_folds.",
            "Segments hold per-word fixation pieces (sensor + sources) for word-level decoder input.",
            "gate_values.npy is the trained soft gate per subject; apply to that subject's sources.",
        ],
    }
    out = work/"runs"/"manifest.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(manifest, f, indent=2)
    return manifest, out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="MindInk raw-route orchestrator (all subjects)")
    ap.add_argument("--raw-root", required=True,
                    help="Folder containing one sub-folder per subject (YAG/, YAC/, ...)")
    ap.add_argument("--work-dir", default=".", help="Where all outputs go (default: cwd)")
    ap.add_argument("--subjects", nargs="+", default=None,
                    help="Explicit subject list; default = auto-discover all")
    ap.add_argument("--from-step", type=int, default=3, choices=range(3,10),
                    help="Start from this step (uses existing outputs of earlier steps)")
    ap.add_argument("--skip-step9", action="store_true",
                    help="Skip gate training (e.g. to run it later on a GPU)")
    ap.add_argument("--force", action="store_true",
                    help="Recompute even if a step's output already exists")
    ap.add_argument("--threshold", type=float, default=0.5, help="ICLabel reject threshold (Step 7)")
    ap.add_argument("--epochs", type=int, default=100, help="Gate training epochs (Step 9)")
    ap.add_argument("--device", default="cpu", choices=("cpu","cuda"))
    ap.add_argument("--stop-on-error", action="store_true",
                    help="Abort the whole run if any subject fails (default: continue)")
    args = ap.parse_args()

    raw_root = Path(args.raw_root)
    work = Path(args.work_dir)
    (work/"runs").mkdir(parents=True, exist_ok=True)

    global _LOG_FH
    _LOG_FH = open(work/"runs"/"orchestrator.log", "a", encoding="utf-8")

    subjects = args.subjects or discover_subjects(raw_root)
    if not subjects:
        raise SystemExit(f"No subjects found under {raw_root} (expected <subj>/<subj>_NR*_EEG.mat)")

    log("=" * 72)
    log(f"MindInk orchestrator — {len(subjects)} subjects: {subjects}")
    log(f"raw-root={raw_root}  work-dir={work}  from-step={args.from_step}  "
        f"skip9={args.skip_step9}  force={args.force}  device={args.device}")
    log("=" * 72)

    status = {}
    t0 = time.time()
    for i, subj in enumerate(subjects, 1):
        log(f"\n[{i}/{len(subjects)}] ===== SUBJECT {subj} =====")
        ts = time.time()
        try:
            process_subject(subj, raw_root, work, args, status)
            log(f"    subject {subj} done in {(time.time()-ts)/60:.1f} min")
        except Exception as e:
            log(f"    !! subject {subj} FAILED: {e}")
            log(traceback.format_exc())
            status.setdefault(subj, {})["error"] = str(e)
            if args.stop_on_error:
                log("    --stop-on-error set; aborting."); break

    manifest, mpath = build_manifest(work, subjects, status)

    # ---- final summary table ----
    log("\n" + "=" * 72)
    log("RUN SUMMARY")
    log("=" * 72)
    steps = [3,4,5,6,7,8,9]
    log("subject   " + "  ".join(f"S{s}" for s in steps) + "   complete")
    for subj in subjects:
        st = status.get(subj, {})
        cells = []
        for s in steps:
            v = st.get(s)
            cells.append(" ok" if (v and v.get("ok")) else ("---" if args.skip_step9 and s==9 else " XX"))
        comp = "yes" if manifest["subjects"].get(subj,{}).get("complete") else "no"
        log(f"{subj:<9} " + " ".join(cells) + f"    {comp}")
    log(f"\ncomplete subjects: {manifest['n_subjects_complete']}/{manifest['n_subjects_total']}")
    log(f"total wall time: {(time.time()-t0)/60:.1f} min")
    log(f"manifest -> {mpath}")
    log("Next stage (decoder) can load runs/manifest.json to find every subject's artifacts.")
    _LOG_FH.close()


if __name__ == "__main__":
    main()