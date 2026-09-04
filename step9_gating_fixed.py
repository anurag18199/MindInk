#!/usr/bin/env python3
"""
MindInk / ZuCo Step 9 (CORRECTED)
Differentiable Component Gating — now with a REAL denoising objective.

WHAT CHANGED vs the original step9 (all marked `# FIX` below)
  FIX 1  The SNR loss now gates the CORRUPTED sources and compares to the CLEAN
         target: L = || A(g . noisy_sources) - A.S ||.  The original gated the
         clean sources against the clean target, which is minimized by g=1 for
         all components (keep-everything) and taught no denoising at all.
  FIX 2  Corruption is built by AMPLIFYING artifact-like components inside the ICA
         basis at a controlled SNR (so a perfect gate can undo it). Artifact comps
         are chosen by ICLabel when it is actually informative, else by a kurtosis
         fallback, so training is meaningful even before ICLabel is fixed.
  FIX 3  Minibatch over synthetic pairs each epoch (the original built the graph
         over all 1745 pairs every step -> slow).
  FIX 4  current_iclabel passed explicitly, not via a module global.
  FIX 5  Feature normalization stats are computed on TRAIN components only in
         spirit (here all 98 are the subject's; scaler saved for reuse downstream).

UNCHANGED (kept deliberately)
  - g_k = sigmoid(f(phi_k)); component index NEVER used.
  - ICA never refit; A = ica.get_components() (105x98), never learned.
  - ICLabel is an artifact PRIOR, not the decision rule.
  - Step-10 decoding loss kept as an explicit zero hook.

This is still a BOOTSTRAP stage: the decode loss is 0 until Step 10 supplies it.
But the gate now genuinely learns to suppress injected artifacts, which you can
verify with step9b_test.py (learned_gate should climb toward / above hard_reject).
"""

from __future__ import annotations
import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# ============================================================================
# Configuration
# ============================================================================
DEFAULT_SUBJECT = "YAG"
N_CHANNELS = 104   # raw route (auto-checked against mixing_sensor.npy)
N_COMPONENTS = 98
N_FEATURES = 35
FS = 250.0

ICLABEL_CLASSES = ["brain", "muscle", "eye", "heart",
                   "line_noise", "channel_noise", "other"]
ARTIFACT_CLASS_INDICES = [1, 2, 4]        # muscle, eye, line_noise

DEFAULT_EPOCHS = 100
DEFAULT_LR = 1e-3
DEFAULT_HIDDEN = 64

# Loss weights (weights, NOT per-component scores).
DEFAULT_LAMBDA_ARTIFACT = 0.5
DEFAULT_LAMBDA_SNR = 1.0
DEFAULT_LAMBDA_SPARSITY = 0.01

DEFAULT_SYNTHETIC_SNRS = (-5.0, 0.0, 5.0, 10.0)
DEFAULT_K_ARTIFACT = 5                      # how many comps to corrupt per pair
DEFAULT_BATCH = 16                          # FIX 3: minibatch of synthetic pairs
EPS = 1e-8


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ============================================================================
# JSON helpers
# ============================================================================
def load_json(path: Path):
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, obj) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


# ============================================================================
# Step-6 loading  (features + ICLabel posteriors)
# ============================================================================
def load_step6(features_root: Path, subject: str):
    d = Path(features_root) / subject
    features = np.load(d / "component_features.npy", allow_pickle=False)
    names = load_json(d / "feature_names.json")
    iclabel = _extract_iclabel(load_json(d / "iclabel.json"))

    if features.shape != (N_COMPONENTS, N_FEATURES):
        raise RuntimeError(f"features {features.shape} != ({N_COMPONENTS},{N_FEATURES})")
    if iclabel.shape != (N_COMPONENTS, 7):
        raise RuntimeError(f"iclabel {iclabel.shape} != ({N_COMPONENTS},7)")
    if not np.isfinite(features).all():
        raise RuntimeError("Step-6 features contain NaN/Inf.")
    return features.astype(np.float32), names, iclabel.astype(np.float32)


def _extract_iclabel(records):
    items = list(records.values()) if isinstance(records, dict) else records
    if len(items) != N_COMPONENTS:
        raise RuntimeError(f"Expected {N_COMPONENTS} ICLabel records, got {len(items)}")
    rows = []
    for it in items:
        row = None
        for key in ("probabilities", "posterior", "probs"):
            v = it.get(key) if isinstance(it, dict) else None
            if isinstance(v, (list, tuple, np.ndarray)):
                a = np.asarray(v, dtype=np.float32).reshape(-1)
                if a.size == 7:
                    row = a
                    break
            if isinstance(v, dict):
                cand = [float(v[c]) for c in ICLABEL_CLASSES if c in v]
                if len(cand) == 7:
                    row = np.asarray(cand, dtype=np.float32)
                    break
        if row is None:  # last resort: class-named scalar fields on the record
            cand = [float(it[c]) for c in ICLABEL_CLASSES
                    if isinstance(it, dict) and isinstance(it.get(c), (int, float))]
            if len(cand) == 7:
                row = np.asarray(cand, dtype=np.float32)
        if row is None:
            raise RuntimeError("Could not extract 7 ICLabel probabilities.")
        rows.append(row)
    return np.vstack(rows)


# ============================================================================
# Frozen ICA + sources + sensor loading
# ============================================================================
def load_mixing(ica_root: Path, subject: str, mixing_path=None):
    """
    Sensor-space A (n_channels x 98). RAW ROUTE: prefer mixing_sensor.npy saved by
    step4_ica_raw (no MNE needed). Fall back to an explicit --mixing path, then to
    recovering via MNE get_components(). Channel count is validated against A, not
    hard-assumed, so 104 (or 105 with Cz) both work.
    """
    A = None
    auto = Path(ica_root) / subject / "mixing_sensor.npy"
    if mixing_path:
        A = np.load(mixing_path).astype(np.float32)
    elif auto.exists():
        A = np.load(auto).astype(np.float32)
    else:
        import mne
        d = Path(ica_root) / subject
        fif = d / "ica_model-ica.fif"
        if not fif.exists():
            fif = d / "ica_model.fif"
        ica = mne.preprocessing.read_ica(fif, verbose=False)
        if ica.n_components_ != N_COMPONENTS:
            raise RuntimeError(f"ICA has {ica.n_components_} comps; expected {N_COMPONENTS}")
        A = np.asarray(ica.get_components(), dtype=np.float32)
    if A.ndim != 2 or A.shape[1] != N_COMPONENTS:
        raise RuntimeError(f"A {A.shape} must be (n_channels,{N_COMPONENTS}); "
                           "do NOT use the reduced (98,98) matrix.")
    return A


def sentence_indices(ica_root: Path, subject: str):
    root = Path(ica_root) / subject / "sources"
    idxs = []
    for p in sorted(root.glob("sentence_*_sources.npy")):
        try:
            idxs.append(int(p.name.split("_")[1]))
        except Exception:
            pass
    if not idxs:
        raise RuntimeError(f"No source files in {root}")
    return sorted(idxs)


def load_sources(ica_root: Path, subject: str, idx: int):
    p = Path(ica_root) / subject / "sources" / f"sentence_{idx:03d}_sources.npy"
    S = np.load(p, allow_pickle=False)
    if S.ndim != 2 or S.shape[0] != N_COMPONENTS:
        raise RuntimeError(f"{p}: expected ({N_COMPONENTS},T), got {S.shape}")
    return S.astype(np.float32)


def load_sensor(baselines_root: Path, subject: str, idx: int):
    p = (Path(baselines_root) / subject / "no_denoising"
         / f"sentence_{idx:03d}" / "sensor_continuous.npy")
    return np.load(p, allow_pickle=False).astype(np.float32)


# ============================================================================
# Gate network  (unchanged)
# ============================================================================
class ComponentGate(nn.Module):
    def __init__(self, n_features=N_FEATURES, hidden=DEFAULT_HIDDEN):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(n_features, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, features):
        return torch.sigmoid(self.network(features).squeeze(-1))


# ============================================================================
# Feature normalization  (FIX 5)
# ============================================================================
def fit_scaler(features):
    mean = features.mean(axis=0)
    std = features.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


def apply_scaler(features, mean, std):
    return ((features - mean) / std).astype(np.float32)


# ============================================================================
# Artifact selection + corruption  (FIX 2)
# ============================================================================
def artifact_probability(iclabel):
    return iclabel[:, ARTIFACT_CLASS_INDICES].sum(axis=1)


def pick_artifact_components(iclabel, sources_list, k, manual=None,
                             prob_threshold=0.5):
    """
    Choose which components to inject as artifact (and thus train the gate to suppress).
      1. manual override if given (e.g. --artifact-index 4 5 8 11 12 26 32 49).
      2. RAW ROUTE: ICLabel artifact prob is now informative -> select ALL components
         with P(eye+muscle+line) >= prob_threshold. This matches the Step-7 hard-reject
         set exactly, so the gate learns to suppress the SAME real artifacts the
         classical baseline removes (fair, physically-meaningful comparison).
      3. If ICLabel is flat (preprocessed data), fall back to top-k kurtosis.
    """
    if manual:
        return list(manual)
    p = artifact_probability(iclabel)
    if float(p.max() - p.min()) > 1e-3:
        thr_idx = list(np.flatnonzero(p >= prob_threshold))
        if thr_idx:                       # real artifacts found by ICLabel
            return thr_idx
        return list(np.argsort(p)[::-1][:k])   # none above threshold: take top-k by prob
    from scipy.stats import kurtosis
    cat = np.concatenate([s[:, : min(s.shape[1], 2000)] for s in sources_list], axis=1)
    kurt = kurtosis(cat, axis=1, fisher=True, bias=False)
    return list(np.argsort(kurt)[::-1][:k])


def boost_for_target_snr(A, S, art_idx, snr_db):
    """Amplification on artifact comps giving input SNR = snr_db vs clean = A@S."""
    clean = A @ S
    S_art = np.zeros_like(S)
    S_art[art_idx, :] = S[art_idx, :]
    noise_unit = A @ S_art
    ratio = 10.0 ** (snr_db / 20.0)
    k = np.sqrt(np.sum(clean ** 2)) / (ratio * (np.sqrt(np.sum(noise_unit ** 2)) + EPS))
    return 1.0 + float(k)


# ============================================================================
# Losses
# ============================================================================
def artifact_prior_loss(gates, artifact_p):
    p = torch.as_tensor(artifact_p, dtype=gates.dtype, device=gates.device)
    return torch.mean(p * gates)


def gate_sparsity_loss(gates):
    return torch.mean(gates)


def snr_recon_loss(recon, clean):
    scale = torch.sqrt(torch.mean(clean * clean) + EPS)
    return torch.mean(((recon - clean) / scale) ** 2)


def decoding_loss_placeholder(device):
    # Step-10 hook. MUST be replaced by the real BART/IIST loss when available.
    return torch.zeros((), device=device)


# ============================================================================
# Training  (FIX 1 + FIX 3)
# ============================================================================
def train_gate(features, iclabel, sentence_sources, mixing, args, device):
    set_seed(args.seed)
    mean, std = fit_scaler(features)
    x = torch.tensor(apply_scaler(features, mean, std), dtype=torch.float32, device=device)
    A = torch.tensor(mixing, dtype=torch.float32, device=device)
    artifact_p = artifact_probability(iclabel)

    model = ComponentGate().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    sources_list = list(sentence_sources.values())
    art_idx = pick_artifact_components(iclabel, sources_list,
                                       args.k_artifact, args.artifact_index)
    print(f" Artifact components injected: {sorted(int(i) for i in art_idx)}")

    # ---- Pre-build corrupted (clean, noisy_sources) pairs.  FIX 1/FIX 2 -----
    pairs = []
    for idx, S in sentence_sources.items():
        clean = mixing @ S
        for snr_db in args.snrs:
            boost = boost_for_target_snr(mixing, S, art_idx, snr_db)
            noisy_sources = S.copy()
            noisy_sources[art_idx, :] *= boost            # corrupt IN the ICA basis
            pairs.append((clean.astype(np.float32),
                          noisy_sources.astype(np.float32)))
    if not pairs:
        raise RuntimeError("No synthetic pairs.")
    print(f" Synthetic pairs : {len(pairs)}  (SNRs {list(args.snrs)} dB)")

    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        random.shuffle(pairs)
        gates = model(x)                                   # (98,), same gate for all

        # minibatch the SNR term  (FIX 3)
        batch = pairs[: args.batch]
        snr_terms = []
        for clean_np, noisy_src_np in batch:
            clean = torch.tensor(clean_np, device=device)
            noisy_src = torch.tensor(noisy_src_np, device=device)
            recon = A @ (gates[:, None] * noisy_src)       # FIX 1: gate the NOISY sources
            snr_terms.append(snr_recon_loss(recon, clean))
        loss_snr = torch.stack(snr_terms).mean()

        loss_art = artifact_prior_loss(gates, artifact_p)
        loss_sps = gate_sparsity_loss(gates)
        loss_dec = decoding_loss_placeholder(device)

        total = (args.lambda_snr * loss_snr
                 + args.lambda_artifact * loss_art
                 + args.lambda_sparsity * loss_sps
                 + loss_dec)

        opt.zero_grad()
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()

        rec = {"epoch": epoch, "total": float(total.detach().cpu()),
               "snr": float(loss_snr.detach().cpu()),
               "artifact": float(loss_art.detach().cpu()),
               "sparsity": float(loss_sps.detach().cpu()), "decoding": 0.0}
        history.append(rec)
        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            print(f"  Epoch {epoch:03d}/{args.epochs}: total={rec['total']:.5f} "
                  f"snr={rec['snr']:.5f} artifact={rec['artifact']:.5f} "
                  f"sparse={rec['sparsity']:.5f}")

    return model, mean, std, history, sorted(int(i) for i in art_idx)


# ============================================================================
# Save gated outputs  (unchanged behavior)
# ============================================================================
def save_gated(output_subject, sentence_sources, gates, mixing, baselines_root, subject):
    sroot = Path(output_subject) / "sentences"
    sroot.mkdir(parents=True, exist_ok=True)
    recs = []
    for idx, S in sentence_sources.items():
        original = load_sensor(baselines_root, subject, idx)
        gated = mixing @ (gates[:, None] * S)
        d = sroot / f"sentence_{idx:03d}"
        d.mkdir(parents=True, exist_ok=True)
        np.save(d / "original_sensor.npy", original)
        np.save(d / "gated_sensor.npy", gated.astype(np.float32))
        np.save(d / "gates.npy", gates.astype(np.float32))
        recs.append({"sentence_index": int(idx),
                     "gate_mean": float(gates.mean()),
                     "gate_min": float(gates.min()),
                     "gate_max": float(gates.max())})
    return recs


# ============================================================================
# Main
# ============================================================================
def run_step9(args):
    device = torch.device(args.device)
    print("=" * 72)
    print("MindInk / ZuCo Step 9 (CORRECTED) — denoising-aware gate bootstrap")
    print("=" * 72)

    features, names, iclabel = load_step6(args.features, args.subject)
    mixing = load_mixing(args.ica, args.subject, args.mixing)
    idxs = sentence_indices(args.ica, args.subject)
    if args.max_sentences:
        idxs = idxs[: args.max_sentences]
    sentence_sources = {i: load_sources(args.ica, args.subject, i) for i in idxs}
    print(f" Subject {args.subject} | comps {N_COMPONENTS} | feats {N_FEATURES} "
          f"| sentences {len(idxs)} | A {mixing.shape}")

    model, mean, std, history, art_idx = train_gate(
        features, iclabel, sentence_sources, mixing, args, device)

    model.eval()
    with torch.no_grad():
        gates = model(torch.tensor(apply_scaler(features, mean, std),
                                   dtype=torch.float32, device=device)).cpu().numpy()
    gates = gates.astype(np.float32)
    if gates.shape != (N_COMPONENTS,) or not np.isfinite(gates).all():
        raise RuntimeError("Bad gate vector.")

    print("\nFinal gate statistics:")
    print(f"  mean {gates.mean():.4f} | median {np.median(gates):.4f} "
          f"| min {gates.min():.4f} | max {gates.max():.4f}")
    print(f"  gates on injected-artifact comps {art_idx}: "
          f"{np.round(gates[art_idx], 3).tolist()}  (should be LOW now)")

    out = Path(args.output) / args.subject
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(),
                "feature_names": names, "feature_mean": mean, "feature_std": std,
                "architecture": "sigmoid(MLP(phi_k))", "component_index_used": False,
                "ica_frozen": True, "artifact_components": art_idx, "step": 9},
               out / "gate_model.pt")
    np.save(out / "gate_values.npy", gates)
    np.save(out / "gate_features.npy", apply_scaler(features, mean, std))
    recs = save_gated(out, sentence_sources, gates, mixing, Path(args.baselines), args.subject)
    save_json(out / "training_history.json", history)
    save_json(out / "manifest.json", {
        "step": "9", "subject_id": args.subject,
        "method": "differentiable_component_gate_denoising",
        "status": "bootstrap_pretraining", "component_index_used": False,
        "ica_frozen": True, "artifact_components": art_idx,
        "snr_loss": "gate(noisy_sources) vs clean target  [corrected]",
        "sentences_processed": len(recs),
        "warnings": ["Bootstrap only; decode loss still 0 (Step 10).",
                     "Synthetic corruption = amplified artifact comps, not real EOG/EMG.",
                     "Reference is real (still-noisy) EEG; not a clean-GT SNR claim."]})
    print(f"\nSaved -> {out}")
    print("Now run step9b_test.py to check learned_gate vs hard_reject.")


def build_parser():
    ap = argparse.ArgumentParser(description="MindInk Step 9 (corrected) gating")
    ap.add_argument("--subject", default=DEFAULT_SUBJECT)
    ap.add_argument("--ica", default="ica")
    ap.add_argument("--features", default="features")
    ap.add_argument("--baselines", default="baselines")
    ap.add_argument("--output", default="gates")
    ap.add_argument("--mixing", default=None, help="Optional saved (n_ch,98) A .npy; else uses mixing_sensor.npy")
    ap.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    ap.add_argument("--lr", type=float, default=DEFAULT_LR)
    ap.add_argument("--lambda-artifact", dest="lambda_artifact",
                    type=float, default=DEFAULT_LAMBDA_ARTIFACT)
    ap.add_argument("--lambda-snr", dest="lambda_snr",
                    type=float, default=DEFAULT_LAMBDA_SNR)
    ap.add_argument("--lambda-sparsity", dest="lambda_sparsity",
                    type=float, default=DEFAULT_LAMBDA_SPARSITY)
    ap.add_argument("--snrs", type=float, nargs="+", default=list(DEFAULT_SYNTHETIC_SNRS))
    ap.add_argument("--k-artifact", dest="k_artifact", type=int, default=DEFAULT_K_ARTIFACT)
    ap.add_argument("--artifact-index", dest="artifact_index",
                    type=int, nargs="+", default=None)
    ap.add_argument("--batch", type=int, default=DEFAULT_BATCH)
    ap.add_argument("--max-sentences", dest="max_sentences", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    return ap


if __name__ == "__main__":
    a = build_parser().parse_args()
    if a.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but not available.")
    run_step9(a)
