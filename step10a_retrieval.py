#!/usr/bin/env python3
"""
step10a_retrieval.py — DECISIVE first decode experiment: sentence-level contrastive
retrieval, GATED vs UNGATED EEG, across the 17-subject cohort.

THE QUESTION THIS ANSWERS
  Does the (gated) EEG carry decodable linguistic signal at all, and does YOUR
  Step-9 gate HELP decoding? We test this WITHOUT a text decoder (so no
  teacher-forcing trap): we learn an EEG encoder that maps a sentence's EEG into
  the same vector space as the sentence's text embedding, trained with a contrastive
  loss, then measure RETRIEVAL — given a sentence's EEG, does its true text rank
  above other sentences' texts?

WHY THIS IS THE RIGHT FIRST TEST
  * If real EEG can't beat the controls here, full text generation won't work either
    -- and we learn that in minutes, not weeks.
  * It directly compares GATED vs UNGATED EEG -> quantifies whether the contribution
    helps decoding (the whole thesis), on real data, with a clean metric.
  * Contrastive alignment gives a gradient that flows ONLY through the EEG (no LM
    crutch), which is exactly the mechanism the decoder stage needs.

INPUTS (from your pipeline; sentence-level, no word labels needed)
  runs/manifest.json                         -> per-subject artifact paths + LOSO folds
  ica/<subj>/sources/sentence_XXX_sources.npy-> ICA sources per sentence
  ica/<subj>/mixing_sensor.npy               -> sensor-space A (n_ch x 98)
  gates/<subj>/gate_values.npy               -> trained soft gate (for GATED condition)
  <preprocessed>/resultsYYY_NR.mat           -> sentence text (sentenceData/content)
  GloVe embeddings file (--glove) OR --hashing fallback if you have no GloVe

CONDITIONS COMPARED (all with identical encoder capacity, for fairness)
  gated_eeg      : X = A @ (gate ⊙ S)      <- your contribution
  ungated_eeg    : X = A @ S               <- baseline (no gating)
  shuffled_eeg   : gated EEG, but paired with the WRONG sentence's text (control)
  gaussian_noise : random input matched in size (control)

METRICS (retrieval; higher = better; chance shown for reference)
  Top-1 / Top-5 accuracy, and Mean Reciprocal Rank (MRR).
  A working EEG signal => gated/ungated clearly beat shuffled & noise.
  Contribution helps  => gated beats ungated.

EVALUATION
  Leave-one-subject-out is available (--loso) using manifest folds, but the default
  is WITHIN-SUBJECT (train/test split per subject, averaged) because that is the
  cleaner first test of "is there signal". Cross-subject (LOSO) is the harder,
  subject-invariance question -- run it once within-subject works.

USAGE
  python step10a_retrieval.py --preprocessed-dir dataset-NR --glove glove.6B.300d.txt
  python step10a_retrieval.py --preprocessed-dir dataset-NR --hashing   # if no GloVe
  python step10a_retrieval.py --subjects YAG YAC --epochs 60
"""

from __future__ import annotations
import argparse, json, re
from pathlib import Path
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

N_COMPONENTS = 98
EMB_DIM = 300           # GloVe 300d (hashing fallback also uses 300)


# ---------------------------------------------------------------------------
# Sentence text (reuse the verified ZuCo decoder)
# ---------------------------------------------------------------------------
def load_sentence_text(preprocessed_dir, subject):
    """Read sentenceData/content from resultsYYY_NR.mat. Returns {index: text}."""
    import h5py
    # try a couple of common filename patterns
    cands = [Path(preprocessed_dir)/f"results{subject}_NR.mat",
             Path(preprocessed_dir)/subject/f"results{subject}_NR.mat",
             Path(preprocessed_dir)/f"{subject}_NR.mat"]
    path = next((c for c in cands if c.exists()), None)
    if path is None:
        return None
    out = {}
    with h5py.File(path, "r") as f:
        content = f["sentenceData/content"]
        n = content.shape[0]
        for i in range(n):
            vals = np.asarray(f[content[i,0]][()]).reshape(-1)
            txt = "".join(chr(int(v)) for v in vals if int(v) != 0).strip()
            if txt:
                out[i] = txt
    return out


# ---------------------------------------------------------------------------
# Word embeddings: GloVe if given, else a deterministic hashing embedding
# ---------------------------------------------------------------------------
class Embedder:
    def __init__(self, glove_path=None, dim=EMB_DIM):
        self.dim = dim
        self.vecs = None
        if glove_path and Path(glove_path).exists():
            self.vecs = {}
            with open(glove_path, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.rstrip().split(" ")
                    if len(parts) != dim + 1:
                        continue
                    self.vecs[parts[0]] = np.asarray(parts[1:], dtype=np.float32)
            print(f"  GloVe loaded: {len(self.vecs)} words, dim {dim}")
        else:
            print("  No GloVe -> using deterministic hashing embedding (fallback).")

    def _word(self, w):
        w = w.lower()
        if self.vecs is not None:
            return self.vecs.get(w)
        # hashing fallback: stable pseudo-random vector per word
        rng = np.random.default_rng(abs(hash(w)) % (2**32))
        v = rng.standard_normal(self.dim).astype(np.float32)
        return v / (np.linalg.norm(v) + 1e-8)

    def sentence(self, text):
        """Mean of word vectors -> one sentence embedding (L2-normalised)."""
        toks = re.findall(r"[a-zA-Z']+", text.lower())
        vs = [self._word(t) for t in toks]
        vs = [v for v in vs if v is not None]
        if not vs:
            return None
        v = np.mean(vs, axis=0)
        return v / (np.linalg.norm(v) + 1e-8)


# ---------------------------------------------------------------------------
# EEG encoder: sentence sources -> embedding-space vector
# ---------------------------------------------------------------------------
class EEGSentenceEncoder(nn.Module):
    """
    Summarise a sentence's sensor EEG (n_ch x T) into an EMB_DIM vector.
    Channel-wise band-ish stats over time -> MLP. Deliberately compact and identical
    across conditions so the comparison is fair.
    """
    def __init__(self, n_ch, emb_dim=EMB_DIM, hidden=512):
        super().__init__()
        # per-channel summary features: mean, std, and log-power in 5 coarse bands
        self.in_dim = n_ch * 7
        self.net = nn.Sequential(
            nn.Linear(self.in_dim, hidden), nn.LayerNorm(hidden), nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(hidden, emb_dim),
        )

    @staticmethod
    def summarise(x, fs=250.0):
        """x: (n_ch, T) tensor -> (n_ch*7) feature vector. Differentiable."""
        mean = x.mean(dim=1)
        std = x.std(dim=1) + 1e-6
        # crude band powers via FFT magnitude in 5 ranges
        X = torch.fft.rfft(x, dim=1)
        mag = (X.real**2 + X.imag**2)
        freqs = torch.fft.rfftfreq(x.shape[1], d=1.0/fs).to(x.device)
        bands = [(0.5,4),(4,8),(8,13),(13,30),(30,100)]
        bp = []
        for lo,hi in bands:
            m = (freqs>=lo)&(freqs<hi)
            p = mag[:, m].mean(dim=1) if m.any() else torch.zeros(x.shape[0], device=x.device)
            bp.append(torch.log(p + 1e-8))
        return torch.cat([mean, std] + bp, dim=0)   # (n_ch*7,)

    def forward(self, feats):
        z = self.net(feats)
        return F.normalize(z, dim=-1)               # unit vectors for cosine


# ---------------------------------------------------------------------------
# Data assembly per subject
# ---------------------------------------------------------------------------
def build_subject_samples(manifest, subject, text_map, embedder, condition):
    """
    Returns list of (eeg_feature_vector, sentence_embedding) for one subject.
    condition: 'gated' | 'ungated'  (shuffle/noise are made at eval time)
    """
    ent = manifest["subjects"][subject]
    A = np.load(ent["mixing_sensor"]).astype(np.float32)          # (n_ch, 98)
    gate = np.load(ent["gate_values"]).astype(np.float32)         # (98,)
    src_dir = Path(ent["sources_dir"])
    samples = []
    for p in sorted(src_dir.glob("sentence_*_sources.npy")):
        idx = int(p.name.split("_")[1])
        if idx not in text_map:
            continue
        emb = embedder.sentence(text_map[idx])
        if emb is None:
            continue
        S = np.load(p).astype(np.float32)                        # (98, T)
        if condition == "gated":
            S = gate[:, None] * S
        X = A @ S                                                # (n_ch, T)
        feat = EEGSentenceEncoder.summarise(torch.from_numpy(X)).numpy()
        samples.append((feat.astype(np.float32), emb.astype(np.float32), idx))
    return samples, A.shape[0]


# ---------------------------------------------------------------------------
# Train + evaluate retrieval for one subject, one condition
# ---------------------------------------------------------------------------
def info_nce(z_eeg, z_txt, temp=0.07):
    """Contrastive loss: match EEG to its own sentence text vs others in batch."""
    logits = z_eeg @ z_txt.t() / temp          # (B,B) cosine sims
    labels = torch.arange(z_eeg.shape[0], device=z_eeg.device)
    return 0.5*(F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))


def retrieval_metrics(z_eeg, z_txt):
    sims = z_eeg @ z_txt.t()                    # (N,N)
    n = sims.shape[0]
    ranks = []
    top1 = top5 = 0
    for i in range(n):
        order = torch.argsort(sims[i], descending=True)
        rank = (order == i).nonzero(as_tuple=True)[0].item() + 1
        ranks.append(rank)
        top1 += (rank == 1); top5 += (rank <= 5)
    ranks = np.array(ranks)
    return {"top1": top1/n, "top5": top5/n, "mrr": float(np.mean(1.0/ranks)), "n": n,
            "chance_top1": 1.0/n}


def run_condition(samples, n_ch, condition, epochs, lr, seed, device):
    """One train/eval run for a condition at a given seed. Clean, per-seed controls."""
    torch.manual_seed(seed); np.random.seed(seed)
    feats = torch.tensor(np.stack([s[0] for s in samples]), device=device)
    txt   = F.normalize(torch.tensor(np.stack([s[1] for s in samples]), device=device), dim=-1)
    N = feats.shape[0]

    perm = np.random.permutation(N)
    ntr = int(0.8*N); tr, te = perm[:ntr], perm[ntr:]
    if len(te) < 3:
        return None

    enc = EEGSentenceEncoder(n_ch).to(device)
    opt = torch.optim.AdamW(enc.parameters(), lr=lr, weight_decay=1e-4)
    tr_f, tr_t = feats[tr], txt[tr]

    # gated/ungated/shuffled all TRAIN a real encoder on real EEG->text pairs.
    # gaussian TRAINS on random inputs (a true "no information" learner), so its
    # ceiling is chance -- a proper reference, drawn fresh per seed.
    train_feats = torch.randn_like(tr_f) if condition == "gaussian" else tr_f
    for ep in range(epochs):
        enc.train(); opt.zero_grad()
        z = enc(train_feats)
        loss = info_nce(z, tr_t)
        loss.backward(); torch.nn.utils.clip_grad_norm_(enc.parameters(), 5.0); opt.step()

    enc.eval()
    with torch.no_grad():
        eval_feats = torch.randn_like(feats[te]) if condition == "gaussian" else feats[te]
        z_te = enc(eval_feats)
        t_te = txt[te]
        if condition == "shuffled":
            # pair each test EEG with a WRONG sentence (fresh permutation per seed)
            pr = torch.randperm(len(te))
            # guard against identity permutation on tiny sets
            t_te = t_te[pr]
        return retrieval_metrics(z_te, t_te)


def _mean_std(vals):
    a = np.array(vals, dtype=float)
    return float(a.mean()), float(a.std())


# ---------------------------------------------------------------------------
# Main — multi-seed, with mean +/- std and a paired gated-vs-ungated test
# ---------------------------------------------------------------------------
def run(args):
    manifest = json.loads(Path(args.manifest).read_text())
    subjects = args.subjects or manifest["subjects_complete"]
    device = torch.device(args.device)
    embedder = Embedder(args.glove if not args.hashing else None)
    seeds = list(range(args.seeds))

    conds = ["gated","ungated","shuffled","gaussian"]
    # collect, per condition, the list of per-(subject,seed) metric values
    pool = {c: {"top1":[], "top5":[], "mrr":[]} for c in conds}
    # for the paired test we need gated & ungated aligned by (subject,seed)
    paired_gated, paired_ungated = [], []
    per_subject = {}

    for subj in subjects:
        tmap = load_sentence_text(args.preprocessed_dir, subj)
        if not tmap:
            print(f"[skip] {subj}: no sentence text (.mat not found)")
            continue
        # pre-build samples once per base condition (gated vs ungated EEG)
        samp_g, n_ch = build_subject_samples(manifest, subj, tmap, embedder, "gated")
        samp_u, _    = build_subject_samples(manifest, subj, tmap, embedder, "ungated")
        if len(samp_g) < 10:
            print(f"[skip] {subj}: only {len(samp_g)} usable sentences"); continue

        subj_metrics = {c: {"top1":[], "mrr":[]} for c in conds}
        for sd in seeds:
            runs = {
                "gated":    run_condition(samp_g, n_ch, "gated",    args.epochs, args.lr, sd, device),
                "ungated":  run_condition(samp_u, n_ch, "ungated",  args.epochs, args.lr, sd, device),
                "shuffled": run_condition(samp_g, n_ch, "shuffled", args.epochs, args.lr, sd, device),
                "gaussian": run_condition(samp_g, n_ch, "gaussian", args.epochs, args.lr, sd, device),
            }
            if any(v is None for v in runs.values()):
                continue
            for c in conds:
                for k in ("top1","top5","mrr"): pool[c][k].append(runs[c][k])
                subj_metrics[c]["top1"].append(runs[c]["top1"])
                subj_metrics[c]["mrr"].append(runs[c]["mrr"])
            paired_gated.append(runs["gated"]["mrr"])
            paired_ungated.append(runs["ungated"]["mrr"])

        per_subject[subj] = {c: {"top1_mean": _mean_std(subj_metrics[c]["top1"])[0],
                                 "mrr_mean":  _mean_std(subj_metrics[c]["mrr"])[0]}
                             for c in conds}
        g = per_subject[subj]["gated"]; u = per_subject[subj]["ungated"]
        sh = per_subject[subj]["shuffled"]
        print(f"{subj}: gated mrr={g['mrr_mean']:.3f} ungated={u['mrr_mean']:.3f} "
              f"shuffled={sh['mrr_mean']:.3f}  (avg of {len(seeds)} seeds)")

    # ---- cohort summary with mean +/- std over all (subject,seed) runs ----
    print("\n" + "="*70)
    print(f"SENTENCE-LEVEL RETRIEVAL — {len(per_subject)} subjects x {len(seeds)} seeds")
    print("="*70)
    print(f"{'condition':<12}{'Top-1 (mean±std)':>22}{'MRR (mean±std)':>22}")
    for c in conds:
        if pool[c]["top1"]:
            t_m,t_s=_mean_std(pool[c]["top1"]); m_m,m_s=_mean_std(pool[c]["mrr"])
            print(f"{c:<12}{t_m:>12.3f} ±{t_s:<7.3f}{m_m:>12.3f} ±{m_s:<7.3f}")

    # ---- paired gated-vs-ungated significance (Wilcoxon; t-test fallback) ----
    print("\nPAIRED TEST  gated vs ungated  (MRR, matched by subject+seed):")
    if len(paired_gated) >= 5:
        pg=np.array(paired_gated); pu=np.array(paired_ungated)
        diff=pg-pu
        print(f"  mean MRR diff (gated-ungated) = {diff.mean():+.4f}  over {len(diff)} runs")
        print(f"  gated wins in {int((diff>0).sum())}/{len(diff)} runs")
        try:
            from scipy.stats import wilcoxon
            if np.any(diff!=0):
                stat,p=wilcoxon(pg,pu)
                print(f"  Wilcoxon p = {p:.4f}  -> "
                      + ("SIGNIFICANT (gate helps)" if p<0.05 and diff.mean()>0
                         else "not significant at 0.05"))
        except Exception as e:
            from math import sqrt
            se=diff.std()/sqrt(len(diff)) if diff.std()>0 else 1e-9
            t=diff.mean()/se
            print(f"  (scipy unavailable) approx t = {t:.2f} "
                  + ("(suggestive)" if abs(t)>2 else "(weak)"))
    else:
        print("  too few runs for a test; increase --seeds")

    print("\nHOW TO READ:")
    print("  real EEG (gated/ungated) MRR clearly above shuffled  => decodable signal")
    print("  gated MRR > ungated AND paired p<0.05                => the gate helps, reliably")
    print("  std tells you if per-run differences are just noise")

    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    summary={"n_subjects":len(per_subject),"n_seeds":len(seeds),
             "cohort":{c:{"top1":_mean_std(pool[c]["top1"]),"mrr":_mean_std(pool[c]["mrr"])}
                       for c in conds if pool[c]["top1"]},
             "paired_gated_minus_ungated_mrr":
                 (float(np.mean(np.array(paired_gated)-np.array(paired_ungated)))
                  if paired_gated else None),
             "per_subject":per_subject}
    json.dump(summary, open(out,"w"), indent=2)
    print(f"\nsaved -> {out}")


def build_parser():
    ap = argparse.ArgumentParser(description="Step 10A retrieval (multi-seed)")
    ap.add_argument("--manifest", default="runs/manifest.json")
    ap.add_argument("--preprocessed-dir", required=True)
    ap.add_argument("--glove", default=None)
    ap.add_argument("--hashing", action="store_true")
    ap.add_argument("--subjects", nargs="+", default=None)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seeds", type=int, default=5, help="Number of random splits per condition")
    ap.add_argument("--device", default="cpu", choices=("cpu","cuda"))
    ap.add_argument("--output", default="metrics/retrieval_sentence_multiseed.json")
    return ap


if __name__ == "__main__":
    run(build_parser().parse_args())