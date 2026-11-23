#!/usr/bin/env python3
"""
step10b_decode_aware.py — THE decode-aware loop (contrastive, sentence-level).

This is the piece that makes the contribution genuinely "decode-aware": the Step-9
gate is now TRAINABLE and sits INSIDE a loop where a contrastive EEG<->text loss
flows BACK INTO the gate. The gate therefore learns which components help the text
objective -- "text usefulness" emerges from this gradient (it is NOT a precomputed
score). No BART, no teacher forcing, so no language-model crutch can hide the EEG.

THE LOOP (per sentence)
  sources S  ->  gate g=sigmoid(f(phi))  ->  X_hat = A (g . S)  [gate is trainable]
             ->  EEG encoder -> z_eeg (in GloVe space)
             ->  contrastive loss vs the sentence's GloVe vector (pull true, push others)
             ->  backprop updates ENCODER *and* GATE together.

WHAT MAKES IT DECODE-AWARE (vs Step 9)
  Step 9 trained the gate on artifact-prior + synthetic-SNR only (decode loss = 0).
  Here the ONLY loss is the contrastive decode signal, and it updates the gate.
  So for the first time the gate is optimised for decoding, not just cleanliness.

CONDITIONS (each trains its own gate+encoder, for a fair comparison)
  gated_learn : gate trainable from the decode signal   <- the real contribution
  gated_frozen: gate fixed at Step-9 values (artifact-only)  <- ablation: does decode-training help?
  ungated     : no gate (g=1)                            <- baseline
  shuffled    : gated_learn EEG vs WRONG text            <- control (no signal => ~this)
  gaussian    : random EEG in                            <- control (chance)

METRIC: sentence retrieval Top-1 / MRR, multi-seed, with paired significance for
        the comparisons that matter (gated_learn vs ungated; gated_learn vs gated_frozen).

NOTE (expectations): at SENTENCE level the gate benefit may be small (artifacts
average out). The purpose of this stage is to prove THE LOOP WORKS and real EEG
beats controls. The gate's decoding benefit is chased at WORD level next.

USAGE
  python step10b_decode_aware.py --preprocessed-dir dataset-NR-prepr \\
      --glove glove.6B.300d.txt --seeds 5 --epochs 80
"""

from __future__ import annotations
import argparse, json, re
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

N_COMPONENTS = 98
N_FEATURES = 35
EMB_DIM = 300


# ---------------------------------------------------------------------------
# Loaders (manifest-driven, channel-count auto)
# ---------------------------------------------------------------------------
def load_manifest(p): return json.loads(Path(p).read_text())

def load_sentence_text(preprocessed_dir, subject):
    import h5py
    cands = [Path(preprocessed_dir)/f"results{subject}_NR.mat",
             Path(preprocessed_dir)/subject/f"results{subject}_NR.mat"]
    path = next((c for c in cands if c.exists()), None)
    if path is None: return None
    out = {}
    with h5py.File(path, "r") as f:
        content = f["sentenceData/content"]
        for i in range(content.shape[0]):
            vals = np.asarray(f[content[i,0]][()]).reshape(-1)
            txt = "".join(chr(int(v)) for v in vals if int(v)!=0).strip()
            if txt: out[i] = txt
    return out


class Embedder:
    def __init__(self, glove_path, dim=EMB_DIM):
        self.dim = dim; self.vecs = {}
        with open(glove_path, encoding="utf-8") as f:
            for line in f:
                p = line.rstrip().split(" ")
                if len(p) == dim+1: self.vecs[p[0]] = np.asarray(p[1:], dtype=np.float32)
        print(f"  GloVe: {len(self.vecs)} words")
    def sentence(self, text):
        toks = re.findall(r"[a-zA-Z']+", text.lower())
        vs = [self.vecs[t] for t in toks if t in self.vecs]
        if not vs: return None
        v = np.mean(vs, axis=0); return v/(np.linalg.norm(v)+1e-8)


# ---------------------------------------------------------------------------
# Reused-in-spirit components (self-contained here to avoid import coupling)
# ---------------------------------------------------------------------------
class FeatureGate(nn.Module):
    """g_k = sigmoid(f(phi_k)); index-free; init to Step-9 gate mean."""
    def __init__(self, n_features, initial_gates, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, 1))
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.zeros_(m.weight); nn.init.zeros_(m.bias)
        g = np.clip(np.asarray(initial_gates, np.float32), 1e-4, 1-1e-4)
        self.net[-1].bias.data[:] = float(np.mean(np.log(g/(1-g))))
    def forward(self, feats):            # feats: (98,35) -> (98,)
        return torch.sigmoid(self.net(feats).squeeze(-1))


class EEGEncoderToEmb(nn.Module):
    """Sensor EEG (n_ch x T) -> summary features -> GloVe-space vector."""
    def __init__(self, n_ch, emb_dim=EMB_DIM, hidden=512):
        super().__init__()
        self.in_dim = n_ch*7
        self.net = nn.Sequential(
            nn.Linear(self.in_dim, hidden), nn.LayerNorm(hidden), nn.GELU(),
            nn.Dropout(0.3), nn.Linear(hidden, hidden), nn.GELU(),
            nn.Dropout(0.3), nn.Linear(hidden, emb_dim))
    @staticmethod
    def summarise(x, fs=250.0):          # x: (n_ch, T) -> (n_ch*7,)
        mean = x.mean(1); std = x.std(1)+1e-6
        X = torch.fft.rfft(x, dim=1); mag = X.real**2 + X.imag**2
        fr = torch.fft.rfftfreq(x.shape[1], d=1/fs).to(x.device)
        bp = []
        for lo,hi in [(0.5,4),(4,8),(8,13),(13,30),(30,100)]:
            m = (fr>=lo)&(fr<hi)
            p = mag[:,m].mean(1) if m.any() else torch.zeros(x.shape[0],device=x.device)
            bp.append(torch.log(p+1e-8))
        return torch.cat([mean,std]+bp, 0)
    def forward(self, feat): return F.normalize(self.net(feat), dim=-1)


def gated_sensor(sources, gates, A):     # X_hat = A (g . S)
    return A @ (gates[:, None] * sources)


# ---------------------------------------------------------------------------
# Build per-subject sample tensors (sources kept as tensors so the gate can
# operate on them INSIDE the loop; features needed to compute the gate)
# ---------------------------------------------------------------------------
def build_samples(manifest, subject, tmap, embedder, device):
    ent = manifest["subjects"][subject]
    A = torch.tensor(np.load(ent["mixing_sensor"]).astype(np.float32), device=device)
    feats = torch.tensor(np.load(ent["features"]).astype(np.float32), device=device)  # (98,35)
    gate0 = np.load(ent["gate_values"]).astype(np.float32)
    src_dir = Path(ent["sources_dir"])
    samples = []
    for p in sorted(src_dir.glob("sentence_*_sources.npy")):
        idx = int(p.name.split("_")[1])
        if idx not in tmap: continue
        emb = embedder.sentence(tmap[idx])
        if emb is None: continue
        S = np.load(p).astype(np.float32)
        samples.append((torch.tensor(S, device=device),
                        torch.tensor(emb, device=device)))
    return samples, A, feats, gate0, A.shape[0]


def info_nce(z_eeg, z_txt, temp=0.07):
    logits = z_eeg @ z_txt.t() / temp
    labels = torch.arange(z_eeg.shape[0], device=z_eeg.device)
    return 0.5*(F.cross_entropy(logits, labels)+F.cross_entropy(logits.t(), labels))


def retrieval(z_eeg, z_txt):
    sims = z_eeg @ z_txt.t(); n = sims.shape[0]; ranks=[]
    for i in range(n):
        order = torch.argsort(sims[i], descending=True)
        ranks.append((order==i).nonzero(as_tuple=True)[0].item()+1)
    r = np.array(ranks)
    return {"top1": float((r==1).mean()), "mrr": float(np.mean(1/r)), "n": n}


# ---------------------------------------------------------------------------
# One condition, one seed  -- THE DECODE-AWARE LOOP
# ---------------------------------------------------------------------------
def run_condition(samples, A, feats, gate0, n_ch, condition, epochs, lr, seed, device):
    torch.manual_seed(seed); np.random.seed(seed)
    N = len(samples)
    perm = np.random.permutation(N); ntr=int(0.8*N)
    tr, te = perm[:ntr], perm[ntr:]
    if len(te) < 3: return None

    enc = EEGEncoderToEmb(n_ch).to(device)
    gate = FeatureGate(N_FEATURES, gate0).to(device)

    # Which params train? gated_learn trains gate+encoder; others fix the gate.
    params = list(enc.parameters())
    train_gate = (condition == "gated_learn")
    if train_gate: params += list(gate.parameters())
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)

    fixed_gate_full = torch.ones(N_COMPONENTS, device=device)          # ungated
    fixed_gate_step9 = torch.tensor(gate0, device=device)             # frozen

    for ep in range(epochs):
        enc.train(); gate.train()
        opt.zero_grad()
        # compute the current gate vector for this step
        if condition == "gated_learn":
            g = gate(feats)                       # trainable, decode-driven
        elif condition == "gated_frozen":
            g = fixed_gate_step9
        elif condition == "ungated":
            g = fixed_gate_full
        elif condition == "shuffled":
            g = fixed_gate_step9
        elif condition == "gaussian":
            g = fixed_gate_full

        zs, ts = [], []
        for i in tr:
            S, emb = samples[i]
            if condition == "gaussian":
                X = torch.randn(n_ch, S.shape[1], device=device)
            else:
                X = gated_sensor(S, g, A)          # X_hat = A (g . S)
            z = enc(EEGEncoderToEmb.summarise(X))
            zs.append(z); ts.append(emb)
        z = torch.stack(zs); t = F.normalize(torch.stack(ts), dim=-1)
        loss = info_nce(z, t)
        loss.backward(); torch.nn.utils.clip_grad_norm_(params,5.0); opt.step()

    # ---- eval ----
    enc.eval(); gate.eval()
    with torch.no_grad():
        if condition == "gated_learn": g = gate(feats)
        elif condition in ("gated_frozen","shuffled"): g = fixed_gate_step9
        else: g = fixed_gate_full
        zs, ts = [], []
        for i in te:
            S, emb = samples[i]
            X = torch.randn(n_ch,S.shape[1],device=device) if condition=="gaussian" else gated_sensor(S,g,A)
            zs.append(enc(EEGEncoderToEmb.summarise(X))); ts.append(emb)
        z = torch.stack(zs); t = F.normalize(torch.stack(ts), dim=-1)
        if condition == "shuffled":
            t = t[torch.randperm(len(te))]
        m = retrieval(z, t)
        # report the learned gate's suppression of artifacts, for insight
        if condition == "gated_learn":
            m["gate_mean"] = float(g.mean()); m["gate_min"] = float(g.min())
        return m


CONDS = ["gated_learn","gated_frozen","ungated","shuffled","gaussian"]

def _ms(v): a=np.array(v,float); return (float(a.mean()),float(a.std()))

def run(args):
    man = load_manifest(args.manifest)
    subjects = args.subjects or man["subjects_complete"]
    device = torch.device(args.device)
    emb = Embedder(args.glove)
    seeds = list(range(args.seeds))

    pool = {c:{"top1":[],"mrr":[]} for c in CONDS}
    paired = {"gl_vs_ug":[], "gl_vs_gf":[]}   # (gated_learn - ungated), (gated_learn - gated_frozen)
    gate_means=[]

    for subj in subjects:
        tmap = load_sentence_text(args.preprocessed_dir, subj)
        if not tmap: print(f"[skip] {subj}: no text"); continue
        samples, A, feats, gate0, n_ch = build_samples(man, subj, tmap, emb, device)
        if len(samples) < 10: print(f"[skip] {subj}: {len(samples)} sents"); continue
        for sd in seeds:
            res = {c: run_condition(samples,A,feats,gate0,n_ch,c,args.epochs,args.lr,sd,device)
                   for c in CONDS}
            if any(v is None for v in res.values()): continue
            for c in CONDS:
                pool[c]["top1"].append(res[c]["top1"]); pool[c]["mrr"].append(res[c]["mrr"])
            paired["gl_vs_ug"].append(res["gated_learn"]["mrr"]-res["ungated"]["mrr"])
            paired["gl_vs_gf"].append(res["gated_learn"]["mrr"]-res["gated_frozen"]["mrr"])
            if "gate_mean" in res["gated_learn"]: gate_means.append(res["gated_learn"]["gate_mean"])
        print(f"{subj}: done ({len(seeds)} seeds)")

    print("\n"+"="*70)
    print(f"DECODE-AWARE (contrastive, sentence-level) — {len(subjects)} subj x {len(seeds)} seeds")
    print("="*70)
    print(f"{'condition':<14}{'Top-1 (mean±std)':>22}{'MRR (mean±std)':>22}")
    for c in CONDS:
        if pool[c]["top1"]:
            tm,ts=_ms(pool[c]["top1"]); mm,mstd=_ms(pool[c]["mrr"])
            print(f"{c:<14}{tm:>12.3f} ±{ts:<7.3f}{mm:>12.3f} ±{mstd:<7.3f}")

    def paired_report(name, diffs):
        if len(diffs) < 5: print(f"  {name}: too few runs"); return
        d=np.array(diffs); wins=int((d>0).sum())
        print(f"  {name}: mean MRR diff={d.mean():+.4f}  wins {wins}/{len(d)}", end="")
        try:
            from scipy.stats import wilcoxon
            if np.any(d!=0):
                _,p=wilcoxon(d); print(f"  Wilcoxon p={p:.4f} "
                    +("** SIGNIFICANT" if p<0.05 and d.mean()>0 else "(n.s.)"))
            else: print("  (all zero)")
        except Exception:
            print("  (install scipy for p-value)")
    print("\nPAIRED TESTS (MRR):")
    paired_report("gated_learn vs ungated ", paired["gl_vs_ug"])
    paired_report("gated_learn vs frozen  ", paired["gl_vs_gf"])
    if gate_means:
        print(f"\nlearned-gate mean over runs: {np.mean(gate_means):.3f} "
              "(started ~Step-9 mean; movement shows the decode signal reshaped it)")

    print("\nHOW TO READ:")
    print("  real EEG (any gated/ungated) MRR >> shuffled/gaussian => loop works, signal present")
    print("  gated_learn > ungated (sig)  => the gate helps decoding")
    print("  gated_learn > gated_frozen   => TRAINING the gate on the decode signal helps")
    print("     (this is the core 'decode-aware' evidence; may be small at sentence level)")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    json.dump({"cohort":{c:{"top1":_ms(pool[c]["top1"]),"mrr":_ms(pool[c]["mrr"])}
                         for c in CONDS if pool[c]["top1"]},
               "paired_gl_vs_ungated": _ms(paired["gl_vs_ug"]) if paired["gl_vs_ug"] else None,
               "paired_gl_vs_frozen": _ms(paired["gl_vs_gf"]) if paired["gl_vs_gf"] else None},
              open(args.output,"w"), indent=2)
    print(f"\nsaved -> {args.output}")


def build_parser():
    ap = argparse.ArgumentParser(description="Decode-aware contrastive loop (sentence-level)")
    ap.add_argument("--manifest", default="runs/manifest.json")
    ap.add_argument("--preprocessed-dir", required=True)
    ap.add_argument("--glove", required=True)
    ap.add_argument("--subjects", nargs="+", default=None)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--device", default="cpu", choices=("cpu","cuda"))
    ap.add_argument("--output", default="metrics/decode_aware_sentence.json")
    return ap

if __name__ == "__main__":
    run(build_parser().parse_args())
