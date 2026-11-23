#!/usr/bin/env python3
"""
eeg2text.py — Step 11: EEG-to-Text with BART, under the noise-control protocol
==============================================================================
KAGGLE build. Reads:
    ica/<subj>/sources/sentence_XXX_sources.npy   98 x T sources per sentence
    ica/<subj>/mixing_sensor.npy                  104 x 98 mixing matrix A
    gates/<subj>/gate_values.npy                  98 keep-values (gated condition)
    onsets/<subj>_onsets.json                     from export_onsets.py: onset,
                                                  fix_onsets, word_ks, text, xy_dist
Sentence numbering (yours 0..302 vs ZuCo 0..348) and the matched reference text are
resolved by export_onsets.py, so nothing here depends on word_labels.

Each sentence becomes a sequence of word-level EEG features: for every fixated word
(first fixation per distinct word, reading order) an 800 ms window after that
fixation's onset, rebuilt through A (and the gate if gated), baseline-corrected,
8 band amplitudes x 104 channels = 832 dims (Wang & Ji definition).

Scenarios (each scored WITH and WITHOUT teacher forcing):
    EEG->EEG        real task
    Shuffled->EEG   trained on real EEG paired with the WRONG sentence's text
    EEG->Random     trained on EEG, tested on noise
    Random->EEG     trained on noise, tested on EEG
    Random->Random  memorisation floor
Headline: EEG->EEG minus Random->Random, and EEG->EEG minus Shuffled->EEG.

    !pip -q install transformers
    !python eeg2text.py --gating ungated
    !python eeg2text.py --gating gated
"""

import argparse
import glob
import json
import os
import numpy as np
import torch
import torch.nn as nn
from metrics_text import all_metrics

# ------------------------------------------------------------------ CONFIG ----
SFREQ, WIN, BASE = 250, int(0.800 * 250), int(0.100 * 250)
BANDS = [(4, 6), (6.5, 8), (8.5, 10), (10.5, 13),
         (13.5, 18), (18.5, 30), (30.5, 40), (40, 49.5)]   # Wang & Ji 8 bands
MAX_WORDS, MAX_TOKENS = 58, 58
MAX_XY_DIST = 25.0        # drop sentences whose labeller match was poor (px)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# ------------------------------------------------------------------------------


def walk_find(markers, max_depth=9):
    """Find the first directory under /kaggle/input containing all marker folders.
    In : list of folder names that must all be present
    Out: that directory's path, or None. Depth-independent, so the Kaggle mount
         layout (and re-versioning) never matters."""
    base = "/kaggle/input"
    if os.path.isdir(base):
        for dp, dirs, _ in os.walk(base):
            if dp[len(base):].count(os.sep) > max_depth:
                dirs[:] = []
                continue
            if all(m in dirs for m in markers):
                return dp
    return None


def find_root():
    """Dataset root = the folder holding ica/ and gates/. Out: path.
    Falls back to the current directory for local runs."""
    r = walk_find(["ica", "gates"])
    if r:
        return r
    if os.path.isdir("ica") and os.path.isdir("gates"):
        return "."
    raise FileNotFoundError("Could not find a folder containing ica/ and gates/.")


def find_onsets_dir():
    """Locate the onsets/ folder (a sibling of the root in your dataset).
    Out: path to onsets/. Searches Kaggle input first, then the cwd."""
    base = "/kaggle/input"
    if os.path.isdir(base):
        for dp, dirs, _ in os.walk(base):
            if dp[len(base):].count(os.sep) > 9:
                dirs[:] = []
                continue
            if "onsets" in dirs:
                return os.path.join(dp, "onsets")
    if os.path.isdir("onsets"):
        return "onsets"
    raise FileNotFoundError("No onsets/ folder found. Attach the dataset version "
                            "that contains it.")


def list_subjects(root, onsets_dir):
    """Subjects having BOTH ica/<s>/ and onsets/<s>_onsets.json. Out: sorted list."""
    have_ica = {d for d in os.listdir(os.path.join(root, "ica"))
                if os.path.isdir(os.path.join(root, "ica", d))}
    have_on = {f.split("_onsets.json")[0] for f in os.listdir(onsets_dir)
               if f.endswith("_onsets.json")}
    return sorted(have_ica & have_on)


def band_amplitudes(win):
    """Mean band amplitude per channel for one word window (Wang & Ji features).
    In : win (channels, WIN); Out: (channels*8,) spectral amplitude per band."""
    T = win.shape[1]
    freqs = np.fft.rfftfreq(T, 1.0 / SFREQ)
    amp = np.abs(np.fft.rfft(win, axis=1))
    feats = [amp[:, (freqs >= lo) & (freqs < hi)].mean(axis=1)
             if ((freqs >= lo) & (freqs < hi)).any() else np.zeros(win.shape[0])
             for lo, hi in BANDS]
    return np.stack(feats, axis=1).astype(np.float32).ravel()


def build_subject(root, onsets_dir, subject, gating):
    """Build per-sentence word-feature sequences and reference texts.
    In : dataset root, onsets dir, subject, gating in {'ungated','gated'}
    Out: list of {'x': (n_words, 832), 'text': str}. Skips sentences with no source
         file, no on-word fixations, or a poor labeller match (xy_dist > MAX_XY_DIST).
         Prints how many were skipped and why."""
    A = np.load(os.path.join(root, "ica", subject, "mixing_sensor.npy")).astype(np.float32)
    if A.shape == (98, 104):
        A = A.T
    gate = None
    if gating == "gated":
        gate = np.load(os.path.join(root, "gates", subject,
                                    "gate_values.npy")).astype(np.float32).ravel()
    meta = json.load(open(os.path.join(onsets_dir, f"{subject}_onsets.json")))

    out = []
    skip = {"no_sources": 0, "bad_match": 0, "no_words": 0}
    for sid_str in sorted(meta, key=int):
        e, sid = meta[sid_str], int(sid_str)
        if e["xy_dist"] is not None and e["xy_dist"] > MAX_XY_DIST:
            skip["bad_match"] += 1
            continue
        if not e["word_ks"]:
            skip["no_words"] += 1
            continue
        sp = os.path.join(root, "ica", subject, "sources",
                          f"sentence_{sid:03d}_sources.npy")
        if not os.path.exists(sp):
            skip["no_sources"] += 1
            continue
        S = np.load(sp)
        rows = []
        for k in e["word_ks"]:
            if k >= len(e["fix_onsets"]):
                continue
            rel = e["fix_onsets"][k] - e["onset"]          # onset within the sentence
            s0, s1 = rel - BASE, rel + WIN
            a, b = max(0, s0), min(S.shape[1], s1)
            if b <= a:
                continue
            buf = np.zeros((S.shape[0], BASE + WIN), np.float32)
            buf[:, (a - s0):(b - s0)] = S[:, a:b]
            Sg = buf if gate is None else buf * gate[:, None]
            X = A @ Sg                                       # rebuild sensor signal
            X = X - X[:, :BASE].mean(axis=1, keepdims=True)  # baseline correct
            rows.append(band_amplitudes(X[:, BASE:]))
        if rows:
            out.append({"x": np.stack(rows)[:MAX_WORDS].astype(np.float32),
                        "text": e["text"]})
    print(f"[{subject}] {len(out)} sentences built | skipped: {skip}", flush=True)
    return out


class EEGEncoder(nn.Module):
    """Word-feature sequence -> BART encoder embedding space.
    Forward in : (B,L,832), (B,L) pad mask;  out: (B,L,d_model)."""
    def __init__(self, d_in, d_model, layers=6, heads=8):
        super().__init__()
        enc = nn.TransformerEncoderLayer(d_model=d_in, nhead=heads,
                                         dim_feedforward=2048, batch_first=True)
        self.tf = nn.TransformerEncoder(enc, num_layers=layers)
        self.proj = nn.Linear(d_in, d_model)

    def forward(self, x, pad_mask):
        return self.proj(self.tf(x, src_key_padding_mask=pad_mask))


def collate(batch, tok, d_in, randomize):
    """Pad features, tokenize texts; optionally replace features with noise.
    In : batch, tokenizer, feat dim, randomize; Out: feats, pad_mask, labels."""
    L = max(len(b["x"]) for b in batch)
    feats = np.zeros((len(batch), L, d_in), np.float32)
    pad = np.ones((len(batch), L), bool)
    for i, b in enumerate(batch):
        n = len(b["x"]); feats[i, :n] = b["x"]; pad[i, :n] = False
    if randomize:
        feats = np.random.standard_normal(feats.shape).astype(np.float32)
    enc = tok([b["text"] for b in batch], return_tensors="pt", padding=True,
              truncation=True, max_length=MAX_TOKENS)
    labels = enc["input_ids"].clone()
    labels[labels == tok.pad_token_id] = -100
    return torch.tensor(feats), torch.tensor(pad), labels


def run_scenario(train_data, test_data, tok, bart, d_in, train_rand, test_rand,
                 epochs, bs, lr, seed, shuffle_train=False):
    """Train one scenario, evaluate with and without teacher forcing.
    shuffle_train pairs each training EEG with a DIFFERENT sentence's text.
    Out: (metrics_with_tf, metrics_without_tf)."""
    torch.manual_seed(seed)
    if shuffle_train:
        perm = np.random.default_rng(seed).permutation(len(train_data))
        train_data = [{"x": train_data[i]["x"], "text": train_data[perm[i]]["text"]}
                      for i in range(len(train_data))]
    enc = EEGEncoder(d_in, bart.config.d_model).to(DEVICE).float()
    model = bart.to(DEVICE).float()          # force fp32: match encoder embeds
    opt = torch.optim.AdamW(list(enc.parameters()) + list(model.parameters()), lr=lr)
    for ep in range(epochs):
        enc.train(); model.train()
        order = np.random.permutation(len(train_data))
        for i in range(0, len(order), bs):
            batch = [train_data[j] for j in order[i:i + bs]]
            f, pad, lab = collate(batch, tok, d_in, train_rand)
            f, pad, lab = f.to(DEVICE), pad.to(DEVICE), lab.to(DEVICE)
            loss = model(inputs_embeds=enc(f, pad),
                         attention_mask=(~pad).long(), labels=lab).loss
            opt.zero_grad(); loss.backward(); opt.step()
        print(f"      epoch {ep+1}/{epochs} loss {loss.item():.4f}", flush=True)
    enc.eval(); model.eval()
    hyp_tf, hyp_gen, refs = [], [], []
    with torch.no_grad():
        for i in range(0, len(test_data), bs):
            batch = test_data[i:i + bs]
            f, pad, lab = collate(batch, tok, d_in, test_rand)
            f, pad = f.to(DEVICE), pad.to(DEVICE); am = (~pad).long()
            emb = enc(f, pad)
            logits = model(inputs_embeds=emb, attention_mask=am,
                           labels=lab.to(DEVICE)).logits
            for ids in logits.argmax(-1):
                hyp_tf.append(tok.decode(ids, skip_special_tokens=True))
            gen = model.generate(inputs_embeds=emb, attention_mask=am,
                                 max_length=MAX_TOKENS, num_beams=5,
                                 repetition_penalty=5.0, no_repeat_ngram_size=2)
            for ids in gen:
                hyp_gen.append(tok.decode(ids, skip_special_tokens=True))
            refs += [b["text"] for b in batch]
    return all_metrics(hyp_tf, refs), all_metrics(hyp_gen, refs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gating", choices=["ungated", "gated"], default="ungated")
    ap.add_argument("--subjects", nargs="+", default=None)
    ap.add_argument("--model", default="facebook/bart-large")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seeds", nargs="+", type=int, default=None,
                    help="run these seeds and report mean+/-std per delta")
    ap.add_argument("--scenarios", nargs="+", default=None,
                    help="subset by name, e.g. EEG->EEG Shuffled->EEG Random->Random")
    ap.add_argument("--out", default="/kaggle/working/step11")
    ap.add_argument("--only", nargs="+", default=None,
                    help="run only these scenarios, e.g. Random->EEG Random->Random; "
                         "appends to the existing CSV")
    args = ap.parse_args()

    root, onsets_dir = find_root(), find_onsets_dir()
    subs = args.subjects or list_subjects(root, onsets_dir)
    print(f"root:   {root}\nonsets: {onsets_dir}\nsubjects ({len(subs)}): {subs}\n"
          f"device: {DEVICE} | gating: {args.gating}\n", flush=True)

    data = []
    for s in subs:
        try:
            data += build_subject(root, onsets_dir, s, args.gating)
        except Exception as e:
            print(f"[{s}] skipped: {type(e).__name__}: {e}", flush=True)
    if not data:
        print("No data built."); return
    d_in = data[0]["x"].shape[1]
    rng = np.random.default_rng(args.seed); rng.shuffle(data)
    n_te = max(1, int(0.1 * len(data)))
    test, train = data[:n_te], data[n_te:]
    print(f"\ntotal {len(data)} | train {len(train)} test {len(test)} | feat {d_in}\n",
          flush=True)

    from transformers import BartForConditionalGeneration, BartTokenizer
    import pandas as pd
    tok = BartTokenizer.from_pretrained(args.model)
    # (train_name, eval_name, train_rand, test_rand, shuffle_train)
    scenarios = [("EEG", "EEG", False, False, False),
                 ("Shuffled", "EEG", False, False, True),
                 ("EEG", "Random", False, True, False),
                 ("Random", "EEG", True, False, False),
                 ("Random", "Random", True, True, False)]
    if args.scenarios:
        want = set(args.scenarios)
        scenarios = [s for s in scenarios if f"{s[0]}->{s[1]}" in want]
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, f"eeg2text_{args.gating}.csv")
    seeds = args.seeds if args.seeds else [args.seed]
    all_rows, per_seed = [], []
    for sd in seeds:
        print(f"\n==== SEED {sd} ====", flush=True)
        srows = []
        for tr_name, te_name, tr_rand, te_rand, shuf in scenarios:
            print(f"  scenario {tr_name} -> {te_name} (seed {sd})", flush=True)
            bart = BartForConditionalGeneration.from_pretrained(
                args.model, torch_dtype=torch.float32)
            m_tf, m_gen = run_scenario(train, test, tok, bart, d_in, tr_rand, te_rand,
                                       args.epochs, args.batch, args.lr, sd,
                                       shuffle_train=shuf)
            for mode, m in (("with", m_tf), ("without", m_gen)):
                r = {"gating": args.gating, "seed": sd, "train": tr_name,
                     "eval": te_name, "tf": mode, **m}
                srows.append(r); all_rows.append(r)
            pd.DataFrame(all_rows).to_csv(path, index=False)   # checkpoint each scenario
            print(f"    saved -> {path}", flush=True)

        # per-seed deltas (without-TF, the honest metric)
        sdf = pd.DataFrame(srows)
        def row(tr, te, tf="without"):
            q = sdf[(sdf["train"] == tr) & (sdf["eval"] == te) & (sdf["tf"] == tf)]
            return q.iloc[0] if len(q) else None
        e, n, s = row("EEG", "EEG"), row("Random", "Random"), row("Shuffled", "EEG")
        rec = {"seed": sd}
        if e is not None and n is not None:
            rec["d_noise_bleu1"] = e.bleu1 - n.bleu1
            rec["d_noise_rouge1f"] = e.rouge1_f - n.rouge1_f
        if e is not None and s is not None:
            rec["d_shuf_bleu1"] = e.bleu1 - s.bleu1
            rec["d_shuf_rouge1f"] = e.rouge1_f - s.rouge1_f
        per_seed.append(rec)
        print(f"  seed {sd} deltas: {rec}", flush=True)

    # ---- aggregate across seeds: mean +/- std, and a sign check ----
    pdf = pd.DataFrame(per_seed)
    pdf.to_csv(os.path.join(args.out, f"deltas_{args.gating}.csv"), index=False)
    print("\n" + "=" * 60)
    print(f"GATED delta across {len(seeds)} seeds (without teacher forcing):")
    for col, label in [("d_noise_bleu1",  "EEG-EEG  minus Random-Random  BLEU-1"),
                       ("d_noise_rouge1f","EEG-EEG  minus Random-Random  ROUGE1F"),
                       ("d_shuf_bleu1",   "EEG-EEG  minus Shuffled-EEG   BLEU-1"),
                       ("d_shuf_rouge1f", "EEG-EEG  minus Shuffled-EEG   ROUGE1F")]:
        if col in pdf:
            v = pdf[col].to_numpy()
            pos = int((v > 0).sum())
            print(f"  {label:38s} {v.mean():+6.2f} +/- {v.std():5.2f}   "
                  f"[{pos}/{len(v)} seeds positive]")
    print("=" * 60)
    print("Interpretation: beats NOISE  -> gated EEG distinguishable from noise.")
    print("                beats SHUFFLED -> uses sentence-specific content (the strong claim).")
    print(f"\nper-seed deltas saved -> {os.path.join(args.out, f'deltas_{args.gating}.csv')}")
    print(f"all rows saved        -> {path}")


if __name__ == "__main__":
    main()