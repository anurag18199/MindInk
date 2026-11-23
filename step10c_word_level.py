#!/usr/bin/env python3
"""
step10c_word_decode.py — WORD-LEVEL decode-aware experiment (the key test of the gate).

At word level, artifacts do NOT average out (each word's EEG keeps its own blink/noise),
so cleaning should matter -> this is where the gate's decoding benefit can appear.
Same contrastive decode-aware loop as sentence level, fed WORD pairs.

ALIGNMENT (important — verified against real data)
  Your Step-5 stored ONE FOLDER PER FIXATION, in order:
      segments/<subj>/sentence_NNN/words/word_<f>/fixation_000_sources.npy
  (so "word_011" is actually the 12th FIXATION of the sentence, not word #11.)
  ZuCo's labels give, per word, its text AND its fixation index:
      word_labels/<subj>_word_labels.json -> sentences[N].words[W].fixIdx (0-based),
      plus nFix (0 = skipped word, no EEG).
  MAPPING:  ZuCo word  ->  fixIdx  ->  your word_<fixIdx> folder  ->  that word's EEG.
  We use the word with nFix>0, take its FIRST fixation (fixIdx), pair its gated EEG
  with the word-text GloVe vector. Words with nFix=0 are skipped (no EEG).

  SAFETY: we skip a word if fixIdx is out of range for that sentence's folders, and
  report how many pairs were built vs skipped, so mislabelling can't pass silently.

CONDITIONS (per word)
  gated_learn : gate trainable from the decode signal    <- the contribution
  gated_frozen: Step-9 gate fixed                        <- ablation
  ungated     : no gate (g=1)                            <- baseline
  shuffled / gaussian : controls

USAGE
  python step10c_word_decode.py --word-labels word_labels --glove glove.6B.300d.txt --seeds 5
"""
from __future__ import annotations
import argparse, json, re
from pathlib import Path
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

N_COMPONENTS=98; N_FEATURES=35; EMB_DIM=300

# ---------------- embeddings ----------------
class Embedder:
    def __init__(self, glove):
        self.v={}
        with open(glove, encoding="utf-8") as f:
            for line in f:
                p=line.rstrip().split(" ")
                if len(p)==EMB_DIM+1: self.v[p[0]]=np.asarray(p[1:], np.float32)
        print(f"  GloVe: {len(self.v)} words")
    def word(self, w):
        w=re.sub(r"[^a-zA-Z']","",w).lower()        # strip ZuCo punctuation
        if not w: return None
        vec=self.v.get(w)
        return None if vec is None else vec/(np.linalg.norm(vec)+1e-8)

# ---------------- nets ----------------
class FeatureGate(nn.Module):
    def __init__(self, nfeat, init_gates, hidden=64):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(nfeat,hidden),nn.GELU(),
            nn.Linear(hidden,hidden),nn.GELU(),nn.Linear(hidden,1))
        for m in self.net:
            if isinstance(m,nn.Linear): nn.init.zeros_(m.weight); nn.init.zeros_(m.bias)
        g=np.clip(np.asarray(init_gates,np.float32),1e-4,1-1e-4)
        self.net[-1].bias.data[:]=float(np.mean(np.log(g/(1-g))))
    def forward(self,feat): return torch.sigmoid(self.net(feat).squeeze(-1))

class WordEncoder(nn.Module):
    def __init__(self, n_ch, emb=EMB_DIM, hid=256):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(n_ch*7,hid),nn.LayerNorm(hid),nn.GELU(),
            nn.Dropout(0.3),nn.Linear(hid,hid),nn.GELU(),nn.Dropout(0.3),nn.Linear(hid,emb))
    @staticmethod
    def summarise(x, fs=250.0):
        mean=x.mean(1); std=x.std(1)+1e-6
        X=torch.fft.rfft(x,dim=1); mag=X.real**2+X.imag**2
        fr=torch.fft.rfftfreq(x.shape[1],d=1/fs).to(x.device); bp=[]
        for lo,hi in [(0.5,4),(4,8),(8,13),(13,30),(30,100)]:
            m=(fr>=lo)&(fr<hi)
            p=mag[:,m].mean(1) if m.any() else torch.zeros(x.shape[0],device=x.device)
            bp.append(torch.log(p+1e-8))
        return torch.cat([mean,std]+bp,0)
    def forward(self,feat): return F.normalize(self.net(feat),dim=-1)

def gated(sources,g,A): return A@(g[:,None]*sources)

# ---------------- build word pairs ----------------
def load_fix_sources(word_dir):
    """Return the sources for a fixation folder (prefer the per-folder fixation file)."""
    p=word_dir/"fixation_000_sources.npy"
    if p.exists(): return np.load(p).astype(np.float32)
    # fallback: a flat word_<n>_sources.npy next to the folder
    alt=word_dir.parent/(word_dir.name+"_sources.npy")
    if alt.exists(): return np.load(alt).astype(np.float32)
    return None

def build_word_samples(manifest, subject, labels, embedder, device, min_samps=16):
    ent=manifest["subjects"][subject]
    A=torch.tensor(np.load(ent["mixing_sensor"]).astype(np.float32),device=device)
    feats=torch.tensor(np.load(ent["features"]).astype(np.float32),device=device)
    gate0=np.load(ent["gate_values"]).astype(np.float32)
    seg_root=Path(ent["segments_dir"])
    lab_by_idx={s["index"]:s for s in labels["sentences"] if s}

    samples=[]; n_sent=0; miss_text=0; miss_fix=0; oor=0
    for sent_dir in sorted(seg_root.glob("sentence_*")):
        idx=int(sent_dir.name.split("_")[1])
        if idx not in lab_by_idx: continue
        words_meta=lab_by_idx[idx]["words"]
        # fixation folders of this sentence, ordered
        fix_dirs=sorted([d for d in (sent_dir/"words").glob("word_*") if d.is_dir()],
                        key=lambda d:int(d.name.split("_")[1]))
        n_fix=len(fix_dirs)
        if n_fix==0: continue
        n_sent+=1
        for w in words_meta:
            if (w.get("nFix") or 0)<=0: continue          # skipped word, no EEG
            fi=w.get("fixIdx")
            if fi is None: miss_fix+=1; continue
            fi=int(round(fi))
            if fi<0 or fi>=n_fix: oor+=1; continue         # index out of range -> skip safely
            emb=embedder.word(w["text"])
            if emb is None: miss_text+=1; continue
            S=load_fix_sources(fix_dirs[fi])
            if S is None or S.ndim!=2 or S.shape[0]!=N_COMPONENTS or S.shape[1]<min_samps:
                continue
            samples.append((torch.tensor(S,device=device),
                            torch.tensor(emb,device=device)))
    info=dict(n_sentences=n_sent, out_of_range=oor, no_glove=miss_text, no_fixidx=miss_fix)
    return samples,A,feats,gate0,A.shape[0],info

def info_nce(z,t,temp=0.07):
    logits=z@t.t()/temp; lab=torch.arange(z.shape[0],device=z.device)
    return 0.5*(F.cross_entropy(logits,lab)+F.cross_entropy(logits.t(),lab))

def retrieval(z,t):
    sims=z@t.t(); n=sims.shape[0]; r=[]
    for i in range(n):
        o=torch.argsort(sims[i],descending=True)
        r.append((o==i).nonzero(as_tuple=True)[0].item()+1)
    r=np.array(r); return {"top1":float((r==1).mean()),"mrr":float(np.mean(1/r)),"n":n}

def run_condition(samples,A,feats,gate0,n_ch,cond,epochs,lr,seed,device,batch=256):
    torch.manual_seed(seed); np.random.seed(seed)
    N=len(samples); perm=np.random.permutation(N); ntr=int(0.8*N)
    tr,te=perm[:ntr],perm[ntr:]
    if len(te)<5: return None
    enc=WordEncoder(n_ch).to(device); gate=FeatureGate(N_FEATURES,gate0).to(device)
    params=list(enc.parameters())
    if cond=="gated_learn": params+=list(gate.parameters())
    opt=torch.optim.AdamW(params,lr=lr,weight_decay=1e-4)
    full=torch.ones(N_COMPONENTS,device=device); step9=torch.tensor(gate0,device=device)
    def gvec():
        if cond=="gated_learn": return gate(feats)
        if cond in ("gated_frozen","shuffled"): return step9
        return full
    for ep in range(epochs):
        enc.train(); gate.train()
        bb=np.random.choice(tr,size=min(batch,len(tr)),replace=False)
        g=gvec(); zs,ts=[],[]
        for i in bb:
            S,emb=samples[i]
            X=torch.randn(n_ch,S.shape[1],device=device) if cond=="gaussian" else gated(S,g,A)
            zs.append(enc(WordEncoder.summarise(X))); ts.append(emb)
        z=torch.stack(zs); t=F.normalize(torch.stack(ts),dim=-1)
        loss=info_nce(z,t); opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(params,5.0); opt.step()
    enc.eval(); gate.eval()
    with torch.no_grad():
        g=gvec(); zs,ts=[],[]
        for i in te:
            S,emb=samples[i]
            X=torch.randn(n_ch,S.shape[1],device=device) if cond=="gaussian" else gated(S,g,A)
            zs.append(enc(WordEncoder.summarise(X))); ts.append(emb)
        z=torch.stack(zs); t=F.normalize(torch.stack(ts),dim=-1)
        if cond=="shuffled": t=t[torch.randperm(len(te))]
        return retrieval(z,t)

CONDS=["gated_learn","gated_frozen","ungated","shuffled","gaussian"]
def _ms(v): a=np.array(v,float); return (float(a.mean()),float(a.std()))

def run(args):
    man=json.loads(Path(args.manifest).read_text())
    subs=args.subjects or man["subjects_complete"]
    device=torch.device(args.device); emb=Embedder(args.glove); seeds=list(range(args.seeds))
    pool={c:{"top1":[],"mrr":[]} for c in CONDS}
    paired={"gl_ug":[],"gl_gf":[]}
    for subj in subs:
        lp=Path(args.word_labels)/f"{subj}_word_labels.json"
        if not lp.exists(): print(f"[skip] {subj}: no {lp}"); continue
        labels=json.loads(lp.read_text())
        samples,A,feats,gate0,n_ch,info=build_word_samples(man,subj,labels,emb,device)
        if len(samples)<50:
            print(f"[skip] {subj}: only {len(samples)} word pairs ({info})"); continue
        print(f"{subj}: {len(samples)} word pairs  "
              f"[sent {info['n_sentences']}, out-of-range {info['out_of_range']}, "
              f"no-glove {info['no_glove']}]")
        for sd in seeds:
            res={c:run_condition(samples,A,feats,gate0,n_ch,c,args.epochs,args.lr,sd,device) for c in CONDS}
            if any(v is None for v in res.values()): continue
            for c in CONDS:
                pool[c]["top1"].append(res[c]["top1"]); pool[c]["mrr"].append(res[c]["mrr"])
            paired["gl_ug"].append(res["gated_learn"]["mrr"]-res["ungated"]["mrr"])
            paired["gl_gf"].append(res["gated_learn"]["mrr"]-res["gated_frozen"]["mrr"])

    print("\n"+"="*72)
    print(f"WORD-LEVEL decode-aware — {len(subs)} subjects x {len(seeds)} seeds")
    print("="*72)
    print(f"{'condition':<14}{'Top-1 (mean±std)':>22}{'MRR (mean±std)':>22}")
    for c in CONDS:
        if pool[c]["top1"]:
            tm,ts=_ms(pool[c]["top1"]); mm,msd=_ms(pool[c]["mrr"])
            print(f"{c:<14}{tm:>12.3f} ±{ts:<7.3f}{mm:>12.3f} ±{msd:<7.3f}")
    def report(name,d):
        if len(d)<5: print(f"  {name}: too few runs"); return
        d=np.array(d)
        print(f"  {name}: mean MRR diff={d.mean():+.4f}  wins {int((d>0).sum())}/{len(d)}",end="")
        try:
            from scipy.stats import wilcoxon
            if np.any(d!=0):
                _,p=wilcoxon(d)
                print(f"  p={p:.4f} "+("** SIGNIFICANT (gate helps)" if p<0.05 and d.mean()>0 else "(n.s.)"))
            else: print("  (all zero)")
        except Exception: print("  (install scipy for p-value)")
    print("\nPAIRED TESTS (MRR):")
    report("gated_learn vs ungated",paired["gl_ug"])
    report("gated_learn vs frozen ",paired["gl_gf"])
    print("\nHOW TO READ:")
    print("  real EEG >> shuffled/gaussian  => words carry decodable signal")
    print("  gated_learn > ungated (p<0.05) => the GATE HELPS word decoding  <-- the goal")
    print("  gated_learn > gated_frozen     => training the gate on decoding helps")
    Path(args.output).parent.mkdir(parents=True,exist_ok=True)
    json.dump({"cohort":{c:{"top1":_ms(pool[c]["top1"]),"mrr":_ms(pool[c]["mrr"])} for c in CONDS if pool[c]["top1"]},
               "paired_gl_vs_ungated":_ms(paired["gl_ug"]) if paired["gl_ug"] else None,
               "paired_gl_vs_frozen":_ms(paired["gl_gf"]) if paired["gl_gf"] else None},
              open(args.output,"w"),indent=2)
    print(f"\nsaved -> {args.output}")

def build_parser():
    ap=argparse.ArgumentParser(description="Word-level decode-aware experiment")
    ap.add_argument("--manifest",default="runs/manifest.json")
    ap.add_argument("--word-labels",default="word_labels")
    ap.add_argument("--glove",required=True)
    ap.add_argument("--subjects",nargs="+",default=None)
    ap.add_argument("--epochs",type=int,default=40)
    ap.add_argument("--lr",type=float,default=1e-3)
    ap.add_argument("--seeds",type=int,default=5)
    ap.add_argument("--device",default="cpu",choices=("cpu","cuda"))
    ap.add_argument("--output",default="metrics/decode_aware_word.json")
    return ap

if __name__=="__main__":
    run(build_parser().parse_args())
