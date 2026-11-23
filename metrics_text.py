#!/usr/bin/env python3
"""
metrics_text.py — BLEU-N, ROUGE-1, WER for EEG-to-Text evaluation.
Standalone (no nltk/rouge_score dependency) so results are reproducible anywhere.
Definitions match the standard ones used in the ZuCo EEG-to-Text literature.
"""

from collections import Counter
import numpy as np


def _ngrams(toks, n):
    """All n-grams of a token list.
    In : tokens, n;  Out: Counter of n-gram tuples (empty if len(toks) < n)."""
    return Counter(tuple(toks[i:i + n]) for i in range(len(toks) - n + 1))


def corpus_bleu(hyps, refs, max_n=4):
    """Corpus BLEU-1..max_n with brevity penalty (Papineni et al.).
    In : hyps = list of token lists, refs = list of token lists (one ref each)
    Out: list [BLEU-1, ..., BLEU-max_n] as percentages. Uses cumulative geometric
         mean of modified n-gram precisions, like standard corpus BLEU."""
    out = []
    hyp_len = sum(len(h) for h in hyps)
    ref_len = sum(len(r) for r in refs)
    bp = 1.0 if hyp_len > ref_len else np.exp(1 - ref_len / max(hyp_len, 1))
    precs = []
    for n in range(1, max_n + 1):
        num = den = 0
        for h, r in zip(hyps, refs):
            hc, rc = _ngrams(h, n), _ngrams(r, n)
            num += sum(min(c, rc[g]) for g, c in hc.items())   # clipped matches
            den += max(sum(hc.values()), 0)
        precs.append(num / den if den else 0.0)
        valid = [p for p in precs if p > 0]
        if len(valid) < len(precs):            # a zero precision zeroes the score
            out.append(0.0)
        else:
            out.append(100.0 * bp * float(np.exp(np.mean(np.log(precs)))))
    return out


def rouge1(hyps, refs):
    """Corpus ROUGE-1 precision/recall/F1 on unigram overlap.
    In : hyps, refs as token lists;  Out: (P, R, F) as percentages."""
    m = hp = hr = 0
    for h, r in zip(hyps, refs):
        hc, rc = Counter(h), Counter(r)
        m += sum(min(c, rc[g]) for g, c in hc.items())
        hp += len(h); hr += len(r)
    P = 100.0 * m / hp if hp else 0.0
    R = 100.0 * m / hr if hr else 0.0
    F = 2 * P * R / (P + R) if (P + R) else 0.0
    return P, R, F


def wer(hyps, refs):
    """Corpus word error rate via Levenshtein distance over tokens.
    In : hyps, refs as token lists;  Out: WER as a percentage (lower is better)."""
    tot_d = tot_n = 0
    for h, r in zip(hyps, refs):
        d = np.zeros((len(r) + 1, len(h) + 1), int)
        d[:, 0] = np.arange(len(r) + 1); d[0, :] = np.arange(len(h) + 1)
        for i in range(1, len(r) + 1):
            for j in range(1, len(h) + 1):
                cost = 0 if r[i - 1] == h[j - 1] else 1
                d[i, j] = min(d[i - 1, j] + 1, d[i, j - 1] + 1, d[i - 1, j - 1] + cost)
        tot_d += d[len(r), len(h)]; tot_n += len(r)
    return 100.0 * tot_d / tot_n if tot_n else 0.0


def all_metrics(hyp_strs, ref_strs):
    """Compute the full EEG-to-Text metric set from raw strings.
    In : list of hypothesis strings, list of reference strings
    Out: dict with bleu1..bleu4, rouge1_p/r/f, wer — the columns reported in the
         ZuCo EEG-to-Text tables."""
    hyps = [s.lower().split() for s in hyp_strs]
    refs = [s.lower().split() for s in ref_strs]
    b = corpus_bleu(hyps, refs, 4)
    P, R, F = rouge1(hyps, refs)
    return {"bleu1": b[0], "bleu2": b[1], "bleu3": b[2], "bleu4": b[3],
            "rouge1_p": P, "rouge1_r": R, "rouge1_f": F, "wer": wer(hyps, refs)}