# MindInk — Phase 1 starter skeleton (Steps 1–4 + feature stub)

This covers the plumbing so you can spend your time on the actual contribution
(the decode-aware gating network) rather than on data loading.

## What's here

| File | Step | Purpose |
|------|------|---------|
| `config.py` | — | All paths and signal constants. Edit paths first. |
| `utils_h5.py` | — | HDF5 (MATLAB v7.3) dereferencing + string decoding helpers. |
| `step1_inspect.py` | 1 | Print the `.mat` structure and verify you have **continuous raw EEG**, not band-power features. |
| `step2_load.py` | 2 | Load one subject → continuous sentence EEG + per-word raw segments. |
| `step3_preprocess.py` | 3 | Montage, bandpass, notch, avg-ref, resample (MNE). |
| `step4_ica.py` | 4 | Fit **one** ICA per subject, optional ICLabel, save. |
| `features.py` | 6 (stub) | Per-component feature vector `phi_k` the gate will consume. |
| `run_pipeline.py` | 2–4+6 | End-to-end for one subject. |

## Run order

```bash
pip install -r requirements.txt

# 1. LOOK at one file first — do not skip this.
python step1_inspect.py /path/to/one_subject_NR.mat

# 2. If step1 shows continuous raw EEG, edit config.py paths, then:
python run_pipeline.py /path/to/one_subject_NR.mat ZAB
```

## The three things you MUST verify before trusting the output

1. **Field names.** `step2_load.py` assumes `sentenceData / content / rawData / word /
   rawEEG`. Confirm against your `step1_inspect.py` dump; edit the `*_FIELD` constants
   if they differ.
2. **Units.** MNE expects volts. ZuCo raw is usually microvolts. If step1 shows
   amplitudes in the 10–100 range, uncomment the `* 1e-6` line in
   `step3_preprocess.build_continuous_raw`.
3. **One ICA per subject, identical settings across subjects.** Do not refit per word.
   If you fix `N_ICA_COMPONENTS`, keep it the same for every subject or the
   per-component features stop being comparable — which breaks subject-invariance.

## What this skeleton deliberately does NOT do yet

- No gating network, no subject-invariant encoder, no decoder.
- No band-power (840-dim) features are ever loaded as "signal" — that pickle is
  downstream of your entry point and ICA cannot run on it.
- No decoding metric is produced here, so there is nothing to over-claim yet. The
  validity controls (noise input, shuffled EEG, LM-only) attach at the decoder step.

## Next attach points

- **Step 7 (baselines):** use the saved ICA + ICLabel result to build a hard-threshold
  artifact-rejection baseline and a no-denoising passthrough.
- **Step 9 (contribution):** `features.npz` → small net `g_k = sigma(f(phi_k))` →
  reconstruct `X_hat = A · (g ⊙ S)` with the **frozen** mixing matrix `A = ica.get_components()`
  and sources `S = ica.get_sources(raw).get_data()`.
