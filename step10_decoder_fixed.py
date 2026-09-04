#!/usr/bin/env python3
"""
MindInk / ZuCo Step 10A
Decode-aware EEG-to-text integration for ZuCo 2.0 NR.

Scope:
    ZuCo open-vocabulary English only.
    IIST is intentionally NOT included.

Pipeline:
    Step-6 component features
        -> Step-9 feature-conditioned gates
        -> frozen ICA sources
        -> frozen sensor-space mixing matrix
        -> gated sensor EEG
        -> temporal EEG encoder
        -> BART encoder-decoder
        -> free-running text generation

Training:
    Real EEG only.
    The Step-9 gate is initialized from the bootstrap gate values and is
    allowed to receive gradients from the decoding loss.

Evaluation controls:
    real EEG
    Gaussian-noise EEG
    shuffled/mismatched EEG
    LM-only (zero EEG encoder input)

Important:
    - ICA is never refit.
    - Component index is never used as a feature.
    - The headline evaluation is free-running generation.
    - Teacher-forced loss is used only as an optimization objective.
    - BLEU-1 and token F1 are secondary diagnostic metrics.
    - This is ZuCo-only; subject invariance is Step 11.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Dict, List, Tuple

import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from transformers import (
    BartForConditionalGeneration,
    BartTokenizer,
)


# ============================================================================
# Constants
# ============================================================================

DEFAULT_SUBJECT = "YAG"
ICA_ROOT = "ica"
FEATURES_ROOT = "features"
SEGMENTS_ROOT = "segments"
OUTPUT_ROOT = "step10"

N_CHANNELS = 105
N_COMPONENTS = 98
N_FEATURES = 35
FS = 250.0

BART_MODEL_NAME = "facebook/bart-base"

CONTROL_REAL = "real_eeg"
CONTROL_GAUSSIAN = "gaussian_noise"
CONTROL_SHUFFLED = "shuffled_eeg"
CONTROL_LM_ONLY = "lm_only"

CONTROLS = (
    CONTROL_REAL,
    CONTROL_GAUSSIAN,
    CONTROL_SHUFFLED,
    CONTROL_LM_ONLY,
)


# ============================================================================
# Reproducibility
# ============================================================================

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================================
# JSON helpers
# ============================================================================

def save_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


# ============================================================================
# ZuCo sentence text
# ============================================================================

def decode_content_reference(f: h5py.File, ref) -> str:
    """
    Decode sentenceData/content[i] for this verified ZuCo file.

    content[i] -> HDF5 reference -> uint16 code-point array.
    """
    target = f[ref]
    values = np.asarray(target[()]).reshape(-1)

    chars = []
    for value in values:
        code = int(value)
        if code == 0:
            continue
        chars.append(chr(code))

    return "".join(chars).strip()


def load_zuco_sentences(
    mat_file: Path,
    expected_sentences: int = 349,
) -> Dict[int, str]:

    with h5py.File(mat_file, "r") as f:
        content = f["sentenceData/content"]

        if content.shape[0] != expected_sentences:
            raise RuntimeError(
                f"Expected {expected_sentences} ZuCo sentences, "
                f"got {content.shape}"
            )

        sentences = {}

        for i in range(expected_sentences):
            ref = content[i, 0]
            text = decode_content_reference(f, ref)

            if not text:
                raise RuntimeError(
                    f"Empty sentence text at sentence {i}."
                )

            sentences[i] = text

    return sentences


# ============================================================================
# Step-9 / Step-4 loading
# ============================================================================

def load_step6_features(
    features_root: Path,
    subject: str,
) -> np.ndarray:

    subject_dir = features_root / subject
    feature_path = subject_dir / "component_features.npy"

    if not feature_path.exists():
        raise FileNotFoundError(
            f"Missing Step-6 feature matrix: {feature_path}"
        )

    features = np.load(feature_path).astype(np.float32)

    if features.shape != (N_COMPONENTS, N_FEATURES):
        raise RuntimeError(
            f"Expected Step-6 features "
            f"({N_COMPONENTS}, {N_FEATURES}), "
            f"got {features.shape}"
        )

    if not np.isfinite(features).all():
        raise RuntimeError(
            "Step-6 component features contain NaN/Inf."
        )

    return features


def load_step9_gates(
    gates_root: Path,
    subject: str,
) -> np.ndarray:

    path = gates_root / subject / "gate_values.npy"

    if not path.exists():
        raise FileNotFoundError(
            f"Missing Step-9 gates: {path}"
        )

    gates = np.load(path).astype(np.float32)

    if gates.shape != (N_COMPONENTS,):
        raise RuntimeError(
            f"Expected Step-9 gates ({N_COMPONENTS},), "
            f"got {gates.shape}"
        )

    if not np.isfinite(gates).all():
        raise RuntimeError(
            "Step-9 gates contain NaN/Inf."
        )

    if np.any(gates <= 0.0) or np.any(gates >= 1.0):
        raise RuntimeError(
            "Step-9 gates must be strictly inside (0,1) "
            "for logit initialization."
        )

    return gates


def load_mixing_matrix(
    ica_root: Path,
    subject: str,
) -> np.ndarray:
    """
    Load the frozen sensor-space ICA mixing matrix A.

    Step 4 currently contains:
        mixing_matrix.npy   : (98, 98)
        unmixing_matrix.npy : (98, 105)

    The 98x98 file is an ICA-space representation and must NOT be used as
    the sensor reconstruction matrix.

    For the frozen ICA decomposition:
        S = W @ X
    where W is (98, 105).

    Therefore the sensor-space reconstruction operator used by Step 10 is:
        A = pinv(W)
    with shape (105, 98).

    ICA is not refit here.
    """
    subject_dir = ica_root / subject

    mixing_path = subject_dir / "mixing_matrix.npy"
    sensor_mixing_path = subject_dir / "mixing_matrix_sensor.npy"
    unmixing_path = subject_dir / "unmixing_matrix.npy"

    # Prefer an explicitly saved sensor-space matrix if it exists.
    if sensor_mixing_path.exists():
        A = np.load(sensor_mixing_path).astype(np.float32)

        if A.shape != (N_CHANNELS, N_COMPONENTS):
            raise RuntimeError(
                f"{sensor_mixing_path} has shape {A.shape}; expected "
                f"sensor-space A = ({N_CHANNELS}, {N_COMPONENTS})."
            )

        print(f"  Sensor-space mixing A : {A.shape}")
        return A

    if not mixing_path.exists():
        raise FileNotFoundError(
            f"Missing Step-4 mixing matrix: {mixing_path}"
        )

    saved_mixing = np.load(mixing_path).astype(np.float32)

    print(
        f"  Step-4 saved mixing matrix : {saved_mixing.shape}"
    )

    # If a future Step-4 version already stores sensor-space A, accept it.
    if saved_mixing.shape == (N_CHANNELS, N_COMPONENTS):
        print(
            f"  Sensor-space mixing A : {saved_mixing.shape}"
        )
        return saved_mixing

    # Current MindInk Step-4 layout: mixing_matrix.npy is (98,98).
    # Recover the required frozen sensor-space A from W.
    if saved_mixing.shape == (N_COMPONENTS, N_COMPONENTS):
        if not unmixing_path.exists():
            raise RuntimeError(
                f"{mixing_path} is {saved_mixing.shape}, but "
                f"{unmixing_path} is missing. Cannot construct the required "
                f"sensor-space A = pinv(W)."
            )

        W = np.load(unmixing_path).astype(np.float32)

        print(
            f"  Step-4 saved unmixing W : {W.shape}"
        )

        if W.shape != (N_COMPONENTS, N_CHANNELS):
            raise RuntimeError(
                f"{unmixing_path} has shape {W.shape}; expected "
                f"({N_COMPONENTS}, {N_CHANNELS})."
            )

        # For the frozen ICA model, W maps sensors -> ICA sources.
        # pinv(W) maps ICA sources -> sensors.
        A = np.linalg.pinv(W).astype(np.float32)

        if A.shape != (N_CHANNELS, N_COMPONENTS):
            raise RuntimeError(
                f"Constructed sensor-space A has shape {A.shape}; expected "
                f"({N_CHANNELS}, {N_COMPONENTS})."
            )

        # Save the derived matrix so subsequent Step-10 runs use the
        # explicit sensor-space artifact rather than recomputing it.
        np.save(sensor_mixing_path, A)

        print(
            f"  Sensor-space A = pinv(W) : {A.shape}"
        )
        print(
            f"  Saved derived A          : {sensor_mixing_path}"
        )

        return A

    raise RuntimeError(
        f"Unexpected Step-4 mixing matrix shape {saved_mixing.shape} at "
        f"{mixing_path}. Expected either "
        f"({N_CHANNELS}, {N_COMPONENTS}) or "
        f"({N_COMPONENTS}, {N_COMPONENTS})."
    )

def load_sentence_sources(
    ica_root: Path,
    subject: str,
    sentence_index: int,
) -> np.ndarray:

    path = (
        ica_root
        / subject
        / "sources"
        / f"sentence_{sentence_index:03d}_sources.npy"
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Missing ICA source file: {path}"
        )

    sources = np.load(path, mmap_mode="r")

    if sources.ndim != 2:
        raise RuntimeError(
            f"{path}: expected 2-D source array, got {sources.shape}"
        )

    if sources.shape[0] != N_COMPONENTS:
        raise RuntimeError(
            f"{path}: expected {N_COMPONENTS} components, "
            f"got {sources.shape[0]}"
        )

    return np.asarray(sources, dtype=np.float32)


# ============================================================================
# Step-9 gate network
# ============================================================================

class FeatureGate(nn.Module):
    """
    g_k = sigmoid(f(phi_k))

    The network consumes only component features.
    It never receives the ICA component index.
    """

    def __init__(
        self,
        n_features: int,
        initial_gates: np.ndarray,
        hidden: int = 64,
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(n_features, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

        # Initialize the network to reproduce the Step-9 bootstrap gates.
        # Zero weights make output initially independent of feature values;
        # the final bias is initialized to logit(g_k).
        for module in self.net:
            if isinstance(module, nn.Linear):
                nn.init.zeros_(module.weight)
                nn.init.zeros_(module.bias)

        g = np.clip(
            np.asarray(initial_gates, dtype=np.float32),
            1e-4,
            1.0 - 1e-4,
        )

        logits = np.log(g / (1.0 - g))

        final = self.net[-1]
        final.bias.data[:] = torch.from_numpy(
            logits.reshape(1)
        )

        # A per-component bootstrap bias is not allowed: that would make
        # the gate depend on component identity.  Use the mean bootstrap
        # logit for the shared feature-conditioned network.
        mean_logit = float(np.mean(logits))
        final.bias.data[:] = mean_logit

        self.register_buffer(
            "bootstrap_mean_gate",
            torch.tensor(float(np.mean(g))),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(
            self.net(features).squeeze(-1)
        )


# ============================================================================
# EEG temporal encoder
# ============================================================================

class EEGEncoder(nn.Module):
    """
    Compact temporal encoder.

    Input:
        batch x 105 x T

    Output:
        batch x L x 768

    L is reduced by temporal pooling so BART can consume the sequence.
    """

    def __init__(
        self,
        channels: int = N_CHANNELS,
        hidden: int = 256,
        d_model: int = 768,
        layers: int = 2,
        heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv1d(
                channels,
                hidden,
                kernel_size=9,
                stride=4,
                padding=4,
            ),
            nn.GELU(),
            nn.Conv1d(
                hidden,
                hidden,
                kernel_size=7,
                stride=2,
                padding=3,
            ),
            nn.GELU(),
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=heads,
            dim_feedforward=hidden * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=layers,
        )

        self.proj = nn.Linear(
            hidden,
            d_model,
        )

        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        x = self.conv(x)
        x = x.transpose(1, 2)

        x = self.transformer(x)

        x = self.proj(x)
        x = self.norm(x)

        return x


# ============================================================================
# Dataset
# ============================================================================

class ZuCoDataset(Dataset):

    def __init__(
        self,
        indices: List[int],
        sentences: Dict[int, str],
        ica_root: Path,
        subject: str,
        mixing: np.ndarray,
        max_samples: int = 12000,
    ):
        self.indices = list(indices)
        self.sentences = sentences
        self.ica_root = ica_root
        self.subject = subject
        self.mixing = mixing
        self.max_samples = max_samples

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        idx = self.indices[item]

        sources = load_sentence_sources(
            self.ica_root,
            self.subject,
            idx,
        )

        # Avoid excessive memory/sequence lengths during the first
        # decoder experiments.  Uniformly resample only for the decoder
        # input; the underlying Step-4 source files remain untouched.
        if sources.shape[1] > self.max_samples:
            positions = np.linspace(
                0,
                sources.shape[1] - 1,
                self.max_samples,
            ).astype(np.int64)
            sources = sources[:, positions]

        return {
            "index": idx,
            "sources": torch.from_numpy(
                np.asarray(sources, dtype=np.float32)
            ),
            "text": self.sentences[idx],
        }


def collate_single(batch):
    """
    We intentionally use batch_size=1.

    ZuCo sentence durations differ substantially, and this keeps the first
    Step-10 implementation simple and auditable.
    """
    if len(batch) != 1:
        raise RuntimeError(
            "Step 10A currently requires batch_size=1."
        )

    return batch[0]


# ============================================================================
# Metrics
# ============================================================================

def normalize_tokens(text: str) -> List[str]:
    return text.lower().strip().split()


def token_f1(reference: str, hypothesis: str) -> float:
    ref = normalize_tokens(reference)
    hyp = normalize_tokens(hypothesis)

    if not ref and not hyp:
        return 1.0
    if not ref or not hyp:
        return 0.0

    ref_counts = {}
    hyp_counts = {}

    for token in ref:
        ref_counts[token] = ref_counts.get(token, 0) + 1

    for token in hyp:
        hyp_counts[token] = hyp_counts.get(token, 0) + 1

    overlap = 0
    for token, count in hyp_counts.items():
        overlap += min(
            count,
            ref_counts.get(token, 0),
        )

    precision = overlap / len(hyp)
    recall = overlap / len(ref)

    if precision + recall == 0:
        return 0.0

    return (
        2.0 * precision * recall
        / (precision + recall)
    )


def bleu1(reference: str, hypothesis: str) -> float:
    ref = normalize_tokens(reference)
    hyp = normalize_tokens(hypothesis)

    if not hyp:
        return 0.0

    ref_counts = {}
    for token in ref:
        ref_counts[token] = ref_counts.get(token, 0) + 1

    overlap = 0
    used = {}

    for token in hyp:
        count = used.get(token, 0)
        if count < ref_counts.get(token, 0):
            overlap += 1
            used[token] = count + 1

    precision = overlap / len(hyp)

    # Brevity penalty.
    if len(hyp) == 0:
        return 0.0

    if len(hyp) > len(ref):
        bp = 1.0
    else:
        bp = math.exp(
            1.0 - len(ref) / max(len(hyp), 1)
        )

    return float(bp * precision)


# ============================================================================
# BART wrapper
# ============================================================================

class EEGToBART(nn.Module):

    def __init__(
        self,
        eeg_encoder: EEGEncoder,
        bart: BartForConditionalGeneration,
    ):
        super().__init__()

        self.eeg_encoder = eeg_encoder
        self.bart = bart

    def encode_eeg(
        self,
        eeg: torch.Tensor,
    ):
        z = self.eeg_encoder(eeg)

        mask = torch.ones(
            z.shape[:2],
            dtype=torch.long,
            device=z.device,
        )

        return z, mask

    def forward(
        self,
        eeg: torch.Tensor,
        labels: torch.Tensor,
    ):
        z, mask = self.encode_eeg(eeg)

        outputs = self.bart(
            inputs_embeds=z,
            attention_mask=mask,
            labels=labels,
        )

        return outputs

    @torch.no_grad()
    def generate(
        self,
        eeg: torch.Tensor,
        max_new_tokens: int = 64,
        num_beams: int = 1,
    ):
        z, mask = self.encode_eeg(eeg)

        return self.bart.generate(
            inputs_embeds=z,
            attention_mask=mask,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            do_sample=False,
        )


# ============================================================================
# Gated reconstruction
# ============================================================================

def reconstruct_gated_sensor(
    sources: torch.Tensor,
    gates: torch.Tensor,
    mixing: torch.Tensor,
) -> torch.Tensor:
    """
    X_hat = A @ (g * S)

    sources: components x T
    gates: components
    mixing: 105 x 98
    """

    gated_sources = gates[:, None] * sources

    return torch.matmul(
        mixing,
        gated_sources,
    )


# ============================================================================
# Training
# ============================================================================

def train_epoch(
    model: EEGToBART,
    gate_model: FeatureGate,
    loader: DataLoader,
    feature_tensor: torch.Tensor,
    mixing_tensor: torch.Tensor,
    tokenizer: BartTokenizer,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    lambda_artifact: float,
    artifact_probability: torch.Tensor,
    lambda_sparsity: float,
    lambda_reconstruction: float,
) -> Dict[str, float]:

    model.train()
    gate_model.train()

    totals = {
        "total": 0.0,
        "decoding": 0.0,
        "artifact": 0.0,
        "sparsity": 0.0,
        "reconstruction": 0.0,
    }

    count = 0

    for batch in loader:
        sources = batch["sources"].to(device)
        text = batch["text"]

        # sources is 98 x T; batch size is one.
        gates = gate_model(feature_tensor)

        gated_sensor = reconstruct_gated_sensor(
            sources,
            gates,
            mixing_tensor,
        )

        # Per-sentence normalization. This is only an encoder input
        # normalization and does not modify saved EEG files.
        mean = gated_sensor.mean(dim=1, keepdim=True)
        std = gated_sensor.std(dim=1, keepdim=True).clamp_min(1e-5)
        eeg = (gated_sensor - mean) / std

        eeg = eeg.unsqueeze(0)

        tokens = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=128,
        )

        labels = tokens.input_ids.to(device)

        optimizer.zero_grad(set_to_none=True)

        outputs = model(
            eeg,
            labels,
        )

        decoding_loss = outputs.loss

        artifact_loss = torch.mean(
            gates * artifact_probability
        )

        sparsity_loss = torch.mean(gates)

        # Reconstruction auxiliary:
        # Keep gated EEG close to the no-denoising sensor signal in a weak
        # sense. This prevents the gate from collapsing while the decoder
        # learns. It is deliberately small and is NOT presented as a clean
        # ground-truth SNR objective.
        original = torch.matmul(
            mixing_tensor,
            sources,
        )

        reconstruction_loss = torch.mean(
            (gated_sensor - original) ** 2
        ) / (
            torch.mean(original ** 2).detach()
            + 1e-6
        )

        total = (
            decoding_loss
            + lambda_artifact * artifact_loss
            + lambda_sparsity * sparsity_loss
            + lambda_reconstruction * reconstruction_loss
        )

        total.backward()

        torch.nn.utils.clip_grad_norm_(
            list(model.parameters())
            + list(gate_model.parameters()),
            1.0,
        )

        optimizer.step()

        totals["total"] += float(total.detach().cpu())
        totals["decoding"] += float(
            decoding_loss.detach().cpu()
        )
        totals["artifact"] += float(
            artifact_loss.detach().cpu()
        )
        totals["sparsity"] += float(
            sparsity_loss.detach().cpu()
        )
        totals["reconstruction"] += float(
            reconstruction_loss.detach().cpu()
        )

        count += 1

    if count == 0:
        raise RuntimeError("No training samples.")

    return {
        key: value / count
        for key, value in totals.items()
    }


# ============================================================================
# Evaluation controls
# ============================================================================

def make_gaussian_eeg(
    reference: torch.Tensor,
) -> torch.Tensor:

    std = reference.std().clamp_min(1e-5)

    return torch.randn_like(reference) * std


def evaluate_condition(
    model: EEGToBART,
    gate_model: FeatureGate,
    dataset_indices: List[int],
    sentences: Dict[int, str],
    ica_root: Path,
    subject: str,
    mixing: torch.Tensor,
    feature_tensor: torch.Tensor,
    tokenizer: BartTokenizer,
    device: torch.device,
    condition: str,
    shuffle_map: Dict[int, int] | None,
    max_new_tokens: int,
) -> List[dict]:

    model.eval()
    gate_model.eval()

    records = []

    with torch.no_grad():

        gates = gate_model(
            feature_tensor
        )

        for sentence_index in dataset_indices:

            if condition == CONTROL_LM_ONLY:

                # Same temporal shape as a normal EEG representation,
                # but zero input means the decoder receives no EEG content.
                sources = load_sentence_sources(
                    ica_root,
                    subject,
                    sentence_index,
                )

                T = min(
                    sources.shape[1],
                    12000,
                )

                eeg = torch.zeros(
                    1,
                    N_CHANNELS,
                    T,
                    device=device,
                    dtype=torch.float32,
                )

            else:

                source_index = sentence_index

                if condition == CONTROL_SHUFFLED:
                    if shuffle_map is None:
                        raise RuntimeError(
                            "Shuffle map missing."
                        )
                    source_index = shuffle_map[
                        sentence_index
                    ]

                sources = load_sentence_sources(
                    ica_root,
                    subject,
                    source_index,
                )

                if sources.shape[1] > 12000:
                    positions = np.linspace(
                        0,
                        sources.shape[1] - 1,
                        12000,
                    ).astype(np.int64)
                    sources = sources[:, positions]

                source_tensor = torch.from_numpy(
                    sources
                ).to(device)

                gated_sensor = reconstruct_gated_sensor(
                    source_tensor,
                    gates,
                    mixing,
                )

                mean = gated_sensor.mean(
                    dim=1,
                    keepdim=True,
                )
                std = gated_sensor.std(
                    dim=1,
                    keepdim=True,
                ).clamp_min(1e-5)

                eeg = (
                    (gated_sensor - mean) / std
                ).unsqueeze(0)

                if condition == CONTROL_GAUSSIAN:
                    eeg = make_gaussian_eeg(eeg)

            generated_ids = model.generate(
                eeg,
                max_new_tokens=max_new_tokens,
                num_beams=1,
            )

            prediction = tokenizer.decode(
                generated_ids[0],
                skip_special_tokens=True,
            ).strip()

            reference = sentences[
                sentence_index
            ]

            records.append(
                {
                    "sentence_index": int(sentence_index),
                    "condition": condition,
                    "reference": reference,
                    "prediction": prediction,
                    "bleu1": bleu1(
                        reference,
                        prediction,
                    ),
                    "token_f1": token_f1(
                        reference,
                        prediction,
                    ),
                    "exact_match": bool(
                        reference.strip().lower()
                        == prediction.strip().lower()
                    ),
                }
            )

    return records


# ============================================================================
# Main Step 10
# ============================================================================

def run_step10(
    subject: str,
    mat_file: str,
    ica_root: str = ICA_ROOT,
    features_root: str = FEATURES_ROOT,
    segments_root: str = SEGMENTS_ROOT,
    gates_root: str = "gates",
    output_root: str = OUTPUT_ROOT,
    model_name: str = BART_MODEL_NAME,
    epochs: int = 1,
    lr: float = 1e-4,
    max_sentences: int | None = 10,
    max_samples: int = 12000,
    max_new_tokens: int = 64,
    device_name: str = "auto",
    seed: int = 42,
    freeze_bart: bool = True,
    lambda_artifact: float = 0.02,
    lambda_sparsity: float = 0.001,
    lambda_reconstruction: float = 0.01,
):

    set_seed(seed)

    if device_name == "auto":
        device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )
    else:
        device = torch.device(device_name)

    print("=" * 80)
    print("MindInk / ZuCo Step 10")
    print("Decode-aware EEG-to-text Integration")
    print("=" * 80)

    print(f"Subject/session : {subject}")
    print(f"Device          : {device}")
    print(f"Epochs          : {epochs}")
    print(f"Learning rate    : {lr}")
    print(f"BART             : {model_name}")
    print(f"Freeze BART      : {freeze_bart}")
    print()

    # ------------------------------------------------------------------
    # Paths
    # ------------------------------------------------------------------

    ica_subject = Path(ica_root) / subject
    features_subject = Path(features_root) / subject
    gates_subject = Path(gates_root) / subject
    output_subject = Path(output_root) / subject

    output_subject.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ------------------------------------------------------------------
    # Load verified inputs
    # ------------------------------------------------------------------

    print("Loading ZuCo sentence text...")
    sentences = load_zuco_sentences(
        Path(mat_file)
    )
    print(
        f"  Sentences available : {len(sentences)}"
    )

    features = load_step6_features(
        Path(features_root),
        subject,
    )

    bootstrap_gates = load_step9_gates(
        Path(gates_root),
        subject,
    )

    mixing = load_mixing_matrix(
        Path(ica_root),
        subject,
    )

    print()
    print("Frozen inputs:")
    print(
        f"  Component features : {features.shape}"
    )
    print(
        f"  Step-9 gates       : {bootstrap_gates.shape}"
    )
    print(
        f"  Sensor mixing A    : {mixing.shape}"
    )

    # ------------------------------------------------------------------
    # Sentence selection
    # ------------------------------------------------------------------

    all_indices = sorted(sentences.keys())

    if max_sentences is not None:
        selected = all_indices[:max_sentences]
    else:
        selected = all_indices

    if len(selected) < 2:
        raise RuntimeError(
            "Step 10 requires at least 2 sentences."
        )

    # Simple deterministic train/eval split.
    # This is deliberately a first integration experiment, not the final
    # cross-subject protocol.
    split = max(
        1,
        int(round(0.8 * len(selected))),
    )

    if split >= len(selected):
        split = len(selected) - 1

    train_indices = selected[:split]
    eval_indices = selected[split:]

    print()
    print(
        f"Sentences selected : {len(selected)}"
    )
    print(
        f"Training sentences : {len(train_indices)}"
    )
    print(
        f"Evaluation         : {len(eval_indices)}"
    )

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------

    print()
    print("Loading BART tokenizer/model...")

    tokenizer = BartTokenizer.from_pretrained(
        model_name
    )

    bart = BartForConditionalGeneration.from_pretrained(
        model_name
    )

    if freeze_bart:
        for parameter in bart.parameters():
            parameter.requires_grad = False

    eeg_encoder = EEGEncoder(
        d_model=bart.config.d_model
    )

    gate_model = FeatureGate(
        N_FEATURES,
        bootstrap_gates,
    )

    model = EEGToBART(
        eeg_encoder,
        bart,
    )

    model.to(device)
    gate_model.to(device)

    feature_tensor = torch.from_numpy(
        features
    ).to(device)

    mixing_tensor = torch.from_numpy(
        mixing
    ).to(device)

    # ------------------------------------------------------------------
    # ICLabel artifact prior
    # ------------------------------------------------------------------

    iclabel_path = (
        features_subject
        / "iclabel.json"
    )

    if iclabel_path.exists():
        records = load_json(
            iclabel_path
        )

        artifact_probability = []

        for record in records:

            probs = record.get(
                "probabilities",
                record.get(
                    "posterior",
                    None,
                ),
            )

            if isinstance(probs, dict):
                eye = float(
                    probs.get("eye", 0.0)
                )
                muscle = float(
                    probs.get("muscle", 0.0)
                )
                line = float(
                    probs.get("line_noise", 0.0)
                )

                artifact_probability.append(
                    max(eye, muscle, line)
                )

            else:
                artifact_probability.append(0.0)

        if len(artifact_probability) != N_COMPONENTS:
            artifact_probability = [
                0.0
            ] * N_COMPONENTS

    else:
        artifact_probability = [
            0.0
        ] * N_COMPONENTS

    artifact_tensor = torch.tensor(
        artifact_probability,
        dtype=torch.float32,
        device=device,
    )

    # ------------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------------

    dataset = ZuCoDataset(
        train_indices,
        sentences,
        Path(ica_root),
        subject,
        mixing,
        max_samples=max_samples,
    )

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        collate_fn=collate_single,
    )

    trainable_parameters = [
        p for p in model.parameters()
        if p.requires_grad
    ] + list(
        gate_model.parameters()
    )

    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=lr,
        weight_decay=1e-4,
    )

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    print()
    print("Training decode-aware gate + EEG encoder...")

    history = []

    for epoch in range(1, epochs + 1):

        metrics = train_epoch(
            model=model,
            gate_model=gate_model,
            loader=loader,
            feature_tensor=feature_tensor,
            mixing_tensor=mixing_tensor,
            tokenizer=tokenizer,
            optimizer=optimizer,
            device=device,
            lambda_artifact=lambda_artifact,
            artifact_probability=artifact_tensor,
            lambda_sparsity=lambda_sparsity,
            lambda_reconstruction=lambda_reconstruction,
        )

        record = {
            "epoch": epoch,
            **metrics,
        }

        history.append(record)

        print(
            f"  Epoch {epoch:03d}/{epochs}: "
            f"total={metrics['total']:.6f} "
            f"decode={metrics['decoding']:.6f} "
            f"artifact={metrics['artifact']:.6f} "
            f"sparse={metrics['sparsity']:.6f} "
            f"recon={metrics['reconstruction']:.6f}"
        )

    # ------------------------------------------------------------------
    # Save trained gate
    # ------------------------------------------------------------------

    model.eval()
    gate_model.eval()

    with torch.no_grad():
        final_gates = gate_model(
            feature_tensor
        ).cpu().numpy()

    np.save(
        output_subject / "decode_aware_gates.npy",
        final_gates.astype(np.float32),
    )

    np.save(
        output_subject / "component_features.npy",
        features.astype(np.float32),
    )

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "gate_state_dict": gate_model.state_dict(),
            "model_name": model_name,
            "subject": subject,
            "n_components": N_COMPONENTS,
            "n_features": N_FEATURES,
            "n_channels": N_CHANNELS,
            "ica_frozen": True,
            "component_index_used": False,
        },
        output_subject / "step10_checkpoint.pt",
    )

    save_json(
        output_subject / "training_history.json",
        history,
    )

    print()
    print("Final decode-aware gate statistics:")
    print(
        f"  Mean   : {final_gates.mean():.6f}"
    )
    print(
        f"  Median : {np.median(final_gates):.6f}"
    )
    print(
        f"  Min    : {final_gates.min():.6f}"
    )
    print(
        f"  Max    : {final_gates.max():.6f}"
    )

    # ------------------------------------------------------------------
    # Free-running controls
    # ------------------------------------------------------------------

    print()
    print("=" * 80)
    print("STEP 10 — FREE-RUNNING EVALUATION")
    print("=" * 80)

    # Deterministic mismatch: rotate evaluation indices by one.
    shuffle_map = {}

    for position, idx in enumerate(eval_indices):
        shuffle_map[idx] = eval_indices[
            (position + 1) % len(eval_indices)
        ]

    all_results = {}

    for condition in CONTROLS:

        print()
        print(
            f"Condition: {condition}"
        )

        records = evaluate_condition(
            model=model,
            gate_model=gate_model,
            dataset_indices=eval_indices,
            sentences=sentences,
            ica_root=Path(ica_root),
            subject=subject,
            mixing=mixing_tensor,
            feature_tensor=feature_tensor,
            tokenizer=tokenizer,
            device=device,
            condition=condition,
            shuffle_map=shuffle_map,
            max_new_tokens=max_new_tokens,
        )

        all_results[condition] = records

        if records:
            mean_bleu = float(
                np.mean(
                    [r["bleu1"] for r in records]
                )
            )

            mean_f1 = float(
                np.mean(
                    [r["token_f1"] for r in records]
                )
            )

            exact = float(
                np.mean(
                    [
                        r["exact_match"]
                        for r in records
                    ]
                )
            )

            print(
                f"  Mean BLEU-1 : {mean_bleu:.4f}"
            )
            print(
                f"  Mean token F1: {mean_f1:.4f}"
            )
            print(
                f"  Exact match : {exact:.4f}"
            )

    # ------------------------------------------------------------------
    # Save predictions
    # ------------------------------------------------------------------

    save_json(
        output_subject / "free_running_predictions.json",
        all_results,
    )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    summary = {}

    for condition, records in all_results.items():

        if not records:
            continue

        summary[condition] = {
            "n_sentences": len(records),
            "bleu1_mean": float(
                np.mean(
                    [r["bleu1"] for r in records]
                )
            ),
            "token_f1_mean": float(
                np.mean(
                    [r["token_f1"] for r in records]
                )
            ),
            "exact_match_rate": float(
                np.mean(
                    [
                        r["exact_match"]
                        for r in records
                    ]
                )
            ),
        }

    save_json(
        output_subject / "summary.json",
        summary,
    )

    # ------------------------------------------------------------------
    # Manifest
    # ------------------------------------------------------------------

    manifest = {
        "step": "10A",
        "subject_id": subject,
        "scope": "ZuCo open-vocabulary English",
        "sentences_available": len(sentences),
        "sentences_selected": len(selected),
        "train_sentences": train_indices,
        "evaluation_sentences": eval_indices,
        "n_channels": N_CHANNELS,
        "n_components": N_COMPONENTS,
        "n_features": N_FEATURES,
        "sampling_rate": FS,
        "bart_model": model_name,
        "bart_frozen": bool(freeze_bart),
        "ica_frozen": True,
        "ica_refit": False,
        "component_index_used": False,
        "step9_bootstrap_gate_used": True,
        "gate_decode_aware_training": True,
        "evaluation": {
            "free_running": True,
            "teacher_forced_headline_metric": False,
            "controls": list(CONTROLS),
        },
        "controls": {
            "real_eeg": "gated real EEG",
            "gaussian_noise": "Gaussian noise with matched scale",
            "shuffled_eeg": "EEG from another evaluation sentence",
            "lm_only": "zero EEG encoder input",
        },
        "outputs": {
            "checkpoint": "step10_checkpoint.pt",
            "decode_aware_gates": "decode_aware_gates.npy",
            "predictions": "free_running_predictions.json",
            "summary": "summary.json",
            "history": "training_history.json",
        },
        "warnings": [
            "This is the ZuCo-only Step-10A integration.",
            "This single-subject experiment does not establish subject invariance.",
            "Step 11 must perform cross-subject/LOSO evaluation.",
            "Free-running controls are required before claiming EEG decoding improvement.",
            "BLEU-1 and token F1 are diagnostic metrics, not proof by themselves.",
            "No clean-ground-truth SNR improvement is claimed on raw ZuCo.",
        ],
    }

    save_json(
        output_subject / "manifest.json",
        manifest,
    )

    print()
    print("=" * 80)
    print("STEP 10A COMPLETE")
    print("=" * 80)

    print(
        f"Subject/session     : {subject}"
    )
    print(
        f"Training sentences  : {len(train_indices)}"
    )
    print(
        f"Evaluation sentences: {len(eval_indices)}"
    )
    print(
        f"Components          : {N_COMPONENTS}"
    )
    print(
        f"Features/component  : {N_FEATURES}"
    )
    print(
        f"ICA frozen          : YES"
    )
    print(
        f"Gate decode-aware   : YES"
    )
    print(
        f"Free-running eval   : YES"
    )
    print(
        f"Controls            : {', '.join(CONTROLS)}"
    )
    print(
        f"Output              : {output_subject}"
    )


# ============================================================================
# CLI
# ============================================================================

def main():

    parser = argparse.ArgumentParser(
        description="MindInk Step 10A ZuCo EEG-to-text decoder"
    )

    parser.add_argument(
        "--subject",
        default=DEFAULT_SUBJECT,
    )

    parser.add_argument(
        "--mat",
        default="dataset-NR/resultsYAG_NR.mat",
    )

    parser.add_argument(
        "--ica",
        default=ICA_ROOT,
    )

    parser.add_argument(
        "--features",
        default=FEATURES_ROOT,
    )

    parser.add_argument(
        "--segments",
        default=SEGMENTS_ROOT,
    )

    parser.add_argument(
        "--gates",
        default="gates",
    )

    parser.add_argument(
        "--output",
        default=OUTPUT_ROOT,
    )

    parser.add_argument(
        "--model",
        default=BART_MODEL_NAME,
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
    )

    parser.add_argument(
        "--max-sentences",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--max-samples",
        type=int,
        default=12000,
    )

    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--device",
        default="auto",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--unfreeze-bart",
        action="store_true",
        help="Fine-tune BART weights as well as EEG encoder/gate.",
    )

    parser.add_argument(
        "--lambda-artifact",
        type=float,
        default=0.02,
    )

    parser.add_argument(
        "--lambda-sparsity",
        type=float,
        default=0.001,
    )

    parser.add_argument(
        "--lambda-reconstruction",
        type=float,
        default=0.01,
    )

    args = parser.parse_args()

    run_step10(
        subject=args.subject,
        mat_file=args.mat,
        ica_root=args.ica,
        features_root=args.features,
        segments_root=args.segments,
        gates_root=args.gates,
        output_root=args.output,
        model_name=args.model,
        epochs=args.epochs,
        lr=args.lr,
        max_sentences=args.max_sentences,
        max_samples=args.max_samples,
        max_new_tokens=args.max_new_tokens,
        device_name=args.device,
        seed=args.seed,
        freeze_bart=not args.unfreeze_bart,
        lambda_artifact=args.lambda_artifact,
        lambda_sparsity=args.lambda_sparsity,
        lambda_reconstruction=args.lambda_reconstruction,
    )


if __name__ == "__main__":
    main()
