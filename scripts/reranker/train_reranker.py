#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
STEP 2: Supervised Reranking with End-to-End Fine-tuning.
(Modified for Modal remote execution and joint training)

This script loads the pre-trained transformer and a reranker, then trains
them *together*. Gradients from the ranking loss are used to fine-tune
the encoder, specializing it for the ranking task.
"""

import time, random, itertools
from pathlib import Path
from typing import List, Tuple

import torch, torch.nn as nn, torch.nn.functional as F
import modal

from train_unsupervised import TransformerForNextTokenPrediction, encode

# --------------------------- Hyper-parameters -------------------------- #
# Model and Data
DIM                    = 512
MAX_LEN_QUERY          = 100
MAX_LEN_DOC            = 110
MAX_NEG                = 100
TRUNCATE               = 100
PRETRAINED_MODEL_PATH  = "/data/unsupervised_encoder.pt"

# --- CHANGED: Differential Learning Rates ---
RERANKER_LR            = 1e-3 # Higher LR for the new reranker head
ENCODER_LR             = 1e-5 # Lower LR for fine-tuning the large encoder
TAU                    = 0.1
BATCH_QUERIES          = 32
VAL_EVERY              = 100
SAVE_EVERY             = 100
CKPT_DIR               = "/ckpts"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Data Loading (identical logic) ---
_batch_gen = None
def fetch_batch(path: str = "so_ts_markov.txt"):
    global _batch_gen
    if _batch_gen is None:
        def _reader():
            with Path(path).open(encoding="utf-8") as f:
                for gap, grp in itertools.groupby(f, key=lambda l: l.strip() == ""):
                    if not gap:
                        lines = [l.rstrip("\n") for l in grp]
                        if lines: yield lines[0], lines[1:]
        _batch_gen = _reader()
    try:               return next(_batch_gen)
    except StopIteration:
        _batch_gen = None; return fetch_batch(path)

def load_validation(path="so_vs_markov.txt", cap=1000):
    data = []
    with Path(path).open(encoding="utf-8") as f:
        for gap, grp in itertools.groupby(f, key=lambda l: l.strip() == ""):
            if not gap and len(data) < cap:
                lines = [l.rstrip("\n") for l in grp]
                if lines: data.append((lines[0], lines[1:]))
    return data

class Reranker(nn.Module):
    def __init__(self, input_dim=DIM*2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.GELU(),
            nn.Linear(input_dim // 2, 1)
        )

    def forward(self, q_emb, d_emb):
        q_emb_expanded = q_emb.expand(d_emb.size(0), -1)
        combined_emb = torch.cat([q_emb_expanded, d_emb], dim=1)
        return self.net(combined_emb).squeeze(-1)

# --- CHANGED: Removed @torch.no_grad() to allow gradient flow ---
# The function will now respect the context (train() or no_grad()) where it's called.
def get_embedding(text: str, max_len: int, encoder: TransformerForNextTokenPrediction) -> torch.Tensor:
    """Generates a fixed-size embedding for a text using the encoder."""
    encoded_text = encode(text, max_len)
    input_tensor = torch.tensor([encoded_text], dtype=torch.long, device=DEVICE)
    _, hidden_state = encoder(input_tensor)
    embedding = hidden_state[:, 0, :]
    return embedding

def train_reranker(steps: int, ckpt_volume: modal.Volume, val_data_global: List = None):
    print("--- Starting Reranker Training with End-to-End Fine-tuning ---")

    print(f"Loading pre-trained encoder from: {PRETRAINED_MODEL_PATH}")
    if not Path(PRETRAINED_MODEL_PATH).exists():
        raise FileNotFoundError(f"Fatal: Pre-trained model not found at '{PRETRAINED_MODEL_PATH}'.")

    encoder = TransformerForNextTokenPrediction().to(DEVICE)
    encoder.load_state_dict(torch.load(PRETRAINED_MODEL_PATH, map_location=DEVICE))
    reranker = Reranker().to(DEVICE)

    print("Encoder and reranker loaded.")

    # --- CHANGED: Optimizer with differential learning rates ---
    # We create two parameter groups to train the models at different speeds.
    optimizer = torch.optim.AdamW([
        {'params': encoder.parameters(), 'lr': ENCODER_LR},
        {'params': reranker.parameters(), 'lr': RERANKER_LR}
    ])

    if val_data_global is None: val_data_global = []

    for step in range(1, steps + 1):
        # --- CHANGED: Set both models to training mode ---
        encoder.train()
        reranker.train()

        optimizer.zero_grad()
        tot_loss = 0.0

        for _ in range(BATCH_QUERIES):
            q_txt, docs_all = fetch_batch()
            if not docs_all: continue
            docs_all = docs_all[:TRUNCATE]
            pos_doc_text = docs_all[0]
            neg_docs = docs_all[1:]
            neg_sampled = random.sample(neg_docs, min(len(neg_docs), MAX_NEG))
            current_docs = [pos_doc_text] + neg_sampled
            perm = torch.randperm(len(current_docs))
            shuffled_docs = [current_docs[i] for i in perm]
            target_idx = (perm == 0).nonzero(as_tuple=True)[0].item()

            # get_embedding is now called in a gradient-enabled context
            q_emb = get_embedding(q_txt, MAX_LEN_QUERY, encoder)
            d_embs = torch.cat([get_embedding(d, MAX_LEN_DOC, encoder) for d in shuffled_docs], dim=0)

            if d_embs.size(0) == 0: continue
            scores = reranker(q_emb, d_embs) / TAU
            loss = F.cross_entropy(scores.unsqueeze(0), torch.tensor([target_idx], device=DEVICE))

            # This will calculate gradients for both the reranker and the encoder
            loss.backward()
            tot_loss += loss.item()

        torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(reranker.parameters(), 1.0)
        optimizer.step()

        if step % 10 == 0:
            print(f"Step {step:>6} | Avg Loss: {tot_loss / BATCH_QUERIES:.4f}")

        if step % VAL_EVERY == 0:
            # --- CHANGED: Set both models to evaluation mode for validation ---
            encoder.eval()
            reranker.eval()
            val_reciprocal_ranks = []

            # Gradients are disabled here by the context manager
            with torch.no_grad():
                for vq_text, vdocs_all in val_data_global:
                    if not vdocs_all: continue
                    vdocs_all = vdocs_all[:TRUNCATE]

                    vq_emb = get_embedding(vq_text, MAX_LEN_QUERY, encoder)
                    vd_embs = torch.cat([get_embedding(d, MAX_LEN_DOC, encoder) for d in vdocs_all], dim=0)
                    v_scores = reranker(vq_emb, vd_embs)
                    rank = (v_scores.argsort(descending=True) == 0).nonzero(as_tuple=True)[0].item() + 1
                    val_reciprocal_ranks.append(1.0 / rank)

            if val_reciprocal_ranks:
                mrr_val = sum(val_reciprocal_ranks) / len(val_reciprocal_ranks)
                print(f"└─ Step {step} Validation MRR: {mrr_val:.4f} ({len(val_reciprocal_ranks)} queries)")

        if step % SAVE_EVERY == 0:
            print(f"--- Saving models to checkpoint volume ---")
            # --- CHANGED: Save both models in the checkpoint ---
            # It's good practice to save the fine-tuned encoder as well
            torch.save(reranker.state_dict(), f"{CKPT_DIR}/reranker_finetuned_step_{step}.pt")
            torch.save(encoder.state_dict(), f"{CKPT_DIR}/encoder_finetuned_step_{step}.pt")
            ckpt_volume.commit()

def reranker_modal_entrypoint(steps: int, ckpt_volume: modal.Volume):
    """The main entrypoint for the Modal remote training job."""
    validation_data = load_validation()
    train_reranker(steps=steps, ckpt_volume=ckpt_volume, val_data_global=validation_data)