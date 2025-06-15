#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
STEP 2: Supervised Reranking using Pre-trained Embeddings.

This script loads the frozen, pre-trained transformer from Step 1. It uses this
model to generate fixed-size embeddings for queries and documents. A small
reranker model is then trained on top of these embeddings to predict document
relevance for a given query.
"""

import time, random, itertools
from pathlib import Path
from typing import List, Tuple

import torch, torch.nn as nn, torch.nn.functional as F

# Import the model definition from the first script to load its weights
from train_unsupervised import TransformerForNextTokenPrediction, encode

# --------------------------- Hyper-parameters -------------------------- #
# Model and Data
DIM                    = 512            # Must match the unsupervised model
MAX_LEN_QUERY          = 100            # Must match the unsupervised model
MAX_LEN_DOC            = 110            # Must match the unsupervised model
MAX_NEG                = 100            # Max negative samples per positive
TRUNCATE               = 100            # Truncate doc list per query
PRETRAINED_MODEL_PATH  = "unsupervised_encoder.pt"

# Training
LR                     = 1e-3           # Reranker learning rate
TAU                    = 0.1            # Temperature for softmax
BATCH_QUERIES          = 32             # Num of queries in a batch update
VAL_EVERY              = 100            # Steps
SAVE_EVERY             = 500            # Steps
RERANKER_SAVE_PATH     = "reranker_model.pt"

DEVICE = torch.device(
    "mps"  if torch.backends.mps.is_available() else
    "cuda" if torch.cuda.is_available() else "cpu")

# --- Data Loading (identical to original script) ---
_batch_gen = None
def fetch_batch(path: str = "so_ts_markov.txt"):
    """Yield (query, [docs…]) from a file with blank-line delimited blocks."""
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
    """A simple MLP to rerank based on concatenated embeddings."""
    def __init__(self, input_dim=DIM*2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.GELU(),
            nn.Linear(input_dim // 2, 1)
        )

    def forward(self, q_emb, d_emb):
        # q_emb is [1, DIM], d_emb is [N, DIM]
        # Expand query embedding to match the number of documents
        q_emb_expanded = q_emb.expand(d_emb.size(0), -1)
        # Concatenate and score
        combined_emb = torch.cat([q_emb_expanded, d_emb], dim=1)
        return self.net(combined_emb).squeeze(-1)

@torch.no_grad()
def get_embedding(text: str, max_len: int, encoder: TransformerForNextTokenPrediction) -> torch.Tensor:
    """Generates a fixed-size embedding for a text using the frozen encoder."""
    encoded_text = encode(text, max_len)
    input_tensor = torch.tensor([encoded_text], dtype=torch.long, device=DEVICE)
    _, hidden_state = encoder(input_tensor)
    # Use the embedding of the first token as the representation [CLS] style
    embedding = hidden_state[:, 0, :]
    return embedding

def train_reranker(steps=20_000, val_data_global=None):
    print("--- Starting Step 2: Reranker Training ---")

    # 1. Load the pre-trained unsupervised encoder
    print(f"Loading pre-trained encoder from: {PRETRAINED_MODEL_PATH}")
    if not Path(PRETRAINED_MODEL_PATH).exists():
        raise FileNotFoundError(f"Fatal: Pre-trained model not found at '{PRETRAINED_MODEL_PATH}'. Please run `train_unsupervised.py` first.")

    encoder = TransformerForNextTokenPrediction().to(DEVICE)
    encoder.load_state_dict(torch.load(PRETRAINED_MODEL_PATH, map_location=DEVICE))
    encoder.eval() # Set to evaluation mode and freeze
    for param in encoder.parameters():
        param.requires_grad = False

    print("Encoder loaded and frozen.")

    # 2. Initialize the reranker model and optimizer
    reranker = Reranker().to(DEVICE)
    opt = torch.optim.AdamW(reranker.parameters(), lr=LR)

    if val_data_global is None: val_data_global = []

    for step in range(1, steps + 1):
        reranker.train()
        opt.zero_grad()
        tot_loss = 0.0

        # This loop accumulates gradients for a "batch" of queries
        for _ in range(BATCH_QUERIES):
            q_txt, docs_all = fetch_batch()
            if not docs_all: continue

            docs_all = docs_all[:TRUNCATE]
            pos_doc_text = docs_all[0]
            neg_docs = docs_all[1:]

            # Sub-sample negatives
            neg_sampled = random.sample(neg_docs, min(len(neg_docs), MAX_NEG))

            # Create list of documents to be ranked for this query
            current_docs = [pos_doc_text] + neg_sampled

            # Shuffle and find the new index of the positive document
            perm = torch.randperm(len(current_docs))
            shuffled_docs = [current_docs[i] for i in perm]
            target_idx = (perm == 0).nonzero(as_tuple=True)[0].item()

            # 3. Generate embeddings with the frozen encoder
            q_emb = get_embedding(q_txt, MAX_LEN_QUERY, encoder)
            d_embs = torch.cat([get_embedding(d, MAX_LEN_DOC, encoder) for d in shuffled_docs], dim=0)

            if d_embs.size(0) == 0: continue

            # 4. Get scores from the reranker model
            scores = reranker(q_emb, d_embs) / TAU

            # 5. Calculate list-wise cross-entropy loss
            loss = F.cross_entropy(scores.unsqueeze(0), torch.tensor([target_idx], device=DEVICE))
            loss.backward()
            tot_loss += loss.item()

        torch.nn.utils.clip_grad_norm_(reranker.parameters(), 1.0)
        opt.step()

        if step % LOG_EVERY == 0:
            print(f"Step {step:>6} | Avg Loss: {tot_loss / BATCH_QUERIES:.4f}")

        if step % VAL_EVERY == 0:
            reranker.eval()
            val_reciprocal_ranks = []
            for vq_text, vdocs_all in val_data_global:
                if not vdocs_all: continue
                vdocs_all = vdocs_all[:TRUNCATE]

                # In validation, the positive document is always at index 0
                vq_emb = get_embedding(vq_text, MAX_LEN_QUERY, encoder)
                vd_embs = torch.cat([get_embedding(d, MAX_LEN_DOC, encoder) for d in vdocs_all], dim=0)

                v_scores = reranker(vq_emb, vd_embs)
                rank = (v_scores.argsort(descending=True) == 0).nonzero(as_tuple=True)[0].item() + 1
                val_reciprocal_ranks.append(1.0 / rank)

            if val_reciprocal_ranks:
                mrr_val = sum(val_reciprocal_ranks) / len(val_reciprocal_ranks)
                print(f"└─ Step {step} Validation MRR: {mrr_val:.4f} ({len(val_reciprocal_ranks)} queries)")

        if step % SAVE_EVERY == 0:
            print(f"--- Saving reranker model to {RERANKER_SAVE_PATH} ---")
            torch.save(reranker.state_dict(), RERANKER_SAVE_PATH)

if __name__ == "__main__":
    validation_data = load_validation()
    train_reranker(val_data_global=validation_data)