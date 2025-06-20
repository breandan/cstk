#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
STEP 2: Supervised Reranking with End-to-End Fine-tuning using a Transformer-based Reranker.
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

# Differential Learning Rates
RERANKER_LR            = 1e-4
ENCODER_LR             = 1e-5
TAU                    = 0.1
BATCH_QUERIES          = 16
VAL_EVERY              = 100
SAVE_EVERY             = 100
CKPT_DIR               = "/ckpts"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    """
    A transformer-based reranker that models interactions between query and document embeddings
    using a transformer encoder with a [CLS] token for scoring.
    """
    def __init__(self, input_dim=512, num_layers=2, nhead=8):
        super().__init__()
        # Learnable [CLS] token and positional embeddings
        self.cls_token = nn.Parameter(torch.randn(1, 1, input_dim))
        self.pos_emb = nn.Parameter(torch.randn(3, input_dim))

        # Transformer encoder configuration
        transformer_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=nhead,
            dim_feedforward=input_dim * 4,
            dropout=0.1,
            activation=nn.GELU()
        )
        self.transformer_encoder = nn.TransformerEncoder(transformer_layer, num_layers=num_layers)

        # Linear layer for scoring
        self.scorer = nn.Linear(input_dim, 1)

    def forward(self, q_emb, d_embs):
        """
        Forward pass to compute relevance scores for documents given a query.

        Args:
            q_emb (torch.Tensor): Query embedding of shape [1, DIM]
            d_embs (torch.Tensor): Document embeddings of shape [N, DIM]

        Returns:
            torch.Tensor: Scores for each document of shape [N]
        """
        N = d_embs.size(0)

        # Expand [CLS] token and query embedding to match batch size N
        cls_token = self.cls_token.expand(N, 1, -1)  # [N, 1, DIM]
        q_emb_expanded = q_emb.expand(N, 1, -1)      # [N, 1, DIM]
        d_embs_expanded = d_embs.unsqueeze(1)        # [N, 1, DIM]

        # Construct sequences: [CLS, query_emb, doc_emb]
        sequences = torch.cat([cls_token, q_emb_expanded, d_embs_expanded], dim=1)  # [N, 3, DIM]

        # Add positional embeddings
        sequences = sequences + self.pos_emb  # [N, 3, DIM]

        # Transpose for transformer input: [seq_len, batch, dim]
        sequences = sequences.permute(1, 0, 2)  # [3, N, DIM]

        # Pass through transformer encoder
        output = self.transformer_encoder(sequences)  # [3, N, DIM]

        # Extract [CLS] token output
        cls_output = output[0, :, :]  # [N, DIM]

        # Compute scores
        scores = self.scorer(cls_output).squeeze(-1)  # [N]

        return scores

def get_embeddings(texts: List[str], max_len: int, encoder: TransformerForNextTokenPrediction) -> torch.Tensor:
    """Generates embeddings for a list of texts in a single batch without attention masks."""
    encoded_texts = [encode(text, max_len) for text in texts]
    max_seq_len = max(len(enc) for enc in encoded_texts)
    padded_texts = [enc + [0] * (max_seq_len - len(enc)) for enc in encoded_texts]  # Pad with 0s
    input_tensor = torch.tensor(padded_texts, dtype=torch.long, device=DEVICE)
    _, hidden_states = encoder(input_tensor)
    embeddings = hidden_states[:, 0, :]  # Assuming [CLS] token is at position 0
    return embeddings

def train_reranker(steps: int, ckpt_volume: modal.Volume, val_data_global: List = None):
    print("--- Starting Reranker Training with End-to-End Fine-tuning and Transformer-based Reranker ---")

    encoder = TransformerForNextTokenPrediction().to(DEVICE)
    encoder.load_state_dict(torch.load(PRETRAINED_MODEL_PATH, map_location=DEVICE))

    # Initialize the new transformer-based reranker
    reranker = Reranker(input_dim=DIM).to(DEVICE)

    print("Encoder and transformer-based reranker loaded.")

    optimizer = torch.optim.AdamW([
        {'params': encoder.parameters(), 'lr': ENCODER_LR},
        {'params': reranker.parameters(), 'lr': RERANKER_LR}
    ])

    if val_data_global is None: val_data_global = []

    for step in range(1, steps + 1):
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

            q_emb = get_embeddings([q_txt], MAX_LEN_QUERY, encoder)
            d_embs = get_embeddings(shuffled_docs, MAX_LEN_DOC, encoder)

            if d_embs.size(0) == 0: continue
            scores = reranker(q_emb, d_embs) / TAU
            loss = F.cross_entropy(scores.unsqueeze(0), torch.tensor([target_idx], device=DEVICE))

            loss.backward()
            tot_loss += loss.item()

        torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(reranker.parameters(), 1.0)
        optimizer.step()

        if step % 10 == 0:
            print(f"Step {step:>6} | Avg Loss: {tot_loss / BATCH_QUERIES:.4f}")

        if step % VAL_EVERY == 0:
            encoder.eval()
            reranker.eval()
            val_reciprocal_ranks = []

            with torch.no_grad():
                for vq_text, vdocs_all in val_data_global:
                    if not vdocs_all: continue
                    vdocs_all = vdocs_all[:TRUNCATE]

                    vq_emb = get_embeddings([vq_text], MAX_LEN_QUERY, encoder)
                    vd_embs = get_embeddings(vdocs_all, MAX_LEN_DOC, encoder)
                    v_scores = reranker(vq_emb, vd_embs)
                    rank = (v_scores.argsort(descending=True) == 0).nonzero(as_tuple=True)[0].item() + 1
                    val_reciprocal_ranks.append(1.0 / rank)

            if val_reciprocal_ranks:
                mrr_val = sum(val_reciprocal_ranks) / len(val_reciprocal_ranks)
                print(f"└─ Step {step} Validation MRR: {mrr_val:.4f} ({len(val_reciprocal_ranks)} queries)")

        if step % SAVE_EVERY == 0:
            print(f"--- Saving models to checkpoint volume ---")
            torch.save(reranker.state_dict(), f"{CKPT_DIR}/reranker_tx_step_{step}.pt")
            torch.save(encoder.state_dict(), f"{CKPT_DIR}/encoder_tx_step_{step}.pt")
            ckpt_volume.commit()

def reranker_modal_entrypoint(steps: int, ckpt_volume: modal.Volume):
    """The main entrypoint for the Modal remote training job."""
    validation_data = load_validation()
    train_reranker(steps=steps, ckpt_volume=ckpt_volume, val_data_global=validation_data)