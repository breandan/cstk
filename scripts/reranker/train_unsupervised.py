#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
STEP 1: Unsupervised Pre-training on Next-Token Prediction.

This script trains a transformer encoder on the self-supervised task of
predicting the next token in a sequence. The training data consists of
concatenated (query, positive_document) pairs from the dataset. A validation
loss is computed periodically using a separate validation set.

The goal is to produce a model that understands the structure and semantics
of the text, which can then be used to generate high-quality embeddings.
"""

import time, itertools
from pathlib import Path
from typing import List, Tuple

import torch, torch.nn as nn, torch.nn.functional as F

# --------------------------- Hyper-parameters -------------------------- #
DIM, N_HEADS, N_LAYERS = 512, 8, 4      # Model size
MAX_LEN_QUERY          = 100            # Max length for queries
MAX_LEN_DOC            = 110            # Max length for documents
VOCAB                  = 128            # ASCII
BATCH_SIZE             = 32             # Number of (q, d) pairs per batch
LR                     = 1e-4           # AdamW Learning Rate
SAVE_EVERY             = 100            # Save checkpoint and validate every N steps
LOG_EVERY              = 10             # Log loss every N steps

DEVICE = torch.device(
    "mps"  if torch.backends.mps.is_available() else
    "cuda" if torch.cuda.is_available() else "cpu")

MODEL_SAVE_PATH = Path("unsupervised_encoder_new.pt")

def fetch_positive_pairs(path: str = "so_ts_markov.txt"):
    """Yields (query, positive_document) pairs from the specified file."""
    def _reader():
        with Path(path).open(encoding="utf-8") as f:
            for gap, grp in itertools.groupby(f, key=lambda l: l.strip() == ""):
                if not gap:
                    lines = [l.rstrip("\n") for l in grp]
                    if lines and len(lines) > 1:
                        yield lines[0], lines[1] # query, first document
    batch_gen = _reader()
    while True:
        try:
            yield next(batch_gen)
        except StopIteration:
            batch_gen = _reader()
            yield next(batch_gen)

def encode(txt: str, max_len: int) -> List[int]:
    """Encodes a string into a list of token IDs, padded to max_len."""
    ids = [ord(c) % VOCAB for c in txt[:max_len]]
    return ids + [0] * (max_len - len(ids))

def get_batch(generator, batch_size: int, max_len_query: int, max_len_doc: int) -> torch.Tensor:
    """Fetches and encodes a batch of pairs from the generator."""
    batch_q_txt = []
    batch_d_txt = []
    for _ in range(batch_size):
        q, d = next(generator)
        batch_q_txt.append(q)
        batch_d_txt.append(d)
    q_encoded = [encode(q, max_len_query) for q in batch_q_txt]
    d_encoded = [encode(d, max_len_doc) for d in batch_d_txt]
    combined_ids = [q + d for q, d in zip(q_encoded, d_encoded)]
    return torch.tensor(combined_ids, dtype=torch.long, device=DEVICE)

class TransformerForNextTokenPrediction(nn.Module):
    """Transformer Encoder model for next-token prediction."""
    def __init__(self, vocab_size=VOCAB, dim=DIM, n_heads=N_HEADS, n_layers=N_LAYERS, max_len=MAX_LEN_QUERY + MAX_LEN_DOC):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, dim)
        self.pos_emb = nn.Embedding(max_len, dim)
        enc_layer = nn.TransformerEncoderLayer(dim, n_heads, 4*dim, activation="gelu", dropout=0.1, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(enc_layer, n_layers)
        self.head = nn.Linear(dim, vocab_size) # Predict the next token

    def forward(self, x):
        """
        Processes the input sequence to predict the next token at each position.
        Returns:
            - logits (Tensor): Logits for the next token prediction.
            - hidden_state (Tensor): The final hidden state of the transformer,
                                     used for generating embeddings later.
        """
        N, seq_len = x.shape
        pos = torch.arange(0, seq_len, dtype=torch.long, device=x.device).unsqueeze(0)
        h = self.tok_emb(x) + self.pos_emb(pos)
        hidden_state = self.transformer_encoder(h)
        logits = self.head(hidden_state)
        return logits, hidden_state

def train(steps=40_000):
    print(f"Starting unsupervised training on {DEVICE}...")
    model = TransformerForNextTokenPrediction().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()

    # Create separate generators for training and validation
    train_gen = fetch_positive_pairs("so_ts_markov.txt")
    val_gen = fetch_positive_pairs("so_vs_markov.txt")
    start_time = time.time()

    for step in range(1, steps + 1):
        # Training phase
        model.train()
        opt.zero_grad()
        input_tensor = get_batch(train_gen, BATCH_SIZE, MAX_LEN_QUERY, MAX_LEN_DOC)
        targets = input_tensor[:, 1:].contiguous()
        logits, _ = model(input_tensor[:, :-1])
        loss = loss_fn(logits.view(-1, VOCAB), targets.view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        # Log training loss
        if step % LOG_EVERY == 0:
            elapsed = time.time() - start_time
            print(f"Step {step:>6} | Training Loss: {loss.item():.4f} | Time: {elapsed:.2f}s")

        # Save model and compute validation loss
        if step % SAVE_EVERY == 0:
            print(f"--- Saving model checkpoint at step {step} to {MODEL_SAVE_PATH} ---")
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            # Validation phase
            model.eval()
            with torch.no_grad():
                val_input = get_batch(val_gen, BATCH_SIZE, MAX_LEN_QUERY, MAX_LEN_DOC)
                val_targets = val_input[:, 1:].contiguous()
                val_logits, _ = model(val_input[:, :-1])
                val_loss = loss_fn(val_logits.view(-1, VOCAB), val_targets.view(-1))
            print(f"Step {step:>6} | Validation Loss: {val_loss.item():.4f}")

    print("Unsupervised training complete.")
    torch.save(model.state_dict(), MODEL_SAVE_PATH)

if __name__ == "__main__":
    train()