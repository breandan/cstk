#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
STEP 1: Unsupervised Pre-training on Next-Token Prediction.

This script trains a transformer encoder on the self-supervised task of
predicting the next token in a sequence. The training data consists of
concatenated (query, positive_document) pairs from the dataset.

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
SAVE_EVERY             = 100            # Save checkpoint every N steps
LOG_EVERY              = 10             # Log loss every N steps

DEVICE = torch.device(
    "mps"  if torch.backends.mps.is_available() else
    "cuda" if torch.cuda.is_available() else "cpu")
MODEL_SAVE_PATH = Path("unsupervised_encoder.pt")

_batch_gen = None
def fetch_positive_pairs(path: str = "so_ts_markov.txt"):
    """Yields (query, positive_document) pairs from the training file."""
    global _batch_gen
    if _batch_gen is None:
        def _reader():
            with Path(path).open(encoding="utf-8") as f:
                for gap, grp in itertools.groupby(f, key=lambda l: l.strip() == ""):
                    if not gap:
                        lines = [l.rstrip("\n") for l in grp]
                        if lines and len(lines) > 1:
                            yield lines[0], lines[1] # query, first document
        _batch_gen = _reader()
    while True:
        try:
            yield next(_batch_gen)
        except StopIteration:
            _batch_gen = None
            _batch_gen = _reader()
            yield next(_batch_gen)


def encode(txt: str, max_len: int) -> List[int]:
    """Encodes a string into a list of token IDs, padded to max_len."""
    ids = [ord(c) % VOCAB for c in txt[:max_len]]
    return ids + [0] * (max_len - len(ids))

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

    pair_generator = fetch_positive_pairs()
    start_time = time.time()

    for step in range(1, steps + 1):
        model.train()
        opt.zero_grad()

        # 1. Fetch a batch of (query, positive_doc) pairs
        batch_q_txt = []
        batch_d_txt = []
        for _ in range(BATCH_SIZE):
            q, d = next(pair_generator)
            batch_q_txt.append(q)
            batch_d_txt.append(d)

        # 2. Encode and create concatenated input tensor
        q_encoded = [encode(q, MAX_LEN_QUERY) for q in batch_q_txt]
        d_encoded = [encode(d, MAX_LEN_DOC) for d in batch_d_txt]

        # Combine query and document for each pair
        combined_ids = [q + d for q, d in zip(q_encoded, d_encoded)]
        input_tensor = torch.tensor(combined_ids, dtype=torch.long, device=DEVICE)

        # 3. Define targets (the next token)
        # The target for input token at position `i` is the token at position `i+1`
        targets = input_tensor[:, 1:].contiguous()

        # 4. Forward pass
        # We only need to predict up to the second-to-last token
        logits, _ = model(input_tensor[:, :-1])

        # 5. Calculate loss
        # Reshape for CrossEntropyLoss: (Batch * SeqLen, VocabSize) vs (Batch * SeqLen)
        loss = loss_fn(logits.view(-1, VOCAB), targets.view(-1))

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step % LOG_EVERY == 0:
            elapsed = time.time() - start_time
            print(f"Step {step:>6} | Loss: {loss.item():.4f} | Time: {elapsed:.2f}s")

        if step % SAVE_EVERY == 0:
            print(f"--- Saving model checkpoint at step {step} to {MODEL_SAVE_PATH} ---")
            torch.save(model.state_dict(), MODEL_SAVE_PATH)

    print("Unsupervised training complete.")
    torch.save(model.state_dict(), MODEL_SAVE_PATH)

if __name__ == "__main__":
    train()