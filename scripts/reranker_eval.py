#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test set evaluator for the InteractionRanker model.

This script loads a pre-trained model checkpoint and evaluates its performance
on a test set, calculating metrics like Mean Reciprocal Rank (MRR) and
Accuracy@N.
"""

import itertools
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn
from tqdm import tqdm

# --------------------------- Model Hyper-parameters (Must match training) -------------------------- #
DIM, N_HEADS, N_LAYERS = 512, 8, 4      # model size
MAX_LEN                = 100            # truncate / pad length
VOCAB                  = 128            # ASCII
MAX_TOK = 1 + MAX_LEN + 1 + MAX_LEN # Max sequence length for position embeddings

# --- Configuration --- #
MODEL_PATH = "num_reranker.pt"
TEST_SET_PATH = "char_bifi_tst.txt"
BATCH_SIZE = 32 # Adjust based on your GPU memory

DEVICE = torch.device(
    "mps"  if torch.backends.mps.is_available() else
    "cuda" if torch.cuda.is_available()         else "cpu"
)

# --------------------------- Data & Model Utilities (Copied from training script) -------------------------- #

def load_test_data(path: str) -> List[Tuple[str, List[str]]]:
    """Loads the entire test set from a file with blank-line delimited blocks."""
    data = []
    print(f"Loading test data from '{path}'...")
    with Path(path).open(encoding="utf-8") as f:
        for gap, grp in itertools.groupby(f, key=lambda l: l.strip() == ""):
            if not gap:
                lines = [l.rstrip("\n") for l in grp]
                if lines:
                    # The first document is the positive one
                    data.append((lines[0], lines[1:]))
    print(f"Loaded {len(data):,} query-document groups.")
    return data

def encode(txt: str) -> Tuple[List[int], int]:
    """Converts a string to a list of token IDs and its length."""
    ids = [ord(c) % VOCAB for c in txt[:MAX_LEN]]
    return ids + [0]*(MAX_LEN-len(ids)), len(ids)

def to_tensor(strings: List[str]):
    """Converts a list of strings to padded tensors for IDs and lengths."""
    if not strings:
        empty_ids = torch.empty((0, MAX_LEN), dtype=torch.long, device=DEVICE)
        empty_lens = torch.empty((0,), dtype=torch.float, device=DEVICE)
        return empty_ids, empty_lens
    ids, lens = zip(*(encode(s) for s in strings))
    return (torch.tensor(ids,  dtype=torch.long,  device=DEVICE),
            torch.tensor(lens, dtype=torch.float, device=DEVICE))

class InteractionRanker(nn.Module):
    """The reranker model class. Definition must match the trained model."""
    def __init__(self):
        super().__init__()
        self.emb  = nn.Embedding(VOCAB, DIM)
        self.pos  = nn.Embedding(MAX_TOK, DIM)
        enc_layer = nn.TransformerEncoderLayer(DIM, N_HEADS, 4*DIM, activation="gelu", dropout=0.1, batch_first=True)
        self.tf   = nn.TransformerEncoder(enc_layer, N_LAYERS)
        self.head = nn.Linear(DIM, 1)

    def forward(self, q_ids, q_len, d_ids, d_lens):
        N, Lq, Ld = d_ids.size(0), q_ids.size(1), d_ids.size(1)
        if N == 0:
            return torch.empty((0,), device=d_ids.device)
        q_ids_expanded = q_ids.expand(N, -1); q_len_expanded = q_len.expand(N)
        # Note: In the original code, special tokens were defined relative to VOCAB size.
        # This implementation uses fixed large numbers to avoid collision with vocab.
        cls_token_id = VOCAB
        sep_token_id = VOCAB + 1

        # We need to expand the embedding layer to accommodate the new tokens
        if self.emb.num_embeddings < VOCAB + 2:
            # This should ideally be handled more gracefully
            # For this evaluation script, we assume it's handled or we adjust here.
            pass

        x = torch.cat([ torch.full((N,1), cls_token_id, device=DEVICE, dtype=torch.long), q_ids_expanded,
                        torch.full((N,1), sep_token_id, device=DEVICE, dtype=torch.long), d_ids
                        ], dim=1)
        pos_indices  = torch.arange(x.size(1), device=DEVICE).expand(N, -1)
        effective_lengths = (1 + q_len_expanded + 1 + d_lens).unsqueeze(1)
        mask = torch.arange(x.size(1), device=DEVICE).expand(N, -1) >= effective_lengths

        # Adjust embedding layer size if necessary for special tokens
        if self.emb.num_embeddings < VOCAB + 2:
            new_emb = nn.Embedding(VOCAB + 2, DIM).to(DEVICE)
            new_emb.weight.data[:VOCAB] = self.emb.weight.data
            self.emb = new_emb

        h = self.emb(x) + self.pos(pos_indices)
        h = self.tf(h, src_key_padding_mask=mask)
        return self.head(h[:,0]).squeeze(1)


# --------------------------- Main Evaluation Function -------------------------- #

def evaluate():
    """Main function to run the test set evaluation."""
    print(f"Using device: {DEVICE}")

    # 1. Initialize and load the model
    try:
        model = InteractionRanker().to(DEVICE)
        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
        model.load_state_dict(checkpoint['model'])
        model.eval()
        print(f"Successfully loaded model from '{MODEL_PATH}' (trained for {checkpoint.get('step', 'N/A')} steps).")
    except FileNotFoundError:
        print(f"Error: Model file not found at '{MODEL_PATH}'. Please ensure the path is correct.")
        return
    except Exception as e:
        print(f"An error occurred while loading the model: {e}")
        return

    # 2. Load test data
    test_data = load_test_data(TEST_SET_PATH)
    if not test_data:
        print("Test data is empty. Exiting.")
        return

    # 3. Run evaluation
    test_ranks = []
    test_reciprocal_ranks = []

    with torch.no_grad():
        # Wrap the loop with tqdm for a progress bar
        for query_text, doc_texts in tqdm(test_data, desc="Evaluating Test Set"):
            # The positive document is always the first one in the list
            positive_doc = query_text

            # The document list for ranking includes the positive and all negatives
            all_docs_to_rank = [positive_doc] + doc_texts

            # The true positive document is at index 0
            true_positive_idx = 0

            q_ids, q_len = to_tensor([query_text])

            # Process documents in batches to avoid OOM errors
            all_scores = []
            for i in range(0, len(all_docs_to_rank), BATCH_SIZE):
                batch_docs = all_docs_to_rank[i:i+BATCH_SIZE]
                d_ids, d_lens = to_tensor(batch_docs)
                if d_ids.size(0) == 0:
                    continue

                scores = model(q_ids, q_len, d_ids, d_lens)
                all_scores.append(scores)

            if not all_scores:
                continue

            # Combine scores from all batches
            all_scores = torch.cat(all_scores)

            # Find the rank of the true positive document
            sorted_indices = all_scores.argsort(descending=True)
            rank = (sorted_indices == true_positive_idx).nonzero(as_tuple=True)[0].item() + 1

            test_ranks.append(rank)
            test_reciprocal_ranks.append(1.0 / rank)

    # 4. Calculate and print metrics
    if not test_ranks:
        print("No results were generated. Cannot calculate metrics.")
        return

    total_queries = len(test_ranks)
    mrr = sum(test_reciprocal_ranks) / total_queries
    acc1 = sum(r == 1 for r in test_ranks) / total_queries
    acc10 = sum(r <= 10 for r in test_ranks) / total_queries
    acc100 = sum(r <= 100 for r in test_ranks) / total_queries

    print("\n" + "="*35)
    print("      Test Set Results")
    print("="*35)
    print(f"  Queries Evaluated: {total_queries:,}")
    print(f"  Mean Reciprocal Rank (MRR): {mrr:.4f}")
    print(f"  Accuracy@1:  {acc1:.4f}")
    print(f"  Accuracy@10: {acc10:.4f}")
    print(f"  Accuracy@100: {acc100:.4f}")
    print("="*35)


if __name__ == "__main__":
    evaluate()