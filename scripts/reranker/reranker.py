#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Char-level reranker with cross-attention, negative sub-sampling, temperature-scaled list-wise CE.
"""

import time, random, itertools, os
from pathlib import Path
from typing import List, Tuple

import torch, torch.nn as nn, torch.nn.functional as F

# --------------------------- Hyper-parameters -------------------------- #
DIM, N_HEADS, N_LAYERS = 512, 8, 4      # model size
MAX_LEN                = 100            # truncate / pad length
VOCAB                  = 128            # ASCII
MAX_NEG                = 1023
TAU                    = 0.1            # temperature
BATCH_QUERIES          = 1              # optimiser batch
INFERENCE_BATCH_SIZE   = 255
LR                     = 2e-3           # AdamW
SAVE_EVERY             = 100            # steps
VAL_EVERY              = 100            # steps

DEVICE = torch.device(
    # "mps"  if torch.backends.mps.is_available() else
    "cuda" if torch.cuda.is_available()         else "cpu"
)

_batch_gen = None

def _create_generator(path: str = "so_ts.txt"):
    """
    Creates a generator that yields (query, [docs…]) from a file.
    The first doc is positive. Subsequent docs are a random sample of negatives
    obtained via reservoir sampling to avoid loading all negatives into memory.
    """
    with Path(path).open(encoding="utf-8") as f:
        for gap, grp_iterator in itertools.groupby(f, key=lambda l: l.strip() == ""):
            if gap:
                continue

            try:
                query = next(grp_iterator).rstrip('\n')
                positive_doc = next(grp_iterator).rstrip('\n')
            except StopIteration:
                continue

            # --- Reservoir sampling for negative documents ---
            reservoir = []
            for i, line in enumerate(grp_iterator):
                doc = line.rstrip('\n')
                if not doc: continue # Skip empty lines within the block

                if i < MAX_NEG:
                    reservoir.append(doc)
                else:
                    j = random.randint(0, i)
                    if j < MAX_NEG:
                        reservoir[j] = doc

            # The full list of docs for this query is the positive + sampled negatives
            all_docs = [positive_doc] + reservoir
            yield query, all_docs

def fetch_batch(path: str = "so_ts.txt"):
    """Yield (query, [docs…]) from a file with blank-line delimited blocks."""
    global _batch_gen
    if _batch_gen is None:
        _batch_gen = _create_generator(path)
    try:
        return next(_batch_gen)
    except StopIteration:
        print("Data file exhausted. Restarting from the beginning.")
        _batch_gen = _create_generator(path)
        try:
            return next(_batch_gen)
        except StopIteration:
            print("ERROR: Training file appears to be empty. Cannot fetch a batch.")
            return "", []

def load_validation(path="char_bifi_vs.txt", cap=1000):
    data = []
    with Path(path).open(encoding="utf-8") as f:
        for gap, grp in itertools.groupby(f, key=lambda l: l.strip() == ""):
            if not gap and len(data) < cap:
                lines = [l.rstrip("\n") for l in grp]
                if lines: data.append((lines[0], lines[1:]))
    return data

def encode(txt: str) -> Tuple[List[int], int]:
    ids = [ord(c) % VOCAB for c in txt[:MAX_LEN]]
    return ids + [0]*(MAX_LEN-len(ids)), len(ids)

def to_tensor(strings: List[str]):
    if not strings:
        empty_ids = torch.empty((0, MAX_LEN), dtype=torch.long, device=DEVICE)
        empty_lens = torch.empty((0,), dtype=torch.float, device=DEVICE)
        return empty_ids, empty_lens
    ids, lens = zip(*(encode(s) for s in strings))
    return (torch.tensor(ids,  dtype=torch.long,  device=DEVICE),
            torch.tensor(lens, dtype=torch.float, device=DEVICE))

MAX_TOK = 1 + MAX_LEN + 1 + MAX_LEN
class InteractionRanker(nn.Module):
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
        cls_token_id = VOCAB - 1
        sep_token_id = VOCAB - 2
        x = torch.cat([ torch.full((N,1), cls_token_id, device=DEVICE, dtype=torch.long), q_ids_expanded,
                        torch.full((N,1), sep_token_id, device=DEVICE, dtype=torch.long), d_ids
                        ], dim=1)
        pos_indices  = torch.arange(x.size(1), device=DEVICE).expand(N, -1)
        effective_lengths = (1 + q_len_expanded + 1 + d_lens).unsqueeze(1)
        mask = torch.arange(x.size(1), device=DEVICE).expand(N, -1) >= effective_lengths
        h = self.emb(x) + self.pos(pos_indices)
        h = self.tf(h, src_key_padding_mask=mask)
        return self.head(h[:,0]).squeeze(1)

def modal_entrypt(ckpt_volume=None):
    global _batch_gen
    torch.manual_seed(0)
    random.seed(0)

    VAL_DATA = load_validation()

    print("Testing fetch_batch with actual text (first few items):")
    _fetch_batch_gen_orig = _batch_gen
    _batch_gen = None
    try:
        for _i in range(min(3, BATCH_QUERIES if BATCH_QUERIES > 0 else 3)):
            q_example, d_example_list = fetch_batch()
            print(f"  Batch {_i+1}: Query='{q_example[:70]}...', Docs (first doc, {len(d_example_list)} total)='{d_example_list[0][:70]}...'")
            if d_example_list:
                print(f"    Expected Positive (d_example_list[0]): '{d_example_list[0][:70]}...' (should match query)")
    except StopIteration:
        print("  (Not enough data in char_bifi_ts.txt to fetch example batches for testing print)")
    except Exception as e:
        print(f"Error during fetch_batch test: {e}")
    finally:
        _batch_gen = _fetch_batch_gen_orig
    print("-" * 20)

    print(f"Validation data examples (first {min(3, len(VAL_DATA))} items loaded):")
    for i in range(min(10, len(VAL_DATA))):
        vq, vd = VAL_DATA[i]
        print(f"  Val Ex {i+1}: Query='{vq[:70]}...', Docs (first doc, {len(vd)} total)='{vd[0][:70]}...'")
        if vd:
            print(f"    Expected Positive (vd[0]): '{vd[0][:70]}...' (should match query)")
    print("-" * 20)

    train(val_data_global=VAL_DATA, ckpt_volume=ckpt_volume)

def train(steps=20_000, out="num_reranker", val_data_global=None, ckpt_volume=None):
    mdl = InteractionRanker().to(DEVICE)
    total_params = sum(p.numel() for p in mdl.parameters())
    trainable_params = sum(p.numel() for p in mdl.parameters() if p.requires_grad)

    print(f"Model: InteractionRanker")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    opt = torch.optim.AdamW(mdl.parameters(), lr=LR, weight_decay=1e-4)

    if val_data_global is None:
        print("Warning: VAL_DATA not passed to train function. Validation will be empty.")
        val_data_global = []

    for step in range(1, steps+1):
        mdl.train()
        opt.zero_grad(); tot_loss = 0.0
        current_batch_reciprocal_ranks = []

        for _ in range(BATCH_QUERIES):
            q_txt, docs_all = fetch_batch()
            if not docs_all: continue
            pos_doc_text = docs_all[0]
            neg_all_texts = docs_all[1:]
            neg_sampled_texts = random.sample(neg_all_texts, min(len(neg_all_texts), MAX_NEG)) \
                if neg_all_texts else []

            current_docs_texts = [pos_doc_text] + neg_sampled_texts
            perm = torch.randperm(len(current_docs_texts))
            shuffled_docs_texts = [current_docs_texts[i] for i in perm]
            target_idx = (perm == 0).nonzero(as_tuple=False).item()

            q_ids_train, q_len_train = to_tensor([q_txt])

            # --- START: Micro-batching for training scores ---
            all_scores = []
            for i in range(0, len(shuffled_docs_texts), INFERENCE_BATCH_SIZE):
                batch_docs_texts = shuffled_docs_texts[i:i+INFERENCE_BATCH_SIZE]
                if not batch_docs_texts: continue

                d_ids_train, d_lens_train = to_tensor(batch_docs_texts)
                if d_ids_train.size(0) == 0: continue

                with torch.no_grad(): # No grad needed for this part of training forward pass if done chunk by chunk before loss
                    scores_batch = mdl(q_ids_train, q_len_train, d_ids_train, d_lens_train)
                    all_scores.append(scores_batch)

            if not all_scores: continue
            scores_train = torch.cat(all_scores) / TAU
            # --- END: Micro-batching for training scores ---

            scores_train = scores_train.detach().requires_grad_() # Re-attach to graph for loss calculation

            loss = F.cross_entropy(scores_train.unsqueeze(0), torch.tensor([target_idx], device=DEVICE))
            loss.backward(); tot_loss += loss.item()

            with torch.no_grad():
                rank_of_positive_train = (scores_train.argsort(descending=True) == target_idx).nonzero(as_tuple=True)[0].item() + 1
                current_batch_reciprocal_ranks.append(1.0 / rank_of_positive_train)

        if current_batch_reciprocal_ranks:
            torch.nn.utils.clip_grad_norm_(mdl.parameters(), 5.0)
            opt.step()
            if step % 10 == 0:
                mrr_train_batch = sum(current_batch_reciprocal_ranks) / len(current_batch_reciprocal_ranks)
                print(f"{step:>6} | loss {tot_loss/BATCH_QUERIES:.3f} | mrr {mrr_train_batch:.3f}")
        elif step % 10 == 0:
            print(f"{step:>6} | loss {tot_loss/BATCH_QUERIES:.3f} | mrr N/A (no ranks this batch)")

        if step > 0 and step % VAL_EVERY == 0: # Adjusted to not run on step 1
            mdl.eval()
            with torch.no_grad():
                val_ranks, val_reciprocal_ranks = [], []
                for vq_text, vdocs_texts in val_data_global:
                    if not vdocs_texts: continue
                    q_ids_val, q_len_val = to_tensor([vq_text])

                    # --- START: Micro-batching for validation scores ---
                    all_val_scores = []
                    for i in range(0, len(vdocs_texts), INFERENCE_BATCH_SIZE):
                        batch_docs_texts = vdocs_texts[i:i+INFERENCE_BATCH_SIZE]
                        if not batch_docs_texts: continue

                        d_ids_val, d_lens_val = to_tensor(batch_docs_texts)
                        if d_ids_val.size(0) == 0: continue

                        scores_batch = mdl(q_ids_val, q_len_val, d_ids_val, d_lens_val)
                        all_val_scores.append(scores_batch)

                    if not all_val_scores: continue
                    scores_val = torch.cat(all_val_scores)
                    # --- END: Micro-batching for validation scores ---

                    true_positive_idx_in_vdocs = 0
                    rank_of_true_positive = (scores_val.argsort(descending=True) == true_positive_idx_in_vdocs).nonzero(as_tuple=True)[0].item() + 1
                    val_ranks.append(rank_of_true_positive)
                    val_reciprocal_ranks.append(1.0 / rank_of_true_positive)

            if val_ranks:
                acc1 = sum(r == 1 for r in val_ranks) / len(val_ranks)
                acc10 = sum(r <= 10 for r in val_ranks) / len(val_ranks)
                mrr_val = sum(val_reciprocal_ranks) / len(val_reciprocal_ranks)
                print(f"└─ val@1 {acc1:.3f} | val@10 {acc10:.3f} | val_mrr {mrr_val:.3f} | total {len(val_ranks)}")
            else:
                print(f"└─ val metrics N/A")

        if step % SAVE_EVERY == 0:
            torch.save({'step':step, 'model':mdl.state_dict(), 'opt':opt.state_dict()}, f"/ckpts/{out}x{step}.pt")
            if ckpt_volume is not None:
                ckpt_volume.commit()

if __name__ == "__main__":
    torch.manual_seed(0)
    random.seed(0)

    VAL_DATA = load_validation()

    print("Testing fetch_batch with actual text (first few items):")
    _fetch_batch_gen_orig = _batch_gen
    _batch_gen = None
    try:
        for _i in range(min(3, BATCH_QUERIES if BATCH_QUERIES > 0 else 3)):
            q_example, d_example_list = fetch_batch()
            print(f"  Batch {_i+1}: Query='{q_example[:70]}...', Docs (first doc, {len(d_example_list)} total)='{d_example_list[0][:70]}...'")
            if d_example_list:
                print(f"    Expected Positive (d_example_list[0]): '{d_example_list[0][:70]}...' (should match query)")
    except StopIteration:
        print("  (Not enough data in char_bifi_ts.txt to fetch example batches for testing print)")
    except Exception as e:
        print(f"Error during fetch_batch test: {e}")
    finally:
        _batch_gen = _fetch_batch_gen_orig
    print("-" * 20)

    print(f"Validation data examples (first {min(3, len(VAL_DATA))} items loaded):")
    for i in range(min(10, len(VAL_DATA))):
        vq, vd = VAL_DATA[i]
        print(f"  Val Ex {i+1}: Query='{vq[:70]}...', Docs (first doc, {len(vd)} total)='{vd[0][:70]}...'")
        if vd:
            print(f"    Expected Positive (vd[0]): '{vd[0][:70]}...' (should match query)")
    print("-" * 20)

    train(val_data_global=VAL_DATA)