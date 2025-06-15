#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Char-level reranker with cross-attention, negative sub-sampling, trained on next-token prediction.
"""

import time, random, itertools, os
from pathlib import Path
from typing import List, Tuple

import torch, torch.nn as nn, torch.nn.functional as F

# --------------------------- Hyper-parameters -------------------------- #
DIM, N_HEADS, N_LAYERS = 512, 8, 4      # model size
MAX_LEN                = 100            # truncate / pad length
VOCAB                  = 128            # ASCII
MAX_NEG                = 100            # 1 pos + 999 neg = 1000-way softmax
BATCH_QUERIES          = 32             # optimiser batch
TRUNCATE               = 100
LR                     = 1e-3           # AdamW
SAVE_EVERY             = 500            # steps
VAL_EVERY              = 100            # steps

DEVICE = torch.device(
    # "mps"  if torch.backends.mps.is_available() else
    "cuda" if torch.cuda.is_available()         else "cpu"
)

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
        self.head = nn.Linear(DIM, VOCAB)

    def forward(self, q_ids, q_len, d_ids, d_lens):
        N, Lq, Ld = d_ids.size(0), q_ids.size(1), d_ids.size(1)
        if N == 0:
            return torch.empty((0, Ld, VOCAB), device=d_ids.device)

        q_ids_expanded = q_ids.expand(N, -1)
        sep_token_id = VOCAB - 2

        # Concatenate query and documents for transformer input
        x = torch.cat([q_ids_expanded,
                       torch.full((N, 1), sep_token_id, device=DEVICE, dtype=torch.long),
                       d_ids], dim=1)

        pos_indices = torch.arange(x.size(1), device=DEVICE).expand(N, -1)
        h = self.emb(x) + self.pos(pos_indices)
        h = self.tf(h)

        # We only care about the logits for the document part
        doc_logits = self.head(h[:, Lq + 1:])
        return doc_logits

def train(steps=20_000, out="num_reranker_markov", val_data_global=None, ckpt_volume=None):
    mdl = InteractionRanker().to(DEVICE)
    total_params = sum(p.numel() for p in mdl.parameters())
    trainable_params = sum(p.numel() for p in mdl.parameters() if p.requires_grad)

    print(f"Model: InteractionRanker (Next-Token Prediction)")
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

        for i in range(BATCH_QUERIES):
            q_txt, docs_all = fetch_batch()
            if not docs_all: continue
            docs_all = docs_all[:TRUNCATE]
            pos_doc_text = docs_all[0]

            q_ids_train, q_len_train = to_tensor([q_txt])
            d_ids_train, d_lens_train = to_tensor([pos_doc_text])

            if d_ids_train.size(0) == 0: continue

            # Predict logits for each token in the document
            logits = mdl(q_ids_train, q_len_train, d_ids_train, d_lens_train)
            logits = logits.view(-1, VOCAB)

            # The target is the document itself, shifted by one token
            targets = d_ids_train.view(-1)

            loss = F.cross_entropy(logits, targets)
            loss.backward()
            tot_loss += loss.item()

        torch.nn.utils.clip_grad_norm_(mdl.parameters(), 5.0)
        opt.step()

        if step % 10 == 0:
            print(f"{step:>6} | loss {tot_loss/BATCH_QUERIES:.3f}")

        if step % VAL_EVERY == 1:
            mdl.eval()
            with torch.no_grad():
                val_reciprocal_ranks = []
                for vq_idx, (vq_text, vdocs_all) in enumerate(random.sample(val_data_global, len(val_data_global))):
                    if not vdocs_all: continue
                    vdocs_all = vdocs_all[:TRUNCATE]

                    q_ids_val, q_len_val = to_tensor([vq_text])
                    d_ids_val, d_lens_val = to_tensor(vdocs_all)

                    if d_ids_val.size(0) == 0: continue

                    doc_probs = []
                    for i in range(d_ids_val.size(0)):
                        doc_id = d_ids_val[i:i+1]
                        doc_len = d_lens_val[i:i+1]
                        logits = mdl(q_ids_val, q_len_val, doc_id, doc_len)
                        log_probs = F.log_softmax(logits, dim=-1)

                        # Gather the log probabilities of the true tokens
                        true_log_probs = log_probs.gather(2, doc_id.unsqueeze(2)).squeeze(2)

                        # Sum the log probabilities to get the document's log probability
                        doc_log_prob = true_log_probs.sum()
                        doc_probs.append(doc_log_prob.item())

                    # The first document is the positive one
                    true_positive_prob = doc_probs[0]
                    rank_of_true_positive = sorted(doc_probs, reverse=True).index(true_positive_prob) + 1
                    val_reciprocal_ranks.append(1.0 / rank_of_true_positive)

                if val_reciprocal_ranks:
                    mrr_val = sum(val_reciprocal_ranks) / len(val_reciprocal_ranks)
                    print(f"└─ val_mrr {mrr_val:.3f} | total {len(val_reciprocal_ranks)}")
                else:
                    print(f"└─ val metrics N/A (no validation data processed or VAL_DATA empty)")

        if step % SAVE_EVERY == 0:
            if not os.path.exists("/ckpts"):
                os.makedirs("/ckpts")
            torch.save({'step':step, 'model':mdl.state_dict(), 'opt':opt.state_dict()}, f"/ckpts/{out}x{step}.pt")
            if ckpt_volume is not None:
                ckpt_volume.commit()

def modal_entrypt(steps=20_000, ckpt_volume=None):
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
        print("  (Not enough data in so_ts_markov.txt to fetch example batches for testing print)")
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

    train(steps=steps, val_data_global=VAL_DATA, ckpt_volume=ckpt_volume)

if __name__ == "__main__":
    modal_entrypt()