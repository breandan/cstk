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
DIM, N_HEADS, N_LAYERS = 256, 8, 2      # model size -- safe <60 ms
MAX_LEN                = 100            # truncate / pad length
VOCAB                  = 128            # ASCII
MAX_NEG                = 255            # 1 pos + 127 neg = 128-way softmax
TAU                    = 0.1            # temperature
BATCH_QUERIES          = 16              # optimiser batch
LR                     = 2e-3           # AdamW
SAVE_EVERY             = 500            # steps
VAL_EVERY              = 100            # steps

DEVICE = torch.device(
    "mps"  if torch.backends.mps.is_available() else
    "cuda" if torch.cuda.is_available()         else "cpu"
)

if DEVICE.type == "mps":
    print(
        "MPS device detected. Setting PYTORCH_ENABLE_MPS_FALLBACK=1 "
        "to potentially mitigate NotImplementedError for certain Transformer ops. "
        "This may result in slower execution for an unsupported op as it will fall back to CPU."
    )
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

_batch_gen = None
def fetch_batch(path: str = "char_bifi_ts.txt"):
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

def load_validation(path="char_bifi_vs.txt", cap=100):
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

def pairwise_expand(q_ids, q_lens, d_ids, d_lens):
    """
    Broadcast queries and docs so that we can score *every* (qᵢ,dⱼ) pair in
    a single forward pass.

    Returns tensors of shape ((B², L), ...).
    """
    B, L = q_ids.size()
    q_ids_rep = q_ids.unsqueeze(1).expand(B, B, L).reshape(-1, L)
    d_ids_rep = d_ids.unsqueeze(0).expand(B, B, L).reshape(-1, L)

    q_lens_rep = q_lens.unsqueeze(1).expand(B, B).reshape(-1)
    d_lens_rep = d_lens.unsqueeze(0).expand(B, B).reshape(-1)
    return q_ids_rep, q_lens_rep, d_ids_rep, d_lens_rep


def scores_matrix(mdl, q_ids, q_lens, d_ids, d_lens):
    """
    Returns a B × B matrix where entry (i,j) is the (scaled) score for
    query i vs. doc j.
    """
    q_rep, ql_rep, d_rep, dl_rep = pairwise_expand(q_ids, q_lens, d_ids, d_lens)
    flat_scores = mdl(q_rep, ql_rep, d_rep, dl_rep) / TAU        # (B²,)
    B = q_ids.size(0)
    return flat_scores.view(B, B)                                # (B,B)

def train(steps=20_000, out="num_reranker.pt", val_data_global=None, batch_queries=BATCH_QUERIES):
    if DEVICE.type == "cuda":
        mdl = torch.compile(InteractionRanker().to(DEVICE), mode="max-autotune")
    else:
        mdl = InteractionRanker().to(DEVICE)

    opt = torch.optim.AdamW(mdl.parameters(), lr=LR, weight_decay=1e-4)

    if val_data_global is None:
        print("Warning: VAL_DATA not passed to train function. Validation will be empty.")
        val_data_global = []

    for step in range(1, steps + 1):
        mdl.train()
        opt.zero_grad(set_to_none=True)

        q_txts, pos_txts = [], []
        for _ in range(batch_queries):
            q, docs = fetch_batch()
            if not docs:
                continue
            q_txts.append(q)
            pos_txts.append(docs[0])

        if not q_txts:
            continue

        q_ids, q_lens = to_tensor(q_txts)         # (B,L)
        d_ids, d_lens = to_tensor(pos_txts)       # (B,L)

        S = scores_matrix(mdl, q_ids, q_lens, d_ids, d_lens) # (B,B)

        targets = torch.arange(S.size(0), device=DEVICE)
        loss = F.cross_entropy(S, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(mdl.parameters(), 5.0)
        opt.step()

        if step % 10 == 0:
            with torch.no_grad():
                ranks = S.argsort(dim=1, descending=True)
                gold_ranks = ((ranks == targets.unsqueeze(1)).nonzero(as_tuple=False)[:, 1] + 1)
                mrr = (1.0 / gold_ranks.float()).mean().item()
            print(f"{step:>6} | loss {loss.item():.3f} | mrr {mrr:.3f}")

        if step % VAL_EVERY == 0:
            mdl.eval()
            with torch.no_grad():
                val_ranks = [] # For Acc@N
                val_reciprocal_ranks = [] # For MRR
                printed_val_examples = 0
                num_examples_to_print = 3

                for vq_idx, (vq_text, vdocs_texts) in enumerate(val_data_global):
                    if not vdocs_texts: continue
                    q_ids_val, q_len_val  = to_tensor([vq_text])
                    d_ids_val, d_lens_val = to_tensor(vdocs_texts)
                    if d_ids_val.size(0) == 0: continue

                    # For inspecting raw scores vs scaled scores
                    raw_scores_val = mdl(q_ids_val, q_len_val, d_ids_val, d_lens_val)
                    if raw_scores_val.numel() == 0: continue
                    scores_val = raw_scores_val / TAU

                    true_positive_idx_in_vdocs = 0
                    sorted_indices_val = scores_val.argsort(descending=True)
                    rank_of_true_positive = (sorted_indices_val == true_positive_idx_in_vdocs).nonzero(as_tuple=True)[0].item() + 1
                    val_ranks.append(rank_of_true_positive)
                    val_reciprocal_ranks.append(1.0 / rank_of_true_positive)

                    if printed_val_examples < num_examples_to_print and val_data_global:
                        print(f"  --- Validation Example {vq_idx+1}/{len(val_data_global)} ---")
                        print(f"    Query: '{vq_text[:150]}{'...' if len(vq_text) > 150 else ''}'")
                        positive_doc_text_val = vdocs_texts[true_positive_idx_in_vdocs]
                        print(f"    True Positive (Rank {rank_of_true_positive}): '{positive_doc_text_val[:150]}{'...' if len(positive_doc_text_val) > 150 else ''}'")
                        print(f"    Top 5 Ranked Documents by Model:")
                        top_k_val = min(5, len(vdocs_texts))
                        for i in range(top_k_val):
                            doc_idx_in_vdocs = sorted_indices_val[i].item()
                            retrieved_doc_text = vdocs_texts[doc_idx_in_vdocs]
                            retrieved_raw_score = raw_scores_val[doc_idx_in_vdocs].item() # Raw score
                            retrieved_scaled_score = scores_val[doc_idx_in_vdocs].item() # Scaled score
                            is_true_positive_marker = " (*True Positive*)" if doc_idx_in_vdocs == true_positive_idx_in_vdocs else ""
                            print(f"      {i+1}. (Raw: {retrieved_raw_score:.3f}, Scaled: {retrieved_scaled_score:.3f}) '{retrieved_doc_text[:100]}{'...' if len(retrieved_doc_text) > 100 else ''}'{is_true_positive_marker}")
                        printed_val_examples += 1
                        if printed_val_examples == num_examples_to_print and vq_idx < len(val_data_global) -1 : print("    ---")

            if val_ranks:
                acc1   = sum(r == 1 for r in val_ranks) / len(val_ranks)
                acc10  = sum(r <= 10 for r in val_ranks) / len(val_ranks)
                acc100 = sum(r <= 100 for r in val_ranks) / len(val_ranks)
                mrr_val = sum(val_reciprocal_ranks) / len(val_reciprocal_ranks)
                print(f"└─ val@1 {acc1:.3f} | val@10 {acc10:.3f} | val@100 {acc100:.3f} | val_mrr {mrr_val:.3f}")
            else:
                print(f"└─ val metrics N/A (no validation data processed or VAL_DATA empty)")

        if step % SAVE_EVERY == 0:
            torch.save({'step':step, 'model':mdl.state_dict(), 'opt':opt.state_dict()}, out)

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