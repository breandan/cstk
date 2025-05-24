#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tiny char-level reranker.
Forward =  mean( Emb(q) ) · mean( Emb(d) )      # dot product
Loss    =  cross-entropy over the score list    # listwise
"""

import os, time, itertools, json, requests, torch, torch.nn as nn, torch.nn.functional as F
from typing import List, Tuple
from pathlib import Path

_batch_gen = None

# --------------------------- Hyper-parameters --------------------------- #
DIM          = 128          # embedding size
N_LAYERS     = 2
N_HEADS      = 8
MAX_LEN      = 100         # truncate / pad length (query & docs)
VOCAB        = 128         # ASCII subset
LR           = 3e-3
SAVE_EVERY   = 100         # steps
BATCH_SIZE   = 8
DEVICE       = DEVICE = torch.device(
  "mps"  if torch.backends.mps.is_available() else
  "cuda" if torch.cuda.is_available()         else "cpu"
)

# --------------------------- Utilities ---------------------------------- #
def fetch_batch(path: str = 'char_bifi_ts.txt'):
    """
    Read from a single text file where blank lines separate batches.
    In each batch the first non-empty line is the query; the rest are docs.
    When the file ends we rewind so training can run indefinitely.
    """
    global _batch_gen

    if _batch_gen is None:
        def _reader():
            with Path(path).open(encoding='utf-8') as f:
                for is_gap, group in itertools.groupby(f, key=lambda l: l.strip() == ''):
                    if not is_gap:
                        lines = [l.rstrip('\n') for l in group]
                        if lines:
                            yield lines[0], lines[1:]
        _batch_gen = _reader()

    try:
        return next(_batch_gen)
    except StopIteration:
        _batch_gen = None
        return fetch_batch(path)

def load_validation(path: str = 'char_bifi_vs.txt'):
    data = []
    with Path(path).open(encoding="utf-8") as f:
        for is_gap, grp in itertools.groupby(f, key=lambda l: l.strip() == ""):
            if not is_gap:
                lines = [l.rstrip("\n") for l in grp]
                if lines:
                    data.append((lines[0], lines[1:]))       # (query, docs)
    return data

VAL_DATA = load_validation()

def encode(txt: str) -> Tuple[List[int], int]:
    """ASCII-encode and pad; returns ([ids...], valid_len)."""
    ids = [ord(c) % VOCAB for c in txt[:MAX_LEN]]
    pad = [0] * (MAX_LEN - len(ids))
    return ids + pad, len(ids)

def to_tensor(strings: List[str]) -> Tuple[torch.LongTensor, torch.Tensor]:
    """Vectorise a list of strings ⟶ (ids [N×L], lengths [N])."""
    ids, lens = zip(*(encode(s) for s in strings))
    return (torch.tensor(ids,  dtype=torch.long,  device=DEVICE),
            torch.tensor(lens, dtype=torch.float, device=DEVICE))

# --------------------------- Model -------------------------------------- #
class CharRanker(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(VOCAB, DIM)
        self.pos = nn.Embedding(MAX_LEN, DIM)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=DIM,
            nhead=N_HEADS,
            dim_feedforward=4*DIM,
            dropout=0.1,
            activation='gelu',
            batch_first=True)
        self.tf = nn.TransformerEncoder(enc_layer, num_layers=N_LAYERS)

        self.proj = nn.Linear(DIM, DIM, bias=False)

    def encode(self, ids, lens):
        B, L = ids.shape
        pos = torch.arange(L, device=ids.device).expand(B, L)
        x = self.emb(ids) + self.pos(pos)

        pad = torch.arange(L, device=ids.device).expand(B, L) >= lens.unsqueeze(1)
        x = self.tf(x, src_key_padding_mask=pad)

        x = x.masked_fill(pad.unsqueeze(2), 0).sum(1) / lens.unsqueeze(1)
        return F.normalize(self.proj(x), dim=1)        # [B, DIM]

    def forward(self, q_ids, q_len, d_ids, d_lens):
        q = self.encode(q_ids, q_len)                  # [1, D]
        d = self.encode(d_ids, d_lens)                 # [N, D]
        return (d @ q.T).squeeze(1)                    # cosine-like scores

# --------------------------- Training loop ------------------------------ #
def train(steps=14_000, out='char_ranker.pt') -> None:
    mdl = CharRanker().to(DEVICE)
    opt = torch.optim.SGD(mdl.parameters(), lr=LR, momentum=0.9, weight_decay=1e-4)

    for step in range(1, steps + 1):
        opt.zero_grad()
        total_loss, hard_ranks = 0.0, []

        for _ in range(BATCH_SIZE):
            q_txt, docs = fetch_batch()

            perm = torch.randperm(len(docs))
            docs = [docs[i] for i in perm]
            tgt  = (perm == 0).nonzero(as_tuple=False)[0, 0].item()

            q_ids, q_len  = to_tensor([q_txt])
            d_ids, d_lens = to_tensor(docs)

            scores = mdl(q_ids, q_len, d_ids, d_lens)

            target = torch.tensor([tgt], device=DEVICE)
            loss   = F.cross_entropy(scores.unsqueeze(0), target)   # list-wise CE
            loss.backward()
            total_loss += loss.item()

            with torch.no_grad():
                rk = (scores.argsort(descending=True) == tgt).nonzero()[0, 0].item() + 1
                hard_ranks.append(rk)

        opt.step()

        # ---- metrics --------------------------------------------------- #
        if step % 10 == 0:
            avg_loss = total_loss / BATCH_SIZE
            avg_rank = sum(hard_ranks) / len(hard_ranks)
            print(f"{step:>6} | loss {avg_loss:.3f} | mean-rank {avg_rank:.1f}")

        if step % 100 == 0:
            with torch.no_grad():
                ranks = []
                sample_size = 0
                for vq, vdocs in VAL_DATA:
                    q_ids, q_len  = to_tensor([vq])
                    d_ids, d_lens = to_tensor(vdocs)
                    sc  = mdl(q_ids, q_len, d_ids, d_lens)
                    rnk = (sc.argsort(descending=True) == 0).nonzero(as_tuple=False)[0, 0].item() + 1
                    ranks.append(rnk)
                    sample_size += 1
                    if sample_size >= 100: break
            acc1   = sum(r == 1 for r in ranks) / len(ranks)
            acc10  = sum(r <= 10 for r in ranks) / len(ranks)
            acc100 = sum(r <= 100 for r in ranks) / len(ranks)
            mean_r = sum(ranks) / len(ranks)
            print(f"└─ val@1 {acc1:.3f} | val@10 {acc10:.3f} | val@100 {acc100:.3f} | mean_rank {mean_r:.1f}")

        # ---- checkpoint ------------------------------------------------- #
        if step % SAVE_EVERY == 0:
            torch.save({'step': step, 'model': mdl.state_dict(), 'opt': opt.state_dict()}, out)

if __name__ == '__main__':
    torch.manual_seed(0)
    train()