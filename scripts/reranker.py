#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tiny char-level reranker.
Forward =  mean( Emb(q) ) · mean( Emb(d) )      # dot product
Loss    =  cross-entropy over the score list    # listwise
"""

import os, time, json, requests, torch, torch.nn as nn, torch.nn.functional as F
from typing import List, Tuple

# --------------------------- Hyper-parameters --------------------------- #
DIM          = 64          # embedding size
MAX_LEN      = 100         # truncate / pad length (query & docs)
VOCAB        = 128         # ASCII subset
LR           = 3e-3
SAVE_EVERY   = 100         # steps
DEVICE       = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --------------------------- Utilities ---------------------------------- #
def fetch_batch(url: str = 'http://localhost:8001/fetch') -> Tuple[str, List[str]]:
    """Return (query, [doc₀, doc₁, …])  doc₀ is the positive example."""
    txt = requests.get(url, timeout=100).text.strip().splitlines()
    return txt[0], txt[1:]

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
        enc = nn.TransformerEncoderLayer(DIM, nhead=4, dim_feedforward=4*DIM, activation='gelu', batch_first=True)
        self.tf = nn.TransformerEncoder(enc, num_layers=1)

    def mean_embed(self, ids, lens):
        B, L = ids.shape
        pos_ids = torch.arange(L, device=ids.device).expand(B, L)
        x = self.emb(ids) + self.pos(pos_ids)           # [B, L, D]

        pad_mask = torch.arange(L, device=ids.device).expand(B, L) >= lens.unsqueeze(1)
        x = self.tf(x, src_key_padding_mask=pad_mask)   # attention ignores pads

        x = x.masked_fill(pad_mask.unsqueeze(2), 0).sum(1) / lens.unsqueeze(1)
        return x * DIM**-0.5

    def forward(self, q_ids, q_len, d_ids, d_lens) -> torch.Tensor:
        q = self.mean_embed(q_ids, q_len)               # [1×D]
        d = self.mean_embed(d_ids, d_lens)              # [N×D]
        return torch.matmul(d, q.T).squeeze(1)          # [N] scores

# --------------------------- Training loop ------------------------------ #
def train(steps=10_000, out='char_ranker.pt') -> None:
    mdl = CharRanker().to(DEVICE)
    opt = torch.optim.AdamW(mdl.parameters(), lr=LR)
    total_time, num_timed = 0.0, 0

    for step in range(1, steps+1):
        t0 = time.perf_counter()
        # ---- data ------------------------------------------------------ #
        q_txt, docs = fetch_batch()
        perm = torch.randperm(len(docs))
        docs_shuf = [docs[i] for i in perm]                   # shuffle
        tgt = (perm == 0).nonzero(as_tuple=False)[0,0].item() # new idx of pos

        q_ids, q_len  = to_tensor([q_txt])                    #  shape → [1×L], [1]
        d_ids, d_lens = to_tensor(docs_shuf)                  #  shape → [N×L], [N]

        # ---- forward --------------------------------------------------- #
        scores = mdl(q_ids, q_len, d_ids, d_lens)             # [N]
        if DEVICE.type == 'cuda':
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        total_time += dt; num_timed += 1
        avg_ms = 1000 * total_time / num_timed

        # ---- smooth rank & loss ---------------------------------------- #
        diff  = scores - scores[tgt]                          # [N]
        mask  = torch.ones_like(scores, dtype=torch.bool); mask[tgt] = False
        soft_rank = 1 + torch.sigmoid(diff[mask]).sum()
        loss = soft_rank.pow(2)                               # squared rank

        opt.zero_grad(); loss.backward(); opt.step()

        # ---- metrics --------------------------------------------------- #
        with torch.no_grad():
            hard_rank = (scores.argsort(descending=True)==tgt).nonzero()[0,0]+1
        if step % 10 == 0:                          # print every 10 steps
            print(f'{step:>6} | loss {loss.item():.3f} | '
                  f'hard-rank {hard_rank:<4} | '
                  f'avg_fwd {avg_ms:6.2f} ms')

        # ---- checkpoint ------------------------------------------------- #
        if step % SAVE_EVERY == 0:
            torch.save({'step': step, 'model': mdl.state_dict(), 'opt': opt.state_dict()}, out)

if __name__ == '__main__':
    torch.manual_seed(0)
    train()