#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Char-level reranker with cross-attention, negative sub-sampling, temperature-scaled list-wise CE.
"""

import time, random, itertools
from pathlib import Path
from typing import List, Tuple

import torch, torch.nn as nn, torch.nn.functional as F

# --------------------------- Hyper-parameters -------------------------- #
DIM, N_HEADS, N_LAYERS = 128, 8, 2      # model size -- safe <60 ms
MAX_LEN                = 100            # truncate / pad length
VOCAB                  = 128            # ASCII
MAX_NEG                = 127            # 1 pos + 127 neg = 128-way softmax
TAU                    = 0.1            # temperature
BATCH_QUERIES          = 8              # optimiser batch
LR                     = 2e-3           # AdamW
SAVE_EVERY             = 500            # steps
VAL_EVERY              = 100            # steps

DEVICE = torch.device(
    "mps"  if torch.backends.mps.is_available() else
    "cuda" if torch.cuda.is_available()         else "cpu"
)

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
VAL_DATA = load_validation()

def encode(txt: str) -> Tuple[List[int], int]:
    ids = [ord(c) % VOCAB for c in txt[:MAX_LEN]]
    return ids + [0]*(MAX_LEN-len(ids)), len(ids)

def to_tensor(strings: List[str]):
    ids, lens = zip(*(encode(s) for s in strings))
    return (torch.tensor(ids,  dtype=torch.long,  device=DEVICE),
            torch.tensor(lens, dtype=torch.float, device=DEVICE))

MAX_TOK = 1 + MAX_LEN + 1 + MAX_LEN      # [CLS] q [SEP] d
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
        q_ids = q_ids.expand(N, -1); q_len = q_len.expand(N)

        x = torch.cat([ torch.full((N,1), VOCAB-1, device=DEVICE), q_ids,
                        torch.full((N,1), VOCAB-2, device=DEVICE), d_ids ], dim=1)

        pos  = torch.arange(x.size(1), device=DEVICE).expand(N, -1)
        mask = torch.arange(x.size(1), device=DEVICE) >= (1+q_len+1).unsqueeze(1) + d_lens.unsqueeze(1)

        h = self.emb(x) + self.pos(pos)
        h = self.tf(h, src_key_padding_mask=mask)
        return self.head(h[:,0]).squeeze(1)               # logits [N]

def train(steps=20_000, out="reranker.pt"):
    mdl = InteractionRanker().to(DEVICE)
    opt = torch.optim.AdamW(mdl.parameters(), lr=LR, weight_decay=1e-4)

    for step in range(1, steps+1):
        opt.zero_grad(); tot_loss = 0.0; ranks = []

        for _ in range(BATCH_QUERIES):
            q_txt, docs_all = fetch_batch()

            pos, neg_all   = docs_all[0], docs_all[1:]
            neg            = random.sample(neg_all, min(len(neg_all), MAX_NEG))
            docs           = [pos] + neg                      # pos at 0

            perm = torch.randperm(len(docs))                  # shuffle
            docs = [docs[i] for i in perm]
            tgt  = (perm == 0).nonzero(as_tuple=False)[0,0].item()

            q_ids, q_len  = to_tensor([q_txt])
            d_ids, d_lens = to_tensor(docs)
            scores        = mdl(q_ids, q_len, d_ids, d_lens) / TAU

            loss = F.cross_entropy(scores.unsqueeze(0), torch.tensor([tgt], device=DEVICE))
            loss.backward(); tot_loss += loss.item()

            with torch.no_grad():
                rk = (scores.argsort(descending=True)==tgt).nonzero()[0,0].item()+1
                ranks.append(rk)

        torch.nn.utils.clip_grad_norm_(mdl.parameters(), 5.0)
        opt.step()

        if step % 10 == 0:
            print(f"{step:>6} | loss {tot_loss/BATCH_QUERIES:.3f} | mean-rank {sum(ranks)/len(ranks):.1f}")

        if step % VAL_EVERY == 0:
            with torch.no_grad():
                vr = []
                for vq, vdocs in VAL_DATA:
                    q_ids, q_len  = to_tensor([vq])
                    d_ids, d_lens = to_tensor(vdocs)
                    sc = mdl(q_ids,q_len,d_ids,d_lens)/TAU
                    vr.append((sc.argsort(descending=True)==0).nonzero()[0,0].item()+1)
            acc1   = sum(r == 1 for r in vr) / len(vr)
            acc10  = sum(r <= 10 for r in vr) / len(vr)
            acc100 = sum(r <= 100 for r in vr) / len(vr)
            mean_r = sum(vr) / len(vr)
            print(f"└─ val@1 {acc1:.3f} | val@10 {acc10:.3f} | val@100 {acc100:.3f} | mean_rank {mean_r:.1f}")

        if step % SAVE_EVERY == 0:
            torch.save({'step':step, 'model':mdl.state_dict(), 'opt':opt.state_dict()}, out)

if __name__ == "__main__":
    torch.manual_seed(0)
    train()