#!/usr/bin/env python3
import argparse, itertools, random, time, math
from pathlib import Path
from typing import List, Tuple
from array import array

import numpy as np
import os
from tinygrad import Tensor, dtypes, nn
from tinygrad.device import Device
from tinygrad.nn.state import get_state_dict, load_state_dict, safe_save, get_parameters
from tinygrad.nn.optim import AdamW
from extra.export_model import export_model
from tinygrad.device import Device
from tinygrad import TinyJit


Device.DEFAULT = os.environ.get("DEVICE", Device.DEFAULT)

# -------------------------------------------------------------------- #
#  Constants
# -------------------------------------------------------------------- #
DIM, N_HEADS, N_LAYERS = 128, 8, 4
MAX_LEN_Q, MAX_LEN_D   = 100, 110
VOCAB                  = 94                           # ASCII 33–126
CHAR_TO_ID             = {chr(i): i - 33 for i in range(33, 127)}
CLS_Q, CLS_D           = CHAR_TO_ID['{'], CHAR_TO_ID['|']  # 90, 91
MAX_LEN                = MAX_LEN_Q + MAX_LEN_D + 2
NUM_LA_TYPES           = 4                            # 0,1,2,3 (3 unused by current lev_align)
TEMP                   = 0.1

# -------------------------------------------------------------------- #
#  Data helpers
# -------------------------------------------------------------------- #
def encode(txt: str, max_len: int) -> List[int]:
    ids = [CHAR_TO_ID.get(c, 0) for c in txt[:max_len]]
    if len(ids) < max_len: ids.extend([0] * (max_len - len(ids)))
    return ids

def lev_align(q_chars: List[str], d_chars: List[str]) -> List[int]:
    m, n = len(q_chars), len(d_chars)
    n1 = n + 1

    dp = array('H', [0]) * ((m + 1) * (n + 1))

    for i in range(m + 1): dp[i * n1 + 0] = i
    for j in range(n + 1): dp[0 * n1 + j] = j

    for i in range(1, m + 1):
        qi = q_chars[i - 1]
        row = i * n1
        prow = (i - 1) * n1
        for j in range(1, n + 1):
            dj = d_chars[j - 1]
            cost_sub = 0 if qi == dj else 2
            delete = dp[prow + j] + 1
            insert = dp[row + (j - 1)] + 1
            subst  = dp[prow + (j - 1)] + cost_sub
            dp[row + j] = min(delete, insert, subst)

    la = [0] * n
    i, j = m, n
    while j or i:
        cur = dp[i * n1 + j]
        if i and j:
            qi, dj = q_chars[i - 1], d_chars[j - 1]
            cost_sub = 0 if qi == dj else 2
            if cur == dp[(i - 1) * n1 + (j - 1)] + cost_sub:
                la[j - 1] = 0 if cost_sub == 0 else 2
                i -= 1; j -= 1
                continue
        if j and (i == 0 or cur == dp[i * n1 + (j - 1)] + 1):
            la[j - 1] = 1
            j -= 1
            continue
        i -= 1

    return la

def build_pair(q: str, d: str) -> Tuple[List[int], List[int]]:
    q_tr = q[:MAX_LEN_Q]
    d_tr = d[:MAX_LEN_D]
    q_ids = encode(q_tr, MAX_LEN_Q)
    d_ids = encode(d_tr, MAX_LEN_D)

    la = lev_align(list(q_tr), list(d_tr))             # length = len(d_tr)
    la_full = [0] * (MAX_LEN_Q + 2) + la + [0] * (MAX_LEN_D - len(la))
    x_ids = [CLS_Q] + q_ids + [CLS_D] + d_ids
    return x_ids, la_full

def build_query_only(q: str) -> Tuple[List[int], List[int]]:
    q_ids = encode(q[:MAX_LEN_Q], MAX_LEN_Q)
    x_ids = [CLS_Q] + q_ids + [CLS_D] + ([0] * MAX_LEN_D)
    la_ids = [0] * MAX_LEN
    return x_ids, la_ids

def stream_qd(path: str):
    """Yield (query, docs) indefinitely. Same grouping semantics as PyTorch code."""
    p = Path(path)
    while True:
        with p.open(encoding="utf-8") as f:
            for gap, grp in itertools.groupby(f, key=lambda l: not l.strip()):
                if not gap:
                    lines = [l.rstrip("\n") for l in grp]
                    if lines:
                        yield lines[0], lines[1:]

# -------------------------------------------------------------------- #
#  Model Definitions
# -------------------------------------------------------------------- #
class MultiheadAttention:
    def __init__(self, embed_dim, num_heads):
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.in_proj_weight = Tensor.randn(3 * embed_dim, embed_dim) * 0.02
        self.in_proj_bias = Tensor.zeros(3 * embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def __call__(self, x: Tensor) -> Tensor:
        B, T, C = x.shape
        qkv = x.linear(self.in_proj_weight.T, self.in_proj_bias)

        q = qkv[:, :, :C].reshape(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = qkv[:, :, C:2*C].reshape(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = qkv[:, :, 2*C:].reshape(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        out = q.scaled_dot_product_attention(k, v).transpose(1, 2).reshape(B, T, C)
        return self.out_proj(out)

class TransformerEncoderLayer:
    def __init__(self, embed_dim, num_heads, ff_dim):
        self.self_attn = MultiheadAttention(embed_dim, num_heads)
        self.linear1 = nn.Linear(embed_dim, ff_dim)
        self.linear2 = nn.Linear(ff_dim, embed_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def __call__(self, x: Tensor) -> Tensor:
        x = self.norm1(x + self.self_attn(x))
        x = self.norm2(x + self.linear2(self.linear1(x).gelu()))
        return x

class TxEncoder:
    def __init__(self):
        self.tok_emb = nn.Embedding(VOCAB, DIM)
        self.pos_emb = nn.Embedding(MAX_LEN, DIM)
        self.la_emb  = nn.Embedding(NUM_LA_TYPES, DIM)
        self.layers = [TransformerEncoderLayer(DIM, N_HEADS, 4*DIM) for _ in range(N_LAYERS)]

    def __call__(self, x: Tensor, la: Tensor):
        B, S = x.shape
        pos = Tensor.arange(S).reshape(1, S).expand(B, S)
        h = self.tok_emb(x) + self.pos_emb(pos) + self.la_emb(la)
        for layer in self.layers:
            h = layer(h)
        e_q = h[:, 0, :]
        e_d = h[:, MAX_LEN_Q+1, :]
        return e_q, e_d

class RerankerHead:
    def __init__(self):
        self.cls = Tensor.randn(1, 1, DIM) * 0.02
        self.pos = Tensor.randn(3, DIM) * 0.02
        self.layers = [TransformerEncoderLayer(DIM, 8, 4*DIM) for _ in range(4)]
        self.sc = nn.Linear(DIM, 1)

    def __call__(self, q_embs: Tensor, d_embs: Tensor) -> Tensor:
        B, N, D = d_embs.shape
        cls = self.cls.expand(B, N, 1, D)
        q = q_embs.unsqueeze(1).unsqueeze(1).expand(B, N, 1, D)
        d = d_embs.unsqueeze(2)

        seq = cls.cat(q, d, dim=2)
        seq = seq + self.pos.reshape(1, 1, 3, D)
        seq = seq.reshape(B*N, 3, D)

        for layer in self.layers:
            seq = layer(seq)

        out = seq[:, 0, :].reshape(B, N, D)
        return self.sc(out).squeeze(-1)

# Training wrapper: supports batched queries and N docs each
class TrainPipeline:
    def __init__(self, enc: TxEncoder, rer: RerankerHead):
        self.enc = enc
        self.rer = rer

    def __call__(self, x_q: Tensor, la_q: Tensor, x_d: Tensor, la_d: Tensor, B: int, N: int) -> Tensor:
        # x_q, la_q: [B, MAX_LEN]
        # x_d, la_d: [B*N, MAX_LEN] (flattened)
        e_q, _ = self.enc(x_q, la_q)         # [B, D]
        _, e_d = self.enc(x_d, la_d)         # [B*N, D]
        d_embs = e_d.reshape(B, N, DIM)      # [B, N, D]
        scores = self.rer(e_q, d_embs) / TEMP
        return scores                         # [B, N]

# Inference/export wrapper: matches HTML harness signature (1 query vs NUM_DOCS docs)
class InferPipeline:
    def __init__(self, enc: TxEncoder, rer: RerankerHead, num_docs: int):
        self.enc = enc
        self.rer = rer
        self.num_docs = num_docs

    def __call__(self, x_q: Tensor, la_q: Tensor, x_d: Tensor, la_d: Tensor) -> Tensor:
        # x_q, la_q: [1, MAX_LEN]
        # x_d, la_d: [NUM_DOCS, MAX_LEN]
        e_q, _ = self.enc(x_q, la_q)         # [1, D]
        _, e_d = self.enc(x_d, la_d)         # [NUM_DOCS, D]
        scores = self.rer(e_q, e_d.unsqueeze(0)) / TEMP  # [1, NUM_DOCS]
        return scores.squeeze(0)             # [NUM_DOCS]

# -------------------------------------------------------------------- #
#  Loss + grad utils
# -------------------------------------------------------------------- #
def listwise_xent(scores: Tensor) -> Tensor:
    """
    scores: [B, N], target is always index 0 (positive doc)
    cross entropy: -log softmax(scores)[0]
    """
    # stable logsumexp over N
    m = scores.max(axis=1, keepdim=True)
    lse = m + (scores - m).exp().sum(axis=1, keepdim=True).log()
    loss = (lse.squeeze(1) - scores[:, 0]).mean()
    return loss

def clip_grad_norm_(params, max_norm: float, eps: float = 1e-12):
    # Compute global norm
    total = None
    for p in params:
        if p.grad is None: continue
        g2 = (p.grad * p.grad).sum()
        total = g2 if total is None else total + g2
    if total is None: return 0.0
    norm = float(total.sqrt().item())
    if norm > max_norm:
        scale = max_norm / (norm + eps)
        for p in params:
            if p.grad is None: continue
            p.grad *= scale
    return norm

# -------------------------------------------------------------------- #
#  Export helpers
# -------------------------------------------------------------------- #
def export_artifacts(infer_model: InferPipeline, outdir: Path, num_docs: int):
    # 1. Save weights from active CUDA model
    sd = get_state_dict(infer_model)
    safe_save(sd, (outdir / "reranker.safetensors").as_posix())

    # 2. Temporarily switch context to WEBGPU
    old_device = Device.DEFAULT
    Device.DEFAULT = "WEBGPU"

    # 3. Create a fresh model instance strictly on WEBGPU for tracing
    enc_ext = TxEncoder()
    rer_ext = RerankerHead()
    infer_ext = InferPipeline(enc_ext, rer_ext, num_docs=num_docs)

    # 4. Force example inputs to the WEBGPU device
    example_inputs = (
        Tensor(np.zeros((1, MAX_LEN), dtype=np.int32), dtype=dtypes.int32, device="WEBGPU"),
        Tensor(np.zeros((1, MAX_LEN), dtype=np.int32), dtype=dtypes.int32, device="WEBGPU"),
        Tensor(np.zeros((num_docs, MAX_LEN), dtype=np.int32), dtype=dtypes.int32, device="WEBGPU"),
        Tensor(np.zeros((num_docs, MAX_LEN), dtype=np.int32), dtype=dtypes.int32, device="WEBGPU"),
    )

    # 5. Export the JS program (this will now properly output WGSL)
    prg, inp_sizes, out_sizes, state = export_model(infer_ext, "webgpu", *example_inputs)
    (outdir / "reranker.js").write_text(prg, encoding="utf-8")

    # 6. Restore the CUDA context so training can continue smoothly
    Device.DEFAULT = old_device

# -------------------------------------------------------------------- #
#  Training
# -------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=10_000)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--neg-samp", type=int, default=199)     # N = 200 like PyTorch default
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--wd", type=float, default=1e-2)
    ap.add_argument("--grad-clip", type=float, default=5.0)
    ap.add_argument("--export-every", type=int, default=100)
    ap.add_argument("--export-docs", type=int, default=1000) # must match HTML harness
    ap.add_argument("--train-file", type=str, default="so_ts_markov.txt")
    ap.add_argument("--val-file", type=str, default="so_vs_markov.txt")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    outdir = Path(".").resolve()
    tr_path = outdir / args.train_file
    vs_path = outdir / args.val_file
    assert tr_path.exists(), f"missing {tr_path}"
    assert vs_path.exists(), f"missing {vs_path}"

    print(f"Device: {Device.DEFAULT}")
    print(f"Training on: {tr_path.name}  (val: {vs_path.name})")
    print(f"B={args.batch_size}, NEG={args.neg_samp} => N={args.neg_samp+1}")
    print(f"Export every {args.export_every} steps -> reranker.safetensors + reranker.js (overwrite)")

    enc = TxEncoder()
    rer = RerankerHead()
    train_model = TrainPipeline(enc, rer)
    infer_model = InferPipeline(enc, rer, num_docs=args.export_docs)

    params = get_parameters(train_model)
    opt = AdamW(params, lr=args.lr, weight_decay=args.wd)

    tr_gen = stream_qd(tr_path.as_posix())

    def next_batch():
        B = args.batch_size
        N = args.neg_samp + 1
        q_items = []

        # Ensure fixed batch size (skip queries with no negatives)
        while len(q_items) < B:
            q, docs = next(tr_gen)
            if len(docs) <= 1:  # needs at least one negative
                continue
            q_items.append((q, docs))

        # Build query-only tensors [B, MAX_LEN]
        x_q = np.zeros((B, MAX_LEN), dtype=np.int32)
        la_q = np.zeros((B, MAX_LEN), dtype=np.int32)

        # Build flattened docs tensors [B*N, MAX_LEN]
        x_d = np.zeros((B * N, MAX_LEN), dtype=np.int32)
        la_d = np.zeros((B * N, MAX_LEN), dtype=np.int32)

        for i, (q, docs) in enumerate(q_items):
            # query-only
            qx, qla = build_query_only(q)
            x_q[i, :] = np.asarray(qx, dtype=np.int32)
            la_q[i, :] = np.asarray(qla, dtype=np.int32)

            # docs: 1 positive + K negatives (sample w/ replacement if needed)
            pos = docs[0]
            neg_pool = docs[1:]
            if len(neg_pool) >= args.neg_samp:
                negs = random.sample(neg_pool, args.neg_samp)
            else:
                negs = list(neg_pool)
                negs += random.choices(neg_pool, k=args.neg_samp - len(negs))
            docs_batch = [pos] + negs

            # build_pair(q,d) (includes q in the doc input and la_full)
            for j, d in enumerate(docs_batch):
                xd, lad = build_pair(q, d)
                row = i * N + j
                x_d[row, :] = np.asarray(xd, dtype=np.int32)
                la_d[row, :] = np.asarray(lad, dtype=np.int32)

        return q_items, x_q, la_q, x_d, la_d

    def clip_grad_norm_jit(params, max_norm: float, eps: float = 1e-12) -> Tensor:
        grads = [p.grad for p in params if p.grad is not None]
        if not grads: return Tensor([0.0])

        # Compute global norm using tensor math entirely
        total = Tensor([0.0])
        for g in grads:
            total = total + (g * g).sum()
        norm = total.sqrt()

        # Branchless clipping: scale is exactly 1.0 if norm <= max_norm
        scale = (max_norm / (norm + eps)).minimum(1.0)

        # In-place assign the scaled gradients
        for g in grads:
            g.assign(g * scale)

        return norm

    @TinyJit
    def train_step(xq_t: Tensor, laq_t: Tensor, xd_t: Tensor, lad_t: Tensor) -> Tuple[Tensor, Tensor]:
        opt.zero_grad()
        scores = train_model(xq_t, laq_t, xd_t, lad_t, B=args.batch_size, N=args.neg_samp + 1)
        loss = listwise_xent(scores)
        loss.backward()

        gnorm = clip_grad_norm_jit(params, args.grad_clip)
        opt.step()

        # Realize the outputs so the JIT knows to evaluate them
        Tensor.realize(loss, gnorm)
        return loss, gnorm

    t0 = time.time()
    for step in range(1, args.steps + 1):
        Tensor.training = True

        q_items, xq, laq, xd, lad = next_batch()

        # Create tensors (do not put .realize() here, let JIT handle it)
        xq_t  = Tensor(xq, dtype=dtypes.int32)
        laq_t = Tensor(laq, dtype=dtypes.int32)
        xd_t  = Tensor(xd, dtype=dtypes.int32)
        lad_t = Tensor(lad, dtype=dtypes.int32)

        # Call the JIT function
        loss, gnorm = train_step(xq_t, laq_t, xd_t, lad_t)

        if step % 1 == 0:
            dt = time.time() - t0
            t0 = time.time()
            # JIT outputs are tensors, use .item() safely now
            print(f"step {step:>6} | loss {loss.item():.4f} | grad_norm {gnorm.item():.2f} | Δt {dt:.2f}s")

        if step % args.export_every == 0:
            Tensor.training = False
            print(f"--- export @ step {step} ---")
            export_artifacts(infer_model, outdir, num_docs=args.export_docs)
            print(f"wrote: {(outdir/'reranker.safetensors').name}, {(outdir/'reranker.js').name}")

    # Final export at end (handy)
    Tensor.training = False
    export_artifacts(infer_model, outdir, num_docs=args.export_docs)
    print("done.")

if __name__ == "__main__":
    main()