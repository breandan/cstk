#!/usr/bin/env python3
import argparse, itertools, random, time, math, re, shutil
from pathlib import Path
from typing import List, Tuple
from array import array

import numpy as np
import os
from tinygrad import Tensor, dtypes, nn
from tinygrad.device import Device
from tinygrad.nn.state import get_state_dict, load_state_dict, safe_load, safe_save, get_parameters
from tinygrad.nn.optim import AdamW
from extra.export_model import export_model
from tinygrad.device import Device
from tinygrad import TinyJit


Device.DEFAULT = os.environ.get("DEVICE", Device.DEFAULT)

# -------------------------------------------------------------------- #
#  Constants
# -------------------------------------------------------------------- #
DIM, N_HEADS, N_LAYERS = 256, 8, 4
MAX_LEN_Q, MAX_LEN_D   = 100, 110
VOCAB                  = 94                           # ASCII 33–126
CHAR_TO_ID             = {chr(i): i - 33 for i in range(33, 127)}
CLS_Q, CLS_D           = CHAR_TO_ID['{'], CHAR_TO_ID['|']  # 90, 91
MAX_LEN                = MAX_LEN_Q + MAX_LEN_D + 2
NUM_LA_TYPES           = 4                            # 0,1,2,3 (3 unused by current lev_align)
TEMP                   = 0.2
WEBGPU_STORAGE_BINDING_LIMIT = 128 * 1024 * 1024
WEBGPU_EXPORT_QKV_BYTES_PER_DOC = MAX_LEN * DIM * 3 * 4
WEBGPU_EXPORT_ATTN_BYTES_PER_DOC = N_HEADS * MAX_LEN * MAX_LEN * 4
WEBGPU_EXPORT_BYTES_PER_DOC = max(WEBGPU_EXPORT_QKV_BYTES_PER_DOC, WEBGPU_EXPORT_ATTN_BYTES_PER_DOC)
WEBGPU_MAX_EXPORT_DOCS = WEBGPU_STORAGE_BINDING_LIMIT // WEBGPU_EXPORT_BYTES_PER_DOC
WEBGPU_DEFAULT_EXPORT_DOCS = min(64, WEBGPU_MAX_EXPORT_DOCS)

# -------------------------------------------------------------------- #
#  Data helpers
# -------------------------------------------------------------------- #
def encode(txt: str, max_len: int) -> List[int]:
    ids = [CHAR_TO_ID.get(c, 0) for c in txt[:max_len]]
    if len(ids) < max_len: ids.extend([0] * (max_len - len(ids)))
    return ids

def lev_align(q_chars, d_chars) -> List[int]:
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

def build_pair_from_query(q_tr: str, q_ids: List[int], d: str) -> Tuple[List[int], List[int]]:
    d_tr = d[:MAX_LEN_D]
    d_ids = encode(d_tr, MAX_LEN_D)

    la = lev_align(q_tr, d_tr)                         # length = len(d_tr)
    la_full = [0] * (MAX_LEN_Q + 2) + la + [0] * (MAX_LEN_D - len(la))
    x_ids = [CLS_Q] + q_ids + [CLS_D] + d_ids
    return x_ids, la_full

def build_pair(q: str, d: str) -> Tuple[List[int], List[int]]:
    q_tr = q[:MAX_LEN_Q]
    q_ids = encode(q_tr, MAX_LEN_Q)
    return build_pair_from_query(q_tr, q_ids, d)

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

def iter_qd_once(path: str):
    """Yield (query, docs) exactly once."""
    p = Path(path)
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

    def __call__(self, x: Tensor, la: Tensor) -> Tensor:
        B, S = x.shape
        pos = Tensor.arange(S).reshape(1, S).expand(B, S)
        h = self.tok_emb(x) + self.pos_emb(pos) + self.la_emb(la)
        for layer in self.layers:
            h = layer(h)

        # x already contains the complete [CLS_Q] query [CLS_D] document pair.
        # Its leading state is therefore the pair-conditioned cross-encoder embedding.
        return h[:, 0, :]

class RerankerHead:
    """Small pointwise scorer over the pair-conditioned [CLS_Q] embedding."""
    def __init__(self):
        self.proj = nn.Linear(DIM, DIM // 2)
        self.sc = nn.Linear(DIM // 2, 1)

    def __call__(self, pair_embs: Tensor) -> Tensor:
        return self.sc(self.proj(pair_embs).gelu()).squeeze(-1)


def _legacy_input_offset(x_q: Tensor, la_q: Tensor) -> Tensor:
    """Keep the existing four-input WebGPU/HTML signature without a query encoder.

    The offset is constant across all candidates for a query, so it cannot change
    their ordering or the listwise objective. Its only purpose is to keep input0
    and input1 live in the exported graph while downstream harnesses are unchanged.
    """
    return (
            x_q.cast(dtypes.float32).sum(axis=1)
            + la_q.cast(dtypes.float32).sum(axis=1)
    ) * 1e-12


# Training wrapper: supports batched queries and N pair sequences each.
class TrainPipeline:
    def __init__(self, enc: TxEncoder, rer: RerankerHead):
        self.enc = enc
        self.rer = rer

    def __call__(self, x_q: Tensor, la_q: Tensor, x_d: Tensor, la_d: Tensor, B: int, N: int) -> Tensor:
        # x_q, la_q: [B, MAX_LEN], retained only for export/harness compatibility.
        # x_d, la_d: [B*N, MAX_LEN], each row is a complete query-document pair.
        pair_embs = self.enc(x_d, la_d).reshape(B, N, DIM)
        scores = self.rer(pair_embs) / TEMP
        return scores + _legacy_input_offset(x_q, la_q).reshape(B, 1)

# Inference/export wrapper: preserves the existing four-input HTML harness signature.
class InferPipeline:
    def __init__(self, enc: TxEncoder, rer: RerankerHead, num_docs: int):
        self.enc = enc
        self.rer = rer
        self.num_docs = num_docs

    def __call__(self, x_q: Tensor, la_q: Tensor, x_d: Tensor, la_d: Tensor) -> Tensor:
        # x_q, la_q: [1, MAX_LEN], retained only for export/harness compatibility.
        # x_d, la_d: [NUM_DOCS, MAX_LEN], complete query-document pairs.
        pair_embs = self.enc(x_d, la_d)
        scores = self.rer(pair_embs) / TEMP
        return scores + _legacy_input_offset(x_q, la_q).reshape(1)

# -------------------------------------------------------------------- #
#  Loss + grad utils
# -------------------------------------------------------------------- #
def listwise_xent(scores: Tensor, targets: Tensor) -> Tensor:
    """
    scores: [B, N], targets: [B]
    cross entropy: -log softmax(scores)[target]
    """
    # stable logsumexp over N
    m = scores.max(axis=1, keepdim=True)
    lse = m + (scores - m).exp().sum(axis=1, keepdim=True).log()
    n = scores.shape[1]
    target_mask = (Tensor.arange(n).reshape(1, n) == targets.reshape(-1, 1)).cast(scores.dtype)
    target_scores = (scores * target_mask).sum(axis=1)
    loss = (lse.squeeze(1) - target_scores).mean()
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
def _atomic_copy(src: Path, dst: Path):
    tmp = dst.with_name(f".{dst.name}.{os.getpid()}.tmp")
    try:
        shutil.copyfile(src, tmp)
        os.replace(tmp, dst)
    finally:
        tmp.unlink(missing_ok=True)


def _check_webgpu_export_docs(num_docs: int):
    if num_docs <= 0:
        raise ValueError(f"--export-docs must be positive, got {num_docs}")
    binding_bytes = num_docs * WEBGPU_EXPORT_BYTES_PER_DOC
    if binding_bytes > WEBGPU_STORAGE_BINDING_LIMIT:
        raise ValueError(
            f"--export-docs {num_docs} may require a {binding_bytes:,} byte WebGPU storage-buffer binding "
            f"for encoder intermediates, exceeding Dawn's "
            f"{WEBGPU_STORAGE_BINDING_LIMIT:,} byte limit. Use --export-docs <= {WEBGPU_MAX_EXPORT_DOCS} "
            f"(default/recommended: {WEBGPU_DEFAULT_EXPORT_DOCS}) and score larger candidate sets in chunks."
        )


def _validate_webgpu_export(prg: str, inp_sizes, out_sizes, state_keys, num_docs: int):
    """Fail before publishing an incomplete or unweighted generated module."""
    if "\n// ...\n" in prg:
        raise RuntimeError("WebGPU export contains a literal '// ...' placeholder; WGSL was truncated")

    kernel_match = re.search(r"const\s+kernels\s*=\s*\[([^\]]*)\]\s*;", prg, re.S)
    if not kernel_match:
        raise RuntimeError("WebGPU export has no `const kernels = [...]` list")

    kernel_refs = re.findall(r"\b[A-Za-z_$][\w$]*\b", kernel_match.group(1))
    kernel_decls = set(re.findall(r"const\s+([A-Za-z_$][\w$]*)\s*=\s*`", prg))
    missing_kernels = sorted(set(kernel_refs) - kernel_decls)
    if not kernel_refs or not kernel_decls or missing_kernels:
        preview = ", ".join(missing_kernels[:8])
        suffix = " ..." if len(missing_kernels) > 8 else ""
        raise RuntimeError(
            f"WebGPU export is missing {len(missing_kernels)} WGSL declaration(s): {preview}{suffix}"
        )

    # Every trained tensor must be represented by an external safetensor-backed GPU buffer.
    weight_refs = re.findall(r"metadata\[['\"]([^'\"]+)['\"]\]", prg)
    expected_weights = set(state_keys)
    actual_weights = set(weight_refs)
    missing_weights = sorted(expected_weights - actual_weights)
    extra_weights = sorted(actual_weights - expected_weights)
    if missing_weights or extra_weights or len(weight_refs) != len(expected_weights):
        raise RuntimeError(
            "WebGPU export did not bind the trained state correctly: "
            f"expected {len(expected_weights)} tensors, found {len(weight_refs)} references; "
            f"missing={missing_weights[:5]}, extra={extra_weights[:5]}"
        )

    expected_inputs = {
        "input0": MAX_LEN * 4,
        "input1": MAX_LEN * 4,
        "input2": num_docs * MAX_LEN * 4,
        "input3": num_docs * MAX_LEN * 4,
    }
    expected_outputs = {"output0": num_docs * 4}
    if dict(inp_sizes) != expected_inputs:
        raise RuntimeError(f"Unexpected WebGPU input sizes: {inp_sizes}; expected {expected_inputs}")
    if dict(out_sizes) != expected_outputs:
        raise RuntimeError(f"Unexpected WebGPU output sizes: {out_sizes}; expected {expected_outputs}")

    return {
        "kernel_calls": len(kernel_refs),
        "unique_kernels": len(kernel_decls),
        "weight_buffers": len(weight_refs),
    }


def export_artifacts(infer_model: InferPipeline, outdir: Path, step: int, num_docs: int):
    """Export a versioned checkpoint and canonical aliases, publishing only after validation."""
    _check_webgpu_export_docs(num_docs)
    outdir.mkdir(parents=True, exist_ok=True)
    state = get_state_dict(infer_model)

    version_weights = outdir / f"reranker_{step}.safetensors"
    version_js = outdir / f"reranker_{step}.js"
    canonical_weights = outdir / "reranker.safetensors"
    canonical_js = outdir / "reranker.js"

    tmp_weights = outdir / f".{version_weights.name}.{os.getpid()}.tmp"
    tmp_js = outdir / f".{version_js.name}.{os.getpid()}.tmp"

    old_device = Device.DEFAULT
    was_training = Tensor.training
    try:
        # Save to a temporary file first. A failed JS export must not publish a half-pair.
        safe_save(state, tmp_weights.as_posix())

        Tensor.training = False
        Device.DEFAULT = "WEBGPU"

        # The exporter identifies external weights only when the traced model's parameters
        # are realized buffers. Loading the trained state is therefore mandatory.
        enc_ext = TxEncoder()
        rer_ext = RerankerHead()
        infer_ext = InferPipeline(enc_ext, rer_ext, num_docs=num_docs)
        load_state_dict(infer_ext, state, strict=True, verbose=False, realize=True)
        Tensor.realize(*get_parameters(infer_ext))

        example_inputs = (
            Tensor(np.zeros((1, MAX_LEN), dtype=np.int32), dtype=dtypes.int32, device="WEBGPU"),
            Tensor(np.zeros((1, MAX_LEN), dtype=np.int32), dtype=dtypes.int32, device="WEBGPU"),
            Tensor(np.zeros((num_docs, MAX_LEN), dtype=np.int32), dtype=dtypes.int32, device="WEBGPU"),
            Tensor(np.zeros((num_docs, MAX_LEN), dtype=np.int32), dtype=dtypes.int32, device="WEBGPU"),
        )

        prg, inp_sizes, out_sizes, _ = export_model(infer_ext, "webgpu", *example_inputs)
        stats = _validate_webgpu_export(prg, inp_sizes, out_sizes, state.keys(), num_docs)
        tmp_js.write_text(prg, encoding="utf-8")

        # Publish the versioned pair, then refresh the canonical aliases watched by Modal.
        os.replace(tmp_weights, version_weights)
        os.replace(tmp_js, version_js)
        _atomic_copy(version_weights, canonical_weights)
        _atomic_copy(version_js, canonical_js)

        print(
            f"exported step {step}: {version_weights.name}, {version_js.name} | "
            f"{stats['kernel_calls']} calls / {stats['unique_kernels']} WGSL kernels / "
            f"{stats['weight_buffers']} external weights / {num_docs} docs per call"
        )
        return version_weights, version_js
    finally:
        Device.DEFAULT = old_device
        Tensor.training = was_training
        tmp_weights.unlink(missing_ok=True)
        tmp_js.unlink(missing_ok=True)

# -------------------------------------------------------------------- #
#  Training
# -------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=10_000)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--neg-samp", type=int, default=199)     # N = 200 like PyTorch default
    ap.add_argument("--neg-pool", type=int, default=200,
                    help="sample negatives from the first K confounders after the positive; <=0 uses all confounders")
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--wd", type=float, default=1e-2)
    ap.add_argument("--grad-clip", type=float, default=5.0)
    ap.add_argument("--export-every", type=int, default=100)
    ap.add_argument("--export-docs", type=int, default=WEBGPU_DEFAULT_EXPORT_DOCS,
                    help="fixed docs per exported WebGPU call; larger candidate sets are scored in chunks")
    ap.add_argument("--val-every", type=int, default=None,
                    help="validation cadence in steps; defaults to --export-every; <=0 disables validation")
    ap.add_argument("--val-groups", type=int, default=24,
                    help="number of validation groups to score at each validation pass")
    ap.add_argument("--val-docs", type=int, default=0,
                    help="max docs per validation group, including the positive doc; <=0 scores all docs")
    ap.add_argument("--val-batch-size", type=int, default=8,
                    help="deprecated; validation is scored one query at a time in --export-docs chunks")
    ap.add_argument("--no-val-cache", action="store_true",
                    help="rebuild validation features every validation pass instead of caching them")
    ap.add_argument("--train-file", type=str, default="so_ts_markov.txt")
    ap.add_argument("--val-file", type=str, default="so_vs_markov.txt")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--export-from", type=str, default="", metavar="SAFETENSORS",
                    help="regenerate JS from an existing checkpoint and exit")
    ap.add_argument("--export-step", type=int, default=None,
                    help="step label for --export-from; inferred from reranker_<step>.safetensors")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    if args.val_every is None:
        args.val_every = args.export_every
    _check_webgpu_export_docs(args.export_docs)

    outdir = Path(".").resolve()

    if args.export_from:
        checkpoint = Path(args.export_from).expanduser().resolve()
        if not checkpoint.exists():
            raise FileNotFoundError(f"missing export checkpoint: {checkpoint}")
        step = args.export_step
        if step is None:
            match = re.search(r"_(\d+)\.safetensors$", checkpoint.name)
            if not match:
                raise ValueError("--export-step is required when the checkpoint name has no _<step> suffix")
            step = int(match.group(1))

        print(f"Regenerating WebGPU artifacts from {checkpoint.name} at step {step}")
        enc = TxEncoder()
        rer = RerankerHead()
        infer_model = InferPipeline(enc, rer, num_docs=args.export_docs)
        load_state_dict(infer_model, safe_load(checkpoint.as_posix()), strict=True, verbose=True, realize=True)
        export_artifacts(infer_model, outdir, step, num_docs=args.export_docs)
        print("done.")
        return

    tr_path = outdir / args.train_file
    vs_path = outdir / args.val_file
    assert tr_path.exists(), f"missing {tr_path}"
    assert vs_path.exists(), f"missing {vs_path}"

    print(f"Device: {Device.DEFAULT}")
    print(f"Training on: {tr_path.name}  (val: {vs_path.name})")
    neg_pool_msg = "all confounders" if args.neg_pool <= 0 else f"top {args.neg_pool} confounders"
    print(f"B={args.batch_size}, NEG={args.neg_samp} from {neg_pool_msg} => N={args.neg_samp+1}")
    print(f"Export every {args.export_every} steps -> versioned files + canonical reranker.safetensors/reranker.js")
    print(
        f"Export docs/call {args.export_docs} -> estimated largest encoder binding "
        f"{args.export_docs * WEBGPU_EXPORT_BYTES_PER_DOC:,} bytes "
        f"(limit {WEBGPU_STORAGE_BINDING_LIMIT:,}; max docs/call {WEBGPU_MAX_EXPORT_DOCS})"
    )
    if args.val_every > 0:
        val_docs_msg = "all docs" if args.val_docs <= 0 else f"up to {args.val_docs} docs/group"
        cache_msg = "cached" if not args.no_val_cache else "uncached"
        print(
            f"Validate every {args.val_every} steps -> first {args.val_groups} groups, "
            f"{val_docs_msg}, {args.export_docs} docs/call, {cache_msg}"
        )
    else:
        print("Validation disabled")

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
        targets = np.zeros((B,), dtype=np.int32)

        for i, (q, docs) in enumerate(q_items):
            # query-only
            q_tr = q[:MAX_LEN_Q]
            q_ids = encode(q_tr, MAX_LEN_Q)
            qx = [CLS_Q] + q_ids + [CLS_D] + ([0] * MAX_LEN_D)
            x_q[i, :] = qx
            # la_q is already zero-filled.

            # docs: 1 positive + K negatives (sample w/ replacement if needed)
            pos = docs[0]
            all_negs = docs[1:]
            neg_pool = all_negs if args.neg_pool <= 0 else all_negs[:args.neg_pool]
            if len(neg_pool) >= args.neg_samp:
                negs = random.sample(neg_pool, args.neg_samp)
            else:
                negs = list(neg_pool)
                negs += random.choices(neg_pool, k=args.neg_samp - len(negs))
            docs_labeled = [(pos, True)] + [(neg, False) for neg in negs]
            random.shuffle(docs_labeled)
            targets[i] = next(j for j, (_, is_pos) in enumerate(docs_labeled) if is_pos)
            docs_batch = [doc for doc, _ in docs_labeled]

            # build_pair(q,d) (includes q in the doc input and la_full)
            for j, d in enumerate(docs_batch):
                xd, lad = build_pair_from_query(q_tr, q_ids, d)
                row = i * N + j
                x_d[row, :] = xd
                la_d[row, :] = lad

        return q_items, x_q, la_q, x_d, la_d, targets

    val_cache = None

    def select_validation_docs(docs: List[str], group_idx: int) -> List[str]:
        if args.val_docs <= 0 or len(docs) <= args.val_docs:
            return docs
        rng = random.Random(args.seed + group_idx)
        negs = docs[1:]
        return [docs[0]] + rng.sample(negs, args.val_docs - 1)

    def build_validation_cache():
        t_build = time.time()
        docs_per_call = max(1, args.export_docs)
        groups = []
        for group_idx, (q, docs) in enumerate(iter_qd_once(vs_path.as_posix())):
            if len(groups) >= args.val_groups:
                break
            if not docs:
                continue
            docs = select_validation_docs(docs, group_idx)

            q_tr = q[:MAX_LEN_Q]
            q_ids = encode(q_tr, MAX_LEN_Q)
            x_q = np.zeros((1, MAX_LEN), dtype=np.int32)
            la_q = np.zeros((1, MAX_LEN), dtype=np.int32)
            x_q[0, :] = [CLS_Q] + q_ids + [CLS_D] + ([0] * MAX_LEN_D)

            doc_chunks = []
            for start in range(0, len(docs), docs_per_call):
                chunk_docs = docs[start:start + docs_per_call]
                x_d = np.zeros((docs_per_call, MAX_LEN), dtype=np.int32)
                la_d = np.zeros((docs_per_call, MAX_LEN), dtype=np.int32)
                for j, d in enumerate(chunk_docs):
                    xd, lad = build_pair_from_query(q_tr, q_ids, d)
                    x_d[j, :] = xd
                    la_d[j, :] = lad
                doc_chunks.append((x_d, la_d, len(chunk_docs)))

            groups.append((x_q, la_q, doc_chunks, len(docs)))

        if not groups:
            return []

        doc_counts = [n_docs for _, _, _, n_docs in groups]
        calls = sum(len(chunks) for _, _, chunks, _ in groups)

        print(
            f"prepared validation cache: {len(groups)} groups, "
            f"{sum(doc_counts)} docs, docs/group {min(doc_counts)}-{max(doc_counts)}, "
            f"{docs_per_call} docs/call, {calls} calls in {time.time() - t_build:.2f}s"
        )
        return groups

    def run_validation():
        nonlocal val_cache
        was_training = Tensor.training
        Tensor.training = False

        if args.no_val_cache or val_cache is None:
            val_cache = build_validation_cache()

        total = 0
        total_loss = 0.0
        total_rank = 0
        top1 = 0
        hit10 = 0
        p10_slots = 0
        rr_sum = 0.0

        t_val = time.time()
        for x_q, la_q, doc_chunks, n_docs in val_cache:
            xq_t  = Tensor(x_q, dtype=dtypes.int32)
            laq_t = Tensor(la_q, dtype=dtypes.int32)
            score_parts = []
            for x_d, la_d, count in doc_chunks:
                xd_t = Tensor(x_d, dtype=dtypes.int32)
                lad_t = Tensor(la_d, dtype=dtypes.int32)
                scores = train_model(xq_t, laq_t, xd_t, lad_t, B=1, N=x_d.shape[0])
                Tensor.realize(scores)
                score_parts.append(scores.numpy().reshape(-1)[:count])

            if n_docs <= 0:
                continue
            s = np.concatenate(score_parts, axis=0)

            # positive doc is still docs[0] in the validation file; this is used only for metrics.
            pos_rank = 1 + int((s[1:] > s[0]).sum()) if len(s) > 1 else 1

            # numpy version of listwise_xent for logging over the full candidate list
            m = float(s.max())
            lse = m + math.log(float(np.exp(s - m).sum()))
            loss = lse - float(s[0])

            total += 1
            total_loss += loss
            total_rank += pos_rank
            top1 += int(pos_rank == 1)
            hit10 += int(pos_rank <= 10)
            p10_slots += min(10, n_docs)
            rr_sum += 1.0 / pos_rank

        if total:
            print(
                f"val summary ({total} groups, {time.time() - t_val:.2f}s) | "
                f"mean_loss {total_loss/total:.4f} | "
                f"mean_pos_rank {total_rank/total:.2f} | "
                f"top1 {top1/total:.3f} | "
                f"hit@10 {hit10/total:.3f} | "
                f"precision@10 {hit10/p10_slots:.4f} | "
                f"mrr {rr_sum/total:.4f}"
            )
        else:
            print("val summary | no validation groups with docs")

        Tensor.training = was_training

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
    def train_step(xq_t: Tensor, laq_t: Tensor, xd_t: Tensor, lad_t: Tensor, target_t: Tensor) -> Tuple[Tensor, Tensor]:
        opt.zero_grad()
        scores = train_model(xq_t, laq_t, xd_t, lad_t, B=args.batch_size, N=args.neg_samp + 1)
        loss = listwise_xent(scores, target_t)
        loss.backward()

        gnorm = clip_grad_norm_jit(params, args.grad_clip)
        opt.step()

        # Realize the outputs so the JIT knows to evaluate them
        Tensor.realize(loss, gnorm)
        return loss, gnorm

    t0 = time.time()
    for step in range(1, args.steps + 1):
        Tensor.training = True

        q_items, xq, laq, xd, lad, targets = next_batch()

        # Create tensors (do not put .realize() here, let JIT handle it)
        xq_t  = Tensor(xq, dtype=dtypes.int32)
        laq_t = Tensor(laq, dtype=dtypes.int32)
        xd_t  = Tensor(xd, dtype=dtypes.int32)
        lad_t = Tensor(lad, dtype=dtypes.int32)
        target_t = Tensor(targets, dtype=dtypes.int32)

        # Call the JIT function
        loss, gnorm = train_step(xq_t, laq_t, xd_t, lad_t, target_t)

        if step % 10 == 0:
            dt = time.time() - t0
            t0 = time.time()
            # JIT outputs are tensors, use .item() safely now
            print(f"step {step:>6} | loss {loss.item():.4f} | grad_norm {gnorm.item():.2f} | Δt {dt:.2f}s")

        if args.val_every > 0 and step % args.val_every == 0:
            Tensor.training = False
            print(f"--- validation @ step {step} ---")
            run_validation()

        if args.export_every > 0 and step % args.export_every == 0:
            Tensor.training = False
            print(f"--- export @ step {step} ---")
            export_artifacts(infer_model, outdir, step, num_docs=args.export_docs)

    # Export the last step only when it was not already emitted by the periodic cadence.
    Tensor.training = False
    if args.export_every <= 0 or args.steps % args.export_every != 0:
        print(f"--- final export @ step {args.steps} ---")
        export_artifacts(infer_model, outdir, args.steps, num_docs=args.export_docs)
    print("done.")

if __name__ == "__main__":
    main()
