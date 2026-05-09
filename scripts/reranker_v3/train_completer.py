#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_completer.py
------------------
Small causal next-token completer for grammar-constrained DFA rollout.

Training data format:
  - Every nonblank line is treated as one token sequence.
  - Default tokenizer is whitespace/token-level: "NAME = NAME NEWLINE" -> [NAME, =, NAME, NEWLINE]
  - Use --tokenizer char if your corpus is already charified and should be modeled character-by-character.

Serving protocol:
  POST /complete with text/plain body:

      <already-emitted prefix tokens, possibly empty>
      <candidate-token-1>
      <candidate-token-2>
      ...

  Response is text/plain, one log-probability per candidate, aligned with the request order.
  Log-probs are intentionally used so Kotlin can sum scores along a rollout.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from socketserver import ThreadingMixIn
from typing import Dict, Iterable, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Defaults: small enough to serve cheaply, large enough to learn token dynamics.
# ---------------------------------------------------------------------------
PAD, BOS, EOS, UNK = "<pad>", "<bos>", "<eos>", "<unk>"
SPECIALS = [PAD, BOS, EOS, UNK]

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)


@dataclass
class ModelConfig:
    vocab_size: int
    block_size: int = 128
    dim: int = 256
    n_heads: int = 4
    n_layers: int = 4
    dropout: float = 0.10
    tokenizer: str = "ws"  # "ws" or "char"


# ---------------------------------------------------------------------------
# Tokenization / data
# ---------------------------------------------------------------------------
def split_tokens(s: str, tokenizer: str = "ws") -> List[str]:
    s = s.rstrip("\n")
    if tokenizer == "char":
        return list(s)
    # Whitespace tokenizer: this is the natural mode for strings like
    # "for NAME in NAME : NEWLINE INDENT ...".
    return s.strip().split()


def read_sequences(paths: Sequence[str], tokenizer: str) -> List[List[str]]:
    seqs: List[List[str]] = []
    for p in paths:
        path = Path(p)
        if not path.exists():
            raise FileNotFoundError(f"training file not found: {path}")
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                toks = split_tokens(line, tokenizer)
                if toks:
                    seqs.append(toks)
    if not seqs:
        raise ValueError(f"no nonblank sequences found in {paths}")
    return seqs


def build_vocab(seqs: Sequence[Sequence[str]], min_count: int = 1) -> Tuple[Dict[str, int], List[str]]:
    counts = collections.Counter(tok for seq in seqs for tok in seq)
    vocab = SPECIALS + sorted(tok for tok, c in counts.items() if c >= min_count and tok not in SPECIALS)
    stoi = {tok: i for i, tok in enumerate(vocab)}
    return stoi, vocab


def encode_sequences(seqs: Sequence[Sequence[str]], stoi: Dict[str, int]) -> List[List[int]]:
    bos, eos, unk = stoi[BOS], stoi[EOS], stoi[UNK]
    return [[bos] + [stoi.get(tok, unk) for tok in seq] + [eos] for seq in seqs]


def sample_batch(encoded: Sequence[Sequence[int]], batch_size: int, block_size: int, pad_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return x,y of shape [B,T]. y uses pad_id where loss should be ignored."""
    xs: List[List[int]] = []
    ys: List[List[int]] = []
    need = block_size + 1

    for _ in range(batch_size):
        seq = random.choice(encoded)
        if len(seq) >= need:
            start = random.randint(0, len(seq) - need)
            chunk = seq[start:start + need]
        else:
            chunk = list(seq) + [pad_id] * (need - len(seq))

        xs.append(chunk[:-1])
        ys.append(chunk[1:])

    x = torch.tensor(xs, dtype=torch.long, device=DEVICE)
    y = torch.tensor(ys, dtype=torch.long, device=DEVICE)
    return x, y


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class CausalCompleter(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.dim)
        layer = nn.TransformerEncoderLayer(
            d_model=cfg.dim,
            nhead=cfg.n_heads,
            dim_feedforward=4 * cfg.dim,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.tx = nn.TransformerEncoder(layer, num_layers=cfg.n_layers)
        self.ln = nn.LayerNorm(cfg.dim)
        self.head = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)

        # Weight tying is cheap and helps small language models.
        self.head.weight = self.tok_emb.weight

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        b, t = idx.shape
        if t > self.cfg.block_size:
            idx = idx[:, -self.cfg.block_size:]
            t = self.cfg.block_size

        pos = torch.arange(t, device=idx.device).unsqueeze(0)
        h = self.tok_emb(idx) + self.pos_emb(pos)

        # Causal mask: position i cannot attend to positions > i.
        mask = torch.full((t, t), float("-inf"), device=idx.device)
        mask = torch.triu(mask, diagonal=1)
        h = self.tx(h, mask=mask)
        h = self.ln(h)
        return self.head(h)


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------
def save_ckpt(path: str, model: CausalCompleter, cfg: ModelConfig, stoi: Dict[str, int], itos: List[str], step: int, val_loss: float) -> None:
    tmp = f"{path}.tmp"
    torch.save(
        {
            "model": model.state_dict(),
            "cfg": asdict(cfg),
            "stoi": stoi,
            "itos": itos,
            "step": step,
            "val_loss": val_loss,
        },
        tmp,
    )
    os.replace(tmp, path)


def load_ckpt(path: str) -> Tuple[CausalCompleter, ModelConfig, Dict[str, int], List[str]]:
    ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
    cfg = ModelConfig(**ckpt["cfg"])
    model = CausalCompleter(cfg).to(DEVICE)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, cfg, ckpt["stoi"], ckpt["itos"]


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    seqs = read_sequences(args.train, args.tokenizer)
    random.shuffle(seqs)

    if args.val:
        val_seqs = read_sequences(args.val, args.tokenizer)
        train_seqs = seqs
    else:
        cut = max(1, int(0.95 * len(seqs)))
        train_seqs, val_seqs = seqs[:cut], seqs[cut:]
        if not val_seqs:
            val_seqs = train_seqs[:]

    stoi, itos = build_vocab(train_seqs, min_count=args.min_count)
    train_ids = encode_sequences(train_seqs, stoi)
    val_ids = encode_sequences(val_seqs, stoi)

    cfg = ModelConfig(
        vocab_size=len(itos),
        block_size=args.block_size,
        dim=args.dim,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        dropout=args.dropout,
        tokenizer=args.tokenizer,
    )
    model = CausalCompleter(cfg).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    pad_id = stoi[PAD]

    scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE.type == "cuda" and args.amp))
    print(
        json.dumps(
            {
                "device": str(DEVICE),
                "train_sequences": len(train_ids),
                "val_sequences": len(val_ids),
                "vocab_size": len(itos),
                "config": asdict(cfg),
                "ckpt_out": args.ckpt_out,
            },
            indent=2,
        ),
        flush=True,
    )

    best_val = float("inf")
    timer = time.time()

    for step in range(1, args.steps + 1):
        model.train()
        x, y = sample_batch(train_ids, args.batch_size, args.block_size, pad_id)
        opt.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=(DEVICE.type == "cuda" and args.amp)):
            logits = model(x)
            loss = F.cross_entropy(logits.reshape(-1, cfg.vocab_size), y.reshape(-1), ignore_index=pad_id)

        if scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(opt)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()

        if step % args.log_every == 0:
            dt = time.time() - timer
            timer = time.time()
            toks = args.log_every * args.batch_size * args.block_size
            print(f"step {step:>7} | train_loss {loss.item():.4f} | tok/s {toks / max(dt, 1e-9):.0f}", flush=True)

        if step % args.val_every == 0 or step == args.steps:
            model.eval()
            losses = []
            with torch.inference_mode():
                for _ in range(args.val_batches):
                    vx, vy = sample_batch(val_ids, args.batch_size, args.block_size, pad_id)
                    vlogits = model(vx)
                    vloss = F.cross_entropy(vlogits.reshape(-1, cfg.vocab_size), vy.reshape(-1), ignore_index=pad_id)
                    losses.append(float(vloss.item()))
            val_loss = sum(losses) / len(losses)
            print(f"val   {step:>7} | val_loss {val_loss:.4f} | ppl {math.exp(min(val_loss, 20.0)):.2f}", flush=True)

            # Always save latest, and keep best under the requested name.
            latest = args.ckpt_out.replace(".pt", "_latest.pt")
            save_ckpt(latest, model, cfg, stoi, itos, step, val_loss)
            if val_loss <= best_val:
                best_val = val_loss
                save_ckpt(args.ckpt_out, model, cfg, stoi, itos, step, val_loss)
                print(f"saved best checkpoint -> {args.ckpt_out}", flush=True)


# ---------------------------------------------------------------------------
# Inference / serving
# ---------------------------------------------------------------------------
_MODEL: CausalCompleter | None = None
_CFG: ModelConfig | None = None
_STOI: Dict[str, int] | None = None
_INFER_LOCK = None

MAX_CANDIDATES_PER_FORWARD = int(os.environ.get("COMPLETER_MAX_CANDIDATES_PER_FORWARD", "512"))


def _clear_device_cache() -> None:
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
    elif DEVICE.type == "mps" and hasattr(torch, "mps"):
        torch.mps.empty_cache()


def _score_candidates(prefix: str, candidates: List[str]) -> List[float]:
    if not candidates:
        return []

    out: List[float] = []
    bs = MAX_CANDIDATES_PER_FORWARD

    for i in range(0, len(candidates), bs):
        out.extend(_score_candidates_batch(prefix, candidates[i:i + bs]))

    # Optional. Useful on MPS/CUDA if allocator high-water looks like a leak.
    _clear_device_cache()

    return out

def _score_candidates_batch(prefix: str, candidates: List[str]) -> List[float]:
    """Return log P(candidate_string | prefix) for each candidate line.

    Important: candidate lines are *emission continuations*, not necessarily one
    model token. With tokenizer=char, a grammar token like "NAME" may arrive as
    a charified string such as "|NAME}", which tokenizes to many character
    tokens. The old implementation looked up the whole candidate line as one vocab
    item, so every multi-character candidate collapsed to UNK. This version scores
    the whole candidate continuation by summing the next-character log-probs.
    """
    assert _MODEL is not None and _CFG is not None and _STOI is not None
    if not candidates:
        return []

    bos = _STOI[BOS]
    pad = _STOI[PAD]
    unk = _STOI[UNK]

    prefix_toks = split_tokens(prefix, _CFG.tokenizer)
    prefix_ids = [bos] + [_STOI.get(tok, unk) for tok in prefix_toks]
    prefix_len = len(prefix_ids)

    examples = []
    masks = []
    max_len = 1

    for cand in candidates:
        cand_toks = split_tokens(cand.rstrip("\n"), _CFG.tokenizer)
        if not cand_toks:
            # Empty continuation: score neutral. It should normally not occur.
            examples.append([bos])
            masks.append([False])
            continue

        cand_ids = [_STOI.get(tok, unk) for tok in cand_toks]

        # Full teacher-forced sequence. x[j] predicts y[j] = full[j+1].
        full = prefix_ids + cand_ids
        x = full[:-1]
        y = full[1:]

        # Candidate targets begin at y[prefix_len - 1].
        loss_start = max(0, prefix_len - 1)

        if len(x) > _CFG.block_size:
            start = len(x) - _CFG.block_size
            x = x[start:]
            y = y[start:]
            loss_start -= start

        mask = [j >= loss_start for j in range(len(y))]
        examples.append((x, y))
        masks.append(mask)
        max_len = max(max_len, len(x))

    xs, ys, ms = [], [], []
    for ex, mask in zip(examples, masks):
        if isinstance(ex, tuple):
            x, y = ex
        else:
            x = ex
            y = [pad] * len(x)
        n = len(x)
        xs.append(x + [pad] * (max_len - n))
        ys.append(y + [pad] * (max_len - n))
        ms.append(mask + [False] * (max_len - n))

    x_t = torch.tensor(xs, dtype=torch.long, device=DEVICE)
    y_t = torch.tensor(ys, dtype=torch.long, device=DEVICE)
    m_t = torch.tensor(ms, dtype=torch.bool, device=DEVICE)

    with torch.inference_mode():
        logits = _MODEL(x_t)
        logp = F.log_softmax(logits, dim=-1)
        tok_lp = logp.gather(-1, y_t.unsqueeze(-1)).squeeze(-1)
        tok_lp = tok_lp.masked_fill(~m_t, 0.0)
        scores = tok_lp.sum(dim=1).detach().cpu().tolist()

    return [float(s) for s in scores]


class CompleterHTTPServer(HTTPServer):
    request_queue_size = 128


class CompleterHandler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:
        if self.path != "/complete":
            self.send_error(404, "POST to /complete")
            return

        try:
            n = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(n).decode("utf-8")

            # Do not strip leading whitespace: an empty prefix is encoded by a leading newline.
            lines = body.rstrip("\n").split("\n")
            if len(lines) < 2:
                self.send_error(400, "expected first line = prefix, subsequent lines = candidate tokens")
                return

            prefix = lines[0]
            candidates = lines[1:]
            t0 = time.time()
            assert _INFER_LOCK is not None
            with _INFER_LOCK:
                scores = _score_candidates(prefix, candidates)
            dt = time.time() - t0

            print(f"complete | prefix_len={len(split_tokens(prefix, _CFG.tokenizer if _CFG else 'ws'))} "
                  f"candidates={len(candidates)} dt={dt:.4f}s", flush=True)

            resp = "\n".join(f"{s:.9g}" for s in scores).encode("utf-8")

            self.send_response(200)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Content-Length", str(len(resp)))
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(resp)
            self.close_connection = True
        except Exception as e:
            print(f"error: {e}", flush=True)
            self.send_error(500, f"internal error: {e}")

    def do_GET(self) -> None:
        if self.path not in ("/", "/complete"):
            self.send_error(404, "not found")
            return
        msg = (
            "Completer is running.\n\n"
            "POST /complete with body:\n"
            "<prefix tokens>\\n<candidate1>\\n<candidate2>...\n\n"
            "Returns one log-probability per candidate, aligned by line.\n"
        )
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.end_headers()
        self.wfile.write(msg.encode("utf-8"))


def serve(args: argparse.Namespace) -> None:
    global _MODEL, _CFG, _STOI, _INFER_LOCK
    import threading

    torch.set_grad_enabled(False)
    _MODEL, _CFG, _STOI, _itos = load_ckpt(args.ckpt)
    _INFER_LOCK = threading.Lock()
    print(
        json.dumps(
            {
                "device": str(DEVICE),
                "ckpt": args.ckpt,
                "port": args.port,
                "tokenizer": _CFG.tokenizer,
                "vocab_size": _CFG.vocab_size,
                "block_size": _CFG.block_size,
            },
            indent=2,
        ),
        flush=True,
    )
    httpd = CompleterHTTPServer(("", args.port), CompleterHandler)
    httpd.serve_forever()


def probe(args: argparse.Namespace) -> None:
    global _MODEL, _CFG, _STOI
    torch.set_grad_enabled(False)
    _MODEL, _CFG, _STOI, _itos = load_ckpt(args.ckpt)
    scores = _score_candidates(args.prefix, args.candidates)
    for c, s in zip(args.candidates, scores):
        print(f"{s:.9g}\t{c}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser(description="Train or serve a tiny causal next-token completer.")
    p.add_argument("--mode", choices=["train", "serve", "probe"], required=True)

    # Shared-ish
    p.add_argument("--tokenizer", choices=["ws", "char"], default="ws")
    p.add_argument("--seed", type=int, default=0)

    # Train
    p.add_argument("--train", nargs="+", default=["so_ts_markov.txt"], help="training files; every nonblank line is one sequence")
    p.add_argument("--val", nargs="*", default=None, help="optional validation files")
    p.add_argument("--ckpt-out", default="completer.pt")
    p.add_argument("--steps", type=int, default=20_000)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--block-size", type=int, default=128)
    p.add_argument("--dim", type=int, default=256)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--n-layers", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.10)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.10)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--min-count", type=int, default=1)
    p.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--val-every", type=int, default=1000)
    p.add_argument("--val-batches", type=int, default=25)

    # Serve / probe
    p.add_argument("--ckpt", default="completer.pt")
    p.add_argument("--port", type=int, default=8082)
    p.add_argument("--prefix", default="")
    p.add_argument("--candidates", nargs="*", default=[])

    args = p.parse_args()

    if args.mode == "train":
        train(args)
    elif args.mode == "serve":
        serve(args)
    elif args.mode == "probe":
        probe(args)
    else:
        raise ValueError(args.mode)


if __name__ == "__main__":
    main()
