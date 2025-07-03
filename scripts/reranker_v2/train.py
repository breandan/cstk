#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
  train.py
  --------
  • --mode pretrain : next-token + contrastive objective (unsupervised)
  • --mode rerank   : end-to-end fine-tuning with hard-negative reranking
  • --mode serve    : serve the reranker via HTTP API
  All modes share the same encoder, tokenizer, and Levenshtein utilities.
"""
import argparse, random, itertools, time
from pathlib import Path
from typing import List, Tuple
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn
import modal

import torch, torch.nn as nn, torch.nn.functional as F

# -------------------------------------------------------------------- #
#  Shared constants / tokeniser
# -------------------------------------------------------------------- #
DIM, N_HEADS, N_LAYERS = 512, 8, 4
MAX_LEN_Q, MAX_LEN_D   = 100, 110
VOCAB                  = 94                # ASCII 33–126
CHAR_TO_ID             = {chr(i): i - 33 for i in range(33, 127)}
ID_TO_CHAR             = {v: k for k, v in CHAR_TO_ID.items()}
CLS_Q, CLS_D           = CHAR_TO_ID['{'], CHAR_TO_ID['|']           # 90, 91
MAX_LEN                = MAX_LEN_Q + MAX_LEN_D + 2                  # { q | d
NUM_LA_TYPES           = 4                                          # 0=match,
DEVICE                 = torch.device(
    "mps" if torch.backends.mps.is_available() else
    "cuda" if torch.cuda.is_available() else "cpu"
)
CKPT_DIR               = "/ckpts"
MODEL_NAME = "lev_rerank"
BATCH_SIZE = 8
NEG_SAMP = 99

# -------------------------------------------------------------------- #
#  Helpers
# -------------------------------------------------------------------- #
def encode(txt: str, max_len: int) -> List[int]:
    ids = [CHAR_TO_ID.get(c, 0) for c in txt[:max_len]]
    return ids + [0] * (max_len - len(ids))

def lev_align(a: List[str], b: List[str]) -> List[int]:
    m, n = len(a), len(b)
    dp = [[0]*(n+1) for _ in range(m+1)]
    for i in range(m+1): dp[i][0] = i
    for j in range(n+1): dp[0][j] = j
    for i in range(1, m+1):
        for j in range(1, n+1):
            cost_sub = 0 if a[i-1] == b[j-1] else 2
            dp[i][j] = min(dp[i-1][j]+1, dp[i][j-1]+1, dp[i-1][j-1]+cost_sub)
    la, i, j = [0]*n, m, n
    while j or i:
        if i and j and dp[i][j] == dp[i-1][j-1] + (0 if a[i-1]==b[j-1] else 2):
            la[j-1] = 0 if a[i-1]==b[j-1] else 2; i, j = i-1, j-1
        elif j and (i==0 or dp[i][j] == dp[i][j-1]+1):
            la[j-1] = 1; j -= 1
        else:
            i -= 1
    return la

def build_pair(q: str, d: str) -> Tuple[List[int], List[int]]:
    """Return input_ids & LA ids aligned to same length (MAX_LEN)."""
    q_ids, d_ids = encode(q, MAX_LEN_Q), encode(d, MAX_LEN_D)
    la = lev_align(list(q[:MAX_LEN_Q]), list(d[:MAX_LEN_D]))
    la_full = [0]*(MAX_LEN_Q+2) + la + [0]*(MAX_LEN_D - len(la))
    return [CLS_Q]+q_ids+[CLS_D]+d_ids, la_full

# -------------------------------------------------------------------- #
#  Encoder (unchanged logic, now the project’s single source-of-truth)
# -------------------------------------------------------------------- #
class TxEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok_emb = nn.Embedding(VOCAB, DIM)
        self.pos_emb = nn.Embedding(MAX_LEN, DIM)
        self.la_emb  = nn.Embedding(NUM_LA_TYPES, DIM)
        enc = nn.TransformerEncoderLayer(DIM, N_HEADS, 4*DIM, dropout=0.1, activation="gelu", batch_first=True)
        self.tx = nn.TransformerEncoder(enc, N_LAYERS)
        self.head = nn.Linear(DIM, VOCAB)

    def forward(self, x: torch.Tensor, la: torch.Tensor, return_logits=True):
        pos = torch.arange(0, x.size(1), device=x.device).unsqueeze(0)
        h   = self.tok_emb(x) + self.pos_emb(pos) + self.la_emb(la)
        h   = self.tx(h)                                           # [B,S,D]
        if return_logits:
            logits = self.head(h)
            e_q = h[:, 0, :]                 # embedding at '{'
            e_d = h[:, MAX_LEN_Q+1, :]       # embedding at '|'
            return logits, e_q, e_d
        return h

# -------------------------------------------------------------------- #
#  Data readers (unchanged)
# -------------------------------------------------------------------- #
def stream_qd(path: str):
    """Yield (query, docs) indefinitely."""
    while True:
        with Path(path).open(encoding='utf-8') as f:
            for gap, grp in itertools.groupby(f, key=lambda l: not l.strip()):
                if not gap:
                    lines = [l.rstrip('\n') for l in grp]
                    if lines:
                        yield lines[0], lines[1:]

# -------------------------------------------------------------------- #
#  Mode 1 – unsupervised pre-training (next-token + contrastive)
# -------------------------------------------------------------------- #
def pretrain(steps=40_000, tr="so_ts_markov.txt", vs="so_vs_markov.txt", VAL_EVERY=1_000, VAL_BATCHES=10):
    print(f"⚙️  pre-training on {DEVICE}")

    model = TxEncoder().to(DEVICE)
    opt   = torch.optim.AdamW(model.parameters(), lr=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    tr_gen, vs_gen = stream_qd(tr), stream_qd(vs)

    def get_batch(gen, K=8, M=3):
        x_lst, la_lst = [], []
        for _ in range(K):
            q, docs = next(gen)
            docs = docs[:M+1] if len(docs) > M else docs
            for d in docs:
                x_ids, la_ids = build_pair(q, d)
                x_lst.append(x_ids); la_lst.append(la_ids)
        x  = torch.tensor(x_lst, device=DEVICE)
        la = torch.tensor(la_lst, device=DEVICE)
        tgt = x[:, 1:].contiguous()
        return x[:, :-1], la[:, :-1], tgt            # teacher-forced

    timer = time.time()
    for step in range(1, steps + 1):
        # ---------------- Training step ---------------- #
        model.train(); opt.zero_grad()
        x, la, tgt = get_batch(tr_gen)
        logits, e_q, e_d = model(x, la)
        lm_loss  = loss_fn(logits.view(-1, VOCAB), tgt.view(-1))
        sim      = (e_q * e_d).sum(1).view(-1, 4)          # 1 pos + 3 neg
        ctr_loss = F.cross_entropy(sim, torch.zeros(sim.size(0), dtype=torch.long, device=DEVICE))
        (lm_loss + ctr_loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step % 100 == 0:
            dt = time.time() - timer; timer = time.time()
            print(f"step {step:>6} | lm {lm_loss:.3f} | ctr {ctr_loss:.3f} "
                  f"| Δt {dt:.1f}s")

        # ---------------- Validation ------------------- #
        if step % VAL_EVERY == 0:
            model.eval()
            lm_val_tot, ctr_val_tot, n = 0.0, 0.0, 0
            with torch.no_grad():
                for _ in range(VAL_BATCHES):
                    x, la, tgt = get_batch(vs_gen)
                    logits, e_q, e_d = model(x, la)
                    lm_val_tot  += loss_fn(logits.view(-1, VOCAB), tgt.view(-1)).item()
                    sim          = (e_q * e_d).sum(1).view(-1, 4)
                    ctr_val_tot += F.cross_entropy(sim, torch.zeros(sim.size(0), dtype=torch.long, device=DEVICE)).item()
                    n += 1
            print(f"🧪  val @ {step} | lm {lm_val_tot/n:.3f} "
                  f"| ctr {ctr_val_tot/n:.3f}")

            # optional: save checkpoint at every validation
            torch.save(model.state_dict(), "unsupervised_encoder.pt")

# -------------------------------------------------------------------- #
#  Mode 2 – supervised reranking
# -------------------------------------------------------------------- #
class Reranker(nn.Module):
    def __init__(self):
        super().__init__()
        self.cls = nn.Parameter(torch.randn(1,1,DIM))
        self.pos = nn.Parameter(torch.randn(3,DIM))
        enc = nn.TransformerEncoderLayer(DIM,8,4*DIM,dropout=0.1,activation="gelu", batch_first=True)
        self.tx  = nn.TransformerEncoder(enc, 4)
        self.sc  = nn.Linear(DIM,1)
        # self.tau = nn.Parameter(torch.tensor(0.05))

    def forward(self, q_embs, d_embs):
        # q_embs = F.normalize(q_embs, dim=-1)
        # d_embs = F.normalize(d_embs, dim=-1)
        B, N, D = d_embs.size()
        cls = self.cls.expand(B, N, 1, D)
        q = q_embs.unsqueeze(1).unsqueeze(1).expand(B, N, 1, D)
        d = d_embs.unsqueeze(2)
        seq = torch.cat([cls, q, d], dim=2)  # [B, N, 3, D]
        seq = seq + self.pos.unsqueeze(0).unsqueeze(0)
        seq = seq.view(B*N, 3, D)
        out = self.tx(seq)[:, 0, :]  # [B*N, D]
        out = out.view(B, N, D)
        scores = self.sc(out).squeeze(-1)  # [B, N]
        return scores

def rerank(steps=10_000, ckpt_volume: modal.Volume = None, ckpt="/data/unsupervised_encoder.pt", tr="so_ts_markov.txt", vs="so_vs_markov.txt"):
    enc = TxEncoder().to(DEVICE)
    enc.load_state_dict(torch.load(ckpt, map_location=DEVICE, weights_only=True))
    rer = Reranker().to(DEVICE)

    opt = torch.optim.AdamW([
        {'params': enc.parameters(), 'lr':1e-5},
        {'params': rer.parameters(), 'lr':1e-4}
    ])

    def embed_pair_batch(q: str, docs: List[str]):
        x_lst, la_lst = zip(*(build_pair(q,d) for d in docs))
        x = torch.tensor(x_lst, device=DEVICE)
        la= torch.tensor(la_lst, device=DEVICE)
        _, e_q, e_d = enc(x, la)                   # shared q across batch
        return e_q[0:1], e_d                       # [1,D], [N,D]

    tr_gen = stream_qd(tr)
    vs_full = list(itertools.islice(stream_qd(vs),1000))
    vs_data = random.sample(vs_full, 100)
    N = NEG_SAMP + 1  # 1 positive + K negatives

    # --------------------  debug helper ------------------------------ #
    def debug_show_examples(top_k: int = 5):
        print("\n\033[1m=== DEBUG EXAMPLES ===================================\033[0m")
        for q, docs in random.sample(vs_data, 3):
            # forward pass
            q_emb, d_embs = embed_pair_batch(q, docs)
            scores = rer(q_emb, d_embs.unsqueeze(0))[0]
            order  = scores.argsort(descending=True).tolist()
            true_rank = order.index(0) + 1
            print(
                f"\n\033[1mQuery:\033[0m {q}\n"
                f"   ↳ true-doc rank: {true_rank}/{len(docs)}\n"
            )

            # top-k list
            for rank, idx in enumerate(order[:top_k], 1):
                doc      = docs[idx]
                star     = " ★" if idx == 0 else ""
                _, la    = build_pair(q, doc)
                la_slice = la[MAX_LEN_Q + 2 : MAX_LEN_Q + 2 + len(doc)]
                coloured = color_alignment(q, doc, la_slice)
                print(f"{rank:2}. {coloured}{star}")
        print("\033[1m========================================================\033[0m\n")


    for step in range(1, steps+1):
        enc.train(); rer.train(); opt.zero_grad()

        queries = [next(tr_gen) for _ in range(BATCH_SIZE)]
        all_pairs = []
        for q, docs in queries:
            if len(docs) <= 1:
                # Skip queries with no negatives
                continue
            neg = random.sample(docs[1:], min(NEG_SAMP, len(docs) - 1))
            if len(neg) < NEG_SAMP:
                # Sample with replacement if fewer than K negatives
                neg += random.choices(docs[1:], k=NEG_SAMP - len(neg))
            docs_batch = [docs[0]] + neg
            pairs = [build_pair(q, d) for d in docs_batch]
            all_pairs.extend(pairs)

        if not all_pairs:
            continue  # Skip empty batch

        x = torch.tensor([p[0] for p in all_pairs], device=DEVICE)
        la = torch.tensor([p[1] for p in all_pairs], device=DEVICE)
        _, e_q, e_d = enc(x, la)

        # Reshape embeddings for batch processing
        valid_q = [qd for qd in queries if len(qd[1]) > 1]   # keep only used
        B_actual = len(valid_q)

        e_q = e_q.contiguous()
        e_d = e_d.contiguous()

        q_embs = e_q[::N][:B_actual]                # [B_actual, D]
        d_embs = e_d.reshape(B_actual, N, -1)       # [B_actual, N, D]

        scores = rer(q_embs, d_embs) / 0.1
        tgt = torch.zeros(B_actual, dtype=torch.long, device=DEVICE)  # Positive at index 0
        loss = F.cross_entropy(scores, tgt)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(enc.parameters(),1.)
        torch.nn.utils.clip_grad_norm_(rer.parameters(),1.)
        opt.step()

        if step % 50 == 0:
            print(f"step {step:>5} | loss {loss.item():.3f}")

        if step % 500 == 0:
            enc.eval(); rer.eval()
            rr = []
            with torch.no_grad():
                for vq, vdocs in vs_data:
                    q_emb, d_embs = embed_pair_batch(vq, vdocs)
                    scores = rer(q_emb, d_embs.unsqueeze(0))[0]
                    rank   = scores.argsort(descending=True).tolist().index(0) + 1
                    rr.append(1.0 / rank)
            print(f"MRR@{len(vs_data)} = {sum(rr)/len(rr):.4f}")
            debug_show_examples()
            if ckpt_volume is not None:
                print(f"--- Saving models to checkpoint volume ---")
                torch.save(enc.state_dict(), f"{CKPT_DIR}/{MODEL_NAME}_step_{step}.pt")
                torch.save(rer.state_dict(), f"{CKPT_DIR}/encoder_{MODEL_NAME}_step_{step}.pt")
                ckpt_volume.commit()
            else:
                torch.save(enc.state_dict(), "encoder_finetuned.pt")
                torch.save(rer.state_dict(), "reranker.pt")

ANSI  = {0:"\033[0m",         # reset / match
         1:"\033[32m",        # green  (insertion)
         2:"\033[33m",        # yellow (substitution)
         3:"\033[31m"}        # red    (query char deleted – we show next)

def color_alignment(q: str, d: str, la: List[int]) -> str:
    """Return coloured doc string according to its LA vector."""
    out = []
    for ch, t in zip(d, la):
        out.append(f"{ANSI[t]}{ch}")
    out.append("\033[0m")         # reset
    return "".join(out)

# -------------------------------------------------------------------- #
#  Mode 3 – serving the reranker
# -------------------------------------------------------------------- #
# Global models for serving
enc = None
rer = None

def rerank_documents(query: str, documents: List[str]) -> List[str]:
    """Rerank documents based on query using pre-trained encoder and reranker."""
    global enc, rer
    if not documents:
        return []
    x_lst, la_lst = zip(*(build_pair(query, d) for d in documents))
    x = torch.tensor(x_lst, device=DEVICE)
    la = torch.tensor(la_lst, device=DEVICE)
    _, e_q, e_d = enc(x, la)
    q_emb = e_q[0:1]  # [1, D], using the first query embedding (all are same)
    d_embs = e_d      # [N, D]
    scores = rer(q_emb, d_embs.unsqueeze(0))[0]  # [N]
    sorted_indices = scores.argsort(descending=True).tolist()
    ranked_docs = [documents[i] for i in sorted_indices]
    return ranked_docs

class ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
    """Handle requests in a separate thread."""
    daemon_threads = True

class RerankerHandler(BaseHTTPRequestHandler):
    """Request handler for the reranking service."""
    def do_POST(self):
        if self.path != '/rerank':
            self.send_error(404, "Not Found: Please POST to /rerank")
            return

        try:
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length).decode('utf-8')
            lines = post_data.strip().split('\n')

            if len(lines) < 2:
                self.send_error(400, "Bad Request: Expected query on the first line and documents on subsequent lines.")
                return

            query = lines[0]
            documents = lines[1:]

            start_time = time.time()
            ranked_documents = rerank_documents(query, documents)
            end_time = time.time()

            print(f"Reranked {len(documents)} docs for query '{query[:50]}...' in {end_time - start_time:.4f}s.")

            self.send_response(200)
            self.send_header('Content-Type', 'text/plain; charset=utf-8')
            self.end_headers()
            self.wfile.write('\n'.join(ranked_documents).encode('utf-8'))

        except Exception as e:
            self.send_error(500, f"Internal Server Error: {e}")
            print(f"Error processing request: {e}")

    def do_GET(self):
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.end_headers()
            message = """
            <html><head><title>Two-Stage Reranker Service</title></head><body>
                <h1>Two-Stage Reranker Service is Running</h1>
                <p>To use, send a <strong>POST</strong> request to <code>/rerank</code>.</p>
                <p>The request body must be plain text with the query on the first line and documents on subsequent lines.</p>
                <p><b>Example using curl:</b></p>
                <pre><code>QUERY="how to parse json in python"
DOC_1="Parsing JSON is a common task."
DOC_2="Python's json module is standard."
DOC_3="XML is a different data format."

printf "%s\\n%s\\n%s\\n%s" "$QUERY" "$DOC_1" "$DOC_2" "$DOC_3" | curl -X POST --data-binary @- http://localhost:8082/rerank</code></pre>
            </body></html>
            """
            self.wfile.write(message.encode('utf-8'))
        else:
            self.send_error(404, "Not Found")

def serve():
    """Start the HTTP server to serve the reranker."""
    global enc, rer
    enc = TxEncoder().to(DEVICE)
    enc.load_state_dict(torch.load("encoder_finetuned.pt", map_location=DEVICE, weights_only=True))
    rer = Reranker().to(DEVICE)
    rer.load_state_dict(torch.load("reranker.pt", map_location=DEVICE, weights_only=True))

    server_address = ('', 8082)
    httpd = ThreadingHTTPServer(server_address, RerankerHandler)
    print(f"Serving on port 8082...")
    httpd.serve_forever()

# -------------------------------------------------------------------- #
#  Entrypoint
# -------------------------------------------------------------------- #
if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Train or serve a two-stage reranker.")
    p.add_argument("--mode", choices=["pretrain", "rerank", "serve"], required=True, help="Mode: pretrain, rerank, or serve")
    p.add_argument("--steps", type=int, default=40000, help="Number of training steps")
    args = p.parse_args()
    torch.manual_seed(0)
    if args.mode == "pretrain":
        pretrain(steps=args.steps)
    elif args.mode == "rerank":
        rerank(steps=args.steps)
    elif args.mode == "serve":
        serve()