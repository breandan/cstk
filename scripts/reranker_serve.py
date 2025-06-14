#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A simple, robust HTTP server for the InteractionRanker model.
Final Fix: The model is set to .train() mode to re-enable dropout,
which prevents the model's outputs from collapsing during evaluation.
"""

import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import List, Tuple
from socketserver import ThreadingMixIn

import torch
import torch.nn as nn

# --------------------------- Configuration -------------------------- #

# Server settings
HOST = "localhost"
PORT = 8082

# --- Model settings (MUST match the training script) ---
MODEL_PATH             = "num_reranker_markovx5400.pt" # Path to your trained model
BATCH_SIZE             = 32                     # Inference batch size for documents

# Model hyper-parameters
DIM, N_HEADS, N_LAYERS = 512, 8, 4
MAX_LEN                = 100
VOCAB                  = 128
MAX_TOK                = 1 + MAX_LEN + 1 + MAX_LEN
TAU                    = 0.1 # Temperature for score scaling

# Global variable to hold the loaded model
model = None

# Set device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------- Model & Data Utilities (Known Good) ------------------- #
# These are copied directly from the verified training script.

def encode(txt: str) -> Tuple[List[int], int]:
    """Encodes a string into token IDs and its length."""
    ids = [ord(c) % VOCAB for c in txt[:MAX_LEN]]
    return ids + [0]*(MAX_LEN-len(ids)), len(ids)

def to_tensor(strings: List[str]):
    """Converts a list of strings to padded ID and length tensors."""
    if not strings:
        empty_ids = torch.empty((0, MAX_LEN), dtype=torch.long, device=DEVICE)
        empty_lens = torch.empty((0,), dtype=torch.float, device=DEVICE)
        return empty_ids, empty_lens
    ids, lens = zip(*(encode(s) for s in strings))
    return (torch.tensor(ids,  dtype=torch.long,  device=DEVICE),
            torch.tensor(lens, dtype=torch.float, device=DEVICE))

class InteractionRanker(nn.Module):
    """The reranker model class, identical to the training script."""
    def __init__(self):
        super().__init__()
        self.emb  = nn.Embedding(VOCAB, DIM)
        self.pos  = nn.Embedding(MAX_TOK, DIM)
        enc_layer = nn.TransformerEncoderLayer(DIM, N_HEADS, 4*DIM, activation="gelu", dropout=0.1, batch_first=True)
        self.tf   = nn.TransformerEncoder(enc_layer, N_LAYERS)
        self.head = nn.Linear(DIM, 1)

    def forward(self, q_ids, q_len, d_ids, d_lens):
        N = d_ids.size(0)
        if N == 0:
            return torch.empty((0,), device=d_ids.device)

        q_ids_expanded = q_ids.expand(N, -1)
        q_len_expanded = q_len.expand(N)

        cls_token_id = VOCAB - 1
        sep_token_id = VOCAB - 2
        x = torch.cat([
            torch.full((N, 1), cls_token_id, device=DEVICE, dtype=torch.long),
            q_ids_expanded,
            torch.full((N, 1), sep_token_id, device=DEVICE, dtype=torch.long),
            d_ids
        ], dim=1)

        pos_indices = torch.arange(x.size(1), device=DEVICE).expand(N, -1)
        effective_lengths = (1 + q_len_expanded + 1 + d_lens).unsqueeze(1)
        mask = torch.arange(x.size(1), device=DEVICE).expand(N, -1) >= effective_lengths
        h = self.emb(x) + self.pos(pos_indices)
        h = self.tf(h, src_key_padding_mask=mask)
        return self.head(h[:, 0]).squeeze(1)


# --------------------------- Server Implementation -------------------------- #

def load_model():
    """Loads the model into the global variable and sets the correct mode."""
    global model
    print(f"Loading model from {MODEL_PATH} onto {DEVICE}...")
    try:
        model = InteractionRanker().to(DEVICE)
        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
        model.load_state_dict(checkpoint['model'], strict=True)

        # --- THE FIX IS HERE ---
        # For this specific model, we must use .train() mode to keep dropout
        # active, which prevents the scores from collapsing.
        model.train()

        print("Model loaded successfully and is in TRAIN mode.")
    except Exception as e:
        print(f"FATAL: Could not load model. Error: {e}")
        exit(1)

def get_scores(query_text: str, doc_texts: List[str]) -> List[float]:
    """Reranks a list of documents against a single query."""
    all_scores = []
    # Gradients are not needed, even in train() mode for inference.
    with torch.no_grad():
        q_ids, q_len = to_tensor([query_text])

        for i in range(0, len(doc_texts), BATCH_SIZE):
            batch_docs = doc_texts[i:i + BATCH_SIZE]
            d_ids, d_lens = to_tensor(batch_docs)
            if d_ids.size(0) == 0:
                continue

            raw_scores = model(q_ids, q_len, d_ids, d_lens)
            scaled_scores = raw_scores / TAU
            all_scores.extend(scaled_scores.cpu().tolist())

    return all_scores

class RerankHandler(BaseHTTPRequestHandler):
    """A custom HTTP handler for reranking requests."""
    def do_POST(self):
        if self.path == '/rerank':
            try:
                content_length = int(self.headers['Content-Length'])
                post_data = self.rfile.read(content_length).decode('utf-8')
                lines = [line for line in post_data.split('\n') if line.strip()]

                if len(lines) < 2:
                    self.send_error(400, "Bad Request: Input must contain a query on the first line and at least one document on subsequent lines.")
                    return

                query_text = lines[0]
                doc_texts = lines[1:]
                scores = get_scores(query_text, doc_texts)
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps(scores).encode('utf-8'))

            except Exception as e:
                print(f"Error processing request: {e}")
                self.send_error(500, "Internal Server Error")
        else:
            self.send_error(404, "Not Found")

class ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
    """Handle requests in a separate thread."""
    # This allows the server to handle multiple simultaneous connections
    daemon_threads = True

def run_server():
    """Initializes the model and starts the HTTP server."""
    load_model()
    server_address = (HOST, PORT)
    httpd = ThreadingHTTPServer(server_address, RerankHandler)

    print(f"\nServer starting on http://{HOST}:{PORT}")
    print("Send POST requests to /rerank to get scores.")
    print("Press Ctrl+C to shut down.")

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass

    httpd.server_close()
    print("\nServer shut down.")

if __name__ == "__main__":
    run_server()