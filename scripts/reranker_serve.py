#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model serving script for the InteractionRanker.

This script loads a trained reranker model and exposes it via a simple
HTTP server on localhost:8082/rerank.
"""

import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import List, Tuple

import torch
import torch.nn as nn

# --------------------------- Model & Server Configuration -------------------------- #

# Server settings
HOST = "localhost"
PORT = 8082

# Model settings (must match the trained model)
MODEL_PATH = "num_reranker.pt"
BATCH_SIZE = 32  # Batch size for inference to manage memory

# Hyper-parameters from training
DIM, N_HEADS, N_LAYERS = 512, 8, 4
MAX_LEN                = 100
VOCAB                  = 128
MAX_TOK = 1 + MAX_LEN + 1 + MAX_LEN
TAU      = 0.1


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Global variable to hold the loaded model
model = None

# --------------------------- Data & Model Utilities (Copied from training script) -------------------------- #

def encode(txt: str) -> Tuple[List[int], int]:
    """Converts a string to a list of token IDs and its length."""
    ids = [ord(c) % VOCAB for c in txt[:MAX_LEN]]
    return ids + [0] * (MAX_LEN - len(ids)), len(ids)

def to_tensor(strings: List[str]):
    """Converts a list of strings to padded tensors for IDs and lengths."""
    if not strings:
        empty_ids = torch.empty((0, MAX_LEN), dtype=torch.long, device=DEVICE)
        empty_lens = torch.empty((0,), dtype=torch.float, device=DEVICE)
        return empty_ids, empty_lens
    ids, lens = zip(*(encode(s) for s in strings))
    return (torch.tensor(ids, dtype=torch.long, device=DEVICE),
            torch.tensor(lens, dtype=torch.float, device=DEVICE))

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

# --------------------------- Server Logic -------------------------- #

class RerankHandler(BaseHTTPRequestHandler):
    """A custom handler for reranking requests."""

    def do_POST(self):
        if self.path == '/rerank':
            try:
                # 1. Get request body
                content_length = int(self.headers['Content-Length'])
                post_data = self.rfile.read(content_length).decode('utf-8')

                # 2. Parse the input
                lines = [line for line in post_data.split('\n') if line.strip()]
                if len(lines) < 2:
                    self.send_error(400, "Bad Request: Expected at least a query and one document.")
                    return

                query_text = lines[0]
                doc_texts = lines[1:]

                # 3. Get scores from the model
                scores = self.get_scores(query_text, doc_texts)

                # 4. Send the response
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps(scores).encode('utf-8'))

            except Exception as e:
                print(f"Error processing request: {e}")
                self.send_error(500, "Internal Server Error")
        else:
            self.send_error(404, "Not Found")

    @staticmethod
    def get_scores(query_text: str, doc_texts: List[str]) -> List[float]:
        """Uses the loaded model to score documents against a query."""
        global model
        all_scores = []
        with torch.no_grad():
            q_ids, q_len = to_tensor([query_text])

            # Process documents in batches
            for i in range(0, len(doc_texts), BATCH_SIZE):
                batch_docs = doc_texts[i:i+BATCH_SIZE]
                d_ids, d_lens = to_tensor(batch_docs)
                if d_ids.size(0) > 0:
                    scores = model(q_ids, q_len, d_ids, d_lens) / TAU
                    all_scores.extend(scores.cpu().tolist())

        return all_scores

def load_model():
    """Loads the model from disk into the global 'model' variable."""
    global model
    print("Loading model...")
    try:
        model = InteractionRanker().to(DEVICE)
        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)

        model.load_state_dict(checkpoint['model'], strict=True) # Or strict=False to be safe

        model.eval()
        print(f"Model loaded successfully from '{MODEL_PATH}' and is in eval mode.")
    except FileNotFoundError:
        print(f"FATAL: Model file not found at '{MODEL_PATH}'.")
        exit(1)
    except Exception as e:
        print(f"FATAL: An error occurred while loading the model: {e}")
        exit(1)

def run_server():
    """Starts and runs the HTTP server."""
    load_model()

    server_address = (HOST, PORT)
    httpd = HTTPServer(server_address, RerankHandler)

    print(f"\nServer starting on http://{HOST}:{PORT}")
    print("Listening for POST requests to /rerank")
    print("Press Ctrl+C to shut down.")

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass

    httpd.server_close()
    print("\nServer shut down.")

if __name__ == "__main__":
    run_server()