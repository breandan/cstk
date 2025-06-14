#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A simple, threaded HTTP server to host the InteractionRanker model for reranking documents.
Receives a query and a list of documents, and returns the documents in ranked order.
"""
import json
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn
from typing import List, Tuple

import torch
import torch.nn as nn

# --------------------------- Configuration -------------------------- #

# Server settings
HOST = "localhost"
PORT = 8082

# --- Model settings (MUST match the training script) ---
MODEL_PATH = "num_reranker_markovx5400.pt"
BATCH_SIZE = 32  # Inference batch size for documents

# Model hyper-parameters
DIM, N_HEADS, N_LAYERS = 512, 8, 4
MAX_LEN = 100
VOCAB = 128
MAX_TOK = 1 + MAX_LEN + 1 + MAX_LEN

# Global variable to hold the loaded model
model = None

# Set device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------- Model & Data Utilities (from training script) ------------------- #

def encode(txt: str) -> Tuple[List[int], int]:
    """Encodes a string into token IDs and its length."""
    ids = [ord(c) % VOCAB for c in txt[:MAX_LEN]]
    return ids + [0] * (MAX_LEN - len(ids)), len(ids)

def to_tensor(strings: List[str]):
    """Converts a list of strings to padded ID and length tensors."""
    if not strings:
        empty_ids = torch.empty((0, MAX_LEN), dtype=torch.long, device=DEVICE)
        empty_lens = torch.empty((0,), dtype=torch.float, device=DEVICE)
        return empty_ids, empty_lens
    ids, lens = zip(*(encode(s) for s in strings))
    return (torch.tensor(ids, dtype=torch.long, device=DEVICE),
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

def load_model(path: str) -> nn.Module:
    """Loads the trained model from a checkpoint file."""
    print(f"Loading model from {path} onto {DEVICE}...")
    try:
        checkpoint = torch.load(path, map_location=DEVICE)
        m = InteractionRanker().to(DEVICE)
        m.load_state_dict(checkpoint['model'])
        m.eval()  # Set model to evaluation mode
        print("Model loaded successfully.")
        return m
    except FileNotFoundError:
        print(f"ERROR: Model file not found at '{path}'. Please check the MODEL_PATH.")
        exit(1)
    except Exception as e:
        print(f"An error occurred while loading the model: {e}")
        exit(1)

def rerank_documents(query_text: str, doc_texts: List[str]) -> List[str]:
    """Reranks documents for a given query using the loaded model."""
    if not doc_texts:
        return []

    q_ids, q_len = to_tensor([query_text])
    all_scores = []

    with torch.no_grad():
        for i in range(0, len(doc_texts), BATCH_SIZE):
            batch_docs = doc_texts[i:i + BATCH_SIZE]
            if not batch_docs:
                continue

            d_ids, d_lens = to_tensor(batch_docs)
            scores = model(q_ids, q_len, d_ids, d_lens)
            all_scores.append(scores)

    if not all_scores:
        return doc_texts # Return original if no scores were generated

    # Combine scores and get the final ranking
    all_scores_tensor = torch.cat(all_scores)
    sorted_indices = torch.argsort(all_scores_tensor, descending=True)

    # Reorder the original documents based on the sorted indices
    ranked_docs = [doc_texts[i] for i in sorted_indices.cpu().tolist()]
    return ranked_docs

class ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
    """Handle requests in a separate thread."""
    daemon_threads = True


class RerankerHandler(BaseHTTPRequestHandler):
    """
    Request handler for the reranking service.
    Only handles POST requests to /rerank.
    """
    def do_POST(self):
        if self.path != '/rerank':
            self.send_error(404, "Not Found: Please post to /rerank")
            return

        try:
            # Read and parse the request body
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length).decode('utf-8')
            lines = post_data.strip().split('\n')

            if not lines or len(lines) < 2:
                self.send_error(400, "Bad Request: Expected format is a query on the first line and documents on subsequent lines.")
                return

            query = lines[0]
            documents = lines[1:]

            # Perform the reranking
            start_time = time.time()
            ranked_documents = rerank_documents(query, documents)
            end_time = time.time()

            print(f"Reranked {len(documents)} documents for query '{query[:50]}...' in {end_time - start_time:.4f} seconds.")

            # Send the response as raw text, one document per line
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
            self.send_header('Content-Type', 'text/html')
            self.end_headers()
            message = """
            <html>
                <head><title>Reranker Service</title></head>
                <body>
                    <h1>Reranker Model Service is Running</h1>
                    <p>This server provides a reranking service for a trained transformer model.</p>
                    <p>To use it, send a <strong>POST</strong> request to the <code>/rerank</code> endpoint.</p>
                    <p>The request body should be plain text with the query on the first line and each candidate document on a new line.</p>
                    <p>The response will be the documents in ranked order, separated by newlines.</p>
                    <p>Example using <code>curl</code>:</p>
                    <pre><code>curl -s -X POST -H 'Content-Type: text/plain' --data-binary @/tmp/foo.txt http://localhost:8082/rerank</code></pre>
                </body>
            </html>
            """
            self.wfile.write(message.encode('utf-8'))
        else:
            self.send_error(404, "Not Found")

def main():
    """Main function to load the model and start the server."""
    global model
    model = load_model(MODEL_PATH)

    server_address = (HOST, PORT)
    httpd = ThreadingHTTPServer(server_address, RerankerHandler)

    print(f"Starting server on http://{HOST}:{PORT}")
    print("Send POST requests to /rerank to get document rankings.")
    print("Press Ctrl+C to stop the server.")

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server...")
        httpd.socket.close()

if __name__ == "__main__":
    main()