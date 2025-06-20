#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A simple, threaded HTTP server to host the TWO-STAGE reranker pipeline locally.

This server loads a pre-trained transformer encoder and a supervised reranker model.
It receives a query and a list of documents via POST request, generates embeddings
with the encoder, scores them with the reranker, and returns the documents in
ranked order.
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

# --- Model Paths ---
# Make sure these files are in the same directory as the script, or provide full paths.
ENCODER_PATH = "encoder_b200_step_11200.pt"
RERANKER_PATH = "reranker_b200_step_11200.pt"

# --- Inference Settings ---
INFERENCE_BATCH_SIZE = 32  # Batch size for generating document embeddings

# --- Model Hyper-parameters (MUST match training scripts) ---
DIM, N_HEADS, N_LAYERS = 512, 8, 4
VOCAB = 128
# Define max lengths for query and docs separately
MAX_LEN_QUERY = 100
MAX_LEN_DOC = 110
MAX_LEN_ENCODER = MAX_LEN_QUERY + MAX_LEN_DOC

# --- Global variables to hold the loaded models ---
g_encoder = None
g_reranker = None

# Set device
DEVICE = torch.device(
    "mps"  if torch.backends.mps.is_available() else
    "cuda" if torch.cuda.is_available() else "cpu")

# ------------------- Model & Data Utilities (from training scripts) ------------------- #

def encode(txt: str, max_len: int) -> List[int]:
    """Encodes a string into a list of token IDs, padded to max_len."""
    ids = [ord(c) % VOCAB for c in txt[:max_len]]
    return ids + [0] * (max_len - len(ids))

class TransformerForNextTokenPrediction(nn.Module):
    """The Unsupervised Encoder Model Class."""
    def __init__(self, vocab_size=VOCAB, dim=DIM, n_heads=N_HEADS, n_layers=N_LAYERS, max_len=MAX_LEN_ENCODER):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, dim)
        self.pos_emb = nn.Embedding(max_len, dim)
        enc_layer = nn.TransformerEncoderLayer(dim, n_heads, 4*dim, activation="gelu", dropout=0.1, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(enc_layer, n_layers)
        self.head = nn.Linear(dim, vocab_size)

    def forward(self, x):
        N, seq_len = x.shape
        pos = torch.arange(0, seq_len, dtype=torch.long, device=x.device).unsqueeze(0)
        h = self.tok_emb(x) + self.pos_emb(pos)
        hidden_state = self.transformer_encoder(h)
        return hidden_state

class Reranker(nn.Module):
    """
    An enhanced Reranker model that creates explicit interaction features
    (concatenation, difference, and product) from the query and document
    embeddings before passing them to a deeper MLP.
    """
    def __init__(self, input_dim=DIM):
        super().__init__()
        # The input to the network will be 4 times the embedding dimension:
        # [u, v, |u-v|, u*v]
        concat_dim = input_dim * 4

        self.net = nn.Sequential(
            nn.Linear(concat_dim, concat_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(concat_dim // 2, concat_dim // 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(concat_dim // 4, 1)
        )

    def forward(self, q_emb, d_emb):
        # q_emb is [1, DIM], d_emb is [N, DIM]
        q_emb_expanded = q_emb.expand(d_emb.size(0), -1)

        # Create interaction features
        diff = torch.abs(q_emb_expanded - d_emb)
        prod = q_emb_expanded * d_emb

        # Concatenate all features
        combined_features = torch.cat([q_emb_expanded, d_emb, diff, prod], dim=1)

        return self.net(combined_features).squeeze(-1)

@torch.no_grad()
def get_embedding(text: str, max_len: int, encoder: nn.Module) -> torch.Tensor:
    """Generates a fixed-size embedding for a text using the frozen encoder."""
    encoded_text = encode(text, max_len)
    input_tensor = torch.tensor([encoded_text], dtype=torch.long, device=DEVICE)
    hidden_state = encoder(input_tensor)
    # Use the embedding of the first token as the representation ([CLS] style)
    embedding = hidden_state[:, 0, :]
    return embedding

def load_models() -> Tuple[nn.Module, nn.Module]:
    """Loads both the encoder and the reranker models from checkpoint files."""
    print(f"Loading models onto {DEVICE}...")
    try:
        # Load Encoder
        print(f"-> Loading encoder from {ENCODER_PATH}...")
        encoder = TransformerForNextTokenPrediction().to(DEVICE)
        encoder.load_state_dict(torch.load(ENCODER_PATH, map_location=DEVICE))
        encoder.eval()
        print("   Encoder loaded successfully.")

        # Load Reranker
        print(f"-> Loading reranker from {RERANKER_PATH}...")
        reranker = Reranker().to(DEVICE)
        reranker.load_state_dict(torch.load(RERANKER_PATH, map_location=DEVICE))
        reranker.eval()
        print("   Reranker loaded successfully.")

        return encoder, reranker

    except FileNotFoundError as e:
        print(f"ERROR: Model file not found: {e}. Please ensure model files are correctly placed.")
        exit(1)
    except Exception as e:
        print(f"An error occurred while loading models: {e}")
        exit(1)

def rerank_documents(query_text: str, doc_texts: List[str]) -> List[str]:
    """Reranks documents using the two-stage pipeline."""
    if not doc_texts:
        return []

    with torch.no_grad():
        # Stage 1: Generate embeddings using the encoder
        # Get query embedding
        q_emb = get_embedding(query_text, max_len=MAX_LEN_QUERY, encoder=g_encoder)

        # Get document embeddings in batches
        all_doc_embs = []
        for i in range(0, len(doc_texts), INFERENCE_BATCH_SIZE):
            batch_docs = doc_texts[i:i + INFERENCE_BATCH_SIZE]
            d_embs = torch.cat([get_embedding(d, max_len=MAX_LEN_DOC, encoder=g_encoder) for d in batch_docs], dim=0)
            all_doc_embs.append(d_embs)

        if not all_doc_embs:
            return doc_texts

        all_doc_embs_tensor = torch.cat(all_doc_embs, dim=0)

        # Stage 2: Score embeddings with the reranker
        scores = g_reranker(q_emb, all_doc_embs_tensor)

        # Sort original documents based on scores
        sorted_indices = torch.argsort(scores, descending=True)
        ranked_docs = [doc_texts[i] for i in sorted_indices.cpu().tolist()]
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

def main():
    """Main function to load models and start the server."""
    global g_encoder, g_reranker
    g_encoder, g_reranker = load_models()

    server_address = (HOST, PORT)
    httpd = ThreadingHTTPServer(server_address, RerankerHandler)

    print(f"\nServer starting on http://{HOST}:{PORT}")
    print("Send POST requests to /rerank for document ranking.")
    print("Press Ctrl+C to stop.")

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server...")
        httpd.socket.close()

if __name__ == "__main__":
    main()