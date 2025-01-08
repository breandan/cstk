import os
import sys
import torch
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs, unquote

# Import everything from makemore (assuming makemore.py is in the same directory)
# Alternatively, you might integrate this into the makemore script itself.
from makemore import ModelConfig, Transformer, create_datasets, generate


# Load datasets
model_name = 'synt_fixes'
train_dataset, test_dataset = create_datasets(f'{model_name}.txt')
vocab_size = train_dataset.get_vocab_size()
print(f"vocab size: {vocab_size}")
block_size = train_dataset.get_output_length()

# ------------------------------------------------------------
# Load model and datasets (adjust paths and args as needed)
# ------------------------------------------------------------
class Args:
    device = 'mps'                  # or 'cuda' if you have a GPU
    n_layer = 3
    n_head = 3
    n_embd = 90
    top_k = vocab_size
args = Args()

# Load model
config = ModelConfig(vocab_size=vocab_size + 4, # + 4 due to |} and ` ` for contextual repair
                     block_size=block_size,
                     n_layer=args.n_layer, n_head=args.n_head,
                     n_embd=args.n_embd)
model = Transformer(config)
model.to(args.device)

model_path = f'{model_name}_{args.n_embd}_{args.n_layer}_{args.n_head}.pt'
if not os.path.exists(model_path):
    print(f"Model not found at {model_path}. Please train and place {model_name}.pt in the scripts directory.")
    sys.exit(1)

model.load_state_dict(torch.load(model_path, map_location=args.device))
model.eval()

# ------------------------------------------------------------
# Define a request handler for our simple HTTP server
# ------------------------------------------------------------
class MakeMoreHandler(BaseHTTPRequestHandler):

    def do_GET(self):
        # Parse the requested URL
        url = urlparse(self.path)
        if url.path == '/makemore':
            # Parse query parameters
            query = parse_qs(url.query)
            if 'next' in query:
                input_text = query['next'][0]
                input_text = unquote(input_text)  # URL decode
            else:
                input_text = ''

            print(input_text)
            # Encode the input text into indices
            # NOTE: The dataset expects a sequence of single-character tokens.
            #       Ensure that the input_text is in the format expected by your dataset.
            #       For example, if your dataset is character-based, just pass the string.

            # Encode the input sequence
            # The dataset encodes characters to integers.
            # If input_text is empty or doesn't match expected format, handle gracefully.
            if len(input_text) > 0:
                idx = train_dataset.encode(input_text).unsqueeze(0).to(args.device)
            else:
                # If empty, start from the start token (0)
                idx = torch.zeros((1,1), dtype=torch.long, device=args.device)

            # Forward the model with the current idx to get the logits for the next token
            logits, _ = model(idx)
            logits = logits[:, -1, :]  # Get the last timestep's logits

            # If you're using top_k filtering:
            top_k = args.top_k if args.top_k != -1 else None
            if top_k is not None:
                v, _ = torch.topk(logits, top_k, dim=-1)
                logits[logits < v[:, [-1]]] = float('-inf')

            # Convert logits to probabilities
            probs = torch.nn.functional.softmax(logits, dim=-1)

            # Now extract the top n suggestions
            values, indices = torch.topk(probs, k=args.top_k, dim=-1)
            top_n_token_ids = indices[0].tolist()
            top_n_probs = values[0].tolist()

            # Decode these token IDs into characters
            top_n_chars = [
                train_dataset.decode([token_id])
                for token_id in top_n_token_ids
                if 0 < token_id <= len(train_dataset.itos)
            ]

            lines = []
            for ch, p in zip(top_n_chars, top_n_probs):
                # Align the token and print the probability
                # Adjust formatting as needed; here we use a fixed width and 4 decimal places.
                lines.append(f"{ch} {p:.4f}")

            # Join them or handle them as needed
            next_chars = '\n'.join(lines)

            # Send response
            self.send_response(200)
            self.send_header("Content-type", "text/plain; charset=utf-8")
            self.end_headers()
            self.wfile.write(next_chars.encode('utf-8'))
        else:
            # Not found
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b"Not found.")

# ------------------------------------------------------------
# Run the server
# ------------------------------------------------------------
if __name__ == '__main__':
    server_address = ('', 8000)
    httpd = ThreadingHTTPServer(server_address, MakeMoreHandler)
    print("Serving on http://localhost:8000")
    httpd.serve_forever()
